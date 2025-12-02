import os
import json
import argparse
import re
import yaml
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from tqdm import tqdm
import logging
import warnings
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", message="generation flags.*", category=UserWarning)
warnings.filterwarnings("ignore", message="The following generation flags are not valid.*", category=UserWarning)

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

for logger_name in [
    "transformers",
    "transformers.generation",
    "transformers.generation.utils",
    "transformers.generation.configuration_utils",
]:
    tlogger = logging.getLogger(logger_name)
    tlogger.setLevel(logging.ERROR)
    tlogger.propagate = False

from utils import calculate_metrics
from e5_retriever import E5Retriever
from bm25_retriever import BM25Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopOnStringCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, initial_length):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings if isinstance(stop_strings, list) else [stop_strings]
        self.initial_length = initial_length

    def __call__(self, input_ids, scores):
        # Check if any of the stop strings are in newly generated tokens
        new_tokens = input_ids[0][self.initial_length:]
        decoded_new_tokens = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return any(stop_string in decoded_new_tokens for stop_string in self.stop_strings)

@dataclass
class InferenceConfig:
    # Model settings
    reasoner_model_name: str = "Qwen/Qwen3-32B"
    teacher_model_name: str = "Qwen/Qwen3-32B"
    summarizer_model_name: str = "Qwen/Qwen3-32B"
    reasoner_lora_path: Optional[str] = None  # Path to LoRA adapters for reasoner
    summarizer_lora_path: Optional[str] = None  # Path to LoRA adapters for summarizer
    retriever_type: str = "bm25"  # or "e5"
    retriever_index_path: str = "indexes/bm25"
    e5_model_path: str = "intfloat/e5-large-v2"
    # Generation settings
    max_turns: int = 5
    max_new_tokens: int = 2048
    greedy_thinking: bool = False  # Use greedy decoding in thinking mode
    high_randomness_mode: bool = False  # Use more aggressive random generation for diverse trajectories
    # Retrieval settings
    top_k_docs: int = 10
    # Dataset settings
    dataset_name: str = "hotpotqa"
    split: str = "dev"  # train or dev
    max_samples: Optional[int] = None
    output_dir: str = "output/search_o1"
    save_intermediate: bool = False
    start_sample: Optional[int] = None  # 1-based inclusive
    end_sample: Optional[int] = None    # 1-based inclusive
    # Probe settings
    use_probe: bool = False
    probe_path: str = "probe/"
    probe_confidence_threshold: float = 0.7
    # Training settings
    batch_size: int = 8
    num_rollouts: int = 8
    train_micro_batch_size: int = 1
    num_retrieval_threads: int = 8

class StopOnStringCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, initial_length):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings if isinstance(stop_strings, list) else [stop_strings]
        self.initial_length = initial_length

    def __call__(self, input_ids, scores):
        matches = []
        for i in range(input_ids.shape[0]):
            new_tokens = input_ids[i][self.initial_length:]
            decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            row_match = any(stop in decoded_text for stop in self.stop_strings)
            matches.append(row_match)
        return all(matches)

class Reasoner:
    """The main reasoning model that decides when to search and provides answers."""
    def __init__(self, reasoner_model_name: str, teacher_model_name: str, config: InferenceConfig):
        self.student_model_name = reasoner_model_name
        self.teacher_model_name = teacher_model_name
        self.config = config
        self.student_tokenizer = None
        self.teacher_tokenizer = None
        self.student_model = None
        self.teacher_model = None

        # Training metrics tracking
        self.training_step = 0
        self.training_metrics = {
            "loss": [],
            "kl_divergence": [],
        }

        self._load_model()
        self._load_prompt_template()
    
    def _load_model(self):
        """Load the Qwen3 model with recommended settings."""
        logger.info(f"Loading reasoner model: {self.student_model_name}")
        
        self.student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.student_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load LoRA adapters if specified
        if self.config.reasoner_lora_path:
            logger.info(f"Loading LoRA adapters from: {self.config.reasoner_lora_path}")
            self.student_model = PeftModel.from_pretrained(self.student_model, self.config.reasoner_lora_path)
            logger.info("LoRA adapters loaded successfully")
        
        logger.info("Student model loaded successfully")

        logger.info(f"Loading teacher model: {self.teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        logger.info("Teacher model loaded successfully")

        self.optimizer = AdamW(self.student_model.parameters(), lr=1e-6)

        target_words = [
            "<search>"
            "</search>"
            "<answer>"
            "</answer>"
        ]
        
        self.format_token_ids = set()
        for word in target_words:
            ids = self.student_tokenizer(word, add_special_tokens=False)['input_ids']
            if isinstance(ids, list):
                self.format_token_ids.update(ids)
            else:
                self.format_token_ids.add(ids)
        logger.info(f"Format enforcement enabled on {len(self.format_token_ids)} tokens")

        self.accumulation_steps = 4

    def _load_prompt_template(self):
        """Load the prompt template for the reasoner."""
        prompt_path = "Agentic-Rag/prompts/default_QA.yaml"
        with open(prompt_path, 'r') as f:
            prompt_data = yaml.safe_load(f)
        
        self.prompt_template = prompt_data['user_prompt']

    def generate_batch(self, sequences: List[str], current_turn: int, max_turns: int) -> List[str]:
        """
        Only for the inference and sampling the rollouts for training.
        """
        # When generating the rollouts, we need to pad the sequences to the left.
        self.student_tokenizer.padding_side = "left"

        # Encode the batch
        model_inputs = self.student_tokenizer(sequences, return_tensors="pt", padding=True).to(self.student_model.device)
        initial_length = model_inputs['input_ids'].shape[1]

        # Setup Stopping Criteria (Batch-Aware)
        stopping_criteria = StoppingCriteriaList([
            StopOnStringCriteria(self.student_tokenizer, ["</search>", "</answer>"], initial_length)
        ])

        # Determine Settings based on Turn
        if current_turn == 1:
            # Requirement 1: High Temperature for diversity in the first turn
            # This creates the 8 distinct rollout paths
            temperature = 1.2
            top_p = 0.9
            logger.debug(f"Turn 1: Using High Temperature {temperature} for exploration")
        else:
            # Subsequent turns: Normal thinking mode
            temperature = 0.6
            top_p = 0.95
            logger.debug(f"Turn {current_turn}: Using Normal Temperature {temperature}")

        # Teacher Mixing Logic. In turn one , if we use Teacher, we get Teacher's diversity
        user_teacher_forcing = random.random() < 0.2

        # Construct Generation Arguments
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "output_scores": True,
            "return_dict_in_generate": True,
            "temperature": temperature,  # Dynamic
            "top_p": top_p,              # Dynamic
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "pad_token_id": self.student_tokenizer.eos_token_id,
            "eos_token_id": self.student_tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria
        }

        # Generate the rollouts
        with torch.no_grad():
            if user_teacher_forcing:
                bad_words_ids = None
                if current_turn == max_turns:
                    search_token_ids = self.student_tokenizer.encode("<search>", add_special_tokens=False)
                    if search_token_ids: bad_words_ids = [search_token_ids]
                outputs = self.teacher_model.generate(
                    **model_inputs,
                    **gen_kwargs,
                    bad_words_ids=bad_words_ids,
                )
            else:
                outputs = self.student_model.generate(
                    **model_inputs,
                    **gen_kwargs,
                )

        # Decode Batch
        generated_texts = []
        
        # outputs/sequences contains [Batch_Size, Full_Seq_Len]
        for i in range(len(sequences)):
            # Slice only new tokens
            new_ids = outputs.sequences[i][initial_length:]
            text = self.student_tokenizer.decode(new_ids, skip_special_tokens=True)

            for stop in ['</search>', '</answer>']:
                if stop in text:
                    text = text.split(stop)[0] + stop

            generated_texts.append(text)

        return generated_texts

    def train_on_batch(self, trajectories: List[Dict[str, Any]]) -> None:
        """
        New method to train on a batch of full trajectories
        Accepts a list of full trajectories, use mask to ignore the padding tokens and the tool generated <information>, using gradient accumlation
        """
        # Set the model to training mode
        self.student_model.train()

        # Padding on the right while training
        self.student_tokenizer.padding_side = "right"

        # Randomly suffle the batch
        total_trajectories = len(trajectories)
        micro_batch = self.config.train_micro_batch_size

        total_loss = 0

        # Micro-batch processing
        pbar = tqdm(range(0, total_trajectories, micro_batch), desc="Training Step")
        for i in pbar:
            batch_data = trajectories[i: i + micro_batch]

            sequences = [d['sequence'] for d in batch_data]

            # Get the stored trainable range [(start, end)] for each trajectory
            trainable_spans_batch = [d['trainable_spans'] for d in batch_data]
            
            # Tokenize and get the offset mapping
            encoding = self.student_tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                return_offsets_mapping=True, # Key for masking
            ).to(self.student_model.device)

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            offsets_mapping = encoding['offset_mapping']

            # Core logic, build the mask
            loss_mask = torch.zero_like(input_ids, dtype=torch.float)

            for b_idx in range(len(batch_data)):
                spans = trainable_spans_batch[b_idx] # Current sample's trainable span splits list
                offsets = offsets_mapping[b_idx] # Current every token's offset mapping

                for t_idx, (start_char, end_char) in enumerate(offsets):
                    if start_char == end_char == 0:
                        continue # Skip Spectial Tokens

                    # Check current token is fully contained in any of the trainable spans
                    for (span_start, span_end) in spans:
                        if start_char >= span_start and end_char <= span_end:
                            loss_mask[b_idx, t_idx] = 1.0
                            break

            # Calculating Teacher Logits
            with torch.no_grad():
                teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_probs = F.softmax(teacher_outputs.logits, dim=-1)

            # Calculate Student Logits
            student_outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
            # Use Log Softmax to get the log probabilities distribution
            student_log_probs = F.log_softmax(student_outputs.logits, dim=-1)
            student_probs = torch.exp(student_log_probs)

            # Calculate the KL Divergence Loss
            loss_fct = torch.nn.KLDivLoss(reduction="none")
            kl_loss = loss_fct(teacher_probs, student_log_probs).sum(dim=-1)

            # Shift Masking (Because Causal LM predict the next token)
            # Loss at t correpsonds to predicting token at t + 1
            kl_loss = kl_loss[:, :-1]
            
            #Shift the mask by one token to the right
            shift_mask = loss_mask[:, 1:]
            shift_att_mask = attention_mask[:, 1:]
            
            # Token weighting, format tokens
            token_weights = torch.ones_like(kl_loss)
            shift_input_ids = input_ids[:, 1:]
            for tid in self.format_token_ids:
                token_weights[shift_input_ids == tid] = 5.0

            # Finally, Loss = KL * (Trainable Mask *  Padding Mask) * Weights
            final_mask = shift_mask * shift_att_mask

            if final_mask.sum() > 0:
                weighted_loss = kl_loss * final_mask * token_weights
                loss = weighted_loss.sum() / final_mask.sum()
                entropy = -(student_probs * student_log_probs).sum(dim=-1)
                mean_entropy = (entropy * final_mask).sum() / final_mask.sum()
                loss = loss - 0.01 * mean_entropy
            else:
                loss = torch.tensor(0.0, device=self.student_model.device, requires_grad=True)

            # Gradient Accumulation
            accumulation_factor = total_trajectories / micro_batch
            loss = loss / accumulation_factor

            loss.backward()
            total_loss += loss.item()
        
        # Complete Micro-batch and update the parameters
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        #Log metrics(simplified)
        self._update_training_metrics(total_loss, 0.0)

    def save_model(self):
        self.student_model.save_pretrained(f"{self.config.output_dir}/on_policy_distillation")
        self.student_tokenizer.save_pretrained(f"{self.config.output_dir}/on_policy_distillation")
        # Save final training plots
        self._plot_training_metrics(force_save=True)
        logger.info("Final training plots saved")

    def _create_plots_directory(self):
        """Create the plots directory if it doesn't exist."""
        plots_dir = os.path.join(self.config.output_dir, "training_plots")
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir

    def _update_training_metrics(self, loss, avg_kl):
        """Update training metrics for plotting."""
        loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)
        kl_val = float(avg_kl.item()) if torch.is_tensor(avg_kl) else float(avg_kl)
        
        self.training_metrics['loss'].append(loss_val)
        self.training_metrics['kl_divergence'].append(kl_val)

        self.training_step += 1

    def _plot_training_metrics(self, force_save=False):
        """Create and save training plots."""
        # Only plot every 10 steps or when forced
        if self.training_step % 10 != 0 and not force_save:
            return

        plot_dir = self._create_plots_directory()
        steps = range(1, len(self.training_metrics['loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.training_metrics['loss'], 'b-', linewidth=2, label='Distillation Loss')
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Distillation Loss Over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'distillation_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()

        if len(self.training_metrics['kl_divergence']) > 0:
            plt.figure(figsize=(10, 6))
            kl_data = self.training_metrics['kl_divergence']
            plt.plot(steps, kl_data, 'g-', linewidth=2, label='Raw KL')
            window_size = min(20, len(kl_data))
            if window_size > 1:
                moving_avg = np.convolve(kl_data, np.ones(window_size)/window_size, mode='valid')
                ma_steps = range(window_size, len(kl_data)+1)
                plt.plot(ma_steps, moving_avg, 'g-', linewidth=2, label=f'KL Divergence MA-{window_size}')
            plt.xlabel("Training Step")
            plt.ylabel("KL Divergence")
            plt.title("KL Divergence Over Time (Teacher || Student)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'kl_divergence.png'), dpi=150, bbox_inches='tight')
            plt.close()
        # plots_dir = self._create_plots_directory()

        # # Plot 1: PPO Loss over time
        # plt.figure(figsize=(10, 6))
        # steps = range(1, len(self.training_metrics['ppo_loss']) + 1)
        # plt.plot(steps, self.training_metrics['ppo_loss'], 'b-', linewidth=2, label='PPO Loss')
        # plt.xlabel('Training Step')
        # plt.ylabel('Loss')
        # plt.title('PPO Loss Over Time')
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, 'ppo_loss.png'), dpi=150, bbox_inches='tight')
        # plt.close()

        # # Plot 2: Surrogate Losses comparison
        # plt.figure(figsize=(10, 6))
        # plt.plot(steps, self.training_metrics['surr1_loss'], 'r-', linewidth=2, label='Surrogate 1 (Unclipped)', alpha=0.7)
        # plt.plot(steps, self.training_metrics['surr2_loss'], 'g-', linewidth=2, label='Surrogate 2 (Clipped)', alpha=0.7)
        # plt.xlabel('Training Step')
        # plt.ylabel('Surrogate Loss')
        # plt.title('Surrogate Losses Comparison')
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, 'surrogate_losses.png'), dpi=150, bbox_inches='tight')
        # plt.close()

        # # Plot 3: Advantages Distribution (last 1000 values)
        # if len(self.training_metrics['advantages']) > 0:
        #     plt.figure(figsize=(10, 6))
        #     adv_data = self.training_metrics['advantages'][-1000:]  # Last 1000 advantage values
        #     plt.hist(adv_data, bins=50, alpha=0.7, color='purple', edgecolor='black')
        #     plt.axvline(np.mean(adv_data), color='red', linestyle='--', linewidth=2, label='.2f')
        #     plt.xlabel('Advantage Value')
        #     plt.ylabel('Frequency')
        #     plt.title(f'Advantages Distribution (Last {len(adv_data)} values)')
        #     plt.grid(True, alpha=0.3)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(plots_dir, 'advantages_distribution.png'), dpi=150, bbox_inches='tight')
        #     plt.close()

        # # Plot 4: Probability Ratios Distribution (last 1000 values)
        # if len(self.training_metrics['ratios']) > 0:
        #     plt.figure(figsize=(10, 6))
        #     ratio_data = self.training_metrics['ratios'][-1000:]  # Last 1000 ratio values
        #     plt.hist(ratio_data, bins=50, alpha=0.7, color='orange', edgecolor='black')
        #     plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Target Ratio = 1.0')
        #     plt.axvline(1.2, color='green', linestyle=':', linewidth=2, label='Upper Clip = 1.2')
        #     plt.axvline(0.8, color='green', linestyle=':', linewidth=2, label='Lower Clip = 0.8')
        #     plt.xlabel('Probability Ratio')
        #     plt.ylabel('Frequency')
        #     plt.title(f'Probability Ratios Distribution (Last {len(ratio_data)} values)')
        #     plt.grid(True, alpha=0.3)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(plots_dir, 'ratios_distribution.png'), dpi=150, bbox_inches='tight')
        #     plt.close()

        # # Plot 5: KL Divergence over time (moving average)
        # if len(self.training_metrics['kl_divergence']) > 10:
        #     plt.figure(figsize=(10, 6))
        #     kl_data = self.training_metrics['kl_divergence']
        #     # Calculate moving average
        #     window_size = min(50, len(kl_data))
        #     if window_size > 1:
        #         moving_avg = np.convolve(kl_data, np.ones(window_size)/window_size, mode='valid')
        #         plt.plot(range(window_size, len(kl_data)+1), moving_avg, 'b-', linewidth=2, label=f'KL Divergence (MA-{window_size})')
        #     else:
        #         plt.plot(range(1, len(kl_data)+1), kl_data, 'b-', linewidth=2, label='KL Divergence')
        #     plt.xlabel('Training Step')
        #     plt.ylabel('KL Divergence')
        #     plt.title('KL Divergence Between Teacher and Student')
        #     plt.grid(True, alpha=0.3)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(plots_dir, 'kl_divergence.png'), dpi=150, bbox_inches='tight')
        #     plt.close()

        # # Plot 6: Clipping fraction over time
        # if len(self.training_metrics['clipped_ratio_fraction']) > 0:
        #     plt.figure(figsize=(10, 6))
        #     clip_steps = range(1, len(self.training_metrics['clipped_ratio_fraction']) + 1)
        #     plt.plot(clip_steps, self.training_metrics['clipped_ratio_fraction'], 'r-', linewidth=2, marker='o', markersize=3)
        #     plt.xlabel('Training Step')
        #     plt.ylabel('Fraction of Clipped Ratios')
        #     plt.title('PPO Clipping Frequency')
        #     plt.grid(True, alpha=0.3)
        #     plt.ylim(0, 1)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(plots_dir, 'clipping_fraction.png'), dpi=150, bbox_inches='tight')
        #     plt.close()

        # # Plot 7: KL Loss over time
        # if len(self.training_metrics['kl_loss']) > 0:
        #     plt.figure(figsize=(10, 6))
        #     kl_loss_steps = range(1, len(self.training_metrics['kl_loss']) + 1)
        #     plt.plot(kl_loss_steps, self.training_metrics['kl_loss'], 'purple', linewidth=2, label='KL Loss', marker='s', markersize=3)
        #     plt.xlabel('Training Step')
        #     plt.ylabel('KL Loss')
        #     plt.title('KL Loss Trend (Regularization Term)')
        #     plt.grid(True, alpha=0.3)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(plots_dir, 'kl_loss.png'), dpi=150, bbox_inches='tight')
        #     plt.close()

        # # Plot 8: Combined metrics dashboard
        # fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        # fig.suptitle('Training Metrics Dashboard', fontsize=16)

        # # PPO Loss
        # if len(self.training_metrics['ppo_loss']) > 0:
        #     axes[0, 0].plot(steps, self.training_metrics['ppo_loss'], 'b-', linewidth=2)
        #     axes[0, 0].set_title('PPO Loss')
        #     axes[0, 0].grid(True, alpha=0.3)
        #     axes[0, 0].set_xlabel('Step')
        #     axes[0, 0].set_ylabel('Loss')

        # # Advantages histogram
        # if len(self.training_metrics['advantages']) > 0:
        #     adv_data = self.training_metrics['advantages'][-500:]
        #     axes[0, 1].hist(adv_data, bins=30, alpha=0.7, color='purple', edgecolor='black')
        #     axes[0, 1].axvline(np.mean(adv_data), color='red', linestyle='--', linewidth=2)
        #     axes[0, 1].set_title('Advantages Distribution')
        #     axes[0, 1].set_xlabel('Advantage')
        #     axes[0, 1].set_ylabel('Frequency')

        # # Ratios histogram
        # if len(self.training_metrics['ratios']) > 0:
        #     ratio_data = self.training_metrics['ratios'][-500:]
        #     axes[1, 0].hist(ratio_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
        #     axes[1, 0].axvline(1.0, color='red', linestyle='--', linewidth=2)
        #     axes[1, 0].set_title('Probability Ratios')
        #     axes[1, 0].set_xlabel('Ratio')
        #     axes[1, 0].set_ylabel('Frequency')

        # # KL Divergence trend
        # if len(self.training_metrics['kl_divergence']) > 0:
        #     kl_data = self.training_metrics['kl_divergence']
        #     if len(kl_data) > 10:
        #         window_size = min(20, len(kl_data))
        #         moving_avg = np.convolve(kl_data, np.ones(window_size)/window_size, mode='valid')
        #         axes[1, 1].plot(range(window_size, len(kl_data)+1), moving_avg, 'g-', linewidth=2)
        #     else:
        #         axes[1, 1].plot(range(1, len(kl_data)+1), kl_data, 'g-', linewidth=2)
        #     axes[1, 1].set_title('KL Divergence Trend')
        #     axes[1, 1].grid(True, alpha=0.3)
        #     axes[1, 1].set_xlabel('Step')
        #     axes[1, 1].set_ylabel('KL')

        # # KL Loss
        # if len(self.training_metrics['kl_loss']) > 0:
        #     axes[1, 2].plot(range(1, len(self.training_metrics['kl_loss'])+1), self.training_metrics['kl_loss'], 'purple', linewidth=2)
        #     axes[1, 2].set_title('KL Loss (Regularization)')
        #     axes[1, 2].grid(True, alpha=0.3)
        #     axes[1, 2].set_xlabel('Step')
        #     axes[1, 2].set_ylabel('KL Loss')

        # plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, 'training_dashboard.png'), dpi=150, bbox_inches='tight')
        # plt.close()

        # logger.info(f"Training plots updated at step {self.training_step}")

class Summarizer:
    """Summarizes retrieved documents for the reasoner."""
    def __init__(self, model_name: str, config: InferenceConfig):
        self.model_name = model_name
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
        self._load_prompt_template()
    
    def _load_model(self):
        """Load the summarizer model."""
        logger.info(f"Loading summarizer model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load LoRA adapters if specified
        if self.config.summarizer_lora_path:
            logger.info(f"Loading LoRA adapters from: {self.config.summarizer_lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.config.summarizer_lora_path)
            logger.info("LoRA adapters loaded successfully")
        
        logger.info("Summarizer model loaded successfully")
    
    def _load_prompt_template(self):
        """Load the prompt template for summarization."""
        prompt_path = "prompts/default_retrieval_summary.yaml"
        with open(prompt_path, 'r') as f:
            prompt_data = yaml.safe_load(f)
        
        self.prompt_template = prompt_data['user_prompt']
    
    def summarize_documents(self, question: str, documents: List[str]) -> str:
        """
        Summarize retrieved documents for the given question.
        
        Args:
            question: The original question
            documents: List of retrieved document texts
            
        Returns:
            Summarized information
        """
        # Combine documents
        combined_docs = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
        
        # Prepare prompt
        prompt = self.prompt_template.format(question=question, documents=combined_docs)
        
        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs["input_ids"].shape[1]
        
        # Generate summary
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.7,  # non-thinking mode
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (after the input tokens)
        generated_tokens = generated_ids[0][input_length:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract the final summary from the generated text
        # The prompt expects output to start with "### Extracted Information"
        if "### Extracted Information" in summary:
            # Extract everything after "### Extracted Information"
            start_idx = summary.find("### Extracted Information")
            if start_idx != -1:
                # Get the text after the header
                extracted_info = summary[start_idx + len("### Extracted Information"):].strip()
                # Remove any trailing markdown or extra text
                if "\n\n" in extracted_info:
                    extracted_info = extracted_info.split("\n\n")[0]
                if "\n###" in extracted_info:
                    extracted_info = extracted_info.split("\n###")[0]
                return extracted_info.strip()
        
        # If the expected format is not found, return the full summary
        return summary.strip()

class InferenceSystem:
    """Main system that coordinates reasoner, retriever, and summarizer."""
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.reasoner = Reasoner(config.reasoner_model_name, config.teacher_model_name, config)
        self.summarizer = Summarizer(config.summarizer_model_name, config)
        self.retriever = self._load_retriever()
        
        # Load probe if enabled
        self.probe = None
        self.pca = None
        self.scaler = None
        if self.config.use_probe:
            self._load_probe()
        
        os.makedirs(config.output_dir, exist_ok=True)

    def _load_retriever(self):
        if self.config.retriever_type == "bm25":
            return BM25Retriever(self.config.retriever_index_path, self.config.top_k_docs)
        elif self.config.retriever_type == "e5":
            return E5Retriever(self.config.retriever_index_path, self.config.e5_model_path)
        else:
            raise ValueError(f"Unsupported retriever type: {self.config.retriever_type}")

    def _load_probe(self):
        """Load the trained knowledge probe and associated components."""
        import pickle
        
        logger.info(f"Loading probe from {self.config.probe_path}")
        
        try:
            # Load probe configuration
            config_path = os.path.join(self.config.probe_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Probe config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                probe_config = json.load(f)
            
            # Extract probe configuration (handle both single layer and multi-layer formats)
            self.probe_layers = probe_config.get("probe_layers", [probe_config.get("probe_layer", 22)])
            self.probe_input_dim = probe_config.get("input_dim", 5120)
            self.probe_pca_dim = probe_config.get("pca_dim", 64)
            self.probe_hidden_dims = probe_config.get("hidden_dims", [128, 64, 32])
            self.probe_dropout_rate = probe_config.get("dropout_rate", 0.2)
            self.probe_num_layers = probe_config.get("num_layers", len(self.probe_layers))

            logger.info(f"Probe config loaded: layers={self.probe_layers}, input_dim={self.probe_input_dim}, pca_dim={self.probe_pca_dim}")
            logger.info(f"Probe architecture: hidden_dims={self.probe_hidden_dims}, dropout_rate={self.probe_dropout_rate}, num_layers={self.probe_num_layers}")
            
            # Load probe model
            probe_model_path = os.path.join(self.config.probe_path, "best_probe.pth")
            if not os.path.exists(probe_model_path):
                probe_model_path = os.path.join(self.config.probe_path, "final_probe.pth")
            
            if not os.path.exists(probe_model_path):
                raise FileNotFoundError(f"No probe model found in {self.config.probe_path}")
            
            # Load PCA and scaler
            pca_path = os.path.join(self.config.probe_path, "pca.pkl")
            scaler_path = os.path.join(self.config.probe_path, "scaler.pkl")
            
            if not os.path.exists(pca_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"PCA or scaler files not found in {self.config.probe_path}")
            
            # Load components
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load probe model
            from train_probe import KnowledgeProbe
            self.probe = KnowledgeProbe(
                input_dim=self.pca.n_components_,
                hidden_dims=self.probe_hidden_dims,
                dropout_rate=self.probe_dropout_rate,
                num_layers=self.probe_num_layers
            )
            self.probe.load_state_dict(torch.load(probe_model_path, map_location='cpu'))
            self.probe.eval()
            
            logger.info(f"Probe loaded successfully with {self.pca.n_components_} PCA components")
            
        except Exception as e:
            logger.error(f"Failed to load probe: {e}")
            raise
    
    def _get_probe_confidence(self, question: str) -> float:
        """
        Get confidence score from the multi-layer probe for a given question.

        Args:
            question: The question to evaluate

        Returns:
            float: Confidence score between 0 and 1
        """
        if not self.probe:
            return 0.0

        try:
            # Extract hidden states from the reasoner model for multiple layers
            input_text = f"<search> {question} </search>"
            inputs = self.reasoner.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.reasoner.model.device)

            with torch.no_grad():
                outputs = self.reasoner.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # Extract states for all probe layers
                layer_features = []
                last_token_pos = input_ids.shape[1] - 1

                for layer in self.probe_layers:
                    if layer >= len(hidden_states):
                        raise ValueError(f"Target layer {layer} is out of range. Model has {len(hidden_states)} layers.")

                    # Extract hidden state at the target layer and last token position
                    hidden_state = hidden_states[layer][0, last_token_pos, :].float().cpu().numpy()
                    layer_features.append(hidden_state)

                # Concatenate features from multiple layers
                concatenated_features = np.concatenate(layer_features, axis=0)

            # Apply PCA and scaling
            concatenated_scaled = self.scaler.transform(concatenated_features.reshape(1, -1))
            concatenated_pca = self.pca.transform(concatenated_scaled)

            # Get probe prediction
            with torch.no_grad():
                probe_input = torch.FloatTensor(concatenated_pca)
                confidence = self.probe(probe_input).item()

            return confidence

        except Exception as e:
            logger.warning(f"Failed to get probe confidence for question: {question[:50]}... Error: {e}")
            return 0.0
    
    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from <search> tags."""
        search_pattern = r'<search>(.*?)</search>'
        match = re.search(search_pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer> tags."""
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Don't extract 'and' or empty answers as final answers
            if answer.lower() == 'and' or answer == '':
                return None
            return answer
        return None

    def inference_one_turn(self, active_questions: List[Dict[str, Any]], turn_pbar: Optional[tqdm] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Perform one turn of inference for all active questions.
        
        Args:
            active_questions: List of question data with current sequences
            turn_pbar: Optional progress bar for turn-level progress
            
        Returns:
            Tuple of (updated_active_questions, search_queries_dict)
        """
        search_queries = {}

        question_pbar = tqdm(active_questions, desc="Processing questions", unit="question", position=2, leave=False)

        for question_data in question_pbar:
            question_pbar.set_description(f"Processing question {question_data['id']}")

            current_turn = len(question_data["turns"]) + 1
            max_turns = self.config.max_turns
            turns_left = max_turns - current_turn

            urgency_prompt = ""
            if turns_left == 0:
                urgency_prompt = "\n[System Note: This is the FINAL turn. You MUST provide the final answer inside <answer> </answer> tags. Do not search anymore.]\n"
            elif turns_left == 1:
                urgency_prompt = "\n[System Note: You have 1 turn left. Please consolidate information and conclude with an answer inside <answer> </answer> tags.]\n"

            input_sequence = question_data["sequence"] + urgency_prompt


            response = self.reasoner.generate_response(
                sequence=input_sequence,
                current_turn=current_turn,
                max_turns=max_turns
            )

            search_query = self._extract_search_query(response)
            answer = self._extract_answer(response)

            turn_info = {
                "turn": len(question_data["turns"]) + 1,
                "response": response,
                "search_query": search_query,
                "answer": answer
            }
            question_data['turns'].append(turn_info)
            question_data["sequence"] += response
            question_data["response"] += response

            if search_query:
                search_queries[question_data["id"]] = search_query

        question_pbar.close()
        return active_questions, search_queries

    def process_retrievals(self, active_questions: List[Dict[str, Any]], search_queries: Dict[str, str], probe_counters: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Process all search queries and update active questions with retrieved information.
        Includes probe-based confidence handling if probe is enabled.
        
        Args:
            active_questions: List of active question data
            search_queries: Dictionary mapping question_id to search query
            probe_counters: Optional probe statistics counter (for compatibility with process_dataset)
            
        Returns:
            Updated active_questions with retrieval information
        """
        if not search_queries:
            return active_questions
        unique_queries = list(set(search_queries.values()))
        query_to_questions = {}

        for question_id, query in search_queries.items():
            if query not in query_to_questions:
                query_to_questions[query] = []
            query_to_questions[query].append(question_id)

        query_pbar = tqdm(unique_queries, desc="Processing search queries", unit="query", position=3, leave=False)

        for query in query_pbar:
            try:
                query_pbar.set_description(f"Processing query: {query[:30]}...")
                should_retrieve = True
                probe_confidence = 0.0
                
                if self.config.use_probe:
                    original_question = None
                    for question_data in active_questions:
                        if question_data["id"] in query_to_questions[query]:
                            original_question = question_data['question']
                            break

                    if original_question:
                        probe_confidence = self._get_probe_confidence(original_question)
                        should_retrieve = probe_confidence < self.config.probe_confidence_threshold
                        
                        logger.info(f"Query: {query[:50]}... | Probe confidence: {probe_confidence:.4f} | Threshold: {self.config.probe_confidence_threshold} | Should retrieve: {should_retrieve}")

                if should_retrieve:
                    if probe_counters is not None:
                        probe_counters["performed_retrievals"] += 1

                    retrieved_docs = self.retriever.search(query, num=self.config.top_k_docs)

                    doc_texts = []
                    for doc in retrieved_docs:
                        if isinstance(doc, dict):
                            doc_text = doc.get('text', doc.get('contents', str(doc)))
                        else:
                            doc_text = str(doc)
                        doc_texts.append(doc_text)

                    summary = self.summarizer.summarize_documents(query, doc_texts)
                    for question_id in query_to_questions[query]:
                        for question_data in active_questions:
                            if question_data["id"] == question_id:
                                # Append information to the sequence with proper formatting
                                information_block = f"<information> {summary} </information>"
                                question_data["sequence"] += information_block
                                question_data["response"] += information_block
                                
                                # Update turn info with retrieval results
                                for turn_info in question_data["turns"]:
                                    if turn_info["search_query"] == query:
                                        turn_info["retrieved_docs"] = [str(doc) for doc in retrieved_docs]
                                        turn_info["summary"] = summary
                                        turn_info["probe_confidence"] = probe_confidence
                                        turn_info["retrieval_skipped"] = False
                                        break
                                break
                else:
                    # Count skipped retrieval
                    if probe_counters is not None:
                        probe_counters["skipped_retrievals"] += 1
                    
                    # High confidence - let model answer directly in next turn
                    # Add probe confidence info and mark for direct answer generation
                    for question_id in query_to_questions[query]:
                        for question_data in active_questions:
                            if question_data["id"] == question_id:
                                for turn_info in question_data["turns"]:
                                    if turn_info["search_query"] == query:
                                        turn_info["probe_confidence"] = probe_confidence
                                        turn_info["retrieval_skipped"] = True
                                        turn_info["retrieved_docs"] = []
                                        turn_info["summary"] = f"High probe confidence ({probe_confidence:.4f} >= {self.config.probe_confidence_threshold}) - model will answer directly"
                                        break
                                
                                # Add a prompt to encourage the model to answer directly
                                # This simulates the model having "retrieved" information from its internal knowledge
                                internal_knowledge_prompt = f"Based on my internal knowledge (confidence: {probe_confidence:.4f}), I can answer this question directly without external documents. I will put the answer in <information> </information>. <information>"
                                question_data["sequence"] += internal_knowledge_prompt
                                question_data["response"] += internal_knowledge_prompt
                                break
                    
                    logger.info(f"High confidence ({probe_confidence:.4f} >= {self.config.probe_confidence_threshold}) - model will answer directly for query '{query[:50]}...'")
                    
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
        
        query_pbar.close()
        
        # Update probe statistics for answered questions (if probe_counters provided)
        if probe_counters is not None and self.config.use_probe:
            for question_data in active_questions:
                if question_data.get("answer"):
                    # Check if this question had any skipped retrievals
                    had_skipped_retrieval = any(
                        turn.get("retrieval_skipped", False) 
                        for turn in question_data["turns"] 
                        if turn.get("search_query")
                    )
                    
                    if had_skipped_retrieval:
                        probe_counters["self_answered"] += 1
                    else:
                        # Check if it had any performed retrievals
                        had_performed_retrieval = any(
                            not turn.get("retrieval_skipped", True) 
                            for turn in question_data["turns"] 
                            if turn.get("search_query")
                        )
                        if had_performed_retrieval:
                            probe_counters["retrieved_answered"] += 1
        
        return active_questions

    def inference(self, questions: List[Dict[str, Any]], max_turns: Optional[int] = None, probe_counters: Optional[Dict[str, int]] = None, progress_bar: Optional[tqdm] = None) -> List[Dict[str, Any]]:
        """
        Perform full inference for a list of questions.
        
        Args:
            questions: List of question data with 'id', 'question', 'golden_answers'
            max_turns: Maximum number of turns (uses config default if None)
            probe_counters: Optional probe statistics counter (for compatibility with process_dataset)
            progress_bar: Optional tqdm progress bar for tracking question completion
            
        Returns:
            List of completed question results
        """
        if max_turns is None:
            max_turns = self.config.max_turns
        
        # Initialize all questions with their sequences
        active_questions = []
        for question_data in questions:
            # Initialize sequence with the question under prompt template
            prompted_question = self.reasoner.prompt_template.format(question=question_data["question"])
            
            # Prepare messages for chat template
            messages = [{"role": "user", "content": prompted_question}]
            initial_sequence = self.reasoner.student_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            active_questions.append({
                "id": question_data["id"],
                "question": question_data["question"],
                "golden_answers": question_data["golden_answers"],
                "sequence": initial_sequence,
                "response": "",
                "turns": [],
                "final_turn": 0,
                "answer": None,
                "error": None
            })
        
        max_turn_warning = "Time is up. I am not allowed to search anymore. I should give a final answer now with the information I have."
        completed_questions = []
        
        # Process questions in turns with progress tracking
        turn_pbar = tqdm(range(max_turns + 1), desc="Inference turns", unit="turn", position=1, leave=False) if progress_bar else range(max_turns + 1)
        for turn_num in turn_pbar:
            if not active_questions:
                break
            
            # Update turn progress bar description
            if progress_bar and hasattr(turn_pbar, 'set_description'):
                turn_pbar.set_description(f"Turn {turn_num + 1}/{max_turns + 1} (Active: {len(active_questions)})")
            
            # Step 1: Generate responses for all active questions
            active_questions, search_queries = self.inference_one_turn(active_questions, turn_pbar)
            
            # Update probe counters for search queries
            if probe_counters is not None and search_queries:
                probe_counters["total_search_queries"] += len(search_queries)
            
            # Step 2: Check for completed questions (answered or error)
            new_completed = []
            for question_data in active_questions:
                last_turn = question_data["turns"][-1] if question_data["turns"] else {}
                answer = last_turn.get("answer")
                search_query = last_turn.get("search_query")
                
                if answer:
                    # Question answered
                    question_data["answer"] = answer
                    question_data["final_turn"] = turn_num + 1
                    logger.info(f"Question {question_data['id']} completed in turn {turn_num + 1}")
                    new_completed.append(question_data)
                    # Update progress bar with question ID
                    if progress_bar:
                        progress_bar.set_description(f"Completed question {question_data['id']}")
                        progress_bar.update(1)
                elif not search_query:
                    # No search query or answer found
                    question_data["error"] = "No search query or answer found"
                    question_data["final_turn"] = turn_num + 1
                    logger.info(f"Question {question_data['id']} completed with error: No search query or answer found")
                    new_completed.append(question_data)
                    # Update progress bar with question ID
                    if progress_bar:
                        progress_bar.set_description(f"Completed question {question_data['id']} (error)")
                        progress_bar.update(1)
            
            # Add completed questions to results
            completed_questions.extend(new_completed)
            
            # Remove completed questions from active list
            active_questions = [q for q in active_questions if q not in new_completed]
            
            # Step 3: Process retrievals if any search queries
            if search_queries:
                active_questions = self.process_retrievals(active_questions, search_queries, probe_counters)
            
            # Step 4: Check if max turns reached
            if turn_num == max_turns - 1 and active_questions:
                for question_data in active_questions:
                    question_data["sequence"] += max_turn_warning
            elif turn_num == max_turns:
                for question_data in active_questions:
                    question_data["error"] = "Max turns reached"
                    question_data["final_turn"] = max_turns
                    logger.info(f"Question {question_data['id']} completed with error: Max turns reached")
                    completed_questions.append(question_data)
                    # Update progress bar with question ID
                    if progress_bar:
                        progress_bar.set_description(f"Completed question {question_data['id']} (max turns)")
                        progress_bar.update(1)
                active_questions = []
        
        # Calculate metrics for all results
        for result in completed_questions:
            if result["answer"]:
                from utils import calculate_metrics
                metrics = calculate_metrics(result["answer"], result["golden_answers"])
                result["metrics"] = metrics
            else:
                result["metrics"] = {"em": 0.0, "f1": 0.0, "cover_match": 0.0}
        
        # Log probe statistics if probe was used
        if self.config.use_probe and probe_counters is not None:
            logger.info("="*60)
            logger.info("PROBE STATISTICS SUMMARY")
            logger.info("="*60)
            logger.info(f"Total Search Queries: {probe_counters['total_search_queries']}")
            logger.info(f"Skipped Retrievals: {probe_counters['skipped_retrievals']}")
            logger.info(f"Performed Retrievals: {probe_counters['performed_retrievals']}")
            logger.info(f"Self-Answered: {probe_counters['self_answered']}")
            logger.info(f"Retrieved-Answered: {probe_counters['retrieved_answered']}")
            logger.info(f"Confidence Threshold: {self.config.probe_confidence_threshold}")
            logger.info("="*60)
        
        return completed_questions

    # def process_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
    #     """
    #     Process an entire dataset using sequence-based processing.
    #     Each question maintains its own sequence of responses and information.
        
    #     Args:
    #         dataset_path: Path to the dataset file
            
    #     Returns:
    #         List of results for each question
    #     """
    #     logger.info(f"Processing dataset: {dataset_path}")
        
    #     # Load dataset
    #     questions = []
    #     with open(dataset_path, 'r') as f:
    #         for line in f:
    #             data = json.loads(line.strip())
    #             if self.config.dataset_name == "hotpotqa":
    #                 questions.append({
    #                     "id": data["id"],
    #                     "question": data["question"],
    #                     "golden_answers": [data["answer"]]
    #                 })
    #             elif self.config.dataset_name == "2wikimultihop":  # 2wikimultihop
    #                 questions.append({
    #                     "id": data["_id"],
    #                     "question": data["question"],
    #                     "golden_answers": [data["answer"]] if not isinstance(data["answer"], list) else data["answer"]
    #                 })
        
    #     logger.info(f"Loaded {len(questions)} questions from dataset")
        
    #     # Limit samples if specified
    #     if self.config.start_sample is not None or self.config.end_sample is not None:
    #         start = (self.config.start_sample) if self.config.start_sample is not None else 0
    #         end = self.config.end_sample if self.config.end_sample is not None else len(questions)
    #         original_count = len(questions)
    #         questions = questions[start:end]
    #         logger.info(f"Selected questions from {start+1} to {end} (total: {len(questions)}, original: {original_count})")
    #     elif self.config.max_samples:
    #         original_count = len(questions)
    #         questions = questions[:self.config.max_samples]
    #         logger.info(f"Limited to {len(questions)} questions (max_samples: {self.config.max_samples}, original: {original_count})")
        
    #     logger.info(f"Processing {len(questions)} questions")
        
    #     # Initialize probe statistics counters
    #     probe_counters = {
    #         "total_search_queries": 0,
    #         "skipped_retrievals": 0,
    #         "performed_retrievals": 0,
    #         "self_answered": 0,
    #         "retrieved_answered": 0
    #     }
        
    #     # Use the modular inference method with progress tracking
    #     with tqdm(total=len(questions), desc="Processing questions", unit="question") as pbar:
    #         all_results = self.inference(questions, probe_counters=probe_counters, progress_bar=pbar)
        
    #     # Add probe statistics to all results for later analysis (preserve original behavior)
    #     if self.config.use_probe:
    #         for result in all_results:
    #             result["probe_counters"] = probe_counters
        
    #     # Save intermediate results if requested
    #     if self.config.save_intermediate:
    #         self._save_intermediate_results(all_results)
        
    #     logger.info(f"Processed {len(all_results)} questions successfully")
    #     return all_results

    def process_dataset_batchwise(self, dataset_path: str):
        # Read all data
        questions = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if self.config.dataset_name == "hotpotqa":
                    questions.append({
                        "id": data["id"],
                        "question": data["question"],
                        "golden_answers": [data["answer"]]
                    })
                elif self.config.dataset_name == "2wikimultihop":  # 2wikimultihop
                    questions.append({
                        "id": data["_id"],
                        "question": data["question"],
                        "golden_answers": [data["answer"]] if not isinstance(data["answer"], list) else data["answer"]
                    })
        # Batch Loop
        bs = self.config.batch_size
        all_results = []
        for i in range(0, len(questions), bs):
            batch_qs = questions[i: i + bs]
            logger.info(f"Processing batch {i // bs + 1}({len(batch_qs)} questions)...")

            # Generate full trajectory (Expansoion + Agentic Loop)
            trajectories = self.process_batch_rollouts(batch_qs)
            all_results.extend(trajectories)

            # Train (On Policy Distillation)
            logger.info("Training on accumulated trajectories...")
            self.reasoner.train_on_batch(trajectories)

            self.reasoner.save_model()
        
        return all_results

    def process_batch_rollouts(self, questions: List[Dict]) -> List[Dict]:
        """
        Core logic: Process the agentic loop for the batch of questions
        Loop for max_turn times
        After each turn, extract the search query
        execute the search query retrieval
        augmented the retrieval output (not for training)
        """
        active_items = []
        for q in questions:
            for r_idx in range(self.config.num_rollouts):
                active_items.append({
                    "id": f"{q['id']}_r{r_idx}",
                    "question": q["question"],
                    "golden_answers": q["golden_answers"],
                    "sequence": self.reasoner.prompt_template.format(question=q["question"]),
                    "trainable_spans": [], # 记录哪些字符区间是模型生成的 (用于训练)
                    "done": False,
                    "turn": 0,
                    "turns": [],
                    "answer": None,
                    "metrics": {"em": 0.0, "f1": 0.0, "cover_match": 0.0}
                })
        
        active_indices = list(range(len(active_items)))

        pbar = tqdm(total=self.config.max_turns, desc="Rollout Generation")
        while active_indices:
            # filter the time out ones and the complete ones
            current_indices = []
            for idx in active_indices:
                if active_items[idx]['turn'] < self.config.max_turns and not active_items[idx]['done']:
                    current_indices.append(idx)
            active_indices = current_indices
            if not active_indices:
                break

            # Prepare the inputs
            current_seqs = [active_items[idx]['sequence'] for idx in active_indices]
            current_turns = [active_items[idx]['turn'] + 1 for idx in active_indices]
            
            # Update Description: Generation
            current_active_items = [active_items[i] for i in active_indices]
            unique_qids = list(set(item['id'].split('_r')[0] for item in current_active_items))
            qid_str = ",".join(unique_qids[:2])
            if len(unique_qids) > 2:
                qid_str += "..."
            
            status_base = f"Turn {current_turns[0]}/{self.config.max_turns} | Qs({len(unique_qids)}): {qid_str} | Active: {len(active_indices)}"
            pbar.set_description(f"{status_base} | State: Generating")

            # Batch Generation
            generated_texts = self.reasoner.generate_batch(
                current_seqs,
                current_turns[0],
                self.config.max_turns
            )

            # Update the sequence and extract the search query
            search_requests = {}

            for i, idx in enumerate(active_indices):
                item = active_items[idx]
                gen_text = generated_texts[i]

                # Storing the start and end point
                start_char = len(item['sequence'])
                item['sequence'] += gen_text
                end_char = len(item['sequence'])

                # Lable here as trainable spans
                item['trainable_spans'].append((start_char, end_char))

                search_query = None
                # Extract search query
                if "<search>" in gen_text:
                    match = re.search(r'<search>(.*?)</search>', gen_text, re.DOTALL)
                    if match:
                        search_query = match.group(1).strip()
                        search_requests[idx] = search_query
                
                # Extract answer
                answer = None
                if "<answer>" in gen_text:
                    match = re.search(r'<answer>(.*?)</answer>', gen_text, re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                        item['answer'] = answer
                        # Calculate metrics if we have an answer
                        if item.get('golden_answers'):
                            item['metrics'] = calculate_metrics(answer, item['golden_answers'])
                    item['done'] = True
                
                # Record turn info
                turn_info = {
                    "turn": item['turn'] + 1,
                    "response": gen_text,
                    "search_query": search_query,
                    "answer": answer,
                    "retrieved_docs": [] # Will be updated if search occurred
                }
                item['turns'].append(turn_info)
                
                item['turn'] += 1

                # Parallel Execution of retrieval threads
                if search_requests:
                    # Filter the answered and time out ones
                    valid_searches = {k: v for k, v in search_requests.items() if not active_items[k]['done']}
                    
                    num_searches = len(valid_searches)
                    if valid_searches:
                        pbar.set_description(f"{status_base} | State: Retrieving ({num_searches} queries)")
                        results = self.process_retrievals_parallel(valid_searches)

                        # Augment the retrieval results and update turn info
                        for idx, summary in results.items():
                            info_block = f"\n<information>\n{summary}\n</information>\n"
                            active_items[idx]['sequence'] += info_block
                            
                            # Update the last turn info with retrieval results
                            # Note: item was updated in the loop above, so we need to access it again
                            current_item = active_items[idx]
                            if current_item['turns']:
                                current_item['turns'][-1]['summary'] = summary
                                # We don't have raw docs here due to process_retrievals_parallel design, 
                                # but we ensure the key exists for save_final_results
            pbar.update(1)
        pbar.close()
        return active_items

    def process_retrievals_parallel(self, search_requests: Dict[int, str]) -> Dict[int, str]:
        """
        MultiThreading to do retreival and summarization
        """
        results = {}

        def run_single(idx, query):
            # The original retriever and summarizer
            docs = self.retriever.search(query, num=self.config.top_k_docs)
            doc_texts = [str(d) for d in docs]
            summary = self.summarizer.summarize_documents(query, doc_texts)
            return idx, summary

        # Use ThreadPoolExecutor to run the single retrievals in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_retrieval_threads) as executor:
            futures = [executor.submit(run_single, idx, query) for idx, query in search_requests.items()]
            for future in futures:
                idx, res = future.result()
                results[idx] = res

        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to file."""
        # Sort results by question ID/index for consistent output order
        sorted_results = sorted(results, key=lambda x: x.get("id", ""))
        
        output_file = os.path.join(self.config.output_dir, "intermediate_results.json")
        with open(output_file, 'w') as f:
            json.dump(sorted_results, f, indent=2)

    def save_final_results(self, results: List[Dict[str, Any]], dataset_name: str):
        """Save final results and evaluation metrics."""
        # Prepare results without retrieved documents for main results file
        clean_results = []
        retrieval_results = []
        
        for result in results:
            # Create clean result without retrieved documents
            clean_result = result.copy()
            
            # Remove retrieved_docs from turns for main results
            for turn in clean_result["turns"]:
                if "retrieved_docs" in turn:
                    # Store retrieval info separately
                    retrieval_results.append({
                        "question_id": result["id"],
                        "turn": turn["turn"],
                        "search_query": turn["search_query"],
                        "retrieved_docs": turn["retrieved_docs"],
                        "summary": turn.get("summary", "")
                    })
                    # Remove from main results
                    del turn["retrieved_docs"]
                    if "summary" in turn:
                        del turn["summary"]
            
            del clean_result["sequence"]
            clean_results.append(clean_result)
        
        # Save detailed results (without retrieved documents)
        output_file = os.path.join(self.config.output_dir, f"{dataset_name}_results.json")
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        # Save retrieval results separately
        if retrieval_results:
            retrieval_file = os.path.join(self.config.output_dir, f"{dataset_name}_retrieval_results.json")
            with open(retrieval_file, 'w') as f:
                json.dump(retrieval_results, f, indent=2)
            logger.info(f"Retrieval results saved to {retrieval_file}")
        
        # Calculate and save summary metrics
        total_questions = len(clean_results)
        
        if total_questions > 0:
            answered_questions = sum(1 for r in clean_results if r["answer"] is not None)
            answer_rate = answered_questions / total_questions
            
            avg_em = sum(r["metrics"]["em"] for r in clean_results) / total_questions
            avg_f1 = sum(r["metrics"]["f1"] for r in clean_results) / total_questions
            avg_cover_match = sum(r["metrics"]["cover_match"] for r in clean_results) / total_questions
            
            avg_turns = sum(r["final_turn"] for r in clean_results) / total_questions
        else:
            answered_questions = 0
            answer_rate = 0.0
            avg_em = 0.0
            avg_f1 = 0.0
            avg_cover_match = 0.0
            avg_turns = 0.0
        
        summary = {
            "dataset": dataset_name,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "answer_rate": answer_rate,
            "average_em": avg_em,
            "average_f1": avg_f1,
            "average_cover_match": avg_cover_match,
            "average_turns": avg_turns,
            "config": self.config.__dict__
        }
        
        summary_file = os.path.join(self.config.output_dir, f"{dataset_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary saved to {summary_file}")
        logger.info(f"Average EM: {avg_em:.3f}, Average F1: {avg_f1:.3f}, Average Cover Match: {avg_cover_match:.3f}")

        # Only save model if training actually occurred
        if self.reasoner.training_step > 0:
            self.reasoner.save_model()
        else:
            logger.info("Skipping model save - no training occurred (inference-only run)")

def main():
    """Main function to run Search-o1 system."""
    # Start timing
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Agentic RAG Inference System")
    
    # Model settings
    parser.add_argument("--reasoner-model", default="Qwen/Qwen3-32B", help="Reasoner model name")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-32B", help="Teacher model name")
    parser.add_argument("--summarizer-model", default="Qwen/Qwen3-32B", help="Summarizer model name")
    parser.add_argument("--reasoner-lora-path", default=None, help="Path to LoRA adapters for reasoner")
    parser.add_argument("--summarizer-lora-path", default=None, help="Path to LoRA adapters for summarizer")
    parser.add_argument("--retriever-type", default="e5", choices=["bm25", "e5"], help="Retriever type")
    parser.add_argument("--retriever-index-path", default="indexes/bm25", help="Path to retriever index")
    parser.add_argument("--e5-model-path", default="intfloat/e5-large-v2", help="Path to E5 model for retrieval")
    
    # Generation settings
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum number of turns")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--greedy-thinking", action="store_true", help="Use greedy decoding in thinking mode (no sampling)")
    parser.add_argument("--high-randomness-mode", action="store_true", help="Use more aggressive random generation for diverse trajectories")
    
    # Retrieval settings
    parser.add_argument("--top-k-docs", type=int, default=10, help="Number of documents to retrieve")
    
    # Dataset settings
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimultihop", "bamboogle"], help="Dataset to use")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--start-sample", type=int, default=None, help="Start index (1-based, inclusive) of samples to process")
    parser.add_argument("--end-sample", type=int, default=None, help="End index (1-based, inclusive) of samples to process")
    
    # Output settings
    parser.add_argument("--output-dir", default=None, help="Output directory (auto-generated if not specified)")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    
    # Probe settings
    parser.add_argument("--use-probe", action="store_true", help="Enable probe-based inference mode")
    parser.add_argument("--probe-path", default="probe/", help="Path to probe directory containing trained model")
    parser.add_argument("--probe-confidence-threshold", type=float, default=0.7, help="Confidence threshold above which retrieval is skipped")

    # Training settings
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-rollouts", type=int, default=8, help="Number of rollouts per question")
    parser.add_argument("--train-micro-batch-size", type=int, default=1, help="Micro batch size for training")
    parser.add_argument("--num-retrieval-threads", type=int, default=8, help="Number of retrieval threads")
    
    args = parser.parse_args()

    if args.output_dir is None:
        if args.use_probe:
            args.output_dir = f"output/search_o1_probe_{args.probe_confidence_threshold}"
        else:
            args.output_dir = "output/search_o1"
    
    config = InferenceConfig(
        reasoner_model_name=args.reasoner_model,
        teacher_model_name=args.teacher_model,
        summarizer_model_name=args.summarizer_model,
        reasoner_lora_path=args.reasoner_lora_path,
        summarizer_lora_path=args.summarizer_lora_path,
        retriever_type=args.retriever_type,
        retriever_index_path=args.retriever_index_path,
        e5_model_path=args.e5_model_path,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        greedy_thinking=args.greedy_thinking,
        high_randomness_mode=args.high_randomness_mode,
        top_k_docs=args.top_k_docs,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate,
        start_sample=args.start_sample,
        end_sample=args.end_sample,
        use_probe=args.use_probe,
        probe_path=args.probe_path,
        probe_confidence_threshold=args.probe_confidence_threshold,
        batch_size=args.batch_size,
        num_rollouts=args.num_rollouts,
        train_micro_batch_size=args.train_micro_batch_size,
        num_retrieval_threads=args.num_retrieval_threads
    )

    system = InferenceSystem(config)

    if config.reasoner_lora_path:
        logger.info(f"LoRA-enabled reasoner: {config.reasoner_lora_path}")
    else:
        logger.info("Standard reasoner (no LoRA)")
    
    if config.summarizer_lora_path:
        logger.info(f"LoRA-enabled summarizer: {config.summarizer_lora_path}")
    else:
        logger.info("Standard summarizer (no LoRA)")

    if config.use_probe:
        logger.info(f"Probe-based inference mode enabled")
        logger.info(f"Probe path: {config.probe_path}")
        logger.info(f"Confidence threshold: {config.probe_confidence_threshold}")
        logger.info(f"Questions with confidence >= {config.probe_confidence_threshold} will skip retrieval")
        logger.info("Probe configuration (layer, PCA dim, etc.) will be loaded from probe config file")
    else:
        logger.info("Standard inference mode (no probe)")

    if config.greedy_thinking:
        logger.info("Greedy decoding enabled for thinking mode (deterministic reasoning)")
    else:
        logger.info("Sampling-based decoding enabled for thinking mode (creative reasoning)")
    
    # Determine dataset path
    dataset_path = f"datasets/{args.dataset}/{args.split}.jsonl"
    
    # Process dataset
    results = system.process_dataset_batchwise(dataset_path)
    
    # Sort results by question ID/index for consistent output order
    results.sort(key=lambda x: x.get("id", ""))
    
    # Save results
    system.save_final_results(results, args.dataset)
    
    # Calculate and log total running time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Convert to hours, minutes, seconds for readability
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info("="*60)
    logger.info("TIMING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Running Time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.2f} seconds)")
    if results and len(results) > 0:
        logger.info(f"Average Time per Question: {total_time/len(results):.2f} seconds")
    logger.info("="*60)
    
    logger.info("Search-o1 processing completed!")

if __name__ == "__main__":
    main()