import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# 自定义数据集（玩具数学提示）
class MathDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts  # e.g., ["2+3=", "4*5=", ...]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

# 加载模型和tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
student = GPT2LMHeadModel.from_pretrained('gpt2')  # 学生模型
teacher = GPT2LMHeadModel.from_pretrained('gpt2-large')  # 教师模型（更大）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
student.to(device)
teacher.to(device)
teacher.eval()  # 教师不训练

optimizer = AdamW(student.parameters(), lr=1e-5)
dataset = MathDataset(["2+3=", "4*5=", "1-1=", "10/2="] * 100)  # 重复以模拟大批量
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 训练循环
num_steps = 150  # 如文章中150步
for step in range(num_steps):
    for batch in dataloader:
        # 步骤1: 从学生采样轨迹（on-policy采样）
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        prompt_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = student.generate(
                inputs['input_ids'], 
                max_new_tokens=10,  # 生成短序列
                output_scores=True, 
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id
            )
        generated_ids = outputs.sequences  # 形状: [B, prompt_len + new_tokens]
        
        # 收集所有new tokens的student old logprobs（循环over all new tokens）
        student_logprobs = []
        for t in range(len(outputs.scores)):  # len(scores) = new_tokens
            score = outputs.scores[t].log_softmax(dim=-1)  # [B, vocab]
            next_token = generated_ids[:, prompt_len + t].unsqueeze(-1)  # [B, 1]
            logprob = torch.gather(score, dim=1, index=next_token).squeeze(-1)  # [B]
            student_logprobs.append(logprob)
        student_logprobs = torch.stack(student_logprobs, dim=1)  # [B, new_tokens]

        # 步骤2: 计算教师logprobs（在完整学生轨迹上，per-token）
        with torch.no_grad():
            teacher_inputs = {
                'input_ids': generated_ids,
                'attention_mask': (generated_ids != tokenizer.pad_token_id).long()
            }
            teacher_outputs = teacher(**teacher_inputs)
            teacher_logits = teacher_outputs.logits[:, :-1].log_softmax(dim=-1)  # [B, total_len-1, vocab]
            shift_ids = generated_ids[:, 1:]  # [B, total_len-1]
            teacher_logprobs_full = torch.gather(teacher_logits, dim=-1, index=shift_ids.unsqueeze(-1)).squeeze(-1)  # [B, total_len-1]
            # 只取生成的new tokens部分（忽略prompt）
            teacher_logprobs = teacher_logprobs_full[:, prompt_len-1:]  # 调整索引：[B, new_tokens]

        # 步骤3: 计算reverse KL优势（per-token）
        reverse_kl = student_logprobs - teacher_logprobs  # [B, new_tokens]
        advantages = -reverse_kl  # [B, new_tokens]

        # 步骤4: PPO-like损失（importance sampling简化版，per-token）
        # 计算当前学生的新logprobs（在相同轨迹上）
        student_outputs = student(**teacher_inputs)
        student_logits = student_outputs.logits[:, :-1].log_softmax(dim=-1)  # [B, total_len-1, vocab]
        new_logprobs_full = torch.gather(student_logits, dim=-1, index=shift_ids.unsqueeze(-1)).squeeze(-1)  # [B, total_len-1]
        new_logprobs = new_logprobs_full[:, prompt_len-1:]  # [B, new_tokens]

        # 比率（ratio，per-token）
        ratio = (new_logprobs - student_logprobs).exp()  # [B, new_tokens]

        # Clipped surrogate loss（对所有token求mean）
        epsilon = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        loss = -torch.min(surr1, surr2).mean()  # 平均over batch和tokens

        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Step {step+1}/{num_steps} completed.")

# 保存学生模型
student.save_pretrained("distilled_student")