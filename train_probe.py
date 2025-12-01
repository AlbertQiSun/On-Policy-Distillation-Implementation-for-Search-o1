import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
import pickle

# Configs
MODEL_PATH = "/scratch/qs2196/.cache/models/Qwen3-1.7B"
INPUT_FILE = "probe_data_multilayer.pkl"  # Load pre-collected multi-layer hidden states
PROBE_LAYERS = [14, 22, 28]  # Multiple layers to use for probing
PCA_DIM = 64  # Reduce from 2048 to 64 dimensions per layer
PROBE_OUTPUT_DIR = "probe_multilayer/"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100  # Reduced to prevent overfit
PATIENCE = 10  # For early stopping

# Probe architecture configs
HIDDEN_DIMS = [128, 64, 32]  # Multi-layer hidden dimensions
DROPOUT_RATE = 0.2  # Dropout rate for regularization

class KnowledgeProbe(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128, 64, 32], dropout_rate=0.2, num_layers=3):
        """
        Multi-layer MLP probe for knowledge detection with multiple transformer layers.

        Args:
            input_dim: Input dimension per layer (after PCA)
            hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
            dropout_rate: Dropout rate for regularization
            num_layers: Number of transformer layers to concatenate
        """
        super(KnowledgeProbe, self).__init__()

        # Input dimension is multiplied by number of layers
        total_input_dim = input_dim * num_layers

        # Build layers dynamically
        layers = []
        prev_dim = total_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.probe = nn.Sequential(*layers)
        self.num_layers = num_layers

    def forward(self, x):
        return self.probe(x)

class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states, labels):
        self.hidden_states = torch.FloatTensor(hidden_states)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        return self.hidden_states[idx], self.labels[idx]

def expected_calibration_error(confidences, labels, M=15):
    """
    Calculate Expected Calibration Error (ECE) as described in the paper.
    
    Args:
        confidences: numpy array of confidence scores (probabilities)
        labels: numpy array of true labels (0 or 1)
        M: number of bins for calibration calculation (default: 15)
    
    Returns:
        float: ECE value
    """
    # Uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        
        # Calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()
        
        if prob_in_bin > 0:
            # Get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = labels[in_bin].mean()
            # Get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    
    return ece

def main():
    print("Loading multi-layer probe data...")
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    # Load data for multiple layers
    hidden_states_dict = data['features']  # Dict: layer -> numpy array
    labels = data['labels']
    questions = data.get('questions', [])
    probe_layers = data.get('probe_layers', PROBE_LAYERS)

    print(f"Loaded data: {len(labels)} samples")
    print(f"Layers available: {list(hidden_states_dict.keys())}")
    print(f"Using layers: {probe_layers}")
    print(f"Labels shape: {labels.shape}")

    # Concatenate features from multiple layers
    print("Concatenating features from multiple layers...")
    layer_features = []
    for layer in probe_layers:
        if layer in hidden_states_dict:
            layer_features.append(hidden_states_dict[layer])
            print(f"Layer {layer} shape: {hidden_states_dict[layer].shape}")
        else:
            raise ValueError(f"Layer {layer} not found in data")

    # Concatenate along feature dimension
    hidden_states = np.concatenate(layer_features, axis=1)
    print(f"Concatenated hidden states shape: {hidden_states.shape}")

    labels = np.array(labels)

    # Print label distribution to diagnose imbalance
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    print(f"Label distribution: {label_dist}")
    print(f"Positive ratio (label 1): {np.mean(labels):.4f}")
    print(f"Data generation method: {data.get('generation_method', 'unknown')}")

    # Random split instead of fixed
    train_hidden, val_hidden, train_labels, val_labels, train_questions, val_questions = train_test_split(
        hidden_states, labels, questions, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(train_hidden)}")
    print(f"Validation samples: {len(val_hidden)}")

    # Apply PCA
    print("Applying PCA...")
    # Use min of requested components and available dimensions/samples
    max_components = min(PCA_DIM * len(probe_layers), min(train_hidden.shape[0], train_hidden.shape[1]))
    pca = PCA(n_components=max_components)
    scaler = StandardScaler()

    # Fit on training data
    train_hidden_scaled = scaler.fit_transform(train_hidden)
    train_hidden_pca = pca.fit_transform(train_hidden_scaled)

    # Transform validation data
    val_hidden_scaled = scaler.transform(val_hidden)
    val_hidden_pca = pca.transform(val_hidden_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA input dim: {PCA_DIM * len(probe_layers)}, output dim: {pca.n_components_}")

    # Create datasets and dataloaders
    train_dataset = HiddenStateDataset(train_hidden_pca, train_labels)
    val_dataset = HiddenStateDataset(val_hidden_pca, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize probe with concatenated features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = KnowledgeProbe(
        input_dim=pca.n_components_,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
        num_layers=len(probe_layers)
    ).to(device)
    print(f"Probe input dimension: {pca.n_components_} (after PCA)")
    
    # Weighted BCE for imbalance
    pos_weight = len(train_labels) / (2 * np.sum(train_labels)) if np.sum(train_labels) > 0 else 1.0  # Handle all 0 case
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))  # Use logits for stability
    
    optimizer = optim.Adam(probe.parameters(), lr=LEARNING_RATE, weight_decay=0.01)  # Add L2
    
    # Create output directory
    os.makedirs(PROBE_OUTPUT_DIR, exist_ok=True)
    
    # Training loop with early stopping
    print("Training probe...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        probe.train()
        train_loss = 0.0
        for batch_hidden, batch_labels in train_loader:
            batch_hidden = batch_hidden.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_hidden).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        probe.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_hidden, batch_labels in val_loader:
                batch_hidden = batch_hidden.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = probe(batch_hidden).squeeze()
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(probe.state_dict(), os.path.join(PROBE_OUTPUT_DIR, "best_probe.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Save PCA and scaler
    print("Saving models...")
    with open(os.path.join(PROBE_OUTPUT_DIR, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    
    with open(os.path.join(PROBE_OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # Save final probe
    torch.save(probe.state_dict(), os.path.join(PROBE_OUTPUT_DIR, "final_probe.pth"))
    
    # Save configuration
    config = {
        "model_path": MODEL_PATH,
        "probe_layers": PROBE_LAYERS,
        "pca_dim": pca.n_components_,  # Actual PCA components used
        "input_dim": 2048,  # Original hidden state dimension for Qwen3-1.7B
        "hidden_dims": HIDDEN_DIMS,
        "dropout_rate": DROPOUT_RATE,
        "num_layers": len(PROBE_LAYERS),
        "train_samples": len(train_hidden),
        "val_samples": len(val_hidden),
        "epochs": epoch + 1,  # Actual epochs run
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "generation_method": data.get('generation_method', 'unknown')
    }
    
    with open(os.path.join(PROBE_OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Training completed! Models saved in {PROBE_OUTPUT_DIR}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    
    # Validate predictions >= 0.7
    print("\n" + "="*50)
    print("VALIDATING PREDICTIONS >= 0.7")
    print("="*50)
    
    # Load best model for validation
    best_probe = KnowledgeProbe(
        input_dim=pca.n_components_,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
        num_layers=len(PROBE_LAYERS)
    ).to(device)
    best_probe.load_state_dict(torch.load(os.path.join(PROBE_OUTPUT_DIR, "best_probe.pth")))
    best_probe.eval()
    
    # Get all validation predictions
    all_val_predictions = []
    all_val_labels = []
    
    with torch.no_grad():
        for batch_hidden, batch_labels in val_loader:
            batch_hidden = batch_hidden.to(device)
            outputs = best_probe(batch_hidden).squeeze()
            all_val_predictions.extend(outputs.cpu().numpy())
            all_val_labels.extend(batch_labels.cpu().numpy())
    
    all_val_predictions = np.array(all_val_predictions)
    all_val_labels = np.array(all_val_labels)
    
    # Filter predictions >= 0.7
    high_confidence_mask = all_val_predictions >= 0.7
    high_confidence_predictions = all_val_predictions[high_confidence_mask]
    high_confidence_labels = all_val_labels[high_confidence_mask]
    high_confidence_questions = np.array(val_questions)[high_confidence_mask]
    
    print(f"Total validation samples: {len(all_val_predictions)}")
    print(f"High confidence predictions (>= 0.7): {len(high_confidence_predictions)} ({len(high_confidence_predictions)/len(all_val_predictions)*100:.2f}%)")
    
    if len(high_confidence_predictions) > 0:
        # Calculate accuracy for high confidence predictions
        high_confidence_accuracy = np.mean(high_confidence_predictions.round() == high_confidence_labels)
        print(f"Accuracy of high confidence predictions: {high_confidence_accuracy:.4f} ({high_confidence_accuracy*100:.2f}%)")
        
        # Calculate precision, recall, F1 for high confidence predictions
        high_confidence_pred_binary = high_confidence_predictions.round()
        true_positives = np.sum((high_confidence_pred_binary == 1) & (high_confidence_labels == 1))
        false_positives = np.sum((high_confidence_pred_binary == 1) & (high_confidence_labels == 0))
        false_negatives = np.sum((high_confidence_pred_binary == 0) & (high_confidence_labels == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Analyze confidence distribution
        print(f"\nConfidence distribution for high confidence predictions:")
        print(f"Mean confidence: {np.mean(high_confidence_predictions):.4f}")
        print(f"Std confidence: {np.std(high_confidence_predictions):.4f}")
        print(f"Min confidence: {np.min(high_confidence_predictions):.4f}")
        print(f"Max confidence: {np.max(high_confidence_predictions):.4f}")
        
        # Show some examples of correct and incorrect high confidence predictions
        correct_high_conf = (high_confidence_pred_binary == high_confidence_labels)
        incorrect_high_conf = ~correct_high_conf
        
        print(f"\nHigh confidence correct predictions: {np.sum(correct_high_conf)}")
        print(f"High confidence incorrect predictions: {np.sum(incorrect_high_conf)}")
        
        # Show sample questions for incorrect high confidence predictions
        if np.sum(incorrect_high_conf) > 0:
            print(f"\nSample incorrect high confidence predictions:")
            incorrect_indices = np.where(incorrect_high_conf)[0]
            for i in range(min(5, len(incorrect_indices))):
                idx = incorrect_indices[i]
                question = high_confidence_questions[idx]
                pred = high_confidence_predictions[idx]
                label = high_confidence_labels[idx]
                print(f"  Question: {question[:100]}...")
                print(f"  Prediction: {pred:.4f}, True Label: {label}")
                print()
        
        # Save detailed validation results
        validation_results = {
            "total_validation_samples": len(all_val_predictions),
            "high_confidence_threshold": 0.7,
            "high_confidence_count": len(high_confidence_predictions),
            "high_confidence_percentage": len(high_confidence_predictions)/len(all_val_predictions)*100,
            "high_confidence_accuracy": float(high_confidence_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confidence_stats": {
                "mean": float(np.mean(high_confidence_predictions)),
                "std": float(np.std(high_confidence_predictions)),
                "min": float(np.min(high_confidence_predictions)),
                "max": float(np.max(high_confidence_predictions))
            },
            "correct_high_conf": int(np.sum(correct_high_conf)),
            "incorrect_high_conf": int(np.sum(incorrect_high_conf))
        }
        
        with open(os.path.join(PROBE_OUTPUT_DIR, "high_confidence_validation.json"), "w") as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"Detailed validation results saved to {PROBE_OUTPUT_DIR}/high_confidence_validation.json")
        
    else:
        print("No predictions >= 0.7 found in validation set!")
        print("Consider lowering the threshold or improving the model.")
    
    print("="*50)
    
    # Calculate Expected Calibration Error (ECE)
    print("\n" + "="*50)
    print("CALCULATING EXPECTED CALIBRATION ERROR (ECE)")
    print("="*50)
    
    # Get all validation predictions and labels
    all_val_confidences = []
    all_val_binary_labels = []
    
    with torch.no_grad():
        for batch_hidden, batch_labels in val_loader:
            batch_hidden = batch_hidden.to(device)
            outputs = best_probe(batch_hidden).squeeze()
            all_val_confidences.extend(outputs.cpu().numpy())
            all_val_binary_labels.extend(batch_labels.cpu().numpy())
    
    all_val_confidences = np.array(all_val_confidences)
    all_val_binary_labels = np.array(all_val_binary_labels)
    
    # Calculate ECE with different numbers of bins
    ece_15 = expected_calibration_error(all_val_confidences, all_val_binary_labels, M=15)
    ece_10 = expected_calibration_error(all_val_confidences, all_val_binary_labels, M=10)
    ece_5 = expected_calibration_error(all_val_confidences, all_val_binary_labels, M=5)
    
    print(f"ECE with 15 bins: {ece_15:.6f}")
    print(f"ECE with 10 bins: {ece_10:.6f}")
    print(f"ECE with 5 bins: {ece_5:.6f}")
    
    # Calculate ECE for high confidence predictions only
    if len(high_confidence_predictions) > 0:
        high_conf_ece_15 = expected_calibration_error(high_confidence_predictions, high_confidence_labels, M=15)
        high_conf_ece_10 = expected_calibration_error(high_confidence_predictions, high_confidence_labels, M=10)
        high_conf_ece_5 = expected_calibration_error(high_confidence_predictions, high_confidence_labels, M=5)
        
        print(f"\nECE for high confidence predictions (>= 0.7):")
        print(f"  ECE with 15 bins: {high_conf_ece_15:.6f}")
        print(f"  ECE with 10 bins: {high_conf_ece_10:.6f}")
        print(f"  ECE with 5 bins: {high_conf_ece_5:.6f}")
    
    # Analyze calibration by confidence bins
    print(f"\nCalibration analysis by confidence bins:")
    bin_boundaries = np.linspace(0, 1, 11)  # 10 bins for analysis
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    for i in range(len(bin_boundaries) - 1):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        bin_center = bin_centers[i]
        
        # Find samples in this bin
        in_bin = np.logical_and(all_val_confidences > bin_lower, all_val_confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_accuracy = all_val_binary_labels[in_bin].mean()
            bin_confidence = all_val_confidences[in_bin].mean()
            bin_count = in_bin.sum()
            
            print(f"  Bin [{bin_lower:.1f}, {bin_upper:.1f}]: "
                  f"Count={bin_count:3d}, "
                  f"Avg_Conf={bin_confidence:.3f}, "
                  f"Accuracy={bin_accuracy:.3f}, "
                  f"Diff={abs(bin_confidence - bin_accuracy):.3f}")
    
    # Save ECE results
    ece_results = {
        "ece_15_bins": float(ece_15),
        "ece_10_bins": float(ece_10),
        "ece_5_bins": float(ece_5),
        "high_confidence_ece": {
            "ece_15_bins": float(high_conf_ece_15) if len(high_confidence_predictions) > 0 else None,
            "ece_10_bins": float(high_conf_ece_10) if len(high_confidence_predictions) > 0 else None,
            "ece_5_bins": float(high_conf_ece_5) if len(high_confidence_predictions) > 0 else None
        },
        "calibration_analysis": {
            "total_validation_samples": len(all_val_confidences),
            "mean_confidence": float(np.mean(all_val_confidences)),
            "std_confidence": float(np.std(all_val_confidences)),
            "perfect_calibration_threshold": 0.0  # Perfect calibration means ECE = 0
        }
    }
    
    with open(os.path.join(PROBE_OUTPUT_DIR, "ece_results.json"), "w") as f:
        json.dump(ece_results, f, indent=2)
    
    print(f"\nECE results saved to {PROBE_OUTPUT_DIR}/ece_results.json")
    print("="*50)

if __name__ == "__main__":
    main()