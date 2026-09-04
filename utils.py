"""
Utility functions for GNN pattern classification:
- Reproducibility seeding
- Loss weighting for class imbalance
- Training & evaluation loops
- Metric calculations (Accuracy, Macro-F1, Weighted-F1)
- Confusion matrix plotting and summary reporting
"""

from __future__ import annotations

import random
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Computes inverse frequency class weights to balance cross-entropy loss:
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Avoid division by zero
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Trains the GNN model for one epoch and returns the average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass: pass node features x, edge_index, batch index, and global Gemini embeddings
        logits = model(batch.x, batch.edge_index, batch.batch, batch.gemini_emb)
        loss = criterion(logits, batch.y)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluates the model and returns (loss, accuracy, macro_f1, weighted_f1, all_preds, all_targets).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.gemini_emb)
        loss = criterion(logits, batch.y)

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = batch.y.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = total_loss / max(total_samples, 1)
    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return avg_loss, acc, macro_f1, weighted_f1, y_pred, y_true


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """Plots and saves normalized confusion matrix as a high-res image."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel("True Dominant Pattern", fontsize=12)
    plt.xlabel("Predicted Pattern", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {output_path}")


def print_cross_validation_summary(fold_metrics: List[Dict[str, float]], model_name: str) -> None:
    """Prints a neat markdown-ready table of 5-fold cross-validation performance."""
    print(f"\n=================================================================")
    print(f"        5-Fold Cross-Validation Final Summary: {model_name}")
    print(f"=================================================================")
    print(f"{'Fold':<8}{'Accuracy':<14}{'Macro F1':<14}{'Weighted F1':<14}")
    print("-" * 50)
    
    accs = [m["accuracy"] for m in fold_metrics]
    macro_f1s = [m["macro_f1"] for m in fold_metrics]
    weighted_f1s = [m["weighted_f1"] for m in fold_metrics]

    for i, m in enumerate(fold_metrics, start=1):
        print(f"Fold {i:<3} {m['accuracy']*100:>6.2f}%       {m['macro_f1']*100:>6.2f}%       {m['weighted_f1']*100:>6.2f}%")

    print("-" * 50)
    print(f"{'Mean':<8}{np.mean(accs)*100:>6.2f}%       {np.mean(macro_f1s)*100:>6.2f}%       {np.mean(weighted_f1s)*100:>6.2f}%")
    print(f"{'Std':<8}{np.std(accs)*100:>6.2f}%       {np.std(macro_f1s)*100:>6.2f}%       {np.std(weighted_f1s)*100:>6.2f}%")
    print(f"=================================================================\n")
