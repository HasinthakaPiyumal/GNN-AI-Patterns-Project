"""
Graph Attention Network (GAT) Classifier for Code Communities.

Key Architectural Decisions:
1. Multi-Head Attention: Computes dynamic attention coefficients between calling and called functions,
   identifying which function interactions are critical to distinguishing design patterns.
2. Dual Graph Pooling (Mean + Max): Concurrently captures overall community intent (mean)
   and the strongest structural pattern signature / hub function (max).
3. Late Fusion: Combines the graph-level topology embedding with the precomputed 768-d
   Gemini global call-graph embedding.
4. Stratified 5-Fold Cross-Validation: Evaluates generalization across all verified communities
   with class-balanced splits and weighted cross-entropy loss.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from dataset import get_dataset, DEFAULT_CACHE_PATH
from utils import (
    set_seed,
    calculate_class_weights,
    train_one_epoch,
    evaluate,
    plot_and_save_confusion_matrix,
    print_cross_validation_summary
)


class GATCommunityClassifier(nn.Module):
    """
    Two-layer Multi-Head Graph Attention Network with Late Fusion.
    """
    def __init__(
        self,
        node_in_dim: int = 384,
        hidden_dim: int = 64,
        heads: int = 4,
        gemini_dim: int = 768,
        gemini_proj_dim: int = 512,
        num_classes: int = 8,
        dropout: float = 0.25
    ):
        super().__init__()
        self.dropout_rate = dropout

        # GAT Layer 1: [N, 384] -> [N, 64 * 4 = 256]
        self.gat1 = GATConv(
            in_channels=node_in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # GAT Layer 2: [N, 256] -> [N, 64] (averaged across heads)
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Graph pooling combines Mean + Max pool: 64 + 64 = 128
        graph_pooled_dim = hidden_dim * 2

        # Global Gemini Call-Graph Embedding Projection: [B, 768] -> [B, 128]
        self.gemini_projector = nn.Sequential(
            nn.Linear(gemini_dim, gemini_proj_dim),
            nn.LayerNorm(gemini_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Late Fusion Classifier Head: [B, 128 (Graph) + 128 (Gemini) = 256] -> [B, num_classes]
        fusion_dim = graph_pooled_dim + gemini_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        gemini_emb: torch.Tensor
    ) -> torch.Tensor:
        # GAT Layer 1
        h = self.gat1(x, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # GAT Layer 2
        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = F.elu(h)

        # Dual Global Pooling over nodes in each graph
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # [B, 128]

        # Project Global Gemini Embedding
        # Reshape gemini_emb if batching introduces an extra dimension [B, 1, 768] -> [B, 768]
        if gemini_emb.dim() == 3:
            gemini_emb = gemini_emb.squeeze(1)
        h_gemini = self.gemini_projector(gemini_emb)  # [B, 128]

        # Late Fusion: Concatenate graph topological features with global summary features
        h_fused = torch.cat([h_graph, h_gemini], dim=-1)  # [B, 256]

        # Classification Logits
        return self.classifier(h_fused)


def run_gat_cross_validation(
    dataset: List,
    classes: np.ndarray,
    num_classes: int,
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "results_gat"
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """Runs Stratified 5-Fold Cross-Validation for GAT."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(f"\n[GAT] Starting 5-Fold Cross-Validation on device: {device}")

    # Extract targets for Stratified K-Fold
    targets = np.array([data.y.item() for data in dataset])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics: List[Dict[str, float]] = []
    all_oof_preds = np.zeros(len(dataset), dtype=int)
    all_oof_targets = np.zeros(len(dataset), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, targets), start=1):
        print(f"\n---------------- Fold {fold} / 5 ----------------")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Weighted loss to counteract class imbalance
        train_labels = [dataset[i].y.item() for i in train_idx]
        class_weights = calculate_class_weights(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Instantiate Model & Optimizer (dynamically adapts to node feature dimension)
        node_in_dim = dataset[0].x.size(-1)
        model = GATCommunityClassifier(node_in_dim=node_in_dim, num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_macro_f1 = 0.0
        best_metrics: Dict[str, float] = {}
        best_preds = None

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_macro_f1, val_weighted_f1, preds, y_val = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_metrics = {
                    "accuracy": val_acc,
                    "macro_f1": val_macro_f1,
                    "weighted_f1": val_weighted_f1,
                    "val_loss": val_loss
                }
                best_preds = preds
                # Save best fold weights
                torch.save(model.state_dict(), os.path.join(output_dir, f"gat_fold_{fold}_best.pt"))

            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc*100:.1f}% | "
                    f"Val Macro-F1: {val_macro_f1*100:.1f}%"
                )

        print(
            f"Fold {fold} Best Result -> "
            f"Accuracy: {best_metrics['accuracy']*100:.2f}%, "
            f"Macro-F1: {best_metrics['macro_f1']*100:.2f}%, "
            f"Weighted-F1: {best_metrics['weighted_f1']*100:.2f}%"
        )
        fold_metrics.append(best_metrics)
        all_oof_preds[val_idx] = best_preds
        all_oof_targets[val_idx] = targets[val_idx]

    # Print summary table
    print_cross_validation_summary(fold_metrics, model_name="GAT (Graph Attention Network)")

    # Print overall classification report across all folds
    print("\n--- Out-of-Fold Full Classification Report ---")
    print(classification_report(all_oof_targets, all_oof_preds, target_names=classes, digits=4, zero_division=0))

    # Plot & save confusion matrix
    cm_path = os.path.join(output_dir, "gat_confusion_matrix.png")
    plot_and_save_confusion_matrix(all_oof_targets, all_oof_preds, list(classes), cm_path, title="GAT Out-of-Fold Confusion Matrix")

    return fold_metrics, all_oof_preds, all_oof_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT Classifier for Code Communities.")
    parser.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH, help="Path to cached dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="results_gat", help="Directory to save model weights and plots")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild of graph dataset cache")
    parser.add_argument("--min_samples", type=int, default=20, help="Minimum samples per pattern class (default: 20 -> 330 data points)")
    parser.add_argument("--drop_none", action="store_true", default=False, help="Whether to drop 'none' pattern (default: False)")
    args = parser.parse_args()

    set_seed(42)
    dataset, label_encoder, classes = get_dataset(
        cache_path=args.cache_path,
        force_rebuild=args.force_rebuild,
        min_samples_per_class=args.min_samples,
        drop_none=args.drop_none
    )

    run_gat_cross_validation(
        dataset=dataset,
        classes=classes,
        num_classes=len(classes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir
    )
