"""
Graph Isomorphism Network (GIN) Classifier for Code Communities.

Key Architectural Decisions:
1. Theoretical Expressiveness: GIN is provably as powerful as the Weisfeiler-Lehman (1-WL) graph
   isomorphism test. Using sum aggregation with Multi-Layer Perceptrons (MLP), it differentiates
   complex multi-function call topologies that mean/max aggregators cannot distinguish.
2. Sum Graph Pooling (global_add_pool): Retains the full multiset structural information of
   functions and call counts across the entire code community.
3. Learnable Epsilon (train_eps=True): Allows the network to dynamically weight a function's
   own internal representation versus its incoming call neighborhood.
4. Late Fusion with Gemini Call-Graph Embeddings: Bridges topological graph reasoning with the
   high-level semantic embedding of the community.
5. Stratified 5-Fold Cross-Validation: Evaluates generalization across all verified communities
   with class-balanced splits and weighted cross-entropy loss.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List, Tuple

# Suppress PyG scatter acceleration warning
warnings.filterwarnings("ignore", message=".*The usage of scatter.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
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


class GINCommunityClassifier(nn.Module):
    """
    Two-layer Graph Isomorphism Network with Sum Pooling and Late Fusion.
    """
    def __init__(
        self,
        node_in_dim: int = 384,
        hidden_dim: int = 128,
        gemini_dim: int = 768,
        gemini_proj_dim: int = 128,
        num_classes: int = 8,
        dropout: float = 0.25
    ):
        super().__init__()
        self.dropout_rate = dropout

        # MLP 1 for GIN Layer 1: [384] -> [hidden_dim]
        mlp1 = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gin1 = GINConv(mlp1, train_eps=True)

        # MLP 2 for GIN Layer 2: [hidden_dim] -> [hidden_dim]
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gin2 = GINConv(mlp2, train_eps=True)

        # Global Gemini Call-Graph Embedding Projection: [B, 768] -> [B, 128]
        self.gemini_projector = nn.Sequential(
            nn.Linear(gemini_dim, gemini_proj_dim),
            nn.LayerNorm(gemini_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Late Fusion Classifier Head: [B, 128 (Graph) + 128 (Gemini) = 256] -> [B, num_classes]
        fusion_dim = hidden_dim + gemini_proj_dim
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
        # GIN Layer 1
        h = self.gin1(x, edge_index)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # GIN Layer 2
        h = self.gin2(h, edge_index)

        # Graph Pooling: Sum pooling preserves multiset structures
        h_graph = global_add_pool(h, batch)  # [B, hidden_dim = 128]

        # Project Global Gemini Embedding
        if gemini_emb.dim() == 3:
            gemini_emb = gemini_emb.squeeze(1)
        h_gemini = self.gemini_projector(gemini_emb)  # [B, 128]

        # Late Fusion: Concatenate topological graph representation and global Gemini embedding
        h_fused = torch.cat([h_graph, h_gemini], dim=-1)  # [B, 256]

        # Classification Logits
        return self.classifier(h_fused)


def run_gin_cross_validation(
    dataset: List,
    classes: np.ndarray,
    num_classes: int,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "results_gin"
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """Runs Stratified 5-Fold Cross-Validation for GIN."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(f"\n[GIN] Starting 5-Fold Cross-Validation on device: {device}")

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

        # Class weights for imbalanced distributions
        train_labels = [dataset[i].y.item() for i in train_idx]
        class_weights = calculate_class_weights(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Instantiate GIN Model & Optimizer (dynamically adapts to node feature dimension)
        node_in_dim = dataset[0].x.size(-1)
        model = GINCommunityClassifier(node_in_dim=node_in_dim, num_classes=num_classes).to(device)
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
                torch.save(model.state_dict(), os.path.join(output_dir, f"gin_fold_{fold}_best.pt"))

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
    print_cross_validation_summary(fold_metrics, model_name="GIN (Graph Isomorphism Network)")

    # Print overall classification report across all folds
    print("\n--- Out-of-Fold Full Classification Report ---")
    print(classification_report(all_oof_targets, all_oof_preds, target_names=classes, digits=4, zero_division=0))

    # Plot & save confusion matrix
    cm_path = os.path.join(output_dir, "gin_confusion_matrix.png")
    plot_and_save_confusion_matrix(all_oof_targets, all_oof_preds, list(classes), cm_path, title="GIN Out-of-Fold Confusion Matrix")

    return fold_metrics, all_oof_preds, all_oof_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GIN Classifier for Code Communities.")
    parser.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH, help="Path to cached dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="results_gin", help="Directory to save model weights and plots")
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

    run_gin_cross_validation(
        dataset=dataset,
        classes=classes,
        num_classes=len(classes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir
    )
