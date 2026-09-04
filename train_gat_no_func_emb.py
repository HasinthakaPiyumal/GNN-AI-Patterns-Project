"""
GAT Classifier WITHOUT Function Text Embeddings (Ablation Study).

Key Characteristics:
1. NO Function Embeddings: Does NOT use SentenceTransformer or any NLP text embeddings for functions.
2. Pure Graph Topology: Node features are computed solely from call graph structure
   (in-degree / callers, out-degree / callees, total degree, entrypoint/leaf indicators).
3. GAT Graph Vector: Multi-Head Graph Attention learns caller-callee relationship weights
   and pools via Mean + Max pooling to produce a 128-d graph representation vector.
4. Gemini Fusion: The 128-d graph vector is fused (concatenated) with the projected 768-d Gemini embedding.
5. MLP Classifier: A multi-layer perceptron predicts the AI design pattern class.
6. Evaluated via Stratified 5-Fold Cross-Validation for exact comparison with train_gat.py.
"""

from __future__ import annotations

import argparse
import ast
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress PyG scatter acceleration warning
warnings.filterwarnings("ignore", message=".*The usage of scatter.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from dataset import (
    DEFAULT_CACHE_PATH,
    DEFAULT_LABELED_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    ASTCallGraphExtractor,
    load_raw_dataset
)
from utils import (
    set_seed,
    calculate_class_weights,
    train_one_epoch,
    evaluate,
    plot_and_save_confusion_matrix,
    print_cross_validation_summary
)

NO_FUNC_CACHE_PATH = str(Path(__file__).resolve().parent / "call_graphs_no_func_cache.pt")


def extract_structural_node_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    feature_type: str = "degree"
) -> torch.Tensor:
    """
    Constructs topological node features without ANY NLP / SentenceTransformer text embeddings:
    - in_degree: number of functions calling this function
    - out_degree: number of functions this function calls
    - total_degree: sum of caller and callee counts
    - is_source: indicator for entry-point functions (in_degree == 0)
    - is_sink: indicator for leaf functions (out_degree == 0)
    - bias: constant 1.0 term for uniform baseline activation
    """
    if feature_type == "constant":
        return torch.ones((num_nodes, 16), dtype=torch.float32)

    # Separate self-loops to accurately measure caller/callee counts
    if edge_index.numel() > 0:
        mask = edge_index[0] != edge_index[1]
        call_edges = edge_index[:, mask]
    else:
        call_edges = edge_index

    in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    out_deg = torch.zeros(num_nodes, dtype=torch.float32)

    if call_edges.numel() > 0:
        caller_nodes = call_edges[0]
        callee_nodes = call_edges[1]
        out_deg.scatter_add_(0, caller_nodes, torch.ones_like(caller_nodes, dtype=torch.float32))
        in_deg.scatter_add_(0, callee_nodes, torch.ones_like(callee_nodes, dtype=torch.float32))

    total_deg = in_deg + out_deg
    is_source = (in_deg == 0).float()
    is_sink = (out_deg == 0).float()
    bias = torch.ones(num_nodes, dtype=torch.float32)

    # Log1p transforms to handle skewed degree distributions smoothly
    feat = torch.stack([
        torch.log1p(in_deg),
        torch.log1p(out_deg),
        torch.log1p(total_deg),
        is_source,
        is_sink,
        bias
    ], dim=-1)  # Shape: [num_nodes, 6]

    return feat


def get_dataset_without_function_embeddings(
    cache_path: str = NO_FUNC_CACHE_PATH,
    source_cache_path: str = DEFAULT_CACHE_PATH,
    labeled_path: str = DEFAULT_LABELED_PATH,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    force_rebuild: bool = False,
    min_samples_per_class: int = 20,
    drop_none: bool = False,
    feature_type: str = "degree"
) -> Tuple[List[Data], LabelEncoder, np.ndarray]:
    """
    Loads or constructs graph dataset where function nodes DO NOT have NLP embeddings.
    Only graph topology + Gemini global embedding are preserved.
    """
    # 1. Check if dedicated no-func-emb cache exists
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached graph dataset (No Function Embeddings) from: {cache_path}")
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cache_data["dataset"], cache_data["label_encoder"], cache_data["classes"]

    # 2. If main cache exists, adapt it by replacing x with structural features in <0.1s
    if os.path.exists(source_cache_path) and not force_rebuild:
        print(f"Adapting existing graph cache from: {source_cache_path} (replacing text embeddings with graph topology)...")
        cache_data = torch.load(source_cache_path, map_location="cpu", weights_only=False)
        orig_dataset = cache_data["dataset"]
        new_dataset: List[Data] = []

        for data in orig_dataset:
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes else data.x.size(0)
            struct_x = extract_structural_node_features(data.edge_index, num_nodes, feature_type=feature_type)
            new_data = Data(
                x=struct_x,
                edge_index=data.edge_index,
                gemini_emb=data.gemini_emb,
                y=data.y,
                num_nodes=num_nodes,
                file_id=getattr(data, "file_id", "")
            )
            new_dataset.append(new_data)

        # Save to dedicated cache
        torch.save({
            "dataset": new_dataset,
            "label_encoder": cache_data["label_encoder"],
            "classes": cache_data["classes"]
        }, cache_path)
        print(f"Saved adapted dataset to: {cache_path}")
        return new_dataset, cache_data["label_encoder"], cache_data["classes"]

    # 3. Build from raw data without calling SentenceTransformer at all
    print("Building graph dataset directly from raw CSVs (SentenceTransformer skipped entirely)...")
    df, label_encoder = load_raw_dataset(
        labeled_path=labeled_path,
        embeddings_path=embeddings_path,
        min_samples_per_class=min_samples_per_class,
        drop_none=drop_none
    )
    dim_cols = [c for c in df.columns if c.startswith("dim_") or c.startswith("emb_")]

    new_dataset = []
    for g_idx, code_text in enumerate(df["code"]):
        extractor = ASTCallGraphExtractor(str(code_text))
        num_nodes = len(extractor.function_texts)

        if extractor.edges:
            edge_arr = np.array(extractor.edges, dtype=np.int64).T
            edge_index = torch.from_numpy(edge_arr)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        struct_x = extract_structural_node_features(edge_index, num_nodes, feature_type=feature_type)
        gemini_vec = torch.tensor(df.iloc[g_idx][dim_cols].values.astype(np.float32)).unsqueeze(0)
        y_val = torch.tensor([df.iloc[g_idx]["target"]], dtype=torch.long)

        data = Data(
            x=struct_x,
            edge_index=edge_index,
            gemini_emb=gemini_vec,
            y=y_val,
            num_nodes=num_nodes,
            file_id=df.iloc[g_idx]["file_clean"]
        )
        new_dataset.append(data)

    classes = label_encoder.classes_
    torch.save({
        "dataset": new_dataset,
        "label_encoder": label_encoder,
        "classes": classes
    }, cache_path)
    print(f"Generated {len(new_dataset)} graphs without function embeddings and saved to {cache_path}")
    return new_dataset, label_encoder, classes


class GATNoFuncEmbClassifier(nn.Module):
    """
    GAT Classifier operating WITHOUT Function Text Embeddings:
    - Node features: 6-dimensional structural graph features (degree/topology).
    - GAT: 2-layer Multi-Head Attention over call edges.
    - Graph Vector: Dual pooling (Mean + Max) -> 128 dimensions.
    - Gemini Fusion: 128-d graph vector + 128-d Gemini embedding = 256 dimensions.
    - MLP: Multi-layer perceptron predicts the pattern category.
    """
    def __init__(
        self,
        node_in_dim: int = 6,
        hidden_dim: int = 64,
        heads: int = 4,
        gemini_dim: int = 768,
        gemini_proj_dim: int = 128,
        num_classes: int = 8,
        dropout: float = 0.25
    ):
        super().__init__()
        self.dropout_rate = dropout

        # Input projection for structural node features: [N, node_in_dim] -> [N, hidden_dim]
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GAT Layer 1: [N, hidden_dim] -> [N, hidden_dim * heads = 256]
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # GAT Layer 2: [N, 256] -> [N, hidden_dim = 64]
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dual pooling: global_mean_pool (64) + global_max_pool (64) = 128
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
        # Project structural features to hidden_dim
        h = self.node_proj(x)

        # GAT Layer 1: Attention message passing across caller -> callee edges
        h = self.gat1(h, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # GAT Layer 2: Refine 2-hop contextual representations
        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = F.elu(h)

        # Dual Global Pooling: Aggregate all functions into 128-d graph vector
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # [B, 128]

        # Project Global Gemini Embedding
        if gemini_emb.dim() == 3 and gemini_emb.size(1) == 1:
            gemini_emb = gemini_emb.squeeze(1)
        h_gemini = self.gemini_projector(gemini_emb)  # [B, 128]

        # Late Fusion: Graph Vector + Gemini Vector
        h_fused = torch.cat([h_graph, h_gemini], dim=-1)  # [B, 256]

        # Classification Head (MLP)
        return self.classifier(h_fused)


def run_cross_validation(
    dataset: List,
    classes: np.ndarray,
    num_classes: int,
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "results_gat_no_func_emb"
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """Runs Stratified 5-Fold Cross-Validation for GAT without function embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(f"\n[GAT No-Func-Emb] Starting 5-Fold Cross-Validation on device: {device}")
    print("Mode: Topology-Only Graph Vector + Gemini Late Fusion + MLP (No Function NLP Embeddings)")

    targets = np.array([data.y.item() for data in dataset])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics: List[Dict[str, float]] = []
    all_oof_preds = np.zeros(len(dataset), dtype=int)
    all_oof_targets = np.zeros(len(dataset), dtype=int)

    node_in_dim = dataset[0].x.size(-1)
    gemini_dim = dataset[0].gemini_emb.size(-1)

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

        # Instantiate Model
        model = GATNoFuncEmbClassifier(
            node_in_dim=node_in_dim,
            gemini_dim=gemini_dim,
            num_classes=num_classes
        ).to(device)

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
                torch.save(model.state_dict(), os.path.join(output_dir, f"gat_no_func_fold_{fold}_best.pt"))

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
    print_cross_validation_summary(fold_metrics, model_name="GAT (No Function Embeddings + Gemini Late Fusion)")

    # Print overall classification report across all folds
    print("\n--- Out-of-Fold Full Classification Report (No Function Embeddings) ---")
    print(classification_report(all_oof_targets, all_oof_preds, target_names=classes, digits=4, zero_division=0))

    # Plot & save confusion matrix
    cm_path = os.path.join(output_dir, "gat_no_func_emb_confusion_matrix.png")
    plot_and_save_confusion_matrix(
        all_oof_targets,
        all_oof_preds,
        list(classes),
        cm_path,
        title="GAT (No Function Embeddings) Out-of-Fold Confusion Matrix"
    )

    return fold_metrics, all_oof_preds, all_oof_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT without Function Text Embeddings (Topology + Gemini Fusion).")
    parser.add_argument("--cache_path", type=str, default=NO_FUNC_CACHE_PATH, help="Path to save/load no-func-emb cache")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--feature_type", type=str, default="degree", choices=["degree", "constant"], help="Node feature type")
    parser.add_argument("--output_dir", type=str, default="results_gat_no_func_emb", help="Output directory")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild of cache")
    parser.add_argument("--min_samples", type=int, default=20, help="Minimum samples per class (default: 20 -> 330 data points)")
    parser.add_argument("--drop_none", action="store_true", default=False, help="Whether to drop 'none' pattern")
    args = parser.parse_args()

    set_seed(42)
    dataset, label_encoder, classes = get_dataset_without_function_embeddings(
        cache_path=args.cache_path,
        force_rebuild=args.force_rebuild,
        min_samples_per_class=args.min_samples,
        drop_none=args.drop_none,
        feature_type=args.feature_type
    )

    run_cross_validation(
        dataset=dataset,
        classes=classes,
        num_classes=len(classes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir
    )
