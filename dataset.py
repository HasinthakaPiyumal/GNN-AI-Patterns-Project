"""
Dataset Loader, AST Call Graph Extractor, and Feature Builder for Code Communities.

This module parses raw Python code clusters into graph representations:
- Nodes: Functions/Methods defined within the community.
- Edges: Function invocations / calls between functions.
- Node Features: 384-dimensional text embeddings of function signatures, docstrings,
  and code bodies generated via SentenceTransformer ('all-MiniLM-L6-v2').
- Global Graph Features: 768-dimensional Gemini call-graph embeddings from embeddings_gem-emb-2.csv.
- Targets: Design pattern category labels, filtered for well-supported classes.
"""

from __future__ import annotations

import ast
import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Default dataset paths pointing to local project folder
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELED_PATH = str(BASE_DIR / "data" / "labeled_verified_data.csv")
DEFAULT_EMBEDDINGS_PATH = str(BASE_DIR / "data" / "embeddings_gem-emb-2.csv")
DEFAULT_CACHE_PATH = str(BASE_DIR / "call_graphs_cache.pt")


class ASTCallGraphExtractor:
    """
    Extracts functions as nodes and internal function calls as edges using Python AST.
    """

    def __init__(self, code_str: str):
        self.code_str = code_str
        self.functions: Dict[str, int] = {}  # qualified name -> node index
        self.function_texts: List[str] = []
        self.edges: List[Tuple[int, int]] = []
        self._parse()

    def _parse(self) -> None:
        try:
            tree = ast.parse(self.code_str)
        except Exception:
            # Fallback if unparsable
            self.function_texts.append("unknown_function: pass")
            self.functions["unknown_function"] = 0
            self.edges.append((0, 0))
            return

        func_info: List[Dict[str, Any]] = []

        class DefinitionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_class: Optional[str] = None

            def visit_ClassDef(self, node: ast.ClassDef):
                prev = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._record_func(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self._record_func(node)
                self.generic_visit(node)

            def _record_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
                qual_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
                docstring = ast.get_docstring(node) or ""
                
                # Extract arguments as string
                args_list = [arg.arg for arg in node.args.args]
                args_str = ", ".join(args_list)

                # Extract first few non-docstring statements for context
                body_lines = []
                for stmt in node.body[:3]:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        continue  # Skip docstring statement
                    try:
                        body_lines.append(ast.unparse(stmt))
                    except Exception:
                        pass
                body_snippet = "; ".join(body_lines)[:150]

                # Combined text summary for node embedding
                text_repr = f"def {qual_name}({args_str}): {docstring} | {body_snippet}".strip()
                if not text_repr or text_repr == f"def {qual_name}({args_str}): |":
                    text_repr = f"function {qual_name}"

                func_info.append({
                    "name": qual_name,
                    "short_name": node.name,
                    "text": text_repr,
                    "ast_node": node
                })

        visitor = DefinitionVisitor()
        visitor.visit(tree)

        if not func_info:
            # If no functions/classes defined, create a root placeholder node
            self.function_texts.append("community_root: entrypoint")
            self.functions["community_root"] = 0
            self.edges.append((0, 0))
            return

        # Map function names to index
        for idx, item in enumerate(func_info):
            self.functions[item["name"]] = idx
            self.function_texts.append(item["text"])

        # Name lookup mapping (both full qualified name and short name)
        name_to_idx: Dict[str, int] = {}
        for idx, item in enumerate(func_info):
            name_to_idx[item["name"]] = idx
            name_to_idx[item["short_name"]] = idx

        # Second pass: Collect calls (caller -> callee)
        for caller_idx, item in enumerate(func_info):
            fn_node = item["ast_node"]
            for call_node in ast.walk(fn_node):
                if isinstance(call_node, ast.Call):
                    callee_name = None
                    if isinstance(call_node.func, ast.Name):
                        callee_name = call_node.func.id
                    elif isinstance(call_node.func, ast.Attribute):
                        callee_name = call_node.func.attr

                    if callee_name and callee_name in name_to_idx:
                        callee_idx = name_to_idx[callee_name]
                        self.edges.append((caller_idx, callee_idx))

        # Guarantee graph connectivity: add self-loops so isolated functions can propagate
        num_nodes = len(self.function_texts)
        for i in range(num_nodes):
            self.edges.append((i, i))

        # Remove duplicate edges
        self.edges = list(set(self.edges))


def load_raw_dataset(
    labeled_path: str = DEFAULT_LABELED_PATH,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    min_samples_per_class: int = 20,
    drop_none: bool = False
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Loads labeled verified data, merges with graph embeddings, and filters classes.
    """
    print(f"Loading verified dataset from: {labeled_path}")
    df_ver = pd.read_csv(labeled_path)
    
    print(f"Loading global call graph embeddings from: {embeddings_path}")
    df_emb = pd.read_csv(embeddings_path)

    # Clean URL/path strings to match reliably on 'file'
    df_ver["file_clean"] = df_ver["file"].astype(str).str.strip()
    df_emb["file_clean"] = df_emb["file"].astype(str).str.strip()

    merged = pd.merge(df_ver, df_emb, on="file_clean", suffixes=("", "_emb"))
    print(f"Total merged data points: {len(merged)}")

    # Drop 'none' category if requested
    if drop_none:
        merged = merged[merged["label"].str.lower() != "none"].copy()
        print(f"Samples after dropping 'none': {len(merged)}")

    # Filter out rare classes with fewer than min_samples_per_class
    class_counts = merged["label"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    merged = merged[merged["label"].isin(valid_classes)].copy()
    
    print(f"\nFiltered to {len(valid_classes)} classes with >= {min_samples_per_class} samples (Total: {len(merged)} samples):")
    for cls_name, count in merged["label"].value_counts().items():
        print(f"  - {cls_name}: {count}")

    # Encode labels
    label_encoder = LabelEncoder()
    merged["target"] = label_encoder.fit_transform(merged["label"])

    return merged.reset_index(drop=True), label_encoder


def build_pyg_graphs(
    df: pd.DataFrame,
    sentence_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Data]:
    """
    Extracts AST call graphs and encodes function nodes using SentenceTransformer.
    """
    print(f"\nLoading SentenceTransformer encoder: {sentence_model_name} on {device}...")
    encoder = SentenceTransformer(sentence_model_name, device=device)

    # Identify the 768 embedding columns
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if not dim_cols:
        dim_cols = [c for c in df.columns if c.startswith("emb_")]
    print(f"Found {len(dim_cols)} dimensions for global Gemini graph embeddings.")

    # 1. Parse AST for all code communities
    print("Parsing AST call graphs for each code community...")
    graph_extractors: List[ASTCallGraphExtractor] = []
    for code_text in tqdm(df["code"], desc="AST parsing"):
        graph_extractors.append(ASTCallGraphExtractor(str(code_text)))

    # 2. Gather all function texts across all graphs for efficient batch embedding
    all_texts: List[str] = []
    text_to_graph_map: List[int] = []  # maps text index to graph index

    for g_idx, extractor in enumerate(graph_extractors):
        for txt in extractor.function_texts:
            all_texts.append(txt)
            text_to_graph_map.append(g_idx)

    print(f"Embedding {len(all_texts)} total function nodes across {len(df)} communities...")
    node_embeddings = encoder.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    ).cpu().float()

    # 3. Assemble PyG Data objects
    print("Constructing PyTorch Geometric Data objects...")
    pyg_dataset: List[Data] = []
    emb_ptr = 0

    for g_idx, extractor in enumerate(graph_extractors):
        num_nodes = len(extractor.function_texts)
        node_feats = node_embeddings[emb_ptr : emb_ptr + num_nodes]
        emb_ptr += num_nodes

        # Construct edge index tensor
        if extractor.edges:
            edge_arr = np.array(extractor.edges, dtype=np.int64).T
            edge_index = torch.from_numpy(edge_arr)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Global Gemini embedding vector
        gemini_vec = torch.tensor(df.iloc[g_idx][dim_cols].values.astype(np.float32)).unsqueeze(0)

        # Target label
        y_val = torch.tensor([df.iloc[g_idx]["target"]], dtype=torch.long)

        data = Data(
            x=node_feats,
            edge_index=edge_index,
            gemini_emb=gemini_vec,
            y=y_val,
            num_nodes=num_nodes,
            file_id=df.iloc[g_idx]["file_clean"]
        )
        pyg_dataset.append(data)

    print(f"Successfully generated {len(pyg_dataset)} graph data objects.")
    return pyg_dataset


def get_dataset(
    cache_path: str = DEFAULT_CACHE_PATH,
    labeled_path: str = DEFAULT_LABELED_PATH,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    force_rebuild: bool = False,
    min_samples_per_class: int = 20,
    drop_none: bool = False,
    sentence_model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[List[Data], LabelEncoder, np.ndarray]:
    """
    Main entry point: Loads cached PyG graph dataset from disk if present,
    otherwise parses AST, generates embeddings, and saves to cache.
    """
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached graph dataset from: {cache_path}")
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        dataset = cache_data["dataset"]
        label_encoder = cache_data["label_encoder"]
        classes = cache_data["classes"]
        print(f"Loaded {len(dataset)} graphs across {len(classes)} classes from cache.")
        return dataset, label_encoder, classes

    # Rebuild from scratch
    print(f"No cache found at {cache_path} (or force_rebuild=True). Building dataset...")
    df, label_encoder = load_raw_dataset(
        labeled_path=labeled_path,
        embeddings_path=embeddings_path,
        min_samples_per_class=min_samples_per_class,
        drop_none=drop_none
    )
    dataset = build_pyg_graphs(df, sentence_model_name=sentence_model_name, device=device)

    # Save cache
    print(f"Caching processed graph dataset to: {cache_path}")
    torch.save(
        {
            "dataset": dataset,
            "label_encoder": label_encoder,
            "classes": label_encoder.classes_
        },
        cache_path
    )
    print("Cache saved successfully.")
    return dataset, label_encoder, label_encoder.classes_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract call graphs and build cached PyG dataset.")
    parser.add_argument("--labeled_path", type=str, default=DEFAULT_LABELED_PATH, help="Path to labeled_verified_data.csv")
    parser.add_argument("--embeddings_path", type=str, default=DEFAULT_EMBEDDINGS_PATH, help="Path to embeddings_gem-emb-2.csv")
    parser.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH, help="Path to save/load processed_graphs.pt")
    parser.add_argument("--force_rebuild", action="store_true", help="Force recomputation of AST and embeddings")
    parser.add_argument("--min_samples", type=int, default=20, help="Minimum samples per pattern class (default: 20 -> 330 data points)")
    parser.add_argument("--drop_none", action="store_true", default=False, help="Whether to drop 'none' pattern (default: False)")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="HuggingFace model for function node embeddings")
    args = parser.parse_args()

    dataset, le, classes = get_dataset(
        cache_path=args.cache_path,
        labeled_path=args.labeled_path,
        embeddings_path=args.embeddings_path,
        force_rebuild=args.force_rebuild,
        min_samples_per_class=args.min_samples,
        drop_none=args.drop_none,
        sentence_model_name=args.model_name
    )

    print("\n--- Dataset Summary ---")
    print(f"Total Graphs: {len(dataset)}")
    print(f"Number of Classes: {len(classes)}")
    avg_nodes = np.mean([d.num_nodes for d in dataset])
    avg_edges = np.mean([d.edge_index.size(1) for d in dataset])
    print(f"Average Nodes (Functions) per Community: {avg_nodes:.2f}")
    print(f"Average Edges (Calls + Self-loops) per Community: {avg_edges:.2f}")
    print(f"Classes: {list(classes)}")
