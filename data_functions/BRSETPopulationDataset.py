import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.metrics.pairwise import euclidean_distances

def build_population_graph(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    image_dir: Path,
    target_cols: List[str],
    demo_cols: List[str],
    k_neighbors: int = 5,
    multiscale: bool = False
) -> HeteroData:
    """
    Constructs a unified Population Hypergraph from train and validation splits.
    Uses 'train_mask' and 'val_mask' to safely differentiate splits.
    """
    train_frame = train_frame.copy()
    val_frame = val_frame.copy()
    
    train_frame["split"] = "train"
    val_frame["split"] = "val"
    
    df = pd.concat([train_frame, val_frame], ignore_index=True)
    num_images = len(df)
    
    data = HeteroData()
    
    # 1. Image Nodes
    data["image"].x = torch.zeros((num_images, 1), dtype=torch.float32)
    data["image"].image_idx = torch.arange(num_images, dtype=torch.long)
    
    train_mask = torch.tensor(df["split"] == "train", dtype=torch.bool)
    val_mask = torch.tensor(df["split"] == "val", dtype=torch.bool)
    
    data["image"].train_mask = train_mask
    data["image"].val_mask = val_mask
    
    # Target vectors for calculation
    labels = torch.tensor(df[target_cols].to_numpy(dtype=np.float32))
    data["image"].y = labels
    
    # Extract paths for lazy loading
    paths = []
    for _, row in df.iterrows():
        raw_path = row.get("image_path", f"fundus_photos/{row['image_id']}.jpg")
        p = Path(raw_path)
        p = p if p.is_absolute() else Path(image_dir) / p
        paths.append(str(p))
    
    data["image"].image_paths = paths
    
    # 2. Multiscale Patch Nodes & Edges (Optional)
    if multiscale:
        num_patches = num_images * 64 # 8x8 grid for 256x256 image
        data["patch"].x = torch.zeros((num_patches, 1), dtype=torch.float32)
        data["patch"].patch_idx = torch.arange(num_patches, dtype=torch.long)
        
        # Patch Adjacency (8x8 grid)
        p_src, p_dst = [], []
        for i in range(num_images):
            base = i * 64
            for row in range(8):
                for col in range(8):
                    curr = base + row * 8 + col
                    if row > 0: # up
                        p_src.append(curr); p_dst.append(curr - 8)
                    if row < 7: # down
                        p_src.append(curr); p_dst.append(curr + 8)
                    if col > 0: # left
                        p_src.append(curr); p_dst.append(curr - 1)
                    if col < 7: # right
                        p_src.append(curr); p_dst.append(curr + 1)
        
        data[("patch", "adjacent", "patch")].edge_index = torch.tensor([p_src, p_dst], dtype=torch.long)
        
        # Patch <-> Image Hierarchical Edges
        pi_src, pi_dst = [], []
        for i in range(num_images):
            base = i * 64
            for j in range(64):
                pi_src.append(base + j)
                pi_dst.append(i)
                
        data[("patch", "to", "image")].edge_index = torch.tensor([pi_src, pi_dst], dtype=torch.long)
        data[("image", "to", "patch")].edge_index = torch.tensor([pi_dst, pi_src], dtype=torch.long)

    # 3. Same Patient Edges
    if "patient_id" in df.columns:
        sp_src, sp_dst = [], []
        for pid, group in df.groupby("patient_id"):
            indices = group.index.tolist()
            if len(indices) > 1:
                for idx1 in indices:
                    for idx2 in indices:
                        if idx1 != idx2:
                            sp_src.append(idx1)
                            sp_dst.append(idx2)
        if len(sp_src) > 0:
            data[("image", "same_patient", "image")].edge_index = torch.tensor([sp_src, sp_dst], dtype=torch.long)
        else:
            data[("image", "same_patient", "image")].edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        data[("image", "same_patient", "image")].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 4. Context Nodes (Hyperedges)
    # Process age into decades
    if "patient_age" in df.columns:
        df["age_decade"] = (df["patient_age"] // 10 * 10).astype(str) + "s"
        df["age_decade"] = df["age_decade"].replace("nans", "UnknownAge")
        demo_cols_to_use = [c for c in demo_cols if c != "patient_age"] + ["age_decade"]
    else:
        demo_cols_to_use = demo_cols
        
    context_nodes = []
    context_to_idx = {}
    
    for col in demo_cols_to_use:
        unique_vals = df[col].dropna().unique()
        for val in unique_vals:
            node_name = f"{col}:{val}"
            context_to_idx[node_name] = len(context_nodes)
            context_nodes.append(node_name)
            
    num_contexts = len(context_nodes)
    if num_contexts > 0:
        data["context"].x = torch.zeros((num_contexts, 1), dtype=torch.float32)
        data["context"].context_idx = torch.arange(num_contexts, dtype=torch.long)
    
    # 5. Label Nodes
    num_labels = len(target_cols)
    data["label"].x = torch.zeros((num_labels, 1), dtype=torch.float32)
    data["label"].label_idx = torch.arange(num_labels, dtype=torch.long)
    
    # 6. Image <-> Context Edges
    img_ctx_src, img_ctx_dst = [], []
    for i, row in df.iterrows():
        for col in demo_cols_to_use:
            val = row[col]
            if pd.notna(val):
                node_name = f"{col}:{val}"
                if node_name in context_to_idx:
                    img_ctx_src.append(i)
                    img_ctx_dst.append(context_to_idx[node_name])
                    
    if len(img_ctx_src) > 0:
        data[("image", "has_context", "context")].edge_index = torch.tensor([img_ctx_src, img_ctx_dst], dtype=torch.long)
        data[("context", "to", "image")].edge_index = torch.tensor([img_ctx_dst, img_ctx_src], dtype=torch.long)
    else:
        data[("image", "has_context", "context")].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[("context", "to", "image")].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # 7. Image <-> Label Edges (ONLY FOR TRAINING DATA TO PREVENT LEAKAGE!)
    img_lbl_src, img_lbl_dst = [], []
    for i, row in df.iterrows():
        if row["split"] == "train":
            for l_idx, col in enumerate(target_cols):
                if row[col] == 1:
                    img_lbl_src.append(i)
                    img_lbl_dst.append(l_idx)
                    
    if len(img_lbl_src) > 0:
        data[("image", "has_label", "label")].edge_index = torch.tensor([img_lbl_src, img_lbl_dst], dtype=torch.long)
        data[("label", "to", "image")].edge_index = torch.tensor([img_lbl_dst, img_lbl_src], dtype=torch.long)
    else:
        data[("image", "has_label", "label")].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[("label", "to", "image")].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # 8. Macro Edges (Demographic k-NN)
    if num_contexts > 0:
        demo_matrix = np.zeros((num_images, num_contexts))
        for i, row in df.iterrows():
            for col in demo_cols_to_use:
                val = row[col]
                if pd.notna(val):
                    node_name = f"{col}:{val}"
                    if node_name in context_to_idx:
                        demo_matrix[i, context_to_idx[node_name]] = 1.0
                        
        dist = euclidean_distances(demo_matrix, demo_matrix)
        np.fill_diagonal(dist, np.inf) # Don't connect to self
        
        knn_src, knn_dst = [], []
        for i in range(num_images):
            k = min(k_neighbors, num_images - 1)
            top_k = np.argsort(dist[i])[:k]
            knn_src.extend([i] * k)
            knn_dst.extend(top_k.tolist())
            
        data[("image", "similar_to", "image")].edge_index = torch.tensor([knn_src, knn_dst], dtype=torch.long)
    else:
        data[("image", "similar_to", "image")].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data
