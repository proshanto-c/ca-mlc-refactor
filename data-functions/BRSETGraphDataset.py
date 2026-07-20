import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, List, Dict, Union, Optional, Callable

def patchify(image: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if image.dim() != 3:
        raise ValueError("Expected image tensor shaped [C, H, W].")

    c, h, w = image.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h))
        _, h, w = image.shape

    grid_h = h // patch_size
    grid_w = w // patch_size

    patches = (
        image.unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
    ).view(grid_h * grid_w, c * patch_size * patch_size)

    coords = []
    for r in range(grid_h):
        for c_ in range(grid_w):
            coords.append(
                [
                    0.0 if grid_h == 1 else r / float(grid_h - 1),
                    0.0 if grid_w == 1 else c_ / float(grid_w - 1),
                ]
            )
    coord_tensor = torch.tensor(coords, dtype=patches.dtype, device=patches.device)
    return patches, coord_tensor, grid_h, grid_w


def grid_edge_index(grid_h: int, grid_w: int, connectivity: int, device=None) -> torch.Tensor:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    neighbors_4 = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    neighbors_8 = neighbors_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = neighbors_8 if connectivity == 8 else neighbors_4

    def idx(r: int, c: int) -> int:
        return r * grid_w + c

    src, dst = [], []
    for r in range(grid_h):
        for c in range(grid_w):
            i = idx(r, c)
            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid_h and 0 <= cc < grid_w:
                    src.append(i)
                    dst.append(idx(rr, cc))

    return torch.tensor([src, dst], dtype=torch.long, device=device)

class BRSETGraphDataset(Dataset):
    """
    Heterogeneous Graph Dataset for Patient-Level BRSET predictions.
    Consumes a DataFrame and groups by patient_id to build patient-centric graphs.
    """
    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        image_dir: Path,
        target_cols: List[str],
        demo_cols: List[str],
        demo_specs: List[Any], # Passed in from a demographics encoder utility
        image_size: int = 256,
        patch_size: int = 32,
        connectivity: int = 4,
        transform: Optional[Callable] = None,
        validate_paths: bool = True
    ) -> None:
        
        if isinstance(data, (Path, str)):
            self.frame = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.frame = data.copy()
        else:
            raise ValueError("data must be a path-like string/Path or a pandas DataFrame.")
            
        self.image_dir = Path(image_dir)
        self.target_cols = target_cols
        self.demo_cols = demo_cols
        self.demo_specs = demo_specs
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.connectivity = connectivity
        self.transform = transform
        
        # Build grouped patient samples in memory for fast __getitem__
        self.samples = []
        self._build_patient_samples(validate_paths)

    def _build_patient_samples(self, validate: bool) -> None:
        """Groups the flat DataFrame into patient-level data structures."""
        for patient_id, group in self.frame.groupby("patient_id", sort=True):
            image_paths = []
            eye_ids = []
            
            for _, row in group.iterrows():
                # Path resolution
                raw_path = row.get("image_path", f"fundus_photos/{row['image_id']}.jpg")
                p = Path(raw_path)
                p = p if p.is_absolute() else self.image_dir / p
                
                if validate and not p.is_file():
                    raise FileNotFoundError(f"Missing image: {p}")
                
                image_paths.append(p)
                
                # Encode eye (Left/Right) if available
                eye_val = row.get("exam_eye", -1)
                eye_ids.append(int(float(eye_val)) if pd.notna(eye_val) else -1)
            
            # Aggregate targets (If any image is positive for a disease, patient is positive)
            label_vector = group[self.target_cols].max().to_numpy(dtype=np.float32)
            
            # Aggregate demographics (take the mode or first available for the patient)
            demographic_values = {
                col: group[col].dropna().mode()[0] if not group[col].dropna().empty else None 
                for col in self.demo_cols
            }
            
            self.samples.append({
                "patient_id": str(patient_id),
                "image_paths": image_paths,
                "eye_ids": eye_ids,
                "label_vector": label_vector,
                "demographics": demographic_values
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> HeteroData:
        sample = self.samples[index]
        data = HeteroData()
        
        # 1. Global Target (Shape: [1, Num_Labels])
        data.y = torch.tensor(sample["label_vector"], dtype=torch.float32).unsqueeze(0)
        
        # 2. Setup Base Nodes (Patient and Labels)
        num_labels = len(self.target_cols)
        data["patient"].x = torch.zeros((1, 1), dtype=torch.float32)
        
        data["label"].x = torch.zeros((num_labels, 1), dtype=torch.float32)
        data["label"].label_idx = torch.arange(num_labels, dtype=torch.long)
        
        # 3. Setup Demographic Nodes
        self._attach_demographics(data, sample["demographics"])
        
        # 4. Setup Image & Patch Nodes with Edge Connections
        self._attach_vision_nodes(data, sample)
        
        # 5. Setup Structural Edges (Demographics, Labels, Patient)
        self._attach_structural_edges(data, num_labels)
        
        return data

    def _attach_demographics(self, data: HeteroData, demo_dict: Dict) -> None:
            """Encodes demographic specs into node features and attaches them to the graph."""
            num_demo = len(self.demo_specs)
            
            field_idx = torch.arange(num_demo, dtype=torch.long)
            kind = torch.tensor([0 if s.kind == "numeric" else 1 for s in self.demo_specs], dtype=torch.long)
            num_value = torch.zeros((num_demo, 1), dtype=torch.float32)
            cat_value = torch.zeros((num_demo,), dtype=torch.long)
            missing = torch.zeros((num_demo, 1), dtype=torch.float32)

            for i, spec in enumerate(self.demo_specs):
                value = demo_dict.get(spec.name, None)
                
                if spec.kind == "numeric":
                    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                    if pd.isna(num):
                        missing[i, 0] = 1.0
                    else:
                        num_value[i, 0] = (float(num) - spec.mean) / spec.std
                else:
                    token = "__UNK__" if value is None or pd.isna(value) else str(value)
                    cat_value[i] = spec.vocab.get(token, 0) if spec.vocab else 0
                    if token == "__UNK__":
                        missing[i, 0] = 1.0

            # Create the demographic node hub in the HeteroData object
            data["demographic"].x = torch.zeros((num_demo, 1), dtype=torch.float32)
            data["demographic"].field_idx = field_idx
            data["demographic"].kind = kind
            data["demographic"].num_value = num_value
            data["demographic"].cat_value = cat_value
            data["demographic"].missing = missing

    def _attach_vision_nodes(self, data: HeteroData, sample: Dict) -> None:
            """Loads images, creates patches, and draws edges between patches, images, and the patient."""
            num_images = len(sample["image_paths"])
            
            # 1. Initialize Base Image Nodes
            data["image"].x = torch.zeros((num_images, 1), dtype=torch.float32)
            data["image"].image_idx = torch.arange(num_images, dtype=torch.long)
            data["image"].eye_idx = torch.tensor(
                [0 if v in (-1, None) else max(0, int(v) - 1) for v in sample["eye_ids"]],
                dtype=torch.long
            )
            
            patch_features_list: List[torch.Tensor] = []
            
            # Edge accumulators
            patch_adj_src, patch_adj_dst = [], []
            patch_to_img_src, patch_to_img_dst = [], []
            img_to_patch_src, img_to_patch_dst = [], []
            img_to_patient_src, img_to_patient_dst = [], []
            patient_to_img_src, patient_to_img_dst = [], []
            
            patch_offset = 0
            
            for img_idx, path in enumerate(sample["image_paths"]):
                # Load and transform image
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                
                # Create patches and coordinates
                patches, coords, grid_h, grid_w = patchify(img, self.patch_size)
                patch_feat = torch.cat([patches, coords], dim=-1)
                patch_features_list.append(patch_feat)
                
                # Build Patch <-> Patch spatial adjacency edges
                adj = grid_edge_index(grid_h, grid_w, self.connectivity, device=patch_feat.device)
                patch_adj_src.extend((adj[0] + patch_offset).tolist())
                patch_adj_dst.extend((adj[1] + patch_offset).tolist())
                
                n_patches = patch_feat.size(0)
                
                # Build Patch <-> Image hierarchy edges
                for local_patch in range(n_patches):
                    global_patch = patch_offset + local_patch
                    patch_to_img_src.append(global_patch)
                    patch_to_img_dst.append(img_idx)
                    
                    img_to_patch_src.append(img_idx)
                    img_to_patch_dst.append(global_patch)
                    
                # Build Image <-> Patient hierarchy edges
                img_to_patient_src.append(img_idx)
                img_to_patient_dst.append(0)  # Patient node index is always 0
                
                patient_to_img_src.append(0)
                patient_to_img_dst.append(img_idx)
                
                patch_offset += n_patches
                
            # 2. Attach accumulated patches and edges to HeteroData
            data["patch"].x = torch.cat(patch_features_list, dim=0)
            
            data[("patch", "adjacent", "patch")].edge_index = torch.tensor(
                [patch_adj_src, patch_adj_dst], dtype=torch.long
            )
            
            data[("patch", "to", "image")].edge_index = torch.tensor(
                [patch_to_img_src, patch_to_img_dst], dtype=torch.long
            )
            data[("image", "to", "patch")].edge_index = torch.tensor(
                [img_to_patch_src, img_to_patch_dst], dtype=torch.long
            )
            
            data[("image", "to", "patient")].edge_index = torch.tensor(
                [img_to_patient_src, img_to_patient_dst], dtype=torch.long
            )
            data[("patient", "to", "image")].edge_index = torch.tensor(
                [patient_to_img_src, patient_to_img_dst], dtype=torch.long
            )
            
        # Assign populated edge lists to data[...]
        
    def _attach_structural_edges(self, data: HeteroData, num_labels: int) -> None:
        """Builds edges linking Demographics to Patient, and Patient to Labels."""
        num_demo = len(self.demo_cols)
        
        # Demographics <-> Patient
        data[("demographic", "to", "patient")].edge_index = torch.tensor(
            [list(range(num_demo)), [0] * num_demo], dtype=torch.long
        )
        
        # Patient <-> Label
        data[("patient", "to", "label")].edge_index = torch.tensor(
            [[0] * num_labels, list(range(num_labels))], dtype=torch.long
        )
        
        # Label <-> Label (Fully connected, skipping self-loops)
        corr_src, corr_dst = [], []
        for i in range(num_labels):
            for j in range(num_labels):
                if i != j:
                    corr_src.append(i)
                    corr_dst.append(j)
                    
        data[("label", "correlates", "label")].edge_index = torch.tensor(
            [corr_src, corr_dst], dtype=torch.long
        )