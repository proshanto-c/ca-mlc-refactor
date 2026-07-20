import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, List, Dict, Union, Optional, Callable

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
        """Encodes demographic specs into node features."""
        # This logic delegates to your previously built DemographicSpec pipeline.
        # It creates tensors for field_idx, kind, num_value, cat_value, and missing indicators.
        pass # Implement using your existing _encode_demographics logic

    def _attach_vision_nodes(self, data: HeteroData, sample: Dict) -> None:
        """Loads images, creates patches, and draws edges between patches, images, and the patient."""
        num_images = len(sample["image_paths"])
        
        data["image"].x = torch.zeros((num_images, 1), dtype=torch.float32)
        data["image"].image_idx = torch.arange(num_images, dtype=torch.long)
        data["image"].eye_idx = torch.tensor(
            [0 if v in (-1, None) else max(0, int(v) - 1) for v in sample["eye_ids"]],
            dtype=torch.long
        )
        
        patch_features, patch_adj_src, patch_adj_dst = [], [], []
        patch_to_img_src, patch_to_img_dst = [], []
        img_to_patient_src, img_to_patient_dst = [], []
        
        patch_offset = 0
        for img_idx, path in enumerate(sample["image_paths"]):
            # Load and transform
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
            
            # Patchify (Using your existing patchify helper)
            # patches, coords, grid_h, grid_w = patchify(img, self.patch_size)
            
            # 1. Add Patch Features
            # 2. Build Patch Adjacency (grid_edge_index)
            # 3. Build Patch <-> Image Edges
            # 4. Build Image <-> Patient Edges
            pass
            
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