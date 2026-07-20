import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, List, Union, Optional, Callable

class BRSETImageDataset(Dataset):
    """
    Dataset class for use with ConvNeXt-V2.
    Consumes a DataFrame where each row corresponds to a single image (like the format given from BRSETDataSplitter).
    """
    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        image_dir: Path,
        target_cols: List[str],
        transform: Optional[Callable] = None,
        validate_paths: bool = True
    ):
        # 1. Accept either a CSV path or an existing DataFrame
        if isinstance(data, (Path, str)):
            self.frame = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.frame = data.copy()
        else:
            raise ValueError("data must be a path-like string/Path or a pandas DataFrame.")
        
        self.image_dir = Path(image_dir)
        self.target_cols = target_cols
        self.transform = transform
        
        # 2. Strict Schema Validation
        self._validate_schema()
        
        # 3. Resolve and Validate Paths
        self.image_paths = self._resolve_paths(validate_paths)
        
        # 4. Extract Data Arrays for fast indexing in __getitem__
        self.image_ids = self.frame["image_id"].astype(str).tolist()
        self.patient_ids = self.frame["patient_id"].astype(str).tolist()
        self.targets = self.frame[self.target_cols].to_numpy(dtype=np.float32)
    
    def _validate_schema(self) -> None:
        required_cols = {"image_id", "patient_id"}.union(self.target_cols)
        missing = required_cols - set(self.frame.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
            
        # Ensure targets are valid binary floats
        for col in self.target_cols:
            vals = pd.to_numeric(self.frame[col], errors="coerce")
            if vals.isna().any() or not vals.isin([0, 1]).all():
                raise ValueError(f"Target column '{col}' must contain only binary 0/1 values.")
            self.frame[col] = vals.astype(np.float32)

    def _resolve_paths(self, validate: bool) -> List[Path]:
        paths = []
        #  Support pre-constructed paths or build them from IDs
        raw_paths = self.frame.get("image_path", self.frame["image_id"].apply(lambda x: f"fundus_photos/{x}.jpg"))

        for rp in raw_paths:
            p = Path(rp)
            paths.append(p if p.is_absolute() else self.image_dir / p)
        
        if validate:
            missing = [p for p in paths if not p.is_file()]
            if missing:
                sample = "\n".join(str(p) for p in missing[:10])
                raise FileNotFoundError(f"Found {len(missing)} missing images. Sample:\n{sample}")
        
        return paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {path}.") from e
        
        return {
            "image": img,
            "target": torch.from_numpy(self.targets[index].copy()),
            "image_id": self.image_ids[index],
            "patient_id": self.patient_ids[index],
        }