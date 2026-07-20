import os
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> None:
    """
    Sets the random seed across all libraries to ensure reproducibility.
    Optionally enforces deterministic algorithms in PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def select_device(requested: str) -> torch.device:
    """
    Safely resolves the requested hardware device.
    """
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_runtime(device: torch.device) -> None:
    """
    Prints a standardized summary of the hardware environment.
    """
    print("=" * 79)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"GPU: {properties.name}")
        print(f"GPU memory: {properties.total_memory / (1024 ** 3):.2f} GiB")
    
    print(f"Logical CPU cores: {os.cpu_count()}")
    print("=" * 79)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Writes a dictionary to a JSON file, automatically converting Path, 
    NumPy arrays, and NumPy numeric types into standard Python types.
    """
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2, ensure_ascii=False)


def make_grad_scaler(enabled: bool):
    """
    Creates a PyTorch GradScaler, handling version-specific API changes.
    """
    # Modern PyTorch API
    return torch.amp.GradScaler("cuda", enabled=enabled)


def autocast_context(enabled: bool):
    """
    Returns the appropriate autocast context manager for mixed precision.
    """
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)