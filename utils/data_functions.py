import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def calculate_positive_weights(targets: np.ndarray) -> torch.Tensor:
    """
    Calculates positive weights for BCEWithLogitsLoss to handle multilabel class imbalance.
    Safely handles divisions by zero and completely absent classes.
    """
    positives = targets.sum(axis=0).astype(np.float32)
    total_samples = targets.shape[0]
    negatives = total_samples - positives

    # Calculate weights: Negative count / Positive count
    # Use np.maximum to avoid division by zero
    weights = np.divide(negatives, np.maximum(positives, 1.0), dtype=np.float32)

    # Handle edge cases (e.g., no positives -> default weight of 1.0 to avoid NaN)
    weights[~np.isfinite(weights)] = 1.0
    weights[positives == 0] = 1.0

    return torch.tensor(weights, dtype=torch.float32)


def worker_init_fn(worker_id: int) -> None:
    """
    Ensures diverse random seeding across PyTorch DataLoader workers.
    Prevents identical image augmentations across batches when using multiple CPU workers.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    A standardized DataLoader constructor that automatically applies the 
    worker initialization function and a deterministic generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    
    # Persistent workers and prefetch factor are only valid if num_workers > 0
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
        
    return DataLoader(**kwargs)