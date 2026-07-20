import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Safely handle environments where W&B is not installed
try:
    import wandb
except ImportError:
    wandb = None


def is_wandb_enabled(use_wandb: bool, mode: str = "online") -> bool:
    """
    Checks if Weights & Biases is installed, requested by the user, 
    and not explicitly disabled via the mode string.
    """
    return bool(use_wandb and wandb is not None and mode != "disabled")


def start_wandb_run(
    project: str,
    run_name: Optional[str],
    config: Dict[str, Any],
    output_dir: Path,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None,
    mode: str = "online",
):
    """
    Initializes a Weights & Biases run. Automatically sets up standard 
    x-axis step metrics (epochs) for both training and validation logging.
    """
    if not is_wandb_enabled(use_wandb=True, mode=mode):
        return None

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        mode=mode,
        dir=str(output_dir),
        reinit=True,
        config=config,
    )

    # Standardize the x-axis to track against 'epoch' rather than raw global steps.
    # This covers the naming conventions used in both your vision and graph scripts.
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("validation/*", step_metric="epoch")

    return run


def finish_wandb_run(run) -> None:
    """
    Safely closes out the W&B run to ensure all logs and artifacts 
    are synced to the cloud.
    """
    if run is not None:
        run.finish()