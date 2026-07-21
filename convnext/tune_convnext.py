#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm

try:
    import wandb
except ImportError:
    wandb = None

# --- MODULAR IMPORTS ---
from data_functions.BRSETImageDataset import BRSETImageDataset
from utils.metrics import find_optimal_thresholds

from utils.util_functions import (
    set_seed, 
    select_device, 
    print_runtime, 
    write_json, 
    make_grad_scaler
)
from utils.data_functions import calculate_positive_weights
from utils.train_functions import image_epoch
from utils.logger import start_wandb_run

# =====================================================================
# 1. METADATA & CONFIGURATION
# =====================================================================

DEFAULT_LABEL_CANDIDATES = [
    ["diabetic_retinopathy", "macular_edema", "amd", "myopic_fundus", "increased_cup_disc"],
]

def infer_label_columns(frame: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested
    for candidate in DEFAULT_LABEL_CANDIDATES:
        if all(col in frame.columns for col in candidate):
            return candidate
    raise ValueError("Could not infer label columns. Pass --label-columns explicitly.")

# =====================================================================
# 2. ORCHESTRATION ARGUMENTS
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune ConvNeXt V2 hyperparameters for BRSET.")
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train/val manifests.")
    
    p.add_argument("--train-manifest", type=str, default="image_train_42.csv")
    p.add_argument("--validation-manifest", type=str, default="image_validation_42.csv")
    
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--label-columns", nargs="*", default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--num-trials", type=int, default=10, help="Number of random search trials.")

    # W&B Arguments
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args()


# =====================================================================
# 3. MAIN TUNING PIPELINE
# =====================================================================

def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve() if args.prepared_dir else (root / "prepared")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (root / "runs" / "convnext_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Environment & Hardware Setup
    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    print_runtime(device)
    amp_enabled = device.type == "cuda" and not args.no_amp

    # 2. Metadata Extraction
    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    label_columns = infer_label_columns(train_frame, args.label_columns)

    # Define your transforms (crucially including Resize!)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Instantiate Modular Datasets (Memory-Loaded)
    train_dataset = BRSETImageDataset(
        data=prepared_dir / args.train_manifest,
        image_dir=root / "fundus_photos",
        target_cols=label_columns,
        transform=train_transform,
        validate_paths=True
    )
    
    val_dataset = BRSETImageDataset(
        data=prepared_dir / args.validation_manifest,
        image_dir=root / "fundus_photos",
        target_cols=label_columns,
        transform=val_transform,
        validate_paths=False
    )

    # 4. Extract Targets to Calculate Modular Class Imbalance Weights
    train_targets = train_dataset.targets
    pos_weight = calculate_positive_weights(train_targets).to(device)

    # Calculate initial biases
    pos_counts = np.maximum(train_targets.sum(axis=0), 1.0)
    neg_counts = np.maximum(train_targets.shape[0] - pos_counts, 1.0)
    adaptive_biases = np.log(pos_counts / neg_counts).astype(np.float32)

    print(f"\nStarting CONVNEXT hyperparameter tuning over {args.num_trials} trials...")

    # 5. Hyperparameter Tuning Loop
    for trial in range(1, args.num_trials + 1):
        batch_size = random.choice([32, 64])
        learning_rate = random.choice([1e-4, 5e-4, 1e-3])
        dropout = random.uniform(0.1, 0.3)

        print(f"\n=== [TRIAL {trial}/{args.num_trials}] ===")
        print(f"Parameters: drop={dropout:.4f}, batch={batch_size}, lr={learning_rate}")

        # Note: image dataset so just use normal DataLoader
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=(device.type == "cuda")
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=(device.type == "cuda")
        )

        model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=len(label_columns), drop_rate=dropout)
        
        # Apply initial biases to the classification head
        with torch.no_grad():
            if hasattr(model, 'head') and hasattr(model.head, 'fc'):
                model.head.fc.bias.copy_(torch.from_numpy(adaptive_biases))
            elif hasattr(model, 'head'):
                if hasattr(model.head, 'bias') and model.head.bias is not None:
                    model.head.bias.copy_(torch.from_numpy(adaptive_biases))
        
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scaler = make_grad_scaler(amp_enabled)

        run_dir = output_dir / f"trial_{trial}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "final_checkpoint.pt"
        
        trial_config = {
            "trial": trial, "model": "convnextv2_tiny", "dropout": dropout,
            "batch_size": batch_size, "lr": learning_rate, "seed": args.seed,
        }
        write_json(run_dir / "run_config.json", trial_config)

        run_name = f"{args.wandb_run_name}-T{trial}" if args.wandb_run_name else f"tune-convnext-T{trial}"
        wandb_run = start_wandb_run(
            project=args.wandb_project, run_name=run_name, config=trial_config, output_dir=run_dir,
            entity=args.wandb_entity, group=args.wandb_group, tags=args.wandb_tags, mode=args.wandb_mode
        )

        history = []
        for epoch in range(1, args.epochs + 1):
            train_res = image_epoch(
                model=model, loader=train_loader, criterion=criterion, device=device, 
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=optimizer, 
                scaler=scaler, desc=f"T{trial} E{epoch:02d} Train", threshold=args.threshold
            )
            
            val_res = image_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device, 
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None, 
                scaler=None, desc=f"T{trial} E{epoch:02d} Val", threshold=args.threshold
            )

            # Calculate Real-Time Calibrated F1
            _, epoch_threshold_df = find_optimal_thresholds(
                targets=val_res["targets"],
                probabilities=val_res["probabilities"],
                label_names=label_columns,
                step=0.05 
            )
            epoch_calibrated_f1 = epoch_threshold_df["validation_f1"].mean()

            # Update local history CSV
            history.append({
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "val_loss": val_res["loss"],
                "val_macro_f1_static": val_res["summary"]["macro_f1"],
                "val_macro_f1_calibrated": epoch_calibrated_f1,
                "val_macro_auroc": val_res["summary"]["macro_auroc"]
            })
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            # Update WandB Real-Time Dashboard
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "train/loss": train_res["loss"],
                    "val/loss": val_res["loss"],
                    "val/macro_f1_static": val_res["summary"]["macro_f1"],
                    "val/macro_f1_calibrated": epoch_calibrated_f1,
                    "val/macro_auroc": val_res["summary"]["macro_auroc"],
                })

        # Save Final Trial Checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        if wandb_run is not None:
            wandb.finish()

    print("\n✅ Tuning completely finished!")


if __name__ == "__main__":
    main()
