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

try:
    import wandb
except ImportError:
    wandb = None

# --- MODULAR IMPORTS (Based on CA-MLC-REFACTOR layout) ---
from data_functions.BRSETGraphDataset import BRSETGraphDataset
from gnn.patientgraphmodel import PatientGraphModel, DemographicSpec

from utils.util_functions import (
    set_seed, 
    select_device, 
    print_runtime, 
    write_json, 
    make_grad_scaler
)
from utils.data_functions import calculate_positive_weights, create_dataloader
from utils.train_functions import graph_epoch
from utils.metrics import save_predictions
from utils.logger import start_wandb_run, finish_wandb_run


# =====================================================================
# 1. METADATA & DEMOGRAPHICS CONFIGURATION
# =====================================================================

DEFAULT_LABEL_CANDIDATES = [
    ["target_dr", "target_dme", "target_amd", "target_myopic_fundus", "target_increased_cup_disc"],
    ["diabetic_retinopathy", "macular_edema", "amd", "myopic_fundus", "increased_cup_disc"],
]

DEFAULT_DEMOGRAPHIC_CANDIDATES = [
    "patient_age", "age", "patient_sex", "sex", "diabetes_time",
    "diabetes_duration", "insulin_use", "comorbidities", "nationality", "exam_eye",
]

NUMERIC_HINTS = ("age", "time", "years", "duration", "count", "score", "num")


def infer_label_columns(frame: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested
    for candidate in DEFAULT_LABEL_CANDIDATES:
        if all(col in frame.columns for col in candidate):
            return candidate
    raise ValueError("Could not infer label columns. Pass --label-columns explicitly.")


def infer_demographic_columns(frame: pd.DataFrame, requested: Optional[List[str]], label_columns: Sequence[str]) -> List[str]:
    if requested:
        return requested
    excluded = {"image_id", "image_path", "image_name", "patient_id", "patient", "split", *label_columns}
    candidates = [c for c in DEFAULT_DEMOGRAPHIC_CANDIDATES if c in frame.columns and c not in excluded]
    if candidates:
        return candidates
    raise ValueError("No demographic columns found. Pass --demographic-columns explicitly.")


def fit_demographic_specs(frame: pd.DataFrame, demographic_columns: Sequence[str]) -> List[DemographicSpec]:
    specs: List[DemographicSpec] = []
    for col in demographic_columns:
        series = frame[col]
        numeric_mask = pd.to_numeric(series, errors="coerce").notna()
        is_numeric = numeric_mask.mean() >= 0.8 or any(h in col.lower() for h in NUMERIC_HINTS)
        
        if is_numeric:
            numeric = pd.to_numeric(series, errors="coerce")
            mean = float(numeric.mean()) if numeric.notna().any() else 0.0
            std = float(numeric.std(ddof=0)) if numeric.notna().any() else 1.0
            specs.append(DemographicSpec(name=col, kind="numeric", mean=mean, std=max(std, 1e-6)))
        else:
            vocab = {"__UNK__": 0}
            values = [str(v) for v in series.dropna().astype(str).unique().tolist()]
            for i, v in enumerate(sorted(values), start=1):
                vocab[v] = i
            specs.append(DemographicSpec(name=col, kind="categorical", vocab=vocab))
    return specs


# =====================================================================
# 2. ORCHESTRATION ARGUMENTS
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune GNN hyperparameters for BRSET.")
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train.csv/val.csv.")
    p.add_argument("--train-manifest", type=str, default="patient_train_42.csv")
    p.add_argument("--validation-manifest", type=str, default="patient_validation_42.csv")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--label-columns", nargs="*", default=None)
    p.add_argument("--demographic-columns", nargs="*", default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--monitor", choices=["val_loss", "val_macro_f1"], default="val_loss")
    p.add_argument("--num-workers", type=int, default=4)
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
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (root / "runs" / "patient_graph_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Environment & Hardware Setup [cite: 390]
    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    print_runtime(device)
    amp_enabled = device.type == "cuda" and not args.no_amp

    # 2. Metadata Extraction [cite: 276]
    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    label_columns = infer_label_columns(train_frame, args.label_columns)
    demo_columns = infer_demographic_columns(train_frame, args.demographic_columns, label_columns)
    demo_specs = fit_demographic_specs(train_frame, demo_columns)

    # 3. Instantiate Modular Datasets (Memory-Loaded)
    train_dataset = BRSETGraphDataset(
        data=prepared_dir / args.train_manifest,
        image_dir=root / "fundus_photos",
        target_cols=label_columns,
        demo_cols=demo_columns,
        demo_specs=demo_specs,
        image_size=args.image_size,
        patch_size=args.patch_size,
        connectivity=args.connectivity,
        validate_paths=True
    )
    
    val_dataset = BRSETGraphDataset(
        data=prepared_dir / args.validation_manifest,
        image_dir=root / "fundus_photos",
        target_cols=label_columns,
        demo_cols=demo_columns,
        demo_specs=demo_specs,
        image_size=args.image_size,
        patch_size=args.patch_size,
        connectivity=args.connectivity,
        validate_paths=False
    )

    # 4. Extract Targets to Calculate Modular Class Imbalance Weights [cite: 391]
    train_targets = np.stack([s["label_vector"] for s in train_dataset.samples], axis=0)
    pos_weight = calculate_positive_weights(train_targets).to(device)

    print(f"\nStarting hyperparameter tuning over {args.num_trials} random trials...")

    # 5. Hyperparameter Tuning Loop [cite: 278]
    for trial in range(1, args.num_trials + 1):
        # Sample Hyperparameters
        h_dim = random.choice([32, 64, 128, 256])
        n_layers = random.choice([2, 3, 4])
        dropout = random.uniform(0.1, 0.75)
        batch_size = random.choice([16, 32, 64])

        print(f"\n=== [TRIAL {trial}/{args.num_trials}] ===")
        print(f"Parameters: h_dim={h_dim}, n_layers={n_layers}, drop={dropout:.4f}, batch={batch_size}")

        # Instantiate DataLoaders via utils/data_functions.py
        train_loader = create_dataloader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"), seed=args.seed
        )
        val_loader = create_dataloader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"), seed=args.seed
        )

        # Build Model & Optimizer
        patch_dim = 3 * args.patch_size * args.patch_size + 2
        model = PatientGraphModel(
            patch_dim=patch_dim, num_labels=len(label_columns), demographic_specs=demo_specs,
            hidden_dim=h_dim, num_layers=n_layers, dropout=dropout
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scaler = make_grad_scaler(amp_enabled)

        # Setup Logging Directories
        run_dir = output_dir / f"trial_{trial}_h{h_dim}_l{n_layers}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "final_checkpoint.pt"
        
        trial_config = {
            "trial": trial, "h_dim": h_dim, "n_layers": n_layers, "dropout": dropout,
            "batch_size": batch_size, "lr": args.learning_rate, "seed": args.seed,
        }
        write_json(run_dir / "run_config.json", trial_config)

        # Start WandB using utils/logger.py [cite: 392]
        run_name = f"{args.wandb_run_name}-T{trial}-h{h_dim}-l{n_layers}" if args.wandb_run_name else f"tune-T{trial}"
        wandb_run = start_wandb_run(
            project=args.wandb_project, run_name=run_name, config=trial_config, output_dir=run_dir,
            entity=args.wandb_entity, group=args.wandb_group, tags=args.wandb_tags, mode=args.wandb_mode
        )

        # Core Training Loop utilizing graph_epoch from utils/train_functions.py
        history = []
        for epoch in range(1, args.epochs + 1):
            train_res = graph_epoch(
                model=model, loader=train_loader, criterion=criterion, device=device, 
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=optimizer, 
                scaler=scaler, desc=f"T{trial} E{epoch:02d} Train", threshold=args.threshold
            )
            
            val_res = graph_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device, 
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None, 
                scaler=None, desc=f"T{trial} E{epoch:02d} Val", threshold=args.threshold
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "val_loss": val_res["loss"],
                "val_macro_f1": val_res["summary"]["macro_f1"],
                "val_macro_auroc": val_res["summary"]["macro_auroc"]
            })
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            if wandb_run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_res["loss"],
                    "val/loss": val_res["loss"],
                    "val/macro_f1": val_res["summary"]["macro_f1"],
                    "val/macro_auroc": val_res["summary"]["macro_auroc"],
                }, step=epoch)

        # Save Final Model
        torch.save({"model_state_dict": model.state_dict(), "hyperparams": trial_config}, checkpoint_path)

        # Post-Trial Evaluation & Predictions
        val_preds = (val_res["probabilities"] >= args.threshold).astype(np.int8)
        
        # Save Predictions using utils/metrics.py
        save_predictions(
            path=run_dir / "validation_predictions.csv",
            entity_ids=[s["patient_id"] for s in val_dataset.samples],
            id_column_name="patient_id",
            label_names=label_columns,
            targets=val_res["targets"],
            probabilities=val_res["probabilities"],
            predictions=val_preds
        )

        trial_final_metric = float(val_res["summary"]["macro_f1"] if args.monitor == "val_macro_f1" else val_res["loss"])
        write_json(run_dir / "summary.json", {"final_validation_metric": trial_final_metric, "validation": val_res["summary"]})

        # Close WandB Run via utils/logger.py
        if wandb_run is not None:
            wandb_run.summary["final_validation_metric"] = trial_final_metric
            wandb.save(str(run_dir / "summary.json"))
            wandb.save(str(checkpoint_path))
        finish_wandb_run(wandb_run)

    print("\nHyperparameter tuning complete across all trials!")


if __name__ == "__main__":
    main()