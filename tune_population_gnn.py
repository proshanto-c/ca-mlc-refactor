import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
import argparse
import random
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.models as models
from torch.cuda.amp import GradScaler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data_functions.BRSETPopulationDataset import build_population_graph
from utils.data_functions import calculate_positive_weights
from gnn.populationgraphmodel import PopulationGraphModel
from utils.metrics import evaluate_multilabel_predictions, find_optimal_thresholds

class ImageExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, paths, train_mask, train_transform, val_transform):
        self.paths = paths
        self.train_mask = train_mask
        self.train_transform = train_transform
        self.val_transform = val_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
            if self.train_mask[idx]:
                return self.train_transform(img)
            else:
                return self.val_transform(img)

def extract_features_chunked(loader, cnn_model, device, multiscale=False):
    """
    Dynamically extracts ResNet features for the entire graph in chunks to prevent memory explosion.
    Uses pre-built PyTorch DataLoaders with persistent workers to prevent Windows process spawning overhead.
    """
    cnn_model.eval()
    all_feats = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Graph Features", leave=False):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=True):
                if multiscale:
                    # [B, 512, 8, 8] -> [B, 512, 64] -> [B, 64, 512] -> [B*64, 512]
                    feats = cnn_model(batch).flatten(2).transpose(1, 2).reshape(-1, 512)
                else:
                    # [B, 512, 1, 1] -> [B, 512]
                    feats = cnn_model(batch).view(batch.size(0), -1)
            all_feats.append(feats.cpu())
            
    return torch.cat(all_feats, dim=0)


def population_epoch(
    model, graph, criterion, device, amp_enabled, label_columns, mask_type, optimizer=None, scaler=None
):
    is_training = optimizer is not None
    model.train(is_training)
    
    graph = graph.to(device)
    
    if mask_type == "train":
        mask = graph["image"].train_mask
    else:
        mask = graph["image"].val_mask
        
    targets = graph["image"].y[mask]
    
    with torch.set_grad_enabled(is_training):
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            # Forward pass on the ENTIRE graph at once
            logits = model(graph)
            
            # Extract only the nodes we care about for the loss calculation
            logits = logits[mask]
            loss = criterion(logits, targets)
            
        if is_training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
    current_loss = float(loss.item())
    
    targets_np = targets.detach().cpu().numpy()
    probabilities_np = torch.sigmoid(logits).detach().cpu().numpy()
    
    metrics_frame, summary, predictions = evaluate_multilabel_predictions(
        targets=targets_np,
        probabilities=probabilities_np,
        label_names=label_columns
    )
    
    best_thresholds, threshold_df = find_optimal_thresholds(
        targets=targets_np,
        probabilities=probabilities_np,
        label_names=label_columns,
        step=0.05
    )
    
    epoch_calibrated_f1 = threshold_df["validation_f1"].mean()
    calibrated_predictions = (probabilities_np >= threshold_df["optimal_threshold"].to_numpy().reshape(1, -1)).astype(int)
    
    from sklearn.metrics import f1_score
    epoch_calibrated_micro_f1 = f1_score(targets_np, calibrated_predictions, average="micro", zero_division=0)
    
    summary["calibrated_macro_f1"] = epoch_calibrated_f1
    summary["calibrated_micro_f1"] = epoch_calibrated_micro_f1
    
    # Detach graph back to CPU to save memory between passes
    graph.cpu()
    
    return {
        "loss": current_loss,
        "summary": summary,
        "threshold_df": threshold_df,
        "metrics_frame": metrics_frame,
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, required=True)
    p.add_argument("--train-manifest", type=str, required=True)
    p.add_argument("--validation-manifest", type=str, required=True)
    p.add_argument("--label-columns", nargs="*", required=True)
    p.add_argument("--demographic-columns", nargs="*", default=["patient_age", "patient_sex"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--macro-arch", type=str, choices=["bipartite", "hypergraph"], default="bipartite")
    p.add_argument("--wandb-project", type=str, default="PopulationGraph")
    p.add_argument("--num-trials", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_df = pd.read_csv(args.manifest_root / args.train_manifest)
    val_df = pd.read_csv(args.manifest_root / args.validation_manifest)
    
    # Initialize global Feature Extractor
    print("Loading Frozen ResNet18 Feature Extractor...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if args.multiscale:
        resnet = nn.Sequential(*list(resnet.children())[:-2]) # spatial features
    else:
        resnet = nn.Sequential(*list(resnet.children())[:-1]) # GAP features
        
    for param in resnet.parameters():
        param.requires_grad = False
    resnet = resnet.to(device)
    
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"\nStarting Hyperparameter Sweep over {args.num_trials} trials...")
    
    for trial in range(1, args.num_trials + 1):
        k_neighbors = random.choice([3, 5, 10])
        hidden_dim = random.choice([64, 128, 256])
        # if args.multiscale:
        #     hidden_dim = random.choice([32, 64]) # Prevent OOM on massive graphs
        num_layers = random.choice([2, 3, 4])
        dropout = random.choice([0.1, 0.2, 0.3])
        learning_rate = random.choice([1e-4, 5e-4, 1e-3, 5e-3])
        macro_arch = random.choice(["bipartite", "hypergraph"]) if args.macro_arch is None else args.macro_arch

        print(f"\n=== [TRIAL {trial}/{args.num_trials}] ===")
        print(f"Params: k={k_neighbors}, hidden={hidden_dim}, layers={num_layers}, drop={dropout}, lr={learning_rate}, arch={macro_arch}")
        
        graph = build_population_graph(
            train_frame=train_df,
            val_frame=val_df,
            image_dir=args.root,
            target_cols=args.label_columns,
            demo_cols=args.demographic_columns,
            k_neighbors=k_neighbors,
            multiscale=args.multiscale
        )
        
        global_paths = graph["image"].image_paths
        train_mask = graph["image"].train_mask
        val_mask = graph["image"].val_mask
        
        train_targets_np = train_df[args.label_columns].to_numpy()
        pos_counts = np.maximum(train_targets_np.sum(axis=0), 1.0)
        neg_counts = np.maximum(train_targets_np.shape[0] - pos_counts, 1.0)
        adaptive_biases = np.log(pos_counts / neg_counts).astype(np.float32)
        
        model = PopulationGraphModel(
            num_labels=len(args.label_columns),
            num_contexts=graph["context"].num_nodes if "context" in graph.node_types else 0,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            macro_arch=macro_arch,
            initial_biases=adaptive_biases,
            multiscale=args.multiscale
        ).to(device)
        
        pos_weight = calculate_positive_weights(train_targets_np).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scaler = GradScaler()
        
        trial_config = vars(args).copy()
        trial_config.update({
            "k_neighbors": k_neighbors, "hidden_dim": hidden_dim, "num_layers": num_layers,
            "dropout": dropout, "learning_rate": learning_rate, "macro_arch": macro_arch, "trial": trial,
            "multiscale": args.multiscale
        })
        
        if WANDB_AVAILABLE:
            wandb.init(project=args.wandb_project, config=trial_config, reinit=True, name=f"trial_{trial}")
            
        print("\nInitializing Persistent Multiprocessed Workers...")
        # Train loader (applies augs to train nodes)
        train_feat_dataset = ImageExtractionDataset(global_paths, train_mask, train_transform, val_transform)
        train_feat_loader = torch.utils.data.DataLoader(
            train_feat_dataset, batch_size=128, shuffle=False, 
            num_workers=8, pin_memory=True, persistent_workers=True
        )
        
        # Val loader (clean extraction for all nodes)
        val_feat_dataset = ImageExtractionDataset(global_paths, train_mask, val_transform, val_transform)
        val_feat_loader = torch.utils.data.DataLoader(
            val_feat_dataset, batch_size=128, shuffle=False, 
            num_workers=8, pin_memory=True, persistent_workers=True
        )
            
        print("\nStarting Training Loop...")
        for epoch in range(1, args.epochs + 1):
            
            # --- TRAIN PASS ---
            feats = extract_features_chunked(
                train_feat_loader, resnet, device, multiscale=args.multiscale
            )
            if args.multiscale:
                graph["patch"].x = feats
            else:
                graph["image"].x = feats
                
            train_res = population_epoch(model, graph, criterion, device, True, args.label_columns, "train", optimizer, scaler)
            
            # --- EVAL PASS ---
            feats = extract_features_chunked(
                val_feat_loader, resnet, device, multiscale=args.multiscale
            )
            if args.multiscale:
                graph["patch"].x = feats
            else:
                graph["image"].x = feats
                
            val_res = population_epoch(model, graph, criterion, device, True, args.label_columns, "val", None, None)
            
            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_res['loss']:.4f} | Val Loss: {val_res['loss']:.4f} | Val Micro F1: {val_res['summary'].get('calibrated_micro_f1', 0):.4f}")
            
            log_dict = {
                "epoch": epoch,
                "train/loss": train_res['loss'],
                "val/loss": val_res['loss'],
                "val/calibrated_micro_f1": val_res['summary'].get("calibrated_micro_f1", 0),
                "val/calibrated_macro_f1": val_res['summary'].get("calibrated_macro_f1", 0),
            }
            
            # Log per-label metrics
            for _, row in val_res["metrics_frame"].iterrows():
                label = row["label"]
                log_dict[f"val_auroc_per_class/{label}"] = row.get("auroc", 0)
                
            for _, row in val_res["threshold_df"].iterrows():
                label = row["label"]
                log_dict[f"val_optimal_threshold/{label}"] = row["optimal_threshold"]
                log_dict[f"val_calibrated_f1/{label}"] = row["validation_f1"]
                log_dict[f"val_balanced_acc/{label}"] = row["validation_balanced_accuracy"]
            
            if WANDB_AVAILABLE:
                wandb.log(log_dict)
                
        if WANDB_AVAILABLE:
            wandb.finish()
            
if __name__ == "__main__":
    main()
