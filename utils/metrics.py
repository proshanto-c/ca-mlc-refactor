import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_metric(fn, *args, default: float = math.nan, **kwargs) -> float:
    """
    Safely executes a scikit-learn metric function. 
    Returns a default value (usually NaN) if an exception occurs 
    (e.g., calculating AUROC on a batch with only a single class).
    """
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def evaluate_ranking_metrics(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    label_names: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculates threshold-independent metrics (AUROC, AUPRC).
    """
    rows = []
    aucs = []
    aps = []
    
    for index, label in enumerate(label_names):
        y_true = targets[:, index]
        y_score = probabilities[:, index]
        
        # AUROC requires at least one positive and one negative example
        auc = safe_metric(roc_auc_score, y_true, y_score) if np.unique(y_true).size > 1 else math.nan
        ap = safe_metric(average_precision_score, y_true, y_score)
        
        rows.append({
            "label": label,
            "positive_count": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auroc": auc,
            "auprc": ap,
        })
        aucs.append(auc)
        aps.append(ap)

    summary = {
        "macro_auroc": float(np.nanmean(aucs)),
        "macro_auprc": float(np.nanmean(aps)),
    }
    return pd.DataFrame(rows), summary


def find_optimal_thresholds(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    label_names: Sequence[str], 
    step: float = 0.01
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Sweeps through probability thresholds to find the one that maximizes F1 score per label.
    Uses percentiles of the predicted probabilities to dynamically adapt to highly skewed distributions.
    """
    best_thresholds = []
    rows = []

    # Calculate percentiles to sweep based on the step size
    percentiles = np.linspace(1, 99, int(1.0 / step))

    for index, label in enumerate(label_names):
        y_true = targets[:, index]
        y_score = probabilities[:, index]
        
        # Dynamically generate candidates based on the actual probability distribution for this label
        candidates = np.percentile(y_score, percentiles)
        candidates = np.unique(np.clip(candidates, 0.001, 0.999))
        
        is_multilabel = len(label_names) > 1
        
        best_f1_threshold = 0.5
        best_f1_val = -1.0
        
        best_youden_threshold = 0.5
        best_youden_val = -1.0
        
        for threshold in candidates:
            prediction = (y_score >= threshold).astype(np.int8)
            
            # 1. Track F1 Optimization
            f1 = safe_metric(f1_score, y_true, prediction, zero_division=0, default=0.0)
            if f1 > best_f1_val:
                best_f1_val = f1
                best_f1_threshold = float(threshold)
                
            # 2. Track Youden Optimization (if multilabel)
            if is_multilabel:
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    j_stat = sens + spec - 1.0
                except Exception:
                    j_stat = -1.0
                    
                if j_stat > best_youden_val:
                    best_youden_val = j_stat
                    best_youden_threshold = float(threshold)
                    
        # The "primary" optimal threshold is Youden for multilabel, F1 for single-label
        best_primary_threshold = best_youden_threshold if is_multilabel else best_f1_threshold
                
        # --- Calculate Accuracy metrics using the primary winning threshold ---
        primary_prediction = (y_score >= best_primary_threshold).astype(np.int8)
        primary_acc = safe_metric(accuracy_score, y_true, primary_prediction, default=0.0)
        primary_bal_acc = safe_metric(balanced_accuracy_score, y_true, primary_prediction, default=0.0)
        primary_f1_final = safe_metric(f1_score, y_true, primary_prediction, zero_division=0, default=0.0)
        
        # Calculate F1 using the Youden threshold explicitly
        if is_multilabel:
            youden_prediction = (y_score >= best_youden_threshold).astype(np.int8)
            youden_f1 = safe_metric(f1_score, y_true, youden_prediction, zero_division=0, default=0.0)
        else:
            youden_f1 = math.nan
                
        best_thresholds.append(best_primary_threshold)
        rows.append({
            "label": label,
            "optimal_threshold": best_primary_threshold,
            "optimal_f1_threshold": best_f1_threshold,
            "optimal_youden_threshold": best_youden_threshold if is_multilabel else math.nan,
            "validation_f1": primary_f1_final,           # F1 score using primary threshold
            "validation_f1_at_f1_thresh": best_f1_val,   # F1 score using F1-optimized threshold
            "validation_f1_at_youden_thresh": youden_f1, # F1 score using Youden-optimized threshold
            "validation_accuracy": primary_acc, 
            "validation_balanced_accuracy": primary_bal_acc, 
        })

    return np.asarray(best_thresholds, dtype=np.float32), pd.DataFrame(rows)


def evaluate_multilabel_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    thresholds: Union[float, np.ndarray] = 0.5
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    """
    Unified evaluation pipeline. Calculates threshold-dependent metrics (Accuracy, F1, etc.)
    alongside ranking metrics (AUROC, AUPRC) for a comprehensive overview.
    """
    # Standardize thresholds to an array matching the number of labels
    if isinstance(thresholds, (float, int)):
        thresholds = np.full(len(label_names), float(thresholds), dtype=np.float32)
        
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)
    rows = []

    for index, label in enumerate(label_names):
        y_true = targets[:, index].astype(np.int8)
        y_score = probabilities[:, index]
        y_pred = predictions[:, index]

        # Handle edge cases in confusion matrix safely
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except Exception:
            tn, fp, fn, tp = 0, 0, 0, 0

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else math.nan
        has_both_classes = np.unique(y_true).size == 2

        rows.append({
            "label": label,
            "threshold": float(thresholds[index]),
            "accuracy": safe_metric(accuracy_score, y_true, y_pred),
            "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred) if has_both_classes else math.nan,
            "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
            "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
            "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "youden_j": (sensitivity + specificity - 1.0) if not math.isnan(sensitivity) and not math.isnan(specificity) else math.nan,
            "auroc": safe_metric(roc_auc_score, y_true, y_score) if has_both_classes else math.nan,
            "auprc": safe_metric(average_precision_score, y_true, y_score) if y_true.sum() > 0 else math.nan,
            "positive_count": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })

    metrics_frame = pd.DataFrame(rows)
    
    # Generate Macro/Micro Summary
    summary = {
        "macro_accuracy": float(metrics_frame["accuracy"].mean()),
        "macro_balanced_accuracy": float(metrics_frame["balanced_accuracy"].mean(skipna=True)),
        "macro_youden_j": float(metrics_frame["youden_j"].mean(skipna=True)),
        "macro_f1": float(metrics_frame["f1"].mean(skipna=True)),
        "macro_precision": float(metrics_frame["precision"].mean(skipna=True)),
        "macro_recall": float(metrics_frame["recall"].mean(skipna=True)),
        "macro_auroc": float(metrics_frame["auroc"].mean(skipna=True)),
        "macro_auprc": float(metrics_frame["auprc"].mean(skipna=True)),
        "micro_f1": safe_metric(f1_score, targets, predictions, average="micro", zero_division=0),
        "samples_f1": safe_metric(f1_score, targets, predictions, average="samples", zero_division=0),
    }

    return metrics_frame, summary, predictions


def save_predictions(
    path: Path,
    entity_ids: Sequence[str],
    id_column_name: str,
    label_names: Sequence[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    secondary_ids: Sequence[str] = None,
    secondary_id_name: str = None
) -> None:
    """
    Saves targets, probabilities, and binary predictions to a CSV file.
    Flexible enough to handle both patient-level (GNN) and image-level (ConvNeXt) datasets.
    """
    data = {id_column_name: entity_ids}
    
    if secondary_ids is not None and secondary_id_name is not None:
        data[secondary_id_name] = secondary_ids

    for index, label in enumerate(label_names):
        data[f"{label}_target"] = targets[:, index].astype(np.int8)
        data[f"{label}_probability"] = probabilities[:, index]
        data[f"{label}_prediction"] = predictions[:, index].astype(np.int8)
        
    pd.DataFrame(data).to_csv(path, index=False)