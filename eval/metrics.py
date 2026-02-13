"""
Evaluation metrics: precision, recall, F1, AUC, FPR at operational thresholds.
"""

from typing import Dict, Any, List, Tuple, Optional

try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def binary_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """Compute precision, recall, F1 for binary labels (0/1)."""
    assert len(y_true) == len(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_positive": sum(y_true),
        "support_negative": len(y_true) - sum(y_true),
    }


def scores_to_binary(scores: List[float], threshold: float) -> List[int]:
    """Convert continuous scores (0–100) to binary: 1 if score >= threshold else 0."""
    return [1 if s >= threshold else 0 for s in scores]


def compute_auc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    """ROC AUC. Returns None if only one class in y_true or sklearn missing."""
    if not _HAS_SKLEARN:
        return None
    if len(set(y_true)) < 2:
        return None
    try:
        return round(float(roc_auc_score(y_true, y_score)), 4)
    except Exception:
        return None


def metrics_at_threshold(
    y_true: List[int],
    y_score: List[float],
    threshold: float,
) -> Dict[str, Any]:
    """P, R, F1, FPR at a single operational threshold."""
    y_pred = scores_to_binary(y_score, threshold)
    m = binary_metrics(y_true, y_pred)
    m["threshold"] = threshold
    return m


def operational_report(
    y_true: List[int],
    y_score: List[float],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Report precision, recall, F1, AUC, and FPR at operational thresholds.
    thresholds: e.g. [30, 40, 50, 60, 70] for risk levels; default 20–80 step 5.
    """
    if thresholds is None:
        thresholds = list(range(20, 85, 5))
    per_threshold = [metrics_at_threshold(y_true, y_score, t) for t in thresholds]
    auc = compute_auc(y_true, y_score)
    return {
        "auc": auc,
        "n": len(y_true),
        "n_positive": sum(y_true),
        "n_negative": len(y_true) - sum(y_true),
        "operational_thresholds": per_threshold,
    }


def threshold_sweep(
    y_true: List[int],
    y_score: List[float],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Compute P/R/F1/FPR at each threshold."""
    return [metrics_at_threshold(y_true, y_score, t) for t in thresholds]


def best_threshold_by_f1(
    y_true: List[int],
    y_score: List[float],
    thresholds: List[float],
) -> Tuple[float, Dict[str, float]]:
    """Return threshold that maximizes F1 and the corresponding metrics."""
    sweep = threshold_sweep(y_true, y_score, thresholds)
    best = max(sweep, key=lambda x: x["f1"])
    return best["threshold"], {k: v for k, v in best.items() if k != "threshold"}


def value_at_threshold(
    y_true: List[int],
    y_score: List[float],
    threshold: float,
    savings_per_tp: float = 1000.0,
    cost_per_review: float = 50.0,
) -> Dict[str, float]:
    """Value = TP * savings_per_tp - (TP+FP) * cost_per_review (reviews = alerts)."""
    y_pred = scores_to_binary(y_score, threshold)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    reviews = tp + fp
    value = tp * savings_per_tp - reviews * cost_per_review
    return {
        "threshold": threshold,
        "value": value,
        "tp": tp,
        "reviews": reviews,
        "savings": tp * savings_per_tp,
        "cost": reviews * cost_per_review,
    }


def optimize_threshold(
    y_true: List[int],
    y_score: List[float],
    savings_per_tp: float = 1000.0,
    cost_per_review: float = 50.0,
    max_fpr: Optional[float] = None,
    max_workload: Optional[int] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Choose threshold to maximize value (savings - cost of review).
    Optional: max_fpr (e.g. 0.05) or max_workload (max number of reviews).
    """
    if thresholds is None:
        thresholds = list(range(20, 85, 5))
    n_neg = len(y_true) - sum(y_true)
    best = None
    best_value = None
    sweep = []
    for t in thresholds:
        m = metrics_at_threshold(y_true, y_score, t)
        v = value_at_threshold(y_true, y_score, t, savings_per_tp, cost_per_review)
        if max_fpr is not None and m["fpr"] > max_fpr:
            continue
        if max_workload is not None and v["reviews"] > max_workload:
            continue
        sweep.append({**m, **v})
        if best_value is None or v["value"] > best_value:
            best_value = v["value"]
            best = t
    return {
        "best_threshold": best,
        "best_value": best_value,
        "savings_per_tp": savings_per_tp,
        "cost_per_review": cost_per_review,
        "max_fpr": max_fpr,
        "max_workload": max_workload,
        "value_sweep": sweep,
    }
