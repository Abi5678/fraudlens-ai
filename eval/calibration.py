"""Calibration: Platt scaling and isotonic regression for score -> probability."""
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    np = None


def fit_platt(y_true: List[int], y_score: List[float]) -> Dict[str, Any]:
    if not _HAS_SKLEARN or np is None:
        return {"method": "platt", "error": "scikit-learn/numpy required"}
    X = np.clip(np.array(y_score, dtype=float).reshape(-1, 1) / 100.0, 0.0, 1.0)
    y = np.array(y_true, dtype=float)
    clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=500)
    clf.fit(X, y)
    return {"method": "platt", "A": float(clf.coef_[0, 0]), "B": float(clf.intercept_[0])}


def fit_isotonic(y_true: List[int], y_score: List[float]) -> Dict[str, Any]:
    if not _HAS_SKLEARN or np is None:
        return {"method": "isotonic", "error": "scikit-learn/numpy required"}
    X = np.clip(np.array(y_score, dtype=float) / 100.0, 0.0, 1.0)
    y = np.array(y_true, dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X, y)
    x_uniq = np.unique(np.clip(np.linspace(0, 1, 101), 0, 1))
    p_uniq = ir.predict(x_uniq)
    return {"method": "isotonic", "boundaries": [[float(x), float(p)] for x, p in zip(x_uniq, p_uniq)]}


def apply_platt(score: float, params: Dict[str, Any]) -> float:
    import math
    s = max(0, min(100, score)) / 100.0
    A, B = params.get("A", 1.0), params.get("B", 0.0)
    return 1.0 / (1.0 + math.exp(-(A * s + B)))


def apply_isotonic(score: float, params: Dict[str, Any]) -> float:
    s = max(0, min(100, score)) / 100.0
    boundaries = params.get("boundaries", [])
    if not boundaries:
        return s
    for i in range(len(boundaries) - 1):
        x0, p0 = boundaries[i]
        x1, p1 = boundaries[i + 1]
        if x0 <= s <= x1:
            t = (s - x0) / (x1 - x0) if x1 != x0 else 1.0
            return p0 + t * (p1 - p0)
    return boundaries[-1][1] if boundaries else s


def calibrate_score(score: float, calibrator: Dict[str, Any]) -> float:
    if calibrator.get("method") == "isotonic":
        return apply_isotonic(score, calibrator)
    return apply_platt(score, calibrator)


def fit_calibrator(y_true: List[int], y_score: List[float], method: str = "platt") -> Dict[str, Any]:
    return fit_isotonic(y_true, y_score) if method == "isotonic" else fit_platt(y_true, y_score)


def calibration_metrics(y_true: List[int], y_score: List[float], calibrator: Optional[Dict] = None, method: str = "platt") -> Dict[str, Any]:
    if not y_true or not y_score:
        return {"error": "empty data"}
    if calibrator is None:
        calibrator = fit_calibrator(y_true, y_score, method=method)
    if "error" in calibrator:
        return calibrator
    probs = [calibrate_score(s, calibrator) for s in y_score]
    out = {"method": method, "calibrator": calibrator, "example_mapping": [(s, round(calibrate_score(s, calibrator), 3)) for s in [25, 50, 70, 90]]}
    if np is not None:
        probs_arr = np.array(probs)
        y_arr = np.array(y_true, dtype=float)
        out["brier_score"] = round(float(np.mean((probs_arr - y_arr) ** 2)), 4)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs_arr >= bins[i]) & (probs_arr <= bins[i + 1]) if i == n_bins - 1 else (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += mask.sum() * abs(y_arr[mask].mean() - probs_arr[mask].mean())
        out["ece"] = round(float(ece / len(y_arr)), 4)
    return out


def save_calibrator(calibrator: Dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibrator, f, indent=2)


def load_calibrator(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
