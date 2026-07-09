"""Shared metrics, threshold selection, and confidence intervals.

Used by train/ (threshold + Optuna objectives) and eval/ (CI).
"""
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score


def select_threshold(y, prob, min_sens=0.75):
    """Deployment threshold rule: among points with sensitivity >= min_sens pick
    the maximum specificity; fall back to Youden's J if none qualify."""
    fpr, tpr, thr = roc_curve(y, prob); spec = 1 - fpr; ok = tpr >= min_sens
    i = int(np.where(ok)[0][np.argmax(spec[ok])]) if ok.any() else int(np.argmax(tpr + spec - 1))
    return float(thr[i])


def sens_spec(y, pred):
    """(sensitivity, specificity) from hard predictions."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def sens_at_spec(y, proba, min_spec=0.5):
    """Optuna objective: best sensitivity while specificity >= min_spec."""
    fpr, tpr, _ = roc_curve(y, proba); spec = 1.0 - fpr; ok = spec >= min_spec
    return float(np.max(tpr[ok])) if ok.any() else 0.0


def spec_at_sens(y, proba, min_sens=0.75):
    """Optuna objective: best specificity while sensitivity >= min_sens
    (matches the deployment threshold rule)."""
    fpr, tpr, _ = roc_curve(y, proba); spec = 1.0 - fpr; ok = tpr >= min_sens
    return float(np.max(spec[ok])) if ok.any() else 0.0


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n (returns point, lo, hi)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def boot_auc(y, p, n_boot=2000, seed=67):
    """Bootstrap 95% CI for AUC (returns point, lo, hi)."""
    base = roc_auc_score(y, p)
    rng = np.random.default_rng(seed); n = len(y); a = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(a, [2.5, 97.5]) if a else (float("nan"), float("nan"))
    return base, lo, hi
