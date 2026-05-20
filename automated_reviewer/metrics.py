"""Classification metrics for the automated reviewer evaluation.

Implemented with the standard library only (no scikit-learn) so the project
stays dependency-light. The positive class is "Accept" (1); "Reject" is 0.

These are the same metrics reported in Table 1 of the paper: balanced
accuracy, accuracy, F1, AUC, false-positive rate, false-negative rate.
"""

from __future__ import annotations

import random


def confusion(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) with Accept=1 as the positive class."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    tp, fp, fn, tn = confusion(y_true, y_pred)
    return _safe_div(tp + tn, tp + fp + fn + tn)


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Mean of the per-class recall — robust to class imbalance.

    ICLR accepts ~30% of papers, so plain accuracy is misleading; balanced
    accuracy is the headline metric in the paper for this reason.
    """
    tp, fp, fn, tn = confusion(y_true, y_pred)
    tpr = _safe_div(tp, tp + fn)   # recall on Accept
    tnr = _safe_div(tn, tn + fp)   # recall on Reject
    return (tpr + tnr) / 2.0


def f1(y_true: list[int], y_pred: list[int]) -> float:
    tp, fp, fn, tn = confusion(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def fpr(y_true: list[int], y_pred: list[int]) -> float:
    """False-positive rate: rejected papers the reviewer wrongly accepted."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    return _safe_div(fp, fp + tn)


def fnr(y_true: list[int], y_pred: list[int]) -> float:
    """False-negative rate: accepted papers the reviewer wrongly rejected."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    return _safe_div(fn, fn + tp)


def auc(y_true: list[int], scores: list[float]) -> float:
    """Area under the ROC curve, via the Mann-Whitney U statistic.

    `scores` is a continuous predictor (the reviewer's 1-10 overall score):
    AUC is the probability a random accepted paper scores above a random
    rejected one, with ties counted as 0.5.
    """
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def bootstrap_ci(y_true: list[int], y_pred: list[int], metric_fn,
                 n_resamples: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval for a (y_true, y_pred) metric."""
    rng = random.Random(seed)
    idx = list(range(len(y_true)))
    if not idx:
        return (0.0, 0.0)
    vals = []
    for _ in range(n_resamples):
        sample = [rng.choice(idx) for _ in idx]
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        vals.append(metric_fn(yt, yp))
    vals.sort()
    lo = vals[int(0.025 * n_resamples)]
    hi = vals[int(0.975 * n_resamples)]
    return (lo, hi)


def compute_all(y_true: list[int], y_pred: list[int],
                scores: list[float]) -> dict:
    """Compute every Table-1 metric plus a bootstrap CI on balanced accuracy."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    ba = balanced_accuracy(y_true, y_pred)
    ci_lo, ci_hi = bootstrap_ci(y_true, y_pred, balanced_accuracy)
    return {
        "n": len(y_true),
        "n_accept": tp + fn,
        "n_reject": fp + tn,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": ba,
        "balanced_accuracy_95ci": [ci_lo, ci_hi],
        "f1": f1(y_true, y_pred),
        "auc": auc(y_true, scores),
        "fpr": fpr(y_true, y_pred),
        "fnr": fnr(y_true, y_pred),
    }
