"""Evaluation harness shared by every model in the project.

Reports accuracy and macro F1 with bootstrap confidence intervals, a
confusion matrix, and the most confident misclassifications for error
analysis. Every model goes through this same code path so results stay
comparable.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
N_BOOTSTRAP = 2000
SEED = 42


def bootstrap_ci(y_true, y_pred, metric_fn, n_resamples=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower, upper = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def evaluate(model_name, y_true, y_pred, labels, texts=None, probabilities=None):
    """Score predictions and write a full report to results/<model_name>/."""
    out = RESULTS_DIR / model_name
    out.mkdir(parents=True, exist_ok=True)

    macro_f1 = lambda t, p: f1_score(t, p, average="macro")
    metrics = {
        "model": model_name,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci95": bootstrap_ci(y_true, y_pred, accuracy_score),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "macro_f1_ci95": bootstrap_ci(y_true, y_pred, macro_f1),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(labels, f1_score(y_true, y_pred, average=None))
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")

    if texts is not None and probabilities is not None:
        confidence = probabilities.max(axis=1)
        wrong = np.asarray(y_true) != np.asarray(y_pred)
        errors = pd.DataFrame(
            {
                "text": [t[:300] for t in texts],
                "true_label": [labels[i] for i in y_true],
                "predicted_label": [labels[i] for i in y_pred],
                "confidence": confidence,
            }
        )[wrong].sort_values("confidence", ascending=False)
        errors.head(25).to_csv(out / "top_errors.csv", index=False)

    return metrics


def print_report(metrics):
    lo, hi = metrics["macro_f1_ci95"]
    alo, ahi = metrics["accuracy_ci95"]
    print(f"model:     {metrics['model']}")
    print(f"accuracy:  {metrics['accuracy']:.4f}  (95% CI {alo:.4f}–{ahi:.4f})")
    print(f"macro F1:  {metrics['macro_f1']:.4f}  (95% CI {lo:.4f}–{hi:.4f})")
    for label, score in sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1]):
        print(f"  {label:<15} F1 {score:.4f}")
