"""
Evaluate the router on the MedHop validation split.

Generates labels for validation using the same BM25 median-split heuristic
(fit on training scores, applied to validation), then reports accuracy,
confusion matrix, and feature importances.

Saves results to results.json and plots to results/.

Usage: python evaluate.py
"""

import json
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from features import extract_features, FEATURE_NAMES
from label import bm25_answer_score
from train import build_feature_matrix


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def label_split(split, median_score: float) -> dict:
    """Apply the median-split labeling heuristic to any split."""
    labels = {}
    for ex in split:
        score = bm25_answer_score(ex)
        labels[ex["id"]] = 0 if score >= median_score else 1
    return labels


def main():
    print("Loading model and data...")
    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)

    dataset = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    train = dataset["train"]
    val   = dataset["validation"]

    # Recompute the training median so we use the same threshold for validation labels
    print("Recomputing training BM25 median for consistent labeling...")
    train_scores = [bm25_answer_score(ex) for ex in train]
    median_score = float(np.median(train_scores))
    print(f"  Training median BM25 score: {median_score:.3f}")

    print("Labeling and extracting features for validation split...")
    val_labels = label_split(val, median_score)
    X_val, y_val = build_feature_matrix(val, val_labels)
    print(f"  Validation set: {len(y_val)} examples, {y_val.mean()*100:.1f}% hard")

    # Classifier predictions
    y_pred_clf = clf.predict(X_val)
    acc_clf = accuracy_score(y_val, y_pred_clf)

    # Baselines
    y_pred_multi  = np.ones_like(y_val)
    y_pred_single = np.zeros_like(y_val)
    acc_base_multi  = accuracy_score(y_val, y_pred_multi)
    acc_base_single = accuracy_score(y_val, y_pred_single)

    print(f"\n--- Results ---")
    print(f"  Classifier accuracy:      {acc_clf*100:.1f}%")
    print(f"  Always-multi baseline:    {acc_base_multi*100:.1f}%")
    print(f"  Always-single baseline:   {acc_base_single*100:.1f}%")
    print(f"  Improvement over best baseline: {(acc_clf - max(acc_base_multi, acc_base_single))*100:+.1f}%")

    # Confusion matrix plot
    cm = confusion_matrix(y_val, y_pred_clf)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["single", "multi"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Router Confusion Matrix (Validation)")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix → {cm_path}")

    # Feature importance plot (LR coefficients)
    coefs = clf.named_steps["lr"].coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        [coefs[i] for i in sorted_idx],
        color=["#e74c3c" if c < 0 else "#2ecc71" for c in [coefs[i] for i in sorted_idx]],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("Feature Importances")
    plt.tight_layout()
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved feature importance → {fi_path}")

    # Save all results to JSON
    results = {
        "validation_accuracy_classifier": round(acc_clf, 4),
        "validation_accuracy_always_multi_baseline": round(acc_base_multi, 4),
        "validation_accuracy_always_single_baseline": round(acc_base_single, 4),
        "improvement_over_best_baseline": round(acc_clf - max(acc_base_multi, acc_base_single), 4),
        "n_val_examples": int(len(y_val)),
        "val_pct_hard": round(float(y_val.mean()), 4),
        "train_bm25_median": round(median_score, 4),
        "feature_importances": {
            name: round(float(coef), 4)
            for name, coef in zip(FEATURE_NAMES, coefs)
        },
        "confusion_matrix": cm.tolist(),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results → results.json")


if __name__ == "__main__":
    main()
