"""
Train the query router classifier.

Loads MedHop train split + BM25-derived labels from labels_train.json,
extracts features, trains a LogisticRegression, and saves the model.

Output: model.pkl

Usage: python train.py
"""

import json
import pickle
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import extract_features, FEATURE_NAMES


def build_feature_matrix(split, labels: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) for all examples that have a label."""
    X, y = [], []
    for ex in split:
        if ex["id"] not in labels:
            continue
        X.append(extract_features(ex))
        y.append(labels[ex["id"]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    print("Loading labels...")
    with open("labels_train.json") as f:
        labels = json.load(f)

    print("Loading MedHop train split...")
    dataset = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    train   = dataset["train"]

    print("Extracting features (this takes ~2 min)...")
    X, y = build_feature_matrix(train, labels)
    print(f"  Feature matrix shape: {X.shape}  Label shape: {y.shape}")
    print(f"  Class balance: {y.mean()*100:.1f}% hard")

    # StandardScaler + LogisticRegression in a pipeline so scaling is
    # always applied consistently at inference time too
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # 5-fold CV gives a reliable accuracy estimate before we touch validation data
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Train final model on all training data
    clf.fit(X, y)
    print("\nFinal model trained on full training set.")

    # Print feature importances (LR coefficients after scaling)
    coefs = clf.named_steps["lr"].coef_[0]
    print("\nFeature importances (logistic regression coefficients):")
    for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:<28} {coef:+.4f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("\nModel saved to model.pkl")


if __name__ == "__main__":
    main()
