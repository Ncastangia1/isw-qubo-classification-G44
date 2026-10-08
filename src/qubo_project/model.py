

"""
model.py

Supervised classification and prediction module for the ISW QUBO Classification project.

Responsibilities:
- Implement a classifier registry supporting random_forest, logistic_regression, and decision_tree.
- Train the selected classifier on reduced datasets with reproducible seed handling.
- Persist trained models and metadata using joblib.
- Generate predictions, confidence scores, and comprehensive evaluation metrics.
- Export training and classification statistics in JSON format.
- Export test predictions in CSV format.
- Provide a robust CLI interface supporting train and predict commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

LOGGER = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

CLASSIFIER_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
}


def _load_dataset(input_csv: str) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Parameters
    ----------
    input_csv : str
        Path to input CSV.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the CSV cannot be read or is empty.
    """
    path = Path(input_csv)

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: '{input_csv}'")

    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: '{input_csv}'")

    try:
        df = pd.read_csv(path)
        if df.empty or len(df.index) == 0:
            raise ValueError(f"Dataset is empty: '{input_csv}'")
        return df
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise exc
        raise ValueError(f"Failed to read CSV file '{input_csv}': {exc}") from exc


def _validate_train_inputs(
    classifier: str,
    target_column: str,
) -> None:
    """
    Validate training configuration arguments.
    """
    if classifier not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unsupported classifier '{classifier}'. "
            f"Supported classifiers: {list(CLASSIFIER_REGISTRY.keys())}"
        )

    if not target_column:
        raise ValueError("Target column name cannot be empty.")


def _validate_train_dataset(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Validate training dataset structure, features, and target classes.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist in the training dataset."
        )

    feature_columns = [c for c in df.columns if c != target_column]

    if not feature_columns:
        raise ValueError("Training dataset contains no feature columns.")

    target_series = df[target_column]
    unique_classes = target_series.dropna().unique()

    if len(unique_classes) < 2:
        raise ValueError(
            f"Training target column '{target_column}' must contain at least 2 classes, "
            f"found {len(unique_classes)} class(es): {unique_classes}"
        )

    return df[feature_columns], target_series, feature_columns


def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a supervised classification model using the reduced dataset.

    Parameters
    ----------
    classifier : str
        Selected classifier name from registry.
    reducedTrain_csv : str
        Path to reduced training dataset.
    target_column : str
        Binary target column name.
    model_path : str
        Output serialized model path (.joblib).
    metrics_json : str
        Output training metrics JSON path.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Training metrics and metadata dictionary.
    """
    _validate_train_inputs(classifier, target_column)

    start_input = time.perf_counter()
    df = _load_dataset(reducedTrain_csv)
    dataset_input_time = time.perf_counter() - start_input

    X, y, feature_names = _validate_train_dataset(df, target_column)

    LOGGER.info(
        "Training classifier '%s' on %s rows and %s features.",
        classifier,
        len(df.index),
        len(feature_names),
    )

    clf_class = CLASSIFIER_REGISTRY[classifier]
    
    # Instantiate classifier with random seed support where applicable
    try:
        clf = clf_class(random_state=seed)
    except TypeError:
        # Fallback if estimator does not accept random_state
        clf = clf_class()

    training_start = time.perf_counter()
    clf.fit(X, y)
    training_time = time.perf_counter() - training_start

    # Compute training accuracy
    y_pred_train = clf.predict(X)
    train_accuracy = float(accuracy_score(y, y_pred_train))

    target_1_count = int((y == 1).sum())
    target_1_percentage = float((target_1_count / len(y)) * 100)

    metrics = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": str(Path(reducedTrain_csv).name),
        "target_column": target_column,
        "model_path": str(Path(model_path).name),
        "n_samples": int(len(df.index)),
        "n_features": int(len(feature_names)),
        "target_1_percentage": round(target_1_percentage, 2),
        "dataset_input_time": round(float(dataset_input_time), 4),
        "training_time": round(float(training_time), 4),
        "accuracy": round(train_accuracy, 6),
    }

    # Prepare model artifact package containing estimator, features, and metadata
    model_package = {
        "estimator": clf,
        "classifier_name": classifier,
        "feature_names": feature_names,
        "seed": seed,
        "metadata": metrics,
    }

    # Ensure parent directories exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model_package, model_path)
        LOGGER.info("Model saved successfully to '%s'.", model_path)

        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Training metrics exported successfully to '%s'.", metrics_json)

    except Exception as exc:
        raise RuntimeError(f"Failed to persist model or metrics: {exc}") from exc

    return metrics


def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
) -> dict[str, Any]:
    """
    Generate predictions and evaluation metrics using a previously trained model.

    Parameters
    ----------
    reduced_Test_csv : str
        Path to reduced test dataset.
    target_column : str
        Ground-truth target column name.
    model_path : str
        Path to serialized model artifact (.joblib).
    predictions_csv : str
        Output predictions CSV path.
    classif_stats_json : str
        Output classification statistics JSON path.

    Returns
    -------
    dict
        Classification statistics dictionary.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: '{model_path}'")

    try:
        model_package = joblib.load(model_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to load or corrupted model artifact '{model_path}': {exc}") from exc

    if not isinstance(model_package, dict) or "estimator" not in model_package or "feature_names" not in model_package:
        raise ValueError(f"Model artifact '{model_path}' has invalid structure or wrong object type.")

    estimator = model_package["estimator"]
    expected_features = model_package["feature_names"]
    classifier_name = model_package.get("classifier_name", "unknown")

    df_test = _load_dataset(reduced_Test_csv)

    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in test dataset.")

    # Validate feature compatibility
    test_feature_cols = [c for c in df_test.columns if c != target_column]
    if set(test_feature_cols) != set(expected_features):
        missing = set(expected_features) - set(test_feature_cols)
        extra = set(test_feature_cols) - set(expected_features)
        raise ValueError(
            f"Feature mismatch between trained model and test dataset. "
            f"Missing features: {missing}, Extra features: {extra}"
        )

    # Align feature columns to exact training order
    X_test = df_test[expected_features]
    y_true = df_test[target_column]

    LOGGER.info("Generating predictions for %s test samples using '%s'.", len(df_test.index), classifier_name)

    y_pred = estimator.predict(X_test)

    # Generate confidence scores for class 1
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X_test)
        if probs.shape[1] > 1:
            scores = probs[:, 1]
        else:
            scores = probs[:, 0]
    else:
        # Fallback decision function or zero scores if probabilities unavailable
        if hasattr(estimator, "decision_function"):
            dec = estimator.decision_function(X_test)
            scores = 1 / (1 + np.exp(-dec))
        else:
            scores = np.zeros(len(y_pred), dtype=float)

    # Export predictions CSV (row_n, target, prediction, score)[cite: 8]
    pred_df = pd.DataFrame(
        {
            "row_n": range(len(df_test.index)),
            "target": y_true.values,
            "prediction": y_pred,
            "score": scores,
        }
    )

    Path(predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_csv, index=False)
    LOGGER.info("Predictions saved to '%s'.", predictions_csv)

    # Compute evaluation metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    # Compute ROC-AUC (handle single-class edge case gracefully)[cite: 8]
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, scores))
        else:
            LOGGER.warning("ROC-AUC skipped due to single-class test set.")
            roc_auc = 0.0
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    target_1_count = int((y_true == 1).sum())
    target_1_percentage = float((target_1_count / len(y_true)) * 100) if len(y_true) > 0 else 0.0

    stats = {
        "classifier": classifier_name,
        "n_samples": int(len(df_test.index)),
        "target_1_count": target_1_count,
        "target_1_percentage": round(target_1_percentage, 2),
        "accuracy": accuracy,
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
        },
    }

    Path(classif_stats_json).parent.mkdir(parents=True, exist_ok=True)
    with open(classif_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    LOGGER.info("Classification statistics saved to '%s'.", classif_stats_json)

    return stats


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser supporting train and predict subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Supervised model training and prediction module."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command parser
    train_parser = subparsers.add_parser("train", help="Train a classifier.")
    train_parser.add_argument(
        "--classifier",
        required=True,
        choices=list(CLASSIFIER_REGISTRY.keys()),
        help="Classifier name to train.",
    )
    train_parser.add_argument(
        "--in-reduced",
        required=True,
        help="Path to reduced training dataset CSV.",
    )
    train_parser.add_argument(
        "--target",
        required=True,
        help="Target column name.",
    )
    train_parser.add_argument(
        "--out-model",
        required=True,
        help="Output serialized model path (.joblib).",
    )
    train_parser.add_argument(
        "--out-metrics",
        required=True,
        help="Output training metrics JSON path.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # Predict command parser
    predict_parser = subparsers.add_parser("predict", help="Generate predictions using a trained model.")
    predict_parser.add_argument(
        "--input-testset",
        required=True,
        help="Path to reduced test dataset CSV.",
    )
    predict_parser.add_argument(
        "--target",
        required=True,
        help="Ground-truth target column name.",
    )
    predict_parser.add_argument(
        "--model",
        required=True,
        help="Path to serialized model (.joblib).",
    )
    predict_parser.add_argument(
        "--out-predictions",
        required=True,
        help="Output predictions CSV path.",
    )
    predict_parser.add_argument(
        "--out-stats",
        required=True,
        help="Output classification statistics JSON path.",
    )

    return parser


def main() -> int:
    """
    CLI entry point.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            train(
                classifier=args.classifier,
                reducedTrain_csv=args.in_reduced,
                target_column=args.target,
                model_path=args.out_model,
                metrics_json=args.out_metrics,
                seed=args.seed,
            )
        elif args.command == "predict":
            predict(
                reduced_Test_csv=args.input_testset,
                target_column=args.target,
                model_path=args.model,
                predictions_csv=args.out_predictions,
                classif_stats_json=args.out_stats,
            )
        else:
            parser.print_help()
            return 1

        return 0

    except Exception as exc:
        LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

```