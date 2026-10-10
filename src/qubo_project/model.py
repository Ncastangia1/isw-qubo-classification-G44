from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import time
import argparse
import sys
from pathlib import Path
from typing import Any


import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

CLASSIFIER_REGISTRY: dict[str, type[ClassifierMixin]] = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
}


def _get_classifier(
    classifier: str,
    seed: int,
) -> ClassifierMixin:
    """
    Instantiate a supported classifier with deterministic settings.

    Parameters
    ----------
    classifier : str
        Classifier name.
    seed : int
        Random seed.

    Returns
    -------
    ClassifierMixin
        Instantiated classifier.

    Raises
    ------
    ValueError
        If classifier is not supported.
    """
    if classifier not in CLASSIFIER_REGISTRY:
        valid_names = ", ".join(sorted(CLASSIFIER_REGISTRY))
        raise ValueError(
            f"Unsupported classifier '{classifier}'. "
            f"Supported classifiers: {valid_names}."
        )
    if classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        )
    if classifier == "logistic_regression":
        return LogisticRegression(
            random_state=seed,
            max_iter=1000,
        )
    if classifier == "decision_tree":
        return DecisionTreeClassifier(
            random_state=seed,
        )
    valid_names = ", ".join(sorted(CLASSIFIER_REGISTRY))
    raise ValueError(
        f"Unsupported classifier '{classifier}'. "
        f"Supported classifiers: {valid_names}."
    )


def _validate_csv_path(
    csv_path: str,
) -> Path:
    """
    Validate a CSV file path.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.

    Returns
    -------
    Path
        Validated path.

    Raises
    ------
    FileNotFoundError
        If path does not exist or is not a file.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV path is not a file: {path}"
        )
    return path


def _load_dataset(
    csv_path: str,
) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Parameters
    ----------
    csv_path : str
        Dataset path.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    ValueError
        If dataset cannot be loaded or is empty.
    """
    path = _validate_csv_path(csv_path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to load dataset from '{path}'."
        ) from exc
    if df.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )
    return df


def _validate_target_column(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Validate target column existence.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    target_column : str
        Target column name.

    Raises
    ------
    ValueError
        If target column is missing.
    """
    if target_column not in df.columns:
        available_columns = ", ".join(df.columns.astype(str))
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {available_columns}"
        )


def _validate_training_dataset(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Validate a training dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    target_column : str
        Target column name.

    Raises
    ------
    ValueError
        If dataset validation fails.
    """
    if df.empty:
        raise ValueError(
            "Training dataset is empty."
        )
    _validate_target_column(
        df=df,
        target_column=target_column,
    )
    if df.columns.duplicated().any():
        duplicated_columns = (
            df.columns[df.columns.duplicated()]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Training dataset contains duplicate columns: "
            f"{duplicated_columns}"
        )
    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]
    if not feature_columns:
        raise ValueError(
            "Training dataset contains no feature columns."
        )
    target = df[target_column]
    if target.isna().any():
        raise ValueError("Training target cannot contain missing values.")
    unique_classes = target.dropna().unique()
    if len(unique_classes) < 2 or not set(unique_classes).issubset({0, 1}):
        raise ValueError(
            "Training target must contain both binary classes 0 and 1."
        )


def _validate_prediction_dataset(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Validate a prediction dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    target_column : str
        Target column name.

    Raises
    ------
    ValueError
        If dataset validation fails.
    """
    if df.empty:
        raise ValueError(
            "Prediction dataset is empty."
        )
    _validate_target_column(
        df=df,
        target_column=target_column,
    )
    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]
    if not feature_columns:
        raise ValueError(
            "Prediction dataset contains no feature columns."
        )
    target = df[target_column]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(
            "Prediction target must contain only binary classes 0 and 1."
        )


def _split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    target_column : str
        Target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Features and target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def _ensure_parent(
    path: str,
) -> None:
    """
    Ensure parent directory exists.

    Parameters
    ----------
    path : str
        Output path.
    """
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _get_target_distribution(
    target: pd.Series,
) -> dict[str, int]:
    """
    Compute target class distribution.

    Parameters
    ----------
    target : pd.Series
        Target vector.

    Returns
    -------
    dict[str, int]
        Class counts.
    """
    counts = target.value_counts(dropna=False)
    return {
        str(label): int(count)
        for label, count in counts.items()
    }


def _get_library_metadata() -> dict[str, Any]:
    """
    Collect runtime library metadata.

    Returns
    -------
    dict[str, Any]
        Library versions.
    """
    import sklearn
    return {
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }


def _build_training_metrics(
    classifier_name: str,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    training_time: float,
    training_dataset: str,
    target_column: str,
    model_path: str,
) -> dict[str, Any]:
    target_1_count = int((y == 1).sum())
    return {
        "classifier": classifier_name,
        "seed": seed,
        "training_dataset": training_dataset,
        "target_column": target_column,
        "model_path": model_path,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "target_distribution": _get_target_distribution(y),
        "target_1_percentage": float(target_1_count / len(y) * 100),
        "training_time": round(
            float(training_time),
            6,
        ),
    }


def _safe_roc_auc(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> float | None:
    unique_classes = pd.Series(y_true).dropna().unique()
    if len(unique_classes) < 2:
        LOGGER.warning(
            "ROC-AUC cannot be computed because the evaluation "
            "target contains fewer than two classes."
        )
        return None
    try:
        return float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )
    except ValueError:
        LOGGER.warning(
            "ROC-AUC computation failed due to invalid class "
            "distribution."
        )
        return None


def _build_class_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )
    return {
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
    }


def _build_prediction_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classifier_name: str,
) -> dict[str, Any]:
    target_1_count = int((y_true == 1).sum())
    metrics = {
        "classifier": classifier_name,
        "n_samples": int(len(y_true)),
        "target_1_count": target_1_count,
        "target_1_percentage": float(target_1_count / len(y_true) * 100),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc": _safe_roc_auc(
            y_true,
            y_score,
        ),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).tolist(),
        },
    }
    metrics.update(
        _build_class_metrics(
            y_true,
            y_pred,
        )
    )
    return metrics


def _write_json(
    payload: dict[str, Any],
    output_path: str,
) -> None:
    _ensure_parent(output_path)
    with Path(output_path).open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    LOGGER.info(
        "JSON artifact written to '%s'.",
        output_path,
    )


def _build_model_artifact(
    estimator: ClassifierMixin,
    classifier_name: str,
    feature_names: list[str],
    seed: int,
) -> dict[str, Any]:
    return {
        "estimator": estimator,
        "classifier_name": classifier_name,
        "feature_names": feature_names,
        "seed": seed,
        "metadata": {
            "training_timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "library_versions": _get_library_metadata(),
        },
    }


def _save_model(
    artifact: dict[str, Any],
    model_path: str,
) -> None:
    try:
        _ensure_parent(model_path)
        joblib.dump(
            artifact,
            model_path,
            compress=3,
        )
        LOGGER.info(
            "Model artifact saved to '%s'.",
            model_path,
        )
    except Exception:
        LOGGER.error(
            "Failed to save model artifact to '%s'.",
            model_path,
            exc_info=True,
        )
        raise


def _load_model(
    model_path: str,
) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(
            "Loaded model artifact must be a dictionary."
        )
    required_keys = {
        "estimator",
        "classifier_name",
        "feature_names",
    }
    missing_keys = sorted(
        required_keys.difference(artifact.keys())
    )
    if missing_keys:
        raise ValueError(
            "Invalid model artifact. Missing keys: "
            f"{missing_keys}"
        )
    LOGGER.info(
        "Model artifact loaded from '%s'.",
        model_path,
    )
    return artifact


def _validate_feature_compatibility(
    model_feature_names: list[str],
    dataset_feature_names: list[str],
) -> None:
    model_set = set(model_feature_names)
    dataset_set = set(dataset_feature_names)
    missing_features = sorted(
        model_set - dataset_set
    )
    unexpected_features = sorted(
        dataset_set - model_set
    )
    errors: list[str] = []
    if len(model_feature_names) != len(dataset_feature_names):
        errors.append(
            "Feature count mismatch "
            f"(model={len(model_feature_names)}, "
            f"dataset={len(dataset_feature_names)})"
        )
    if missing_features:
        errors.append(
            f"Missing features: {missing_features}"
        )
    if unexpected_features:
        errors.append(
            f"Unexpected features: {unexpected_features}"
        )
    if not errors and model_feature_names != dataset_feature_names:
        errors.append(
            "Feature order mismatch between model and dataset."
        )
    if errors:
        message = " | ".join(errors)
        LOGGER.error(
            "Feature compatibility validation failed: %s",
            message,
        )
        raise ValueError(
            f"Feature compatibility validation failed. {message}"
        )
        
def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
 ) -> None:
    try:
        LOGGER.info(
            "Training started | classifier=%s | dataset=%s",
            classifier,
            reducedTrain_csv,
        )

        df = _load_dataset(reducedTrain_csv)

        LOGGER.info(
            "Dataset loaded | rows=%s | columns=%s",
            len(df),
            len(df.columns),
        )

        _validate_training_dataset(
            df=df,
            target_column=target_column,
        )

        X, y = _split_features_target(
            df=df,
            target_column=target_column,
        )

        if len(X.columns) <= 0:
            raise ValueError(
                "Training dataset contains no feature columns."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Training target must contain at least two classes."
            )

        class_distribution = y.value_counts(normalize=True)

        if not class_distribution.empty:
            min_frequency = float(class_distribution.min())

            if min_frequency < 0.05:
                LOGGER.warning(
                    "Class imbalance detected | "
                    "smallest_class_frequency=%.6f",
                    min_frequency,
                )

        estimator = _get_classifier(
            classifier=classifier,
            seed=seed,
        )

        LOGGER.info(
            "Classifier selected | classifier=%s",
            classifier,
        )

        LOGGER.info(
            "Training in progress | classifier=%s",
            classifier,
        )

        training_start = time.perf_counter()

        try:
            estimator.fit(X, y)
        except Exception:
            LOGGER.error(
                "Model training failed.",
                exc_info=True,
            )
            raise

        training_end = time.perf_counter()
        training_time_seconds = (
        training_end - training_start
        )

        y_pred = estimator.predict(X)

        training_accuracy = float(
            accuracy_score(
                y,
                y_pred,
            )
        )

        training_metrics = _build_training_metrics(
            classifier_name=classifier,
            seed=seed,
            X=X,
            y=y,
            training_time=training_time_seconds,
            training_dataset=str(reducedTrain_csv),
            target_column=target_column,
            model_path=str(model_path),
        )

        training_metrics["accuracy"] = training_accuracy

        artifact = _build_model_artifact(
            estimator=estimator,
            classifier_name=classifier,
            feature_names=list(X.columns),
            seed=seed,
        )

        _save_model(
            artifact=artifact,
            model_path=model_path,
        )

        LOGGER.info(
            "Model saved | path=%s",
            model_path,
        )

        _write_json(
            payload=training_metrics,
            output_path=metrics_json,
        )

        LOGGER.info(
            "Metrics exported | path=%s",
             metrics_json,
        )

        LOGGER.info(
            "Training completed successfully | "
            "classifier=%s | rows=%s | features=%s | "
            "training_time_seconds=%.6f",
            classifier,
            len(X),
            X.shape[1],
            training_time_seconds,
        )

    except Exception:
        LOGGER.error(
            "Training workflow failed.",
            exc_info=True,
        )
        raise


def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
) -> None:
    try:
        LOGGER.info(
            "Prediction started | dataset=%s | model=%s",
            reduced_Test_csv,
            model_path,
        )

        df = _load_dataset(reduced_Test_csv)

        _validate_prediction_dataset(
            df=df,
            target_column=target_column,
        )

        X, y_true = _split_features_target(
            df=df,
            target_column=target_column,
        )

        artifact = _load_model(model_path)

        estimator = artifact["estimator"]
        feature_names = artifact["feature_names"]
        classifier_name = artifact["classifier_name"]

        _validate_feature_compatibility(
            model_feature_names=feature_names,
            dataset_feature_names=list(X.columns),
        )

        y_pred = estimator.predict(X)

        try:
            if hasattr(estimator, "predict_proba"):
                positive_column = np.flatnonzero(
                    np.asarray(estimator.classes_) == 1
                )
                if len(positive_column) != 1:
                    raise ValueError(
                        "Trained classifier does not expose class 1."
                    )
                y_score = estimator.predict_proba(X)[:, positive_column[0]]
            elif hasattr(estimator, "decision_function"):
                raw_scores = estimator.decision_function(X)

                score_min = float(np.min(raw_scores))
                score_max = float(np.max(raw_scores))

                if score_max > score_min:
                    y_score = (
                        (raw_scores - score_min)
                        / (score_max - score_min)
                    )
                else:
                    y_score = np.zeros(
                        len(y_pred),
                        dtype=float,
                    )
            else:
                LOGGER.warning(
                    "Estimator does not expose probability "
                    "scores. Using zero scores."
                )
                y_score = np.zeros(
                    len(y_pred),
                    dtype=float,
                )

        except Exception:
            LOGGER.warning(
                "Failed to compute prediction scores. "
                "Using zero scores.",
                exc_info=True,
            )
            y_score = np.zeros(
                len(y_pred),
                dtype=float,
            )

        metrics = _build_prediction_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            classifier_name=classifier_name,
        )

        predictions_df = pd.DataFrame(
            {
                "row_n": range(len(df)),
                "target": y_true,
                "prediction": y_pred,
                "score": y_score,
            }
        )

        _ensure_parent(predictions_csv)

        predictions_df.to_csv(
            predictions_csv,
            index=False,
        )

        _write_json(
            payload=metrics,
            output_path=classif_stats_json,
        )

        LOGGER.info(
            "Prediction completed successfully | "
            "classifier=%s | rows=%s | "
            "predictions=%s | stats=%s",
            classifier_name,
            len(df),
            predictions_csv,
            classif_stats_json,
        )

    except Exception:
        LOGGER.error(
            "Prediction workflow failed.",
            exc_info=True,
        )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QUBO Classification Model Utilities"
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a classification model.",
    )

    train_parser.add_argument(
        "--classifier",
        required=True,
    )
    train_parser.add_argument(
        "--in-reduced",
        dest="in_reduced",
        required=True,
    )
    train_parser.add_argument(
        "--target",
        required=True,
    )
    train_parser.add_argument(
        "--out-model",
        dest="out_model",
        required=True,
    )
    train_parser.add_argument(
        "--out-metrics",
        dest="out_metrics",
        required=True,
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run model predictions.",
    )

    predict_parser.add_argument(
        "--input-testset",
        dest="input_testset",
        required=True,
    )
    predict_parser.add_argument(
        "--target",
        required=True,
    )
    predict_parser.add_argument(
        "--model",
        required=True,
    )
    predict_parser.add_argument(
        "--out-predictions",
        dest="out_predictions",
        required=True,
    )
    predict_parser.add_argument(
        "--out-stats",
        dest="out_stats",
        required=True,
    )

    return parser


def main() -> int:
    parser = _build_parser()

    try:
        args = parser.parse_args()

        if args.command == "train":
            train(
                classifier=args.classifier,
                reducedTrain_csv=args.in_reduced,
                target_column=args.target,
                model_path=args.out_model,
                metrics_json=args.out_metrics,
                seed=args.seed,
            )
            return 0

        if args.command == "predict":
            predict(
                reduced_Test_csv=args.input_testset,
                target_column=args.target,
                model_path=args.model,
                predictions_csv=args.out_predictions,
                classif_stats_json=args.out_stats,
            )
            return 0

        parser.print_help()
        return 1

    except ValueError:
        LOGGER.error(
            "Validation error occurred.",
            exc_info=True,
        )
        return 1

    except Exception:
        LOGGER.error(
            "Runtime failure occurred.",
            exc_info=True,
        )
        return 2


__all__ = [
    "train",
    "predict",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
     