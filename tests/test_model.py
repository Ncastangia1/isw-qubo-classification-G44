from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from qubo_project.model import (
    _ensure_parent,
    _get_library_metadata,
    _get_target_distribution,
    _load_dataset,
    _split_features_target,
    _validate_csv_path,
    _validate_prediction_dataset,
    _validate_target_column,
    _validate_training_dataset,
    _get_classifier,
    _build_training_metrics,
    _safe_roc_auc,
    _build_class_metrics,
    _build_prediction_metrics,
    _write_json,
    _build_model_artifact,
    _save_model,
    _load_model,
    _validate_feature_compatibility,
    train,
    predict,
    _build_parser,
    main,
)


@pytest.fixture
def valid_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [10, 11, 12, 13, 14, 15],
            "feature_c": [100, 101, 102, 103, 104, 105],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def valid_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [6, 5, 4, 3, 2, 1],
            "feature_b": [15, 14, 13, 12, 11, 10],
            "feature_c": [105, 104, 103, 102, 101, 100],
            "target": [1, 0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def tmp_csv(tmp_path: Path):
    def _writer(df: pd.DataFrame, filename: str = "dataset.csv") -> Path:
        csv_path = tmp_path / filename
        df.to_csv(csv_path, index=False)
        return csv_path
    return _writer


def test_validate_csv_path_success(
    valid_training_df: pd.DataFrame,
    tmp_csv,
) -> None:
    csv_path = tmp_csv(valid_training_df)
    result = _validate_csv_path(str(csv_path))
    assert result == csv_path


def test_validate_csv_path_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        _validate_csv_path(str(missing_path))


def test_validate_csv_path_not_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _validate_csv_path(str(tmp_path))


def test_load_dataset_success(
    valid_training_df: pd.DataFrame,
    tmp_csv,
) -> None:
    csv_path = tmp_csv(valid_training_df)
    result = _load_dataset(str(csv_path))
    assert result.shape == valid_training_df.shape


def test_load_dataset_empty_csv(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        _load_dataset(str(csv_path))


def test_load_dataset_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _load_dataset(str(tmp_path / "missing.csv"))


def test_validate_target_column_success(
    valid_training_df: pd.DataFrame,
) -> None:
    _validate_target_column(
        valid_training_df,
        "target",
    )


def test_validate_target_column_missing_target(
    valid_training_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        _validate_target_column(
            valid_training_df,
            "missing_target",
        )


def test_validate_training_dataset_success(
    valid_training_df: pd.DataFrame,
) -> None:
    _validate_training_dataset(
        valid_training_df,
        "target",
    )


def test_validate_training_dataset_empty() -> None:
    with pytest.raises(ValueError):
        _validate_training_dataset(
            pd.DataFrame(),
            "target",
        )


def test_validate_training_dataset_missing_target(
    valid_training_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        _validate_training_dataset(
            valid_training_df.drop(columns=["target"]),
            "target",
        )


def test_validate_training_dataset_no_features() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError):
        _validate_training_dataset(
            df,
            "target",
        )


def test_validate_training_dataset_single_class_target() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [5, 6, 7, 8],
            "feature_c": [9, 10, 11, 12],
            "target": [1, 1, 1, 1],
        }
    )
    with pytest.raises(ValueError):
        _validate_training_dataset(
            df,
            "target",
        )


def test_validate_training_dataset_duplicate_columns() -> None:
    df = pd.DataFrame(
        [[1, 2, 3, 0], [4, 5, 6, 1]],
        columns=[
            "feature_a",
            "feature_a",
            "feature_c",
            "target",
        ],
    )
    with pytest.raises(ValueError):
        _validate_training_dataset(
            df,
            "target",
        )


def test_validate_prediction_dataset_success(
    valid_prediction_df: pd.DataFrame,
) -> None:
    _validate_prediction_dataset(
        valid_prediction_df,
        "target",
    )


def test_validate_prediction_dataset_empty() -> None:
    with pytest.raises(ValueError):
        _validate_prediction_dataset(
            pd.DataFrame(),
            "target",
        )


def test_validate_prediction_dataset_missing_target(
    valid_prediction_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        _validate_prediction_dataset(
            valid_prediction_df.drop(columns=["target"]),
            "target",
        )


def test_validate_prediction_dataset_no_features() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError):
        _validate_prediction_dataset(
            df,
            "target",
        )


def test_split_features_target(
    valid_training_df: pd.DataFrame,
) -> None:
    X, y = _split_features_target(
        valid_training_df,
        "target",
    )
    assert "target" not in X.columns
    assert len(X) == len(valid_training_df)
    assert y.equals(valid_training_df["target"])


def test_ensure_parent_creates_directory(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "nested" / "dir" / "file.json"
    _ensure_parent(str(output_path))
    assert output_path.parent.exists()


def test_get_target_distribution() -> None:
    target = pd.Series([0, 0, 1, 1, 1])
    distribution = _get_target_distribution(target)
    assert distribution == {
        "0": 2,
        "1": 3,
    }


def test_get_library_metadata() -> None:
    metadata = _get_library_metadata()
    assert {
        "pandas_version",
        "numpy_version",
        "sklearn_version",
    }.issubset(metadata.keys())


def test_get_classifier_random_forest() -> None:
    classifier = _get_classifier(
        "random_forest",
        seed=123,
    )
    assert isinstance(
        classifier,
        RandomForestClassifier,
    )
    assert classifier.random_state == 123


def test_get_classifier_logistic_regression() -> None:
    classifier = _get_classifier(
        "logistic_regression",
        seed=456,
    )
    assert isinstance(
        classifier,
        LogisticRegression,
    )
    assert classifier.random_state == 456


def test_get_classifier_decision_tree() -> None:
    classifier = _get_classifier(
        "decision_tree",
        seed=789,
    )
    assert isinstance(
        classifier,
        DecisionTreeClassifier,
    )
    assert classifier.random_state == 789


def test_get_classifier_invalid_name() -> None:
    with pytest.raises(ValueError):
        _get_classifier(
            "invalid_classifier",
            seed=42,
        )


def test_build_training_metrics(
    valid_training_df: pd.DataFrame,
) -> None:
    X, y = _split_features_target(
        valid_training_df,
        "target",
    )
    metrics = _build_training_metrics(
        classifier_name="random_forest",
        seed=42,
        X=X,
        y=y,
        training_time_seconds=1.23456789,
    )
    assert metrics["classifier"] == "random_forest"
    assert metrics["seed"] == 42
    assert metrics["n_samples"] == len(X)
    assert metrics["n_features"] == X.shape[1]
    assert metrics["target_distribution"] == {
        "0": 3,
        "1": 3,
    }
    assert metrics["training_time_seconds"] == round(
        1.23456789,
        6,
    )


def test_safe_roc_auc_valid_case() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    result = _safe_roc_auc(
        y_true,
        y_score,
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_safe_roc_auc_single_class_returns_none() -> None:
    y_true = pd.Series([1, 1, 1, 1])
    y_score = np.array([0.8, 0.9, 0.7, 0.6])
    result = _safe_roc_auc(
        y_true,
        y_score,
    )
    assert result is None


def test_build_class_metrics_complete_binary_case() -> None:
    metrics = _build_class_metrics(
        pd.Series([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
    )
    assert "class_0" in metrics
    assert "class_1" in metrics
    assert metrics["class_0"]["f1"] == pytest.approx(1.0)
    assert metrics["class_1"]["f1"] == pytest.approx(1.0)


def test_build_class_metrics_missing_positive_class() -> None:
    metrics = _build_class_metrics(
        pd.Series([0, 0, 0, 0]),
        np.array([0, 0, 0, 0]),
    )
    assert "class_0" in metrics
    assert "class_1" in metrics
    assert isinstance(
        metrics["class_0"]["precision"],
        float,
    )
    assert isinstance(
        metrics["class_1"]["precision"],
        float,
    )


def test_build_prediction_metrics() -> None:
    metrics = _build_prediction_metrics(
        y_true=pd.Series([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "class_0" in metrics
    assert "class_1" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_write_json(
    tmp_path: Path,
) -> None:
    payload = {
        "accuracy": 0.95,
        "classifier": "random_forest",
    }
    output_path = tmp_path / "metrics.json"
    _write_json(
        payload,
        str(output_path),
    )
    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        loaded = json.load(file_handle)
    assert loaded == payload


def test_build_model_artifact(
    valid_training_df: pd.DataFrame,
) -> None:
    X, y = _split_features_target(
        valid_training_df,
        "target",
    )
    estimator = _get_classifier(
        "decision_tree",
        seed=42,
    )
    estimator.fit(X, y)
    artifact = _build_model_artifact(
        estimator=estimator,
        classifier_name="decision_tree",
        feature_names=list(X.columns),
        seed=42,
    )
    assert "estimator" in artifact
    assert "classifier_name" in artifact
    assert "feature_names" in artifact
    assert "seed" in artifact
    assert "metadata" in artifact
    assert "training_timestamp" in artifact["metadata"]
    assert "library_versions" in artifact["metadata"]


def test_save_and_load_model_roundtrip(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    X, y = _split_features_target(
        valid_training_df,
        "target",
    )
    estimator = _get_classifier(
        "random_forest",
        seed=42,
    )
    estimator.fit(X, y)
    artifact = _build_model_artifact(
        estimator=estimator,
        classifier_name="random_forest",
        feature_names=list(X.columns),
        seed=42,
    )
    model_path = tmp_path / "model.joblib"
    _save_model(
        artifact,
        str(model_path),
    )
    loaded = _load_model(str(model_path))
    assert loaded["classifier_name"] == "random_forest"
    assert loaded["feature_names"] == list(X.columns)
    assert loaded["seed"] == 42


def test_load_model_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _load_model(
            str(tmp_path / "missing.joblib")
        )


def test_load_model_invalid_artifact(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.joblib"
    joblib.dump(
        {"invalid": True},
        invalid_path,
    )
    with pytest.raises(ValueError):
        _load_model(str(invalid_path))


def test_validate_feature_compatibility_success() -> None:
    _validate_feature_compatibility(
        ["a", "b", "c"],
        ["a", "b", "c"],
    )


def test_validate_feature_compatibility_missing_feature() -> None:
    with pytest.raises(ValueError):
        _validate_feature_compatibility(
            ["a", "b", "c"],
            ["a", "b"],
        )


def test_validate_feature_compatibility_unexpected_feature() -> None:
    with pytest.raises(ValueError):
        _validate_feature_compatibility(
            ["a", "b"],
            ["a", "b", "c"],
        )


def test_validate_feature_compatibility_order_mismatch() -> None:
    with pytest.raises(ValueError):
        _validate_feature_compatibility(
            ["a", "b", "c"],
            ["b", "a", "c"],
        )


def test_validate_feature_compatibility_count_mismatch() -> None:
    with pytest.raises(ValueError):
        _validate_feature_compatibility(
            ["a", "b", "c", "d"],
            ["a", "b", "c"],
        )


def test_train_random_forest_end_to_end(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "training_metrics.json"
    valid_training_df.to_csv(train_csv, index=False)
    train(
        classifier="random_forest",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
        seed=42,
    )
    assert model_path.exists()
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["classifier"] == "random_forest"
    assert metrics["seed"] == 42
    assert metrics["n_samples"] == len(valid_training_df)
    assert metrics["n_features"] == 3
    assert "accuracy" in metrics


def test_train_logistic_regression_end_to_end(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    model_path = tmp_path / "lr.joblib"
    metrics_path = tmp_path / "lr_metrics.json"
    valid_training_df.to_csv(train_csv, index=False)
    train(
        classifier="logistic_regression",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
    )
    assert model_path.exists()
    assert metrics_path.exists()


def test_train_decision_tree_end_to_end(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    model_path = tmp_path / "dt.joblib"
    metrics_path = tmp_path / "dt_metrics.json"
    valid_training_df.to_csv(train_csv, index=False)
    train(
        classifier="decision_tree",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
    )
    assert model_path.exists()
    assert metrics_path.exists()


def test_train_missing_target(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    valid_training_df.to_csv(train_csv, index=False)
    with pytest.raises(ValueError):
        train(
            classifier="random_forest",
            reducedTrain_csv=str(train_csv),
            target_column="missing",
            model_path=str(tmp_path / "m.joblib"),
            metrics_json=str(tmp_path / "m.json"),
        )


def test_train_invalid_classifier(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    valid_training_df.to_csv(train_csv, index=False)
    with pytest.raises(ValueError):
        train(
            classifier="invalid",
            reducedTrain_csv=str(train_csv),
            target_column="target",
            model_path=str(tmp_path / "m.joblib"),
            metrics_json=str(tmp_path / "m.json"),
        )


def test_train_single_class_target(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "f2": [5, 6, 7, 8],
            "f3": [9, 10, 11, 12],
            "target": [1, 1, 1, 1],
        }
    )
    train_csv = tmp_path / "train.csv"
    df.to_csv(train_csv, index=False)
    with pytest.raises(ValueError):
        train(
            classifier="random_forest",
            reducedTrain_csv=str(train_csv),
            target_column="target",
            model_path=str(tmp_path / "m.joblib"),
            metrics_json=str(tmp_path / "m.json"),
        )


def test_train_empty_dataset(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        train(
            classifier="random_forest",
            reducedTrain_csv=str(csv_path),
            target_column="target",
            model_path=str(tmp_path / "m.joblib"),
            metrics_json=str(tmp_path / "m.json"),
        )


def test_predict_end_to_end(
    valid_training_df: pd.DataFrame,
    valid_prediction_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "train_metrics.json"
    predictions_path = tmp_path / "predictions.csv"
    stats_path = tmp_path / "classification_stats.json"
    valid_training_df.to_csv(train_csv, index=False)
    valid_prediction_df.to_csv(test_csv, index=False)
    train(
        classifier="random_forest",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
    )
    predict(
        reduced_Test_csv=str(test_csv),
        target_column="target",
        model_path=str(model_path),
        predictions_csv=str(predictions_path),
        classif_stats_json=str(stats_path),
    )
    assert predictions_path.exists()
    assert stats_path.exists()
    predictions_df = pd.read_csv(predictions_path)
    assert {
        "row_n",
        "target",
        "prediction",
        "score",
    }.issubset(predictions_df.columns)
    assert predictions_df["score"].between(0, 1).all()
    stats = json.loads(
        stats_path.read_text(encoding="utf-8")
    )
    assert "accuracy" in stats
    assert "class_0" in stats
    assert "class_1" in stats
    assert "confusion_matrix" in stats


def test_predict_missing_model(
    valid_prediction_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    test_csv = tmp_path / "test.csv"
    valid_prediction_df.to_csv(test_csv, index=False)
    with pytest.raises(FileNotFoundError):
        predict(
            reduced_Test_csv=str(test_csv),
            target_column="target",
            model_path=str(tmp_path / "missing.joblib"),
            predictions_csv=str(tmp_path / "pred.csv"),
            classif_stats_json=str(tmp_path / "stats.json"),
        )


def test_predict_feature_mismatch(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    valid_training_df.to_csv(train_csv, index=False)
    model_path = tmp_path / "model.joblib"
    train(
        classifier="random_forest",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(tmp_path / "metrics.json"),
    )
    mismatch_df = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_x": [3, 4],
            "feature_c": [5, 6],
            "target": [0, 1],
        }
    )
    test_csv = tmp_path / "test.csv"
    mismatch_df.to_csv(test_csv, index=False)
    with pytest.raises(ValueError):
        predict(
            reduced_Test_csv=str(test_csv),
            target_column="target",
            model_path=str(model_path),
            predictions_csv=str(tmp_path / "pred.csv"),
            classif_stats_json=str(tmp_path / "stats.json"),
        )


def test_predict_missing_target(
    valid_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    valid_training_df.to_csv(train_csv, index=False)
    model_path = tmp_path / "model.joblib"
    train(
        classifier="random_forest",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(tmp_path / "metrics.json"),
    )
    test_df = valid_training_df.drop(columns=["target"])
    test_csv = tmp_path / "test.csv"
    test_df.to_csv(test_csv, index=False)
    with pytest.raises(ValueError):
        predict(
            reduced_Test_csv=str(test_csv),
            target_column="target",
            model_path=str(model_path),
            predictions_csv=str(tmp_path / "pred.csv"),
            classif_stats_json=str(tmp_path / "stats.json"),
        )


def test_build_parser_train_arguments() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train",
            "--classifier",
            "random_forest",
            "--in-reduced",
            "train.csv",
            "--target",
            "target",
            "--out-model",
            "model.joblib",
            "--out-metrics",
            "metrics.json",
        ]
    )
    assert args.command == "train"
    assert args.classifier == "random_forest"


def test_build_parser_predict_arguments() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--input-testset",
            "test.csv",
            "--target",
            "target",
            "--model",
            "model.joblib",
            "--out-predictions",
            "predictions.csv",
            "--out-stats",
            "stats.json",
        ]
    )
    assert args.command == "predict"
    assert args.model == "model.joblib"


def test_main_without_command_returns_1(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["model.py"],
    )
    assert main() == 1


def test_main_train_success_returns_0(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model.py",
            "train",
            "--classifier",
            "random_forest",
            "--in-reduced",
            "x.csv",
            "--target",
            "target",
            "--out-model",
            "m.joblib",
            "--out-metrics",
            "m.json",
        ],
    )
    monkeypatch.setattr(
        "qubo_project.model.train",
        lambda **kwargs: None,
    )
    assert main() == 0


def test_main_predict_success_returns_0(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model.py",
            "predict",
            "--input-testset",
            "x.csv",
            "--target",
            "target",
            "--model",
            "m.joblib",
            "--out-predictions",
            "p.csv",
            "--out-stats",
            "s.json",
        ],
    )
    monkeypatch.setattr(
        "qubo_project.model.predict",
        lambda **kwargs: None,
    )
    assert main() == 0


def test_main_validation_error_returns_1(
    monkeypatch,
) -> None:
    def _raise_value_error(**kwargs):
        raise ValueError("validation")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model.py",
            "train",
            "--classifier",
            "random_forest",
            "--in-reduced",
            "x.csv",
            "--target",
            "target",
            "--out-model",
            "m.joblib",
            "--out-metrics",
            "m.json",
        ],
    )
    monkeypatch.setattr(
        "qubo_project.model.train",
        _raise_value_error,
    )
    assert main() == 1


def test_main_runtime_error_returns_2(
    monkeypatch,
) -> None:
    def _raise_runtime_error(**kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model.py",
            "train",
            "--classifier",
            "random_forest",
            "--in-reduced",
            "x.csv",
            "--target",
            "target",
            "--out-model",
            "m.joblib",
            "--out-metrics",
            "m.json",
        ],
    )
    monkeypatch.setattr(
        "qubo_project.model.train",
        _raise_runtime_error,
    )
    assert main() == 2