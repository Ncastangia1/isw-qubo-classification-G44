from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
from qubo_project.feature_selection import select_features
from qubo_project.model import predict, train
from qubo_project.preprocessing import fit_normalize


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    rows = 120
    return pd.DataFrame(
        {
            "feature_1": list(range(rows)),
            "feature_2": [x * 2 for x in range(rows)],
            "feature_3": [x % 7 for x in range(rows)],
            "feature_4": [x % 5 for x in range(rows)],
            "feature_5": [x % 3 for x in range(rows)],
            "target": [0 if x % 2 == 0 else 1 for x in range(rows)],
        }
    )


def _run_pipeline(
    base_dir: Path,
    dataset: pd.DataFrame,
    seed: int = 42,
    perc_selected: float = 1.0,
    allowance: int = 0,
) -> dict[str, Path]:
    
    base_dir.mkdir(parents=True, exist_ok=True)

    input_csv = base_dir / "input_dataset.csv"
    normalized_csv = base_dir / "normalized.csv"
    preprocessing_json = (
        base_dir / "preprocessing_result.json"
    )
    training_reduced_csv = (
        base_dir / "training_reduced.csv"
    )
    test_reduced_csv = (
        base_dir / "test_reduced.csv"
    )
    optimizations_csv = (
        base_dir / "optimizations.csv"
    )
    feature_selection_json = (
        base_dir / "feature_selection_result.json"
    )
    model_path = base_dir / "model.joblib"
    training_metrics_json = (
        base_dir / "training_metrics.json"
    )
    predictions_csv = (
        base_dir / "predictions.csv"
    )
    classification_stats_json = (
        base_dir / "classification_stats.json"
    )
    dataset.to_csv(input_csv, index=False)
    
    fit_normalize(
        input_csv=str(input_csv),
        target_column="target",
        normalized_csv=str(normalized_csv),
        outInitalRes_json=str(preprocessing_json),
    )
    
    
    select_features(
        normalized_csv=str(normalized_csv),
        reducedTrain_csv=str(training_reduced_csv),
        reducedTest_csv=str(test_reduced_csv),
        output_ottim_csv=str(optimizations_csv),
        output_json=str(feature_selection_json),
        target_column="target",
        percTest=0.30,
        percSelected=perc_selected,
        allowance=allowance,
        seed=seed,
    )
    train(
        classifier="random_forest",
        reducedTrain_csv=str(training_reduced_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(training_metrics_json),
        seed=seed,
    )
    predict(
        reduced_Test_csv=str(test_reduced_csv),
        target_column="target",
        model_path=str(model_path),
        predictions_csv=str(predictions_csv),
        classif_stats_json=str(
            classification_stats_json
        ),
    )
    return {
        "input_csv": input_csv,
        "normalized_csv": normalized_csv,
        "preprocessing_json": preprocessing_json,
        "training_reduced_csv": training_reduced_csv,
        "test_reduced_csv": test_reduced_csv,
        "optimizations_csv": optimizations_csv,
        "feature_selection_json": feature_selection_json,
        "model_path": model_path,
        "training_metrics_json": (
            training_metrics_json
        ),
        "predictions_csv": predictions_csv,
        "classification_stats_json": (
            classification_stats_json
        ),
    }


def test_complete_pipeline_end_to_end(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    outputs = _run_pipeline(
        tmp_path,
        sample_dataset,
        seed=42,
    )
    expected_files = [
        "normalized_csv",
        "preprocessing_json",
        "training_reduced_csv",
        "test_reduced_csv",
        "optimizations_csv",
        "feature_selection_json",
        "model_path",
        "training_metrics_json",
        "predictions_csv",
        "classification_stats_json",
    ]
    for key in expected_files:
        assert outputs[key].exists()


def test_sample_dataset_pipeline_end_to_end(tmp_path: Path) -> None:
    sample_path = Path("data/sample_test_dataset.csv")
    if not sample_path.exists():
        pytest.skip("data/sample_test_dataset.csv is not available")

    outputs = _run_pipeline(
        tmp_path,
        pd.read_csv(sample_path),
        perc_selected=0.20,
        allowance=1,
    )

    predictions = pd.read_csv(outputs["predictions_csv"])
    training = pd.read_csv(outputs["training_reduced_csv"])
    test_rows = pd.read_csv(outputs["test_reduced_csv"])
    assert len(training.columns) > 1
    assert len(predictions) == len(test_rows)
    assert list(predictions.columns) == [
        "row_n",
        "target",
        "prediction",
        "score",
    ]


def test_pipeline_data_compatibility(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    outputs = _run_pipeline(
        tmp_path,
        sample_dataset,
    )
    normalized_df = pd.read_csv(
        outputs["normalized_csv"]
    )
    training_df = pd.read_csv(
        outputs["training_reduced_csv"]
    )
    test_df = pd.read_csv(
        outputs["test_reduced_csv"]
    )
    assert "target" in normalized_df.columns
    assert "target" in training_df.columns
    assert "target" in test_df.columns
    train_features = [
        c
        for c in training_df.columns
        if c != "target"
    ]
    test_features = [
        c
        for c in test_df.columns
        if c != "target"
    ]
    assert train_features == test_features


def test_training_test_split_validation(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    outputs = _run_pipeline(
        tmp_path,
        sample_dataset,
    )
    training_df = pd.read_csv(
        outputs["training_reduced_csv"]
    )
    test_df = pd.read_csv(
        outputs["test_reduced_csv"]
    )
    assert (
        len(training_df) + len(test_df)
        == len(sample_dataset)
    )
    assert len(training_df) > 0
    assert len(test_df) > 0


def test_feature_selection_validation(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    outputs = _run_pipeline(
        tmp_path,
        sample_dataset,
    )
    training_df = pd.read_csv(
        outputs["training_reduced_csv"]
    )
    test_df = pd.read_csv(
        outputs["test_reduced_csv"]
    )
    feature_selection_result = json.loads(
        outputs["feature_selection_json"].read_text(
            encoding="utf-8"
        )
    )
    selected_vector = (
        feature_selection_result.get(
            "selected_vector"
        )
    )
    if selected_vector is not None:
        assert all(
            value in (0, 1)
            for value in selected_vector
        )
    train_features = {
        c
        for c in training_df.columns
        if c != "target"
    }
    test_features = {
        c
        for c in test_df.columns
        if c != "target"
    }
    assert train_features == test_features
    assert "target" in training_df.columns
    assert "target" in test_df.columns


def test_model_validation(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    outputs = _run_pipeline(
        tmp_path,
        sample_dataset,
    )
    assert outputs["model_path"].exists()
    predictions_df = pd.read_csv(
        outputs["predictions_csv"]
    )
    test_df = pd.read_csv(
        outputs["test_reduced_csv"]
    )
    assert len(predictions_df) == len(test_df)


def test_predictions_csv_validation(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    outputs = _run_pipeline(
        tmp_path,
        sample_dataset,
    )
    predictions_df = pd.read_csv(
        outputs["predictions_csv"]
    )
    assert list(predictions_df.columns) == [
        "row_n",
        "target",
        "prediction",
        "score",
    ]
    assert pd.api.types.is_numeric_dtype(
        predictions_df["score"]
    )
    assert predictions_df["score"].between(
        0,
        1,
    ).all()


def test_classification_json_validation(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    outputs = _run_pipeline(
        tmp_path,
        sample_dataset,
    )
    payload = json.loads(
        outputs[
            "classification_stats_json"
        ].read_text(encoding="utf-8")
    )
    assert "accuracy" in payload
    assert "roc_auc" in payload
    assert "class_0" in payload
    assert "class_1" in payload
    assert payload["confusion_matrix"]["labels"] == [0, 1]
    assert len(payload["confusion_matrix"]["matrix"]) == 2
    assert payload["n_samples"] == len(
        pd.read_csv(outputs["test_reduced_csv"])
    )


def test_pipeline_reproducibility(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
) -> None:
    run_1 = _run_pipeline(
        tmp_path / "run1",
        sample_dataset,
        seed=42,
    )
    run_2 = _run_pipeline(
        tmp_path / "run2",
        sample_dataset,
        seed=42,
    )
    pred_1 = pd.read_csv(
        run_1["predictions_csv"]
    )
    pred_2 = pd.read_csv(
        run_2["predictions_csv"]
    )
    fs_1 = json.loads(
        run_1["feature_selection_json"].read_text(
            encoding="utf-8"
        )
    )
    fs_2 = json.loads(
        run_2["feature_selection_json"].read_text(
            encoding="utf-8"
        )
    )
    if (
        "selected_vector" in fs_1
        and "selected_vector" in fs_2
    ):
        assert (
            fs_1["selected_vector"]
            == fs_2["selected_vector"]
        )
    pd.testing.assert_frame_equal(
        pred_1,
        pred_2,
    )


@pytest.mark.integration
def test_cli_pipeline_execution(
    sample_dataset: pd.DataFrame,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            filter(None, (source_path, os.environ.get("PYTHONPATH", "")))
        ),
    )
    input_csv = tmp_path / "input.csv"
    sample_dataset.to_csv(
        input_csv,
        index=False,
    )
    normalized_csv = tmp_path / "normalized.csv"
    preprocessing_json = (
        tmp_path / "preprocessing_result.json"
    )
    training_csv = (
        tmp_path / "training_reduced.csv"
    )
    test_csv = (
        tmp_path / "test_reduced.csv"
    )
    optimizations_csv = (
        tmp_path / "optimizations.csv"
    )
    feature_selection_json = (
        tmp_path / "feature_selection_result.json"
    )
    model_path = tmp_path / "model.joblib"
    training_metrics_json = (
        tmp_path / "training_metrics.json"
    )
    predictions_csv = (
        tmp_path / "predictions.csv"
    )
    stats_json = (
        tmp_path / "classification_stats.json"
    )
    
    
    subprocess.run(
        [
            sys.executable,
            "-m",
            "qubo_project.preprocessing",
            "--input",
            str(input_csv),
            "--target",
            "target",
            "--out-data",
            str(normalized_csv),
            "--out-json",
            str(preprocessing_json),
        ],
        check=True,
    )
    
    
    subprocess.run(
        [
            sys.executable,
            "-m",
            "qubo_project.feature_selection",
            "--in-normalized",
            str(normalized_csv),
            "--out-train",
            str(training_csv),
            "--out-test",
            str(test_csv),
            "--out-optimizations",
            str(optimizations_csv),
            "--out-json",
            str(feature_selection_json),
            "--target",
            "target",
            "--seed",
            "42",
            "--perc-selected",
            "1.0",
            "--allowance",
            "0",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "qubo_project.model",
            "train",
            "--classifier",
            "random_forest",
            "--in-reduced",
            str(training_csv),
            "--target",
            "target",
            "--out-model",
            str(model_path),
            "--out-metrics",
            str(training_metrics_json),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "qubo_project.model",
            "predict",
            "--input-testset",
            str(test_csv),
            "--target",
            "target",
            "--model",
            str(model_path),
            "--out-predictions",
            str(predictions_csv),
            "--out-stats",
            str(stats_json),
        ],
        check=True,
    )
    assert normalized_csv.exists()
    assert training_csv.exists()
    assert test_csv.exists()
    assert model_path.exists()
    assert predictions_csv.exists()
    assert stats_json.exists()