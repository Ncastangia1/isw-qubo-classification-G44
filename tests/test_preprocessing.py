import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.qubo_project.preprocessing import fit_normalize

SAMPLE_DATASET = Path("data/sample_test_dataset.csv")
TARGET_COLUMN = "target"

@pytest.fixture
def output_paths(tmp_path):
    return {
        "csv": tmp_path / "normalized.csv",
        "json": tmp_path / "preprocessing_result.json",
    }

@pytest.fixture
def sample_dataset_exists():
    if not SAMPLE_DATASET.exists():
        pytest.skip(f"Sample dataset not found: {SAMPLE_DATASET}")
    return SAMPLE_DATASET

def _run_preprocessing(
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    target: str = TARGET_COLUMN,
    min_perc_valid: float = 0.05,
):
    fit_normalize(
        input_csv=str(input_csv),
        target_column=target,
        normalized_csv=str(output_csv),
        outInitalRes_json=str(output_json),
        minPercValid=min_perc_valid,
    )

def test_fit_normalize_creates_output_files(sample_dataset_exists, output_paths):
    _run_preprocessing(
        sample_dataset_exists,
        output_paths["csv"],
        output_paths["json"],
    )
    assert output_paths["csv"].exists()
    assert output_paths["json"].exists()
    assert output_paths["csv"].stat().st_size > 0
    assert output_paths["json"].stat().st_size > 0

def test_target_column_is_preserved(sample_dataset_exists, output_paths):
    input_df = pd.read_csv(sample_dataset_exists)
    _run_preprocessing(
        sample_dataset_exists,
        output_paths["csv"],
        output_paths["json"],
    )
    output_df = pd.read_csv(output_paths["csv"])
    assert TARGET_COLUMN in output_df.columns
    pd.testing.assert_series_equal(
        input_df[TARGET_COLUMN].reset_index(drop=True),
        output_df[TARGET_COLUMN].reset_index(drop=True),
        check_names=False,
    )

def test_output_contains_only_numeric_features(sample_dataset_exists, output_paths):
    _run_preprocessing(
        sample_dataset_exists,
        output_paths["csv"],
        output_paths["json"],
    )
    output_df = pd.read_csv(output_paths["csv"])
    feature_columns = [c for c in output_df.columns if c != TARGET_COLUMN]
    for column in feature_columns:
        assert pd.api.types.is_numeric_dtype(output_df[column])

def test_json_schema_matches_specification(sample_dataset_exists, output_paths):
    _run_preprocessing(
        sample_dataset_exists,
        output_paths["csv"],
        output_paths["json"],
    )
    with open(output_paths["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    expected_keys = {
        "n_input_features",
        "n_kept_features",
        "dataset_size",
        "dataset_input_time",
        "dataset_processing_time",
        "dropped_feature_names",
    }
    assert expected_keys.issubset(data.keys())
    assert expected_keys == set(data.keys())

def test_dataset_size_matches_original_dataset(sample_dataset_exists, output_paths):
    input_df = pd.read_csv(sample_dataset_exists)
    _run_preprocessing(
        sample_dataset_exists,
        output_paths["csv"],
        output_paths["json"],
    )
    with open(output_paths["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_size"] == len(input_df)

def test_normalization_produces_numeric_values(sample_dataset_exists, output_paths):
    _run_preprocessing(
        sample_dataset_exists,
        output_paths["csv"],
        output_paths["json"],
    )
    output_df = pd.read_csv(output_paths["csv"])
    feature_df = output_df.drop(columns=[TARGET_COLUMN])
    values = feature_df.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert not np.isnan(values).any()

def test_missing_input_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        fit_normalize(
            input_csv=str(tmp_path / "missing.csv"),
            target_column=TARGET_COLUMN,
            normalized_csv=str(tmp_path / "out.csv"),
            outInitalRes_json=str(tmp_path / "out.json"),
        )

def test_missing_target_column_raises_error(tmp_path):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [4, 5, 6],
        "target": [0, 1, 0],
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        fit_normalize(
            input_csv=str(csv_path),
            target_column="does_not_exist",
            normalized_csv=str(tmp_path / "out.csv"),
            outInitalRes_json=str(tmp_path / "out.json"),
        )

@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.5])
def test_invalid_minPercValid_rejected(tmp_path, invalid_threshold):
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0],
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        fit_normalize(
            input_csv=str(csv_path),
            target_column="target",
            normalized_csv=str(tmp_path / "out.csv"),
            outInitalRes_json=str(tmp_path / "out.json"),
            minPercValid=invalid_threshold,
        )

def test_constant_column_is_handled_correctly(tmp_path):
    df = pd.DataFrame({
        "constant_feature": [5, 5, 5, 5, 5],
        "variable_feature": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
    })
    csv_path = tmp_path / "dataset.csv"
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "result.json"
    df.to_csv(csv_path, index=False)
    _run_preprocessing(csv_path, out_csv, out_json, target="target")
    output_df = pd.read_csv(out_csv)
    assert "constant_feature" in output_df.columns
    assert set(output_df["constant_feature"].unique()) == {0}

def test_duplicate_columns_raise_error(tmp_path):
    csv_content = "dup,dup,target\n1,2,0\n3,4,1\n"
    csv_path = tmp_path / "duplicate_columns.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    with pytest.raises(ValueError):
        fit_normalize(
            input_csv=str(csv_path),
            target_column="target",
            normalized_csv=str(tmp_path / "out.csv"),
            outInitalRes_json=str(tmp_path / "out.json"),
        )

def test_non_numeric_feature_rejected(tmp_path):
    df = pd.DataFrame({
        "feature_a": [1, 2],
        "feature_b": ["hello", "world"],
        "target": [0, 1],
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        fit_normalize(
            input_csv=str(csv_path),
            target_column="target",
            normalized_csv=str(tmp_path / "out.csv"),
            outInitalRes_json=str(tmp_path / "out.json"),
        )

@pytest.mark.parametrize("inf_value", [np.inf, -np.inf])
def test_infinite_values_rejected(tmp_path, inf_value):
    df = pd.DataFrame({
        "feature_a": [1.0, inf_value, 3.0],
        "target": [0, 1, 0],
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        fit_normalize(
            input_csv=str(csv_path),
            target_column="target",
            normalized_csv=str(tmp_path / "out.csv"),
            outInitalRes_json=str(tmp_path / "out.json"),
        )

def test_reproducibility_same_input_same_output(sample_dataset_exists, tmp_path):
    out_csv_1 = tmp_path / "run1.csv"
    out_json_1 = tmp_path / "run1.json"
    out_csv_2 = tmp_path / "run2.csv"
    out_json_2 = tmp_path / "run2.json"
    _run_preprocessing(sample_dataset_exists, out_csv_1, out_json_1)
    _run_preprocessing(sample_dataset_exists, out_csv_2, out_json_2)
    df1 = pd.read_csv(out_csv_1)
    df2 = pd.read_csv(out_csv_2)
    pd.testing.assert_frame_equal(df1, df2)
    with open(out_json_1, "r", encoding="utf-8") as f:
        j1 = json.load(f)
    with open(out_json_2, "r", encoding="utf-8") as f:
        j2 = json.load(f)
    j1.pop("dataset_input_time")
    j1.pop("dataset_processing_time")
    j2.pop("dataset_input_time")
    j2.pop("dataset_processing_time")
    assert j1 == j2

def test_column_order_is_preserved(sample_dataset_exists, output_paths):
    input_df = pd.read_csv(sample_dataset_exists)
    _run_preprocessing(sample_dataset_exists, output_paths["csv"], output_paths["json"])
    output_df = pd.read_csv(output_paths["csv"])
    with open(output_paths["json"], "r", encoding="utf-8") as f:
        report = json.load(f)
    dropped = set(report["dropped_feature_names"])
    original_features = [c for c in input_df.columns if c != TARGET_COLUMN and c not in dropped]
    expected_columns = original_features + [TARGET_COLUMN]
    assert list(output_df.columns) == expected_columns
    assert output_df.columns[-1] == TARGET_COLUMN

def test_threshold_boundary_behavior(tmp_path):
    boundary_feature = [1] + [0] * 19
    df = pd.DataFrame({
        "boundary_feature": boundary_feature,
        "target": [0, 1] * 10,
    })
    csv_path = tmp_path / "dataset.csv"
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "result.json"
    df.to_csv(csv_path, index=False)
    _run_preprocessing(csv_path, out_csv, out_json, target="target", min_perc_valid=0.05)
    output_df = pd.read_csv(out_csv)
    assert "boundary_feature" in output_df.columns
    with open(out_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert "boundary_feature" not in report["dropped_feature_names"]