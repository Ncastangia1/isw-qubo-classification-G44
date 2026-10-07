"""
preprocessing.py

Dataset preprocessing module for the ISW QUBO Classification project.

Responsibilities:
- Load and validate a CSV dataset.
- Separate the target column.
- Remove features with insufficient valid values.
- Normalize remaining features using z-score standardization.
- Preserve the target column unchanged.
- Write the normalized dataset and preprocessing statistics.

This module can be used both:
- as an importable Python module
- as a standalone CLI application
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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
        If the CSV cannot be read.
    """
    path = Path(input_csv)

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: '{input_csv}'")

    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: '{input_csv}'")

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{input_csv}': {exc}") from exc


def _validate_dataset(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Validate dataset structure and content.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.index) == 0:
        raise ValueError("Dataset contains no rows.")

    if len(df.index) == 1:
        raise ValueError(
            "Dataset contains a single row; normalization is not meaningful."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist in the dataset."
        )

    if df.columns.duplicated().any():
        duplicates = list(df.columns[df.columns.duplicated()])
        raise ValueError(
            f"Duplicate column names detected: {duplicates}"
        )

    feature_columns = [c for c in df.columns if c != target_column]

    if not feature_columns:
        raise ValueError("Dataset contains no feature columns.")

    # Check numeric types
    non_numeric: list[str] = []

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric.append(column)

    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns detected: {non_numeric}"
        )

    # Check infinities before expensive operations
    numeric_df = df[feature_columns]

    if np.isinf(numeric_df.to_numpy()).any():
        raise ValueError(
            "Dataset contains +inf or -inf values in feature columns."
        )


def _compute_valid_ratios(features_df: pd.DataFrame) -> pd.Series:
    """
    Compute the valid-value ratio for each feature.

    Valid value:
    NOT NaN AND NOT 0
    """
    valid_mask = features_df.notna() & features_df.ne(0)

    return valid_mask.sum(axis=0) / len(features_df.index)


def _select_valid_features(
    features_df: pd.DataFrame,
    minPercValid: float,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select features whose valid ratio satisfies:

    valid_ratio >= minPercValid
    """
    ratios = _compute_valid_ratios(features_df)

    kept_columns: list[str] = []
    dropped_columns: list[str] = []

    for column, ratio in ratios.items():
        if np.isclose(ratio, minPercValid):
            LOGGER.warning(
                "Feature '%s' is at threshold boundary (ratio=%s).",
                column,
                ratio,
            )

        if ratio >= minPercValid:
            kept_columns.append(column)
        else:
            dropped_columns.append(column)

    LOGGER.info("Features retained: %s", len(kept_columns))
    LOGGER.info("Features removed: %s", len(dropped_columns))

    if not kept_columns:
        raise ValueError(
            "All features were removed by minPercValid filtering."
        )

    return features_df.loc[:, kept_columns], dropped_columns


def _normalize_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using z-score standardization.

    Constant columns are transformed into all-zero columns.
    """
    if features_df.isna().any().any():
        raise ValueError(
            "NaN values remain in retained features after filtering."
        )

    result = pd.DataFrame(
        index=features_df.index,
        columns=features_df.columns,
        dtype=np.float64,
    )

    constant_columns: list[str] = []
    normal_columns: list[str] = []

    for column in features_df.columns:
        if features_df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)
        else:
            normal_columns.append(column)

    if normal_columns:
        scaler = StandardScaler()

        result.loc[:, normal_columns] = scaler.fit_transform(
            features_df[normal_columns]
        )

    for column in constant_columns:
        LOGGER.warning(
            "Feature '%s' has zero standard deviation. "
            "Replacing normalized values with 0.",
            column,
        )
        result.loc[:, column] = 0.0

    if result.isna().any().any():
        raise ValueError(
            "Normalization produced NaN values."
        )

    if np.isinf(result.to_numpy()).any():
        raise ValueError(
            "Normalization produced infinite values."
        )

    LOGGER.info("Normalization completed.")

    return result


def _build_statistics(
    n_input_features: int,
    n_kept_features: int,
    dataset_size: int,
    dataset_input_time: float,
    dataset_processing_time: float,
    dropped_feature_names: list[str],
) -> dict[str, Any]:
    """
    Build preprocessing statistics dictionary.
    """
    return {
        "n_input_features": int(n_input_features),
        "n_kept_features": int(n_kept_features),
        "dataset_size": int(dataset_size),
        "dataset_input_time": round(float(dataset_input_time), 6),
        "dataset_processing_time": round(float(dataset_processing_time), 6),
        "dropped_feature_names": list(dropped_feature_names),
    }


def _write_outputs(
    output_df: pd.DataFrame,
    statistics: dict[str, Any],
    normalized_csv: str,
    outInitalRes_json: str,
) -> None:
    """
    Write CSV and JSON outputs.
    """
    csv_path = Path(normalized_csv)
    json_path = Path(outInitalRes_json)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_df.to_csv(csv_path, index=False)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to write output files: {exc}"
        ) from exc

    LOGGER.info("Outputs written successfully.")


def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str,
    minPercValid: float = 0.05,
):
    """
    Load, validate, filter and normalize a dataset.

    Parameters
    ----------
    input_csv : str
        Path to source CSV.
    target_column : str
        Name of target column.
    normalized_csv : str
        Destination CSV path.
    outInitalRes_json : str
        Destination statistics JSON path.
    minPercValid : float, default=0.05
        Minimum ratio of valid values required for a feature
        to survive filtering.

    Returns
    -------
    dict
        Statistics dictionary.

    Raises
    ------
    ValueError
        On invalid inputs or invalid dataset.
    RuntimeError
        On output generation errors.
    """
    if not (0.0 <= minPercValid <= 1.0):
        raise ValueError(
            "minPercValid must satisfy 0 <= minPercValid <= 1."
        )

    start_input = time.perf_counter()

    df = _load_dataset(input_csv)

    dataset_input_time = time.perf_counter() - start_input

    LOGGER.info(
        "Dataset loaded: rows=%s columns=%s",
        len(df.index),
        len(df.columns),
    )

    processing_start = time.perf_counter()

    _validate_dataset(df, target_column)

    target_series = df[target_column]

    feature_columns = [c for c in df.columns if c != target_column]

    n_input_features = len(feature_columns)

    features_df = df[feature_columns]

    filtered_features_df, dropped_features = _select_valid_features(
        features_df,
        minPercValid=minPercValid,
    )

    normalized_features_df = _normalize_features(filtered_features_df)

    output_df = normalized_features_df.copy()
    output_df[target_column] = target_series

    statistics = _build_statistics(
        n_input_features=n_input_features,
        n_kept_features=len(normalized_features_df.columns),
        dataset_size=len(df.index),
        dataset_input_time=dataset_input_time,
        dataset_processing_time=(
            time.perf_counter() - processing_start
        ),
        dropped_feature_names=dropped_features,
    )

    _write_outputs(
        output_df=output_df,
        statistics=statistics,
        normalized_csv=normalized_csv,
        outInitalRes_json=outInitalRes_json,
    )

    return statistics


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Dataset preprocessing and normalization."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV dataset.",
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name.",
    )

    parser.add_argument(
        "--out-data",
        required=True,
        help="Output normalized CSV.",
    )

    parser.add_argument(
        "--out-json",
        required=True,
        help="Output preprocessing JSON.",
    )

    parser.add_argument(
        "--min-perc-valid",
        type=float,
        default=0.05,
        help="Minimum valid ratio threshold.",
    )

    return parser


def main() -> int:
    """
    CLI entry point.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        fit_normalize(
            input_csv=args.input,
            target_column=args.target,
            normalized_csv=args.out_data,
            outInitalRes_json=args.out_json,
            minPercValid=args.min_perc_valid,
        )

        return 0

    except Exception as exc:
        LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())