from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import neal
except ImportError as exc:
    raise ImportError(
        "dwave-neal is required. Install with: pip install dwave-neal"
    ) from exc


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )


def _validate_inputs(
    normalized_csv: str,
    target_column: str,
    perc_test: float,
    perc_selected: float,
    allowance: int,
    alpha_computations: int,
) -> None:
    path = Path(normalized_csv)

    if not path.exists():
        raise FileNotFoundError(f"Normalized dataset not found: {normalized_csv}")

    if not 0 < perc_selected <= 1:
        raise ValueError("percSelected must be in (0, 1].")

    if not 0 < perc_test < 1:
        raise ValueError("percTest must be in (0, 1).")

    if allowance < 0:
        raise ValueError("allowance must be >= 0.")

    if alpha_computations <= 0:
        raise ValueError("alpha_computations must be > 0.")

    if not target_column:
        raise ValueError("target_column cannot be empty.")


def _load_dataset(
    normalized_csv: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(normalized_csv)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    features = df.drop(columns=[target_column])
    target = df[target_column]

    if features.shape[1] == 0:
        raise ValueError("No feature columns available.")

    LOGGER.info(
        "Dataset loaded: %s rows, %s features",
        len(df),
        features.shape[1],
    )

    return df, features, target


def _safe_abs_spearman(x: pd.Series, y: pd.Series) -> float:
    corr, _ = spearmanr(x, y)

    if np.isnan(corr):
        return 0.0

    return float(abs(corr))


def _compute_feature_target_correlations(
    features: pd.DataFrame,
    target: pd.Series,
) -> np.ndarray:
    correlations = np.array(
        [
            _safe_abs_spearman(features[col], target)
            for col in features.columns
        ],
        dtype=float,
    )

    return correlations


def _compute_feature_feature_correlations(
    features: pd.DataFrame,
) -> np.ndarray:
    corr_matrix = features.corr(method="spearman").abs().to_numpy(dtype=float)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix


def _build_qubo_matrix(
    alpha: float,
    feature_target_corr: np.ndarray,
    feature_feature_corr: np.ndarray,
) -> np.ndarray:
    n_features = len(feature_target_corr)
    q = np.zeros((n_features, n_features), dtype=float)

    np.fill_diagonal(q, -alpha * np.abs(feature_target_corr))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = (1.0 - alpha) * abs(feature_feature_corr[i, j])
            q[i, j] = value
            q[j, i] = value

    return q


def _qubo_to_dict(q: np.ndarray) -> dict[tuple[int, int], float]:
    qubo: dict[tuple[int, int], float] = {}

    rows, cols = q.shape

    for i in range(rows):
        for j in range(i, cols):
            value = float(q[i, j])
            if i != j:
                value *= 2.0
            if value != 0.0:
                qubo[(i, j)] = value

    return qubo


def _compute_cost(q: np.ndarray, vector: np.ndarray) -> float:
    return float(vector.T @ q @ vector)


def _solve_qubo(
    q: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    sampler = neal.SimulatedAnnealingSampler()

    sample_set = sampler.sample_qubo(
        _qubo_to_dict(q),
        num_reads=50,
        seed=seed,
    )

    best = sample_set.first.sample
    n_features = q.shape[0]

    # CORRETTO: Garantisce che il vettore copra sempre tutte le feature (default a 0 se omesse dal sampler)
    vector = np.array(
        [int(best.get(k, 0)) for k in range(n_features)],
        dtype=int,
    )

    cost = _compute_cost(q, vector)

    return vector, cost


def _alpha_search(
    feature_target_corr: np.ndarray,
    feature_feature_corr: np.ndarray,
    target_k: int,
    allowance: int,
    seed: int,
    alpha_computations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []

    best_result: dict[str, Any] | None = None
    best_distance = float("inf")

    low = 0.0
    high = 1.0
    endpoint_alphas = iter(
        [0.0, 1.0] if alpha_computations > 1 else [0.0]
    )

    while len(history) < alpha_computations:
        try:
            alpha = next(endpoint_alphas)
            is_endpoint = True
        except StopIteration:
            alpha = (low + high) / 2.0
            is_endpoint = False

        q_start = time.perf_counter()

        q = _build_qubo_matrix(
            alpha,
            feature_target_corr,
            feature_feature_corr,
        )

        q_matrix_creation_time = time.perf_counter() - q_start

        opt_start = time.perf_counter()
        if alpha == 0.0:
            vector = np.zeros(len(feature_target_corr), dtype=int)
            cost = _compute_cost(q, vector)
        elif alpha == 1.0:
            vector = np.ones(len(feature_target_corr), dtype=int)
            cost = _compute_cost(q, vector)
        else:
            vector, cost = _solve_qubo(q, seed)
        opt_time = time.perf_counter() - opt_start

        selected = int(vector.sum())

        history.append(
            {
                "alpha": alpha,
                "optimization_time": opt_time,
                "q_matrix_creation_time": q_matrix_creation_time,
                "n_selected_features": selected,
                "cost_value": cost,
            }
        )

        distance = abs(selected - target_k)

        if distance < best_distance:
            best_distance = distance
            best_result = {
                "alpha": alpha,
                "vector": vector,
                "selected": selected,
                "cost": cost,
                "iterations": len(history),
            }

        if target_k - allowance <= selected <= target_k + allowance:
            return best_result, history

        if not is_endpoint:
            if selected > target_k:
                high = alpha
            else:
                low = alpha

    LOGGER.warning("Alpha search did not converge within the allowed tolerance.")

    assert best_result is not None
    raise RuntimeError(
        "Feature selection did not find a solution within "
        f"target_k={target_k} +/- {allowance}; "
        f"best_count={best_result['selected']} after {len(history)} "
        "alpha computations."
    )


def _create_reduced_datasets(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    selected_features: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = selected_features + [target_column]
    return (
        training_dataset[columns].copy(),
        test_dataset[columns].copy(),
    )


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    percSelected: float = 0.20,
    allowance: int = 1,
    seed: int = 42,
    alpha_computations: int = 100,
):
    _configure_logging()

    _validate_inputs(
        normalized_csv=normalized_csv,
        target_column=target_column,
        perc_test=percTest,
        perc_selected=percSelected,
        allowance=allowance,
        alpha_computations=alpha_computations,
    )

    dataset, _, _ = _load_dataset(
        normalized_csv,
        target_column,
    )

    split_index = round(len(dataset) * (1.0 - percTest))
    training_dataset = dataset.iloc[:split_index].copy()
    test_dataset = dataset.iloc[split_index:].copy()
    features = training_dataset.drop(columns=[target_column])
    target = training_dataset[target_column]

    feature_target_corr = _compute_feature_target_correlations(
        features,
        target,
    )

    feature_feature_corr = _compute_feature_feature_correlations(
        features,
    )

    LOGGER.info("Spearman correlations computed.")

    feature_names = list(features.columns)

    n_features = len(feature_names)

    target_k = round(percSelected * n_features)

    result, history = _alpha_search(
        feature_target_corr=feature_target_corr,
        feature_feature_corr=feature_feature_corr,
        target_k=target_k,
        allowance=allowance,
        seed=seed,
        alpha_computations=alpha_computations,
    )

    selected_vector = result["vector"].tolist()

    selected_feature_names = [
        feature_names[i]
        for i, value in enumerate(selected_vector)
        if value == 1
    ]

    train_df, test_df = _create_reduced_datasets(
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        selected_features=selected_feature_names,
        target_column=target_column,
    )

    for output_path in (
        reducedTrain_csv,
        reducedTest_csv,
        output_ottim_csv,
        output_json,
    ):
        _ensure_parent(output_path)

    train_df.to_csv(reducedTrain_csv, index=False)
    test_df.to_csv(reducedTest_csv, index=False)

    history_df = pd.DataFrame(history).sort_values("alpha").reset_index(drop=True)
    history_df[
        ["alpha", "optimization_time", "n_selected_features", "cost_value"]
    ].to_csv(output_ottim_csv, index=False)

    optimization_times = history_df["optimization_time"].to_numpy(dtype=float)

    stats = {
        "n_features": n_features,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": int(result["selected"]),
        "alpha": float(result["alpha"]),
        "selected_vector": selected_vector,
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(history),
        "selection_converged": (
            abs(int(result["selected"]) - target_k) <= allowance
        ),
        "percTest": percTest,
        "training_dataset_size": int(len(train_df)),
        "test_dataset_size": int(len(test_df)),
        "q_matrix_creation_time": float(
            history_df["q_matrix_creation_time"].sum()
        ),
        "mean_optimization_time": float(np.mean(optimization_times)),
        "std_dev_optimization_time": float(
            np.std(optimization_times)
        ),
    }

    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    LOGGER.info("Optimization completed.")
    LOGGER.info("Reduced datasets written.")
    LOGGER.info("Output files created.")

    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QUBO feature selection",
    )

    parser.add_argument(
        "--in-normalized",
        required=True,
    )
    parser.add_argument(
        "--out-train",
        required=True,
    )
    parser.add_argument(
        "--out-test",
        required=True,
    )
    parser.add_argument(
        "--out-optimizations",
        required=True,
    )
    parser.add_argument(
        "--out-json",
        required=True,
    )
    parser.add_argument(
        "--target",
        required=True,
    )

    parser.add_argument(
        "--perc-selected",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--allowance",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--perc-test",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--alpha-computations",
        type=int,
        default=100,
    )

    return parser


def main() -> int:
    try:
        args = _build_parser().parse_args()

        select_features(
            normalized_csv=args.in_normalized,
            reducedTrain_csv=args.out_train,
            reducedTest_csv=args.out_test,
            output_ottim_csv=args.out_optimizations,
            output_json=args.out_json,
            target_column=args.target,
            percTest=args.perc_test,
            percSelected=args.perc_selected,
            allowance=args.allowance,
            seed=args.seed,
            alpha_computations=args.alpha_computations,
        )

        return 0

    except Exception as exc:
        LOGGER.error(str(exc))
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())