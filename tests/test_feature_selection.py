from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qubo_project import feature_selection as feature_selection_module
from qubo_project.feature_selection import select_features

REQUIRED_JSON_KEYS = {
    "n_features",
    "target_ratio",
    "target_k",
    "allowance",
    "n_selected",
    "alpha",
    "selected_vector",
    "selected_feature_names",
    "algorithm",
    "seed",
    "alpha_computations",
    "percTest",
    "training_dataset_size",
    "test_dataset_size",
    "q_matrix_creation_time",
    "mean_optimization_time",
    "std_dev_optimization_time",
    "selection_converged",
}


@pytest.fixture
def dataset_a(tmp_path):
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "feature_1": rng.normal(size=20),
            "feature_2": rng.normal(size=20),
            "feature_3": rng.normal(size=20),
            "feature_4": rng.normal(size=20),
            "feature_5": rng.normal(size=20),
            "feature_6": rng.normal(size=20),
            "target": [0, 1] * 10,
        }
    )

    csv_path = tmp_path / "dataset_a.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def dataset_b(tmp_path):
    x = np.linspace(0, 1, 20)

    df = pd.DataFrame(
        {
            "feature_1": x,
            "feature_2": x * 1.001,
            "feature_3": x * 0.999,
            "feature_4": x + 0.01,
            "feature_5": np.sin(x),
            "feature_6": np.cos(x),
            "target": [0, 1] * 10,
        }
    )

    csv_path = tmp_path / "dataset_b.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def dataset_c(tmp_path):
    df = pd.DataFrame(
        {
            "feature_1": np.linspace(0, 1, 20),
            "target": [0, 1] * 10,
        }
    )

    csv_path = tmp_path / "dataset_c.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def dataset_d(tmp_path):
    df = pd.DataFrame(
        {
            "feature_1": np.arange(20),
            "feature_2": np.arange(20) * 2,
        }
    )

    csv_path = tmp_path / "dataset_d.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def dataset_e(tmp_path):
    rng = np.random.default_rng(123)

    df = pd.DataFrame(
        {
            "feature_1": rng.normal(size=20),
            "feature_2": rng.normal(size=20),
            "feature_3": rng.normal(size=20),
            "target": [0] * 20,
        }
    )

    csv_path = tmp_path / "dataset_e.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def run_selection(tmp_path, dataset_a):
    dataset_path, df = dataset_a

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    optim_csv = tmp_path / "optim.csv"
    stats_json = tmp_path / "stats.json"

    result = select_features(
        normalized_csv=str(dataset_path),
        reducedTrain_csv=str(train_csv),
        reducedTest_csv=str(test_csv),
        output_ottim_csv=str(optim_csv),
        output_json=str(stats_json),
        target_column="target",
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=20,
    )

    with open(stats_json, encoding="utf-8") as fp:
        stats = json.load(fp)

    return {
        "dataset": df,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "optim_csv": optim_csv,
        "stats_json": stats_json,
        "stats": stats,
        "result": result,
    }


def test_select_features_creates_all_output_files(run_selection):
    for path in (
        run_selection["train_csv"],
        run_selection["test_csv"],
        run_selection["optim_csv"],
        run_selection["stats_json"],
    ):
        assert path.exists()
        assert path.stat().st_size > 0


def test_json_schema_matches_specification(run_selection):
    stats = run_selection["stats"]
    assert REQUIRED_JSON_KEYS.issubset(stats.keys())


def test_selected_vector_is_binary(run_selection):
    vector = run_selection["stats"]["selected_vector"]
    assert set(vector).issubset({0, 1})


def test_selected_vector_length_matches_feature_count(run_selection):
    stats = run_selection["stats"]
    assert len(stats["selected_vector"]) == stats["n_features"]


def test_selected_feature_names_match_selected_vector(run_selection):
    stats = run_selection["stats"]
    n_selected = stats["n_selected"]
    ones_count = sum(stats["selected_vector"])

    assert len(stats["selected_feature_names"]) == n_selected
    assert ones_count == n_selected


def test_qubo_matrix_has_expected_signs_at_alpha_endpoints():
    feature_target_corr = np.array([0.2, 0.4, 0.6])
    feature_feature_corr = np.array(
        [
            [1.0, 0.3, 0.5],
            [0.3, 1.0, 0.7],
            [0.5, 0.7, 1.0],
        ]
    )
    q_at_zero = feature_selection_module._build_qubo_matrix(
        0.0,
        feature_target_corr,
        feature_feature_corr,
    )
    q_at_one = feature_selection_module._build_qubo_matrix(
        1.0,
        feature_target_corr,
        feature_feature_corr,
    )

    np.testing.assert_array_equal(np.diag(q_at_zero), np.zeros(3))
    np.testing.assert_array_equal(np.diag(q_at_one), -feature_target_corr)
    expected_at_zero = feature_feature_corr - np.diag(
        np.diag(feature_feature_corr)
    )
    np.testing.assert_array_equal(q_at_zero, expected_at_zero)
    np.testing.assert_array_equal(
        q_at_one - np.diag(np.diag(q_at_one)),
        np.zeros((3, 3)),
    )


def test_qubo_dictionary_and_cost_use_each_pair_once():
    feature_target_corr = np.array([0.4, -0.8])
    feature_feature_corr = np.array([[1.0, -0.6], [-0.6, 1.0]])
    q = feature_selection_module._build_qubo_matrix(
        0.5,
        feature_target_corr,
        feature_feature_corr,
    )
    qubo = feature_selection_module._qubo_to_dict(q)

    assert qubo == {
        (0, 0): -0.2,
        (0, 1): 0.6,
        (1, 1): -0.4,
    }
    for bits in product((0, 1), repeat=2):
        vector = np.asarray(bits)
        expected = (
            -0.5 * np.dot(np.abs(feature_target_corr), vector)
            + 0.5
            * sum(
                abs(feature_feature_corr[row, column])
                * vector[row]
                * vector[column]
                for row in range(2)
                for column in range(2)
                if row != column
            )
        )
        sampler_cost = sum(
            value * vector[row] * vector[column]
            for (row, column), value in qubo.items()
        )
        assert feature_selection_module._compute_cost(q, vector) == pytest.approx(
            expected
        )
        assert sampler_cost == pytest.approx(expected)


def test_alpha_endpoints_have_deterministic_solutions(monkeypatch):
    def unexpected_sampler_call(q, seed):
        raise AssertionError("endpoint solutions must be deterministic")

    monkeypatch.setattr(
        feature_selection_module,
        "_solve_qubo",
        unexpected_sampler_call,
    )

    result, history = feature_selection_module._alpha_search(
        feature_target_corr=np.zeros(3),
        feature_feature_corr=np.zeros((3, 3)),
        target_k=3,
        allowance=0,
        seed=42,
        alpha_computations=2,
    )

    np.testing.assert_array_equal(result["vector"], np.ones(3, dtype=int))
    assert [entry["alpha"] for entry in history] == [0.0, 1.0]


def test_alpha_zero_endpoint_is_empty_even_when_tied(monkeypatch):
    def unexpected_sampler_call(q, seed):
        raise AssertionError("endpoint solutions must be deterministic")

    monkeypatch.setattr(
        feature_selection_module,
        "_solve_qubo",
        unexpected_sampler_call,
    )

    result, history = feature_selection_module._alpha_search(
        feature_target_corr=np.zeros(3),
        feature_feature_corr=np.zeros((3, 3)),
        target_k=3,
        allowance=3,
        seed=42,
        alpha_computations=1,
    )

    np.testing.assert_array_equal(result["vector"], np.zeros(3, dtype=int))
    assert history[0]["alpha"] == 0.0


def test_alpha_search_increases_alpha_when_selecting_too_few(monkeypatch):
    def solve_qubo(q, seed):
        alpha = -q[0, 0]
        selected_count = 2 if alpha < 0.75 else 3
        vector = np.zeros(q.shape[0], dtype=int)
        vector[:selected_count] = 1
        return vector, feature_selection_module._compute_cost(q, vector)

    monkeypatch.setattr(
        feature_selection_module,
        "_solve_qubo",
        solve_qubo,
    )

    result, history = feature_selection_module._alpha_search(
        feature_target_corr=np.ones(6),
        feature_feature_corr=np.zeros((6, 6)),
        target_k=3,
        allowance=0,
        seed=42,
        alpha_computations=4,
    )

    assert [entry["alpha"] for entry in history] == [0.0, 1.0, 0.5, 0.75]
    assert result["selected"] == 3


def test_alpha_search_reports_unreachable_tolerance(monkeypatch):
    def solve_qubo(q, seed):
        vector = np.ones(q.shape[0], dtype=int)
        return vector, float(vector.T @ q @ vector)

    monkeypatch.setattr(
        feature_selection_module,
        "_solve_qubo",
        solve_qubo,
    )

    with pytest.raises(RuntimeError, match="after 4 alpha computations"):
        feature_selection_module._alpha_search(
            feature_target_corr=np.ones(6),
            feature_feature_corr=np.zeros((6, 6)),
            target_k=3,
            allowance=0,
            seed=42,
            alpha_computations=4,
        )


def test_feature_correlations_use_training_rows_only(
    dataset_a,
    tmp_path,
    monkeypatch,
):
    dataset_path, original = dataset_a
    split_index = round(len(original) * 0.7)
    observed = []

    def capture_target_correlations(features, target):
        observed.append(
            {
                "features": features.copy(),
                "target": target.copy(),
            }
        )
        return np.zeros(features.shape[1])

    def capture_feature_correlations(features):
        observed[-1]["correlation_rows"] = len(features)
        return np.zeros((features.shape[1], features.shape[1]))

    def fixed_alpha_search(**kwargs):
        vector = np.array([1, 0, 0, 0, 0, 0])
        return (
            {
                "alpha": 0.5,
                "vector": vector,
                "selected": 1,
                "cost": 0.0,
                "iterations": 1,
            },
            [
                {
                    "alpha": 0.5,
                    "optimization_time": 0.01,
                    "q_matrix_creation_time": 0.001,
                    "n_selected_features": 1,
                    "cost_value": 0.0,
                }
            ],
        )

    monkeypatch.setattr(
        feature_selection_module,
        "_compute_feature_target_correlations",
        capture_target_correlations,
    )
    monkeypatch.setattr(
        feature_selection_module,
        "_compute_feature_feature_correlations",
        capture_feature_correlations,
    )
    monkeypatch.setattr(
        feature_selection_module,
        "_alpha_search",
        fixed_alpha_search,
    )

    altered = original.copy()
    altered.loc[split_index:, "target"] = 1 - altered.loc[
        split_index:, "target"
    ]
    altered_path = tmp_path / "test_targets_changed.csv"
    altered.to_csv(altered_path, index=False)

    selected_feature_names = []
    for index, input_path in enumerate((dataset_path, altered_path)):
        select_features(
            normalized_csv=str(input_path),
            reducedTrain_csv=str(tmp_path / f"train_{index}.csv"),
            reducedTest_csv=str(tmp_path / f"test_{index}.csv"),
            output_ottim_csv=str(tmp_path / f"optimizations_{index}.csv"),
            output_json=str(tmp_path / f"stats_{index}.json"),
            target_column="target",
            percTest=0.30,
        )
        stats = json.loads(
            (tmp_path / f"stats_{index}.json").read_text(encoding="utf-8")
        )
        selected_feature_names.append(stats["selected_feature_names"])

    assert selected_feature_names[0] == selected_feature_names[1]
    assert len(observed) == 2
    for captured in observed:
        assert len(captured["features"]) == split_index
        assert captured["correlation_rows"] == split_index
        pd.testing.assert_series_equal(
            captured["target"].reset_index(drop=True),
            original["target"].iloc[:split_index].reset_index(drop=True),
        )


def test_target_column_preserved_in_reduced_datasets(run_selection):
    for csv_file in (
        run_selection["train_csv"],
        run_selection["test_csv"],
    ):
        df = pd.read_csv(csv_file)
        assert "target" in df.columns
        assert df.columns[-1] == "target"


def test_train_test_split_respects_specification(run_selection):
    stats = run_selection["stats"]
    original_df = run_selection["dataset"]

    m = round(len(original_df) * (1 - stats["percTest"]))

    assert stats["training_dataset_size"] == m
    assert stats["test_dataset_size"] == len(original_df) - m

    train_df = pd.read_csv(run_selection["train_csv"])
    test_df = pd.read_csv(run_selection["test_csv"])

    selected = stats["selected_feature_names"] + ["target"]

    expected_train = original_df[selected].iloc[:m].reset_index(drop=True)
    expected_test = original_df[selected].iloc[m:].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        train_df.reset_index(drop=True),
        expected_train,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        test_df.reset_index(drop=True),
        expected_test,
        check_dtype=False,
    )


def test_output_feature_count_respects_target_ratio(run_selection):
    stats = run_selection["stats"]
    target_k = stats["target_k"]
    allowance = stats["allowance"]
    selected = stats["n_selected"]

    assert stats["selection_converged"] is (
        abs(selected - target_k) <= allowance
    )


def test_optimization_history_csv_created_correctly(run_selection):
    df = pd.read_csv(run_selection["optim_csv"])
    stats = run_selection["stats"]
    expected_columns = [
        "alpha",
        "optimization_time",
        "n_selected_features",
        "cost_value",
    ]

    assert list(df.columns) == expected_columns
    assert len(df) >= 1
    assert pd.api.types.is_numeric_dtype(df["alpha"])
    assert df["alpha"].is_monotonic_increasing
    assert stats["alpha_computations"] == len(df)
    assert stats["q_matrix_creation_time"] >= 0.0


def test_alpha_values_within_valid_range(run_selection):
    df = pd.read_csv(run_selection["optim_csv"])
    assert ((df["alpha"] >= 0) & (df["alpha"] <= 1)).all()


def test_invalid_dataset_path_raises_error(tmp_path):
    with pytest.raises(Exception):
        select_features(
            normalized_csv=str(tmp_path / "missing.csv"),
            reducedTrain_csv=str(tmp_path / "train.csv"),
            reducedTest_csv=str(tmp_path / "test.csv"),
            output_ottim_csv=str(tmp_path / "optim.csv"),
            output_json=str(tmp_path / "stats.json"),
            target_column="target",
        )


def test_missing_target_column_raises_error(tmp_path, dataset_d):
    dataset_path, _ = dataset_d
    with pytest.raises(ValueError):
        select_features(
            normalized_csv=str(dataset_path),
            reducedTrain_csv=str(tmp_path / "train.csv"),
            reducedTest_csv=str(tmp_path / "test.csv"),
            output_ottim_csv=str(tmp_path / "optim.csv"),
            output_json=str(tmp_path / "stats.json"),
            target_column="target",
        )


def test_invalid_perc_selected_rejected(tmp_path, dataset_a):
    dataset_path, _ = dataset_a
    for value in (-0.1, 1.2):
        with pytest.raises(Exception):
            select_features(
                normalized_csv=str(dataset_path),
                reducedTrain_csv=str(tmp_path / f"train_{value}.csv"),
                reducedTest_csv=str(tmp_path / f"test_{value}.csv"),
                output_ottim_csv=str(tmp_path / f"optim_{value}.csv"),
                output_json=str(tmp_path / f"stats_{value}.json"),
                target_column="target",
                percSelected=value,
            )


def test_invalid_perc_test_rejected(tmp_path, dataset_a):
    dataset_path, _ = dataset_a
    for value in (0, 1, -0.5):
        with pytest.raises(Exception):
            select_features(
                normalized_csv=str(dataset_path),
                reducedTrain_csv=str(tmp_path / f"train_{value}.csv"),
                reducedTest_csv=str(tmp_path / f"test_{value}.csv"),
                output_ottim_csv=str(tmp_path / f"optim_{value}.csv"),
                output_json=str(tmp_path / f"stats_{value}.json"),
                target_column="target",
                percTest=value,
            )


def test_negative_allowance_rejected(tmp_path, dataset_a):
    dataset_path, _ = dataset_a
    with pytest.raises(Exception):
        select_features(
            normalized_csv=str(dataset_path),
            reducedTrain_csv=str(tmp_path / "train.csv"),
            reducedTest_csv=str(tmp_path / "test.csv"),
            output_ottim_csv=str(tmp_path / "optim.csv"),
            output_json=str(tmp_path / "stats.json"),
            target_column="target",
            allowance=-1,
        )


def test_invalid_alpha_computations_rejected(tmp_path, dataset_a):
    dataset_path, _ = dataset_a
    for value in (0, -5):
        with pytest.raises(Exception):
            select_features(
                normalized_csv=str(dataset_path),
                reducedTrain_csv=str(tmp_path / f"train_{value}.csv"),
                reducedTest_csv=str(tmp_path / f"test_{value}.csv"),
                output_ottim_csv=str(tmp_path / f"optim_{value}.csv"),
                output_json=str(tmp_path / f"stats_{value}.json"),
                target_column="target",
                alpha_computations=value,
            )


def test_output_dataset_contains_only_selected_features(run_selection):
    stats = run_selection["stats"]
    expected_columns = stats["selected_feature_names"] + ["target"]

    train_df = pd.read_csv(run_selection["train_csv"])
    test_df = pd.read_csv(run_selection["test_csv"])

    assert list(train_df.columns) == expected_columns
    assert list(test_df.columns) == expected_columns


def test_feature_order_is_preserved(run_selection):
    original_columns = [
        c for c in run_selection["dataset"].columns if c != "target"
    ]
    selected_names = run_selection["stats"]["selected_feature_names"]
    expected_order = [
        c for c in original_columns if c in selected_names
    ]
    assert selected_names == expected_order


def test_reproducibility_same_seed_same_result(tmp_path, dataset_a):
    dataset_path, _ = dataset_a
    outputs = []

    for run_id in (1, 2):
        stats_json = tmp_path / f"stats_{run_id}.json"
        select_features(
            normalized_csv=str(dataset_path),
            reducedTrain_csv=str(tmp_path / f"train_{run_id}.csv"),
            reducedTest_csv=str(tmp_path / f"test_{run_id}.csv"),
            output_ottim_csv=str(tmp_path / f"optim_{run_id}.csv"),
            output_json=str(stats_json),
            target_column="target",
            seed=42,
            alpha_computations=20,
        )
        outputs.append(json.loads(stats_json.read_text(encoding="utf-8")))

    a, b = outputs
    assert a["selected_vector"] == b["selected_vector"]
    assert a["selected_feature_names"] == b["selected_feature_names"]
    assert a["alpha"] == b["alpha"]
    assert a["n_selected"] == b["n_selected"]


def test_different_seed_still_produces_valid_solution(tmp_path, dataset_a):
    dataset_path, _ = dataset_a
    stats_json = tmp_path / "stats.json"

    select_features(
        normalized_csv=str(dataset_path),
        reducedTrain_csv=str(tmp_path / "train.csv"),
        reducedTest_csv=str(tmp_path / "test.csv"),
        output_ottim_csv=str(tmp_path / "optim.csv"),
        output_json=str(stats_json),
        target_column="target",
        seed=999,
    )

    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    assert set(stats["selected_vector"]).issubset({0, 1})
    assert REQUIRED_JSON_KEYS.issubset(stats.keys())

    train_df = pd.read_csv(tmp_path / "train.csv")
    test_df = pd.read_csv(tmp_path / "test.csv")
    assert not train_df.empty
    assert not test_df.empty


def test_single_feature_dataset_handled_gracefully(tmp_path, dataset_c):
    dataset_path, _ = dataset_c
    try:
        select_features(
            normalized_csv=str(dataset_path),
            reducedTrain_csv=str(tmp_path / "train.csv"),
            reducedTest_csv=str(tmp_path / "test.csv"),
            output_ottim_csv=str(tmp_path / "optim.csv"),
            output_json=str(tmp_path / "stats.json"),
            target_column="target",
        )
    except Exception as exc:
        assert str(exc)


def test_one_class_target_handled_gracefully(tmp_path, dataset_e):
    dataset_path, _ = dataset_e
    try:
        select_features(
            normalized_csv=str(dataset_path),
            reducedTrain_csv=str(tmp_path / "train.csv"),
            reducedTest_csv=str(tmp_path / "test.csv"),
            output_ottim_csv=str(tmp_path / "optim.csv"),
            output_json=str(tmp_path / "stats.json"),
            target_column="target",
        )
    except Exception as exc:
        assert str(exc)


def test_highly_correlated_features_produce_valid_output(tmp_path, dataset_b):
    dataset_path, _ = dataset_b
    stats_json = tmp_path / "stats.json"

    select_features(
        normalized_csv=str(dataset_path),
        reducedTrain_csv=str(tmp_path / "train.csv"),
        reducedTest_csv=str(tmp_path / "test.csv"),
        output_ottim_csv=str(tmp_path / "optim.csv"),
        output_json=str(stats_json),
        target_column="target",
        alpha_computations=20,
    )

    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    assert REQUIRED_JSON_KEYS.issubset(stats.keys())
    assert set(stats["selected_vector"]).issubset({0, 1})


def test_alpha_computations_limit_respected(run_selection):
    stats = run_selection["stats"]
    assert stats["alpha_computations"] <= 20


def test_q_matrix_statistics_are_valid(run_selection):
    stats = run_selection["stats"]
    assert stats["q_matrix_creation_time"] >= 0
    assert stats["mean_optimization_time"] >= 0
    assert stats["std_dev_optimization_time"] >= 0