# QUBO Binary Classification

Python pipeline for numeric binary-classification datasets. It filters and
normalizes features, performs QUBO feature selection, trains a classifier, and
evaluates predictions on an ordered holdout set.

## Setup

Use Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Run the automated tests with:

```powershell
python -m pytest
```

## Run the Pipeline

The example below uses `data/input_dataset.csv` and the `target` column. The
test fraction must be identical in preprocessing and feature selection. Rows
are split in their original order; filtering thresholds, missing-value
medians, normalization parameters, and feature correlations are determined
from the training prefix only. The fitted preprocessing is applied to both
partitions.

```powershell
python -m qubo_project.preprocessing --input data/input_dataset.csv --target target --out-data outputs/preprocessing/normalized.csv --out-json outputs/preprocessing/preprocessing_result.json --perc-test 0.30
python -m qubo_project.feature_selection --in-normalized outputs/preprocessing/normalized.csv --out-train outputs/feature_selection/training_reduced.csv --out-test outputs/feature_selection/test_reduced.csv --out-optimizations outputs/feature_selection/optimizations.csv --out-json outputs/feature_selection/feature_selection_result.json --target target --perc-test 0.30 --perc-selected 0.20 --allowance 1 --seed 42
python -m qubo_project.model train --classifier random_forest --in-reduced outputs/feature_selection/training_reduced.csv --target target --out-model outputs/training/model.joblib --out-metrics outputs/training/training_metrics.json --seed 42
python -m qubo_project.model predict --input-testset outputs/feature_selection/test_reduced.csv --target target --model outputs/training/model.joblib --out-predictions outputs/prediction/predictions.csv --out-stats outputs/prediction/classification_stats.json
```

To launch the graphical interface:

```powershell
python -m qubo_project.gui
```

The sample file `data/sample_test_dataset.csv` is a small example dataset. It
can be used for pipeline testing, but it is not interchangeable with a model's
holdout data unless its feature columns match the model's selected features.

