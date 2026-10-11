# QUBO Binary Classification — G44

This project implements a binary-classification pipeline with QUBO-based
feature reduction. It preprocesses a numeric dataset, selects features,
trains a classifier, and evaluates predictions on a held-out test set. A
Tkinter graphical interface is included.

The workflow and repository documentation follow the official project specification
(reports/llm_usage/files/Progetto_ISW_2025-26_specifiche.pdf).

## Project Structure

```text
isw-qubo-classification-G44/
├── README.md
├── requirements.txt
├── pyproject.toml
├── group_info.yaml
├── data/
│   ├── input_dataset.csv
│   └── sample_test_dataset.csv
├── src/
│   └── qubo_project/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── feature_selection.py
│       ├── model.py
│       └── gui.py
├── tests/
├── outputs/
└── reports/
    ├── project_report.yaml
    └── llm_usage/
        ├── LOG-G44-01.md
        ├── LOG-G44-02.md
        ├── LOG-G44-03.md
        └── files/
```

## Installation

Python 3.11 or newer is required. From the repository root, create and activate
a virtual environment, then install the project dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell does not allow environment activation, run commands using
`.venv\Scripts\python.exe` instead of `python`.

## Dataset Format

Input data must be a CSV file with a header row, numeric feature columns, and
one binary target column containing `0` and `1`. The target column name is
provided as an argument; it does not need to be named `target`.

The repository includes:

- `data/input_dataset.csv` — dataset for trying the pipeline.
- `data/sample_test_dataset.csv` — small dataset used by the automated tests.

## Phase 1: Preprocessing

The preprocessing module:

- Reads the input CSV and separates the target column.
- Removes features whose non-missing, non-zero value ratio is below
  `minPercValid`.
- Imputes missing feature values using medians computed from the training
  partition.
- Applies z-score normalization to features, using statistics fitted on the
  training partition; the target is retained and not normalized.
- Writes the normalized dataset and preprocessing statistics.

### Command Line Usage

```powershell
python -m qubo_project.preprocessing `
  --input data/input_dataset.csv `
  --target target `
  --out-data outputs/normalized.csv `
  --out-json outputs/preprocessing_result.json `
  --min-perc-valid 0.05 `
  --perc-test 0.30
```

### Outputs

- `outputs/normalized.csv`
- `outputs/preprocessing_result.json`

## Phase 2: QUBO Feature Selection

Feature selection is formulated as a Quadratic Unconstrained Binary
Optimization (QUBO) problem. Absolute Spearman correlations are used to
maximize feature relevance to the target while minimizing redundancy between
features. Simulated annealing (`dwave-neal`) searches for an `alpha` value
that yields approximately the requested number of features, within the
specified allowance.

### Command Line Usage

```powershell
python -m qubo_project.feature_selection `
  --in-normalized outputs/normalized.csv `
  --out-train outputs/training_reduced.csv `
  --out-test outputs/test_reduced.csv `
  --out-optimizations outputs/optimizations.csv `
  --out-json outputs/feature_selection_result.json `
  --target target `
  --perc-selected 0.20 `
  --allowance 1 `
  --perc-test 0.30 `
  --seed 42 `
  --alpha-computations 100
```

Use the same `--perc-test` value for preprocessing and feature selection so
that both stages use the same leading rows for training and trailing rows for
testing.

### Outputs

- `outputs/training_reduced.csv`
- `outputs/test_reduced.csv`
- `outputs/optimizations.csv`
- `outputs/feature_selection_result.json`

## Phase 3: Model Training

The project supports three classifiers:

- Random Forest (`random_forest`)
- Logistic Regression (`logistic_regression`)
- Decision Tree (`decision_tree`)

Random Forest is used in this example. The model is trained using the reduced
training dataset and saved as a Joblib artifact.

### Command Line Usage

```powershell
python -m qubo_project.model train `
  --classifier random_forest `
  --in-reduced outputs/training_reduced.csv `
  --target target `
  --out-model outputs/model.joblib `
  --out-metrics outputs/training_metrics.json `
  --seed 42
```

### Outputs

- `outputs/model.joblib`
- `outputs/training_metrics.json`

## Phase 4: Prediction

The trained model predicts labels for the reduced test dataset. The output
includes the true target, predicted label, and positive-class score for every
test record. Classification statistics include per-class precision, recall,
F1-score, ROC-AUC, and the confusion matrix.

### Command Line Usage

```powershell
python -m qubo_project.model predict `
  --input-testset outputs/test_reduced.csv `
  --target target `
  --model outputs/model.joblib `
  --out-predictions outputs/predictions.csv `
  --out-stats outputs/classification_stats.json
```

### Outputs

- `outputs/predictions.csv`
- `outputs/classification_stats.json`

## GUI

The Tkinter interface lets users select a dataset and target column, run
preprocessing, perform QUBO feature selection, train a classifier, run
predictions, and inspect generated outputs.

Launch the GUI from the repository root:

```powershell
python -m qubo_project.gui
```

## Running Tests

Run the complete automated test suite with:

```powershell
python -m pytest
```

The tests cover preprocessing, feature selection, model training and
prediction, and end-to-end pipeline execution. The 1.5-million-row
preprocessing performance test is skipped by default. To run it in PowerShell:

```powershell
$env:RUN_PERFORMANCE_TESTS = "1"
python -m pytest -m performance
```

## Reproducibility

- Requires Python 3.11 or newer.
- Runtime dependencies are listed in `requirements.txt`.
- Randomized operations accept a seed; the examples use `42`.
- The pipeline runs from the command line independently of the GUI.
- Input and output paths are supplied at runtime, so external datasets can be
  processed without code changes.

## Technologies

- Python and Tkinter
- pandas and NumPy
- SciPy
- scikit-learn
- dimod and dwave-neal
- Joblib
- pytest

## Authors

Group: G44

- Nicolò Castangia — 60/61/66404
- Silvia Sotgia — 60/61/66397

See [`group_info.yaml`](group_info.yaml) for group configuration and
[`reports/project_report.yaml`](reports/project_report.yaml) for the project
report.

## License

This project was developed for the Software Engineering course.
