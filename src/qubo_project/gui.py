from __future__ import annotations

import json
import logging
import threading
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from qubo_project.feature_selection import select_features
from qubo_project.model import predict, train
from qubo_project.preprocessing import fit_normalize


OUTPUT_DIR = Path("outputs")
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing"
FEATURE_SELECTION_DIR = OUTPUT_DIR / "feature_selection"
TRAINING_DIR = OUTPUT_DIR / "training"
PREDICTION_DIR = OUTPUT_DIR / "prediction"


for directory in (
    OUTPUT_DIR,
    PREPROCESSING_DIR,
    FEATURE_SELECTION_DIR,
    TRAINING_DIR,
    PREDICTION_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        "gui.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)


class TextHandler(logging.Handler):
    """
    Logging handler writing records to a Tk text widget.
    """

    def __init__(self, widget: tk.Text) -> None:
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)

        def append() -> None:
            self.widget.configure(state=tk.NORMAL)
            self.widget.insert(tk.END, message + "\n")
            self.widget.see(tk.END)
            self.widget.configure(state=tk.DISABLED)

        try:
            self.widget.after(0, append)
        except RuntimeError:
            pass


class QuboClassificationGUI:
    """
    Main GUI application.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        self.root.title("QUBO Classification Project")
        self.root.geometry("1400x900")

        self.dataset_path = tk.StringVar()
        self.target_column = tk.StringVar(value="target")

        self.model_path = tk.StringVar()
        self.test_dataset_path = tk.StringVar()

        self.status_var = tk.StringVar(value="Ready")

        self._build_layout()
        self._configure_logging()

    def _build_layout(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5,
        )

        self.dataset_tab = ttk.Frame(self.notebook)
        self.preprocessing_tab = ttk.Frame(self.notebook)
        self.feature_selection_tab = ttk.Frame(
            self.notebook
        )
        self.training_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)

        self.notebook.add(
            self.dataset_tab,
            text="Dataset",
        )
        self.notebook.add(
            self.preprocessing_tab,
            text="Preprocessing",
        )
        self.notebook.add(
            self.feature_selection_tab,
            text="Feature Selection",
        )
        self.notebook.add(
            self.training_tab,
            text="Training",
        )
        self.notebook.add(
            self.prediction_tab,
            text="Prediction",
        )
        self.notebook.add(
            self.results_tab,
            text="Results",
        )

        self._build_dataset_tab()
        self._build_preprocessing_tab()
        self._build_feature_selection_tab()
        self._build_training_tab()
        self._build_prediction_tab()
        self._build_results_tab()

        self.progress = ttk.Progressbar(
            self.root,
            mode="indeterminate",
        )
        self.progress.pack(
            fill=tk.X,
            padx=5,
        )

        status_frame = ttk.Frame(self.root)
        status_frame.pack(
            fill=tk.X,
            padx=5,
            pady=5,
        )

        ttk.Label(
            status_frame,
            text="Status:",
        ).pack(side=tk.LEFT)

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
        ).pack(side=tk.LEFT)

        ttk.Label(
            self.root,
            text="Application Log",
        ).pack(anchor=tk.W, padx=5)

        self.log_widget = tk.Text(
            self.root,
            height=10,
            state=tk.DISABLED,
        )
        self.log_widget.pack(
            fill=tk.X,
            padx=5,
            pady=5,
        )

    def _configure_logging(self) -> None:
        handler = TextHandler(self.log_widget)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
        )
        LOGGER.addHandler(handler)

    def _build_dataset_tab(self) -> None:
        frame = ttk.Frame(self.dataset_tab)
        frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

        ttk.Label(
            frame,
            text="Dataset",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Entry(
            frame,
            textvariable=self.dataset_path,
            width=90,
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=5,
        )

        ttk.Button(
            frame,
            text="Browse",
            command=self._browse_dataset,
        ).grid(
            row=1,
            column=1,
            padx=5,
        )

        ttk.Label(
            frame,
            text="Target Column",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

        ttk.Entry(
            frame,
            textvariable=self.target_column,
            width=30,
        ).grid(
            row=3,
            column=0,
            sticky="w",
        )

        ttk.Button(
            frame,
            text="Load Dataset",
            command=self._load_dataset_preview,
        ).grid(
            row=3,
            column=1,
            padx=5,
        )

        self.dataset_info = tk.Text(
            frame,
            height=4,
        )
        self.dataset_info.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="nsew",
            pady=10,
        )

        self.dataset_preview = ttk.Treeview(
            frame,
            show="headings",
        )

        self.dataset_preview.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="nsew",
        )

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

    def _build_preprocessing_tab(self) -> None:
        self.min_perc_valid = tk.StringVar(
            value="0.05"
        )

        top = ttk.Frame(self.preprocessing_tab)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            top,
            text="minPercValid",
        ).pack(side=tk.LEFT)

        ttk.Entry(
            top,
            textvariable=self.min_perc_valid,
            width=10,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        ttk.Button(
            top,
            text="Run Preprocessing",
            command=self._run_preprocessing,
        ).pack(side=tk.LEFT, padx=10)

        self.preprocessing_output = tk.Text(
            self.preprocessing_tab
        )
        self.preprocessing_output.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

    def _build_feature_selection_tab(self) -> None:
        self.perc_selected = tk.StringVar(
            value="0.20"
        )
        self.allowance = tk.StringVar(value="1")
        self.perc_test = tk.StringVar(value="0.30")
        self.seed = tk.StringVar(value="42")
        self.alpha_computations = tk.StringVar(
            value="100"
        )

        frame = ttk.Frame(
            self.feature_selection_tab
        )
        frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            frame,
            text="percSelected",
        ).grid(row=0, column=0)

        ttk.Entry(
            frame,
            textvariable=self.perc_selected,
            width=10,
        ).grid(row=0, column=1)

        ttk.Label(
            frame,
            text="allowance",
        ).grid(row=0, column=2)

        ttk.Entry(
            frame,
            textvariable=self.allowance,
            width=10,
        ).grid(row=0, column=3)

        ttk.Label(
            frame,
            text="percTest",
        ).grid(row=1, column=0)

        ttk.Entry(
            frame,
            textvariable=self.perc_test,
            width=10,
        ).grid(row=1, column=1)

        ttk.Label(
            frame,
            text="seed",
        ).grid(row=1, column=2)

        ttk.Entry(
            frame,
            textvariable=self.seed,
            width=10,
        ).grid(row=1, column=3)

        ttk.Label(
            frame,
            text="alphaComputations",
        ).grid(row=2, column=0)

        ttk.Entry(
            frame,
            textvariable=self.alpha_computations,
            width=10,
        ).grid(row=2, column=1)

        ttk.Button(
            frame,
            text="Run Feature Selection",
            command=self._run_feature_selection,
        ).grid(
            row=3,
            column=0,
            columnspan=4,
            pady=10,
        )

        self.feature_selection_output = tk.Text(
            self.feature_selection_tab
        )
        self.feature_selection_output.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

    def _build_training_tab(self) -> None:
        self.classifier = tk.StringVar(
            value="random_forest"
        )
        self.training_seed = tk.StringVar(
            value="42"
        )

        frame = ttk.Frame(self.training_tab)
        frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            frame,
            text="Classifier",
        ).grid(row=0, column=0)

        ttk.Combobox(
            frame,
            textvariable=self.classifier,
            values=[
                "random_forest",
                "logistic_regression",
                "decision_tree",
            ],
            state="readonly",
        ).grid(row=0, column=1)

        ttk.Label(
            frame,
            text="Seed",
        ).grid(row=0, column=2)

        ttk.Entry(
            frame,
            textvariable=self.training_seed,
            width=10,
        ).grid(row=0, column=3)

        ttk.Button(
            frame,
            text="Train Model",
            command=self._run_training,
        ).grid(
            row=1,
            column=0,
            columnspan=4,
            pady=10,
        )

        self.training_output = tk.Text(
            self.training_tab
        )
        self.training_output.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

    def _build_prediction_tab(self) -> None:
        frame = ttk.Frame(self.prediction_tab)
        frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            frame,
            text="Model File",
        ).grid(row=0, column=0)

        ttk.Entry(
            frame,
            textvariable=self.model_path,
            width=80,
        ).grid(row=0, column=1)

        ttk.Button(
            frame,
            text="Browse",
            command=self._browse_model,
        ).grid(row=0, column=2)

        ttk.Label(
            frame,
            text="Test Dataset",
        ).grid(row=1, column=0)

        ttk.Entry(
            frame,
            textvariable=self.test_dataset_path,
            width=80,
        ).grid(row=1, column=1)

        ttk.Button(
            frame,
            text="Browse",
            command=self._browse_test_dataset,
        ).grid(row=1, column=2)

        ttk.Button(
            frame,
            text="Run Prediction",
            command=self._run_prediction,
        ).grid(
            row=2,
            column=0,
            columnspan=3,
            pady=10,
        )

        self.prediction_output = tk.Text(
            self.prediction_tab
        )
        self.prediction_output.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

    def _build_results_tab(self) -> None:
        self.results_listbox = tk.Listbox(
            self.results_tab
        )
        self.results_listbox.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

        ttk.Button(
            self.results_tab,
            text="Refresh",
            command=self._refresh_results,
        ).pack(pady=5)

    def _browse_dataset(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if path:
            self.dataset_path.set(path)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename()

        if path:
            self.model_path.set(path)

    def _browse_test_dataset(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if path:
            self.test_dataset_path.set(path)

    def _update_status(
        self,
        status: str,
    ) -> None:
        self.status_var.set(status)

    def _start_busy(self, status: str) -> None:
        self._update_status(status)
        self.progress.start(10)

    def _stop_busy(
        self,
        status: str,
    ) -> None:
        self.progress.stop()
        self._update_status(status)

    def _load_dataset_preview(self) -> None:
        try:
            dataset_path = Path(
                self.dataset_path.get()
            )

            if not dataset_path.exists():
                raise FileNotFoundError(
                    "Dataset not found."
                )

            df = pd.read_csv(dataset_path)

            preview = df.head(20)

            self.dataset_info.delete(
                "1.0",
                tk.END,
            )
            self.dataset_info.insert(
                tk.END,
                (
                    f"Rows: {len(df)}\n"
                    f"Columns: {len(df.columns)}\n"
                    f"Target: {self.target_column.get()}"
                ),
            )

            self.dataset_preview.delete(
                *self.dataset_preview.get_children()
            )

            self.dataset_preview["columns"] = list(
                preview.columns
            )

            for column in preview.columns:
                self.dataset_preview.heading(
                    column,
                    text=column,
                )

            for row in preview.itertuples(
                index=False
            ):
                self.dataset_preview.insert(
                    "",
                    tk.END,
                    values=list(row),
                )

            LOGGER.info(
                "Dataset preview loaded."
            )

        except Exception as exc:
            LOGGER.exception(
                "Failed loading dataset preview."
            )

            messagebox.showerror(
                "Dataset Error",
                str(exc),
            )

    # CORRETTO: Spostato fuori dal blocco except, al livello corretto della classe
    def _display_json(
        self,
        widget: tk.Text,
        json_path: Path,
    ) -> None:
        if not json_path.exists():
            return

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        widget.delete(
            "1.0",
            tk.END,
        )

        widget.insert(
            tk.END,
            json.dumps(
                payload,
                indent=2,
            ),
        )

    def _run_in_thread(
        self,
        target: Any,
    ) -> None:
        thread = threading.Thread(
            target=target,
            daemon=True,
        )
        thread.start()

    def _run_preprocessing(self) -> None:
        def worker() -> None:
            try:
                self.root.after(
                    0,
                    lambda: self._start_busy(
                        "Running preprocessing..."
                    ),
                )

                normalized_csv = (
                    PREPROCESSING_DIR
                    / "normalized.csv"
                )

                result_json = (
                    PREPROCESSING_DIR
                    / "preprocessing_result.json"
                )

                fit_normalize(
                    input_csv=self.dataset_path.get(),
                    target_column=self.target_column.get(),
                    normalized_csv=str(
                        normalized_csv
                    ),
                    outInitalRes_json=str(
                        result_json
                    ),
                    minPercValid=float(
                        self.min_perc_valid.get()
                    ),
                )

                self.root.after(
                    0,
                    lambda: self._display_json(
                        self.preprocessing_output,
                        result_json,
                    ),
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Completed"
                    ),
                )

                self.root.after(
                    0,
                    self._refresh_results,
                )

                LOGGER.info(
                    "Preprocessing completed."
                )

            except Exception:
                LOGGER.exception(
                    "Preprocessing failed."
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Failed"
                    ),
                )

                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Preprocessing Error",
                        "Preprocessing failed. "
                        "See gui.log for details.",
                    ),
                )

        self._run_in_thread(worker)

    def _run_feature_selection(self) -> None:
        def worker() -> None:
            try:
                self.root.after(
                    0,
                    lambda: self._start_busy(
                        "Running feature selection..."
                    ),
                )

                normalized_csv = (
                    PREPROCESSING_DIR
                    / "normalized.csv"
                )

                train_csv = (
                    FEATURE_SELECTION_DIR
                    / "training_reduced.csv"
                )

                test_csv = (
                    FEATURE_SELECTION_DIR
                    / "test_reduced.csv"
                )

                optimizations_csv = (
                    FEATURE_SELECTION_DIR
                    / "optimizations.csv"
                )

                result_json = (
                    FEATURE_SELECTION_DIR
                    / "feature_selection_result.json"
                )

                select_features(
                    normalized_csv=str(
                        normalized_csv
                    ),
                    reducedTrain_csv=str(
                        train_csv
                    ),
                    reducedTest_csv=str(
                        test_csv
                    ),
                    output_ottim_csv=str(
                        optimizations_csv
                    ),
                    output_json=str(
                        result_json
                    ),
                    target_column=self.target_column.get(),
                    percSelected=float(
                        self.perc_selected.get()
                    ),
                    allowance=int(
                        self.allowance.get()
                    ),
                    percTest=float(
                        self.perc_test.get()
                    ),
                    seed=int(
                        self.seed.get()
                    ),
                    alpha_computations=int(
                        self.alpha_computations.get()
                    ),
                )

                self.test_dataset_path.set(
                    str(test_csv)
                )

                self.root.after(
                    0,
                    lambda: self._display_json(
                        self.feature_selection_output,
                        result_json,
                    ),
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Completed"
                    ),
                )

                self.root.after(
                    0,
                    self._refresh_results,
                )

                LOGGER.info(
                    "Feature selection completed."
                )

            except Exception:
                LOGGER.exception(
                    "Feature selection failed."
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Failed"
                    ),
                )

                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Feature Selection Error",
                        "Feature selection failed.",
                    ),
                )

        self._run_in_thread(worker)

    def _run_training(self) -> None:
        def worker() -> None:
            try:
                self.root.after(
                    0,
                    lambda: self._start_busy(
                        "Training model..."
                    ),
                )

                train_csv = (
                    FEATURE_SELECTION_DIR
                    / "training_reduced.csv"
                )

                model_file = (
                    TRAINING_DIR
                    / "model.joblib"
                )

                metrics_json = (
                    TRAINING_DIR
                    / "training_metrics.json"
                )

                train(
                    classifier=self.classifier.get(),
                    reducedTrain_csv=str(
                        train_csv
                    ),
                    target_column=self.target_column.get(),
                    model_path=str(
                        model_file
                    ),
                    metrics_json=str(
                        metrics_json
                    ),
                    seed=int(
                        self.training_seed.get()
                    ),
                )

                self.model_path.set(
                    str(model_file)
                )

                self.root.after(
                    0,
                    lambda: self._display_json(
                        self.training_output,
                        metrics_json,
                    ),
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Completed"
                    ),
                )

                self.root.after(
                    0,
                    self._refresh_results,
                )

                LOGGER.info(
                    "Training completed."
                )

            except Exception:
                LOGGER.exception(
                    "Training failed."
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Failed"
                    ),
                )

                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Training Error",
                        "Model training failed.",
                    ),
                )

        self._run_in_thread(worker)

    def _run_prediction(self) -> None:
        def worker() -> None:
            try:
                self.root.after(
                    0,
                    lambda: self._start_busy(
                        "Generating predictions..."
                    ),
                )

                predictions_csv = (
                    PREDICTION_DIR
                    / "predictions.csv"
                )

                stats_json = (
                    PREDICTION_DIR
                    / "classification_stats.json"
                )

                predict(
                    reduced_Test_csv=(
                        self.test_dataset_path.get()
                    ),
                    target_column=(
                        self.target_column.get()
                    ),
                    model_path=self.model_path.get(),
                    predictions_csv=str(
                        predictions_csv
                    ),
                    classif_stats_json=str(
                        stats_json
                    ),
                )

                self.root.after(
                    0,
                    lambda: self._display_json(
                        self.prediction_output,
                        stats_json,
                    ),
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Completed"
                    ),
                )

                self.root.after(
                    0,
                    self._refresh_results,
                )

                LOGGER.info(
                    "Prediction completed."
                )

            except Exception:
                LOGGER.exception(
                    "Prediction failed."
                )

                self.root.after(
                    0,
                    lambda: self._stop_busy(
                        "Failed"
                    ),
                )

                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Prediction Error",
                        "Prediction failed.",
                    ),
                )

        self._run_in_thread(worker)

    def _refresh_results(self) -> None:
        self.results_listbox.delete(
            0,
            tk.END,
        )

        for file_path in sorted(
            OUTPUT_DIR.rglob("*")
        ):
            if file_path.is_file():
                self.results_listbox.insert(
                    tk.END,
                    str(file_path),
                )


def main() -> int:
    root = tk.Tk()

    QuboClassificationGUI(root)

    root.mainloop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# END OF FILE
# FILE COMPLETE