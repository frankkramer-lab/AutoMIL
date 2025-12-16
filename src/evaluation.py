"""
Module for ``automil.Evaluator``, which evaluates MIL models, calculates metrics, and generates plots.
"""
import os
from inspect import signature
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import slideflow as sf
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from slideflow.mil import eval_mil, predict_mil

from utils import LogLevel, format_ensemble_summary, get_vlog


# === Helpers === #
def is_model_directory(path: Path) -> bool:
    """Check if a given directory `path` contains a (trained) model as stored by slideflow
    
    Expected Structure:
    - path/
        - models/
            - best_valid.pth
        - predictions.parquet
        - history.csv
        - mil_params.json
        - slide_manifest.csv
        - (various plots)
        - ...

    Args:
        path (Path): Path to potential model directory

    Returns:
        bool: Whether `path` is a model directory
    """
    required_files = [
        path / "models" / "best_valid.pth",
        path / "predictions.parquet",
        path / "history.csv",
        path / "mil_params.json",
        path / "slide_manifest.csv",
    ]
    return all(file.exists() for file in required_files)

class Evaluator:
    """Evaluates trained MIL models, calculates metrics, generates plots, and creates ensemble predictions."""    

    def __init__(self,
        dataset: sf.Dataset,
        model_dir: Path,
        out_dir: Path,
        bags_dir: Path,
        verbose: bool = True
    ) -> None:
        """Initializes a Evaluator Instance

        Args:
            dataset (sf.Dataset): Slideflow dataset
            model_dir (Path): Directory in which to store trained models
            out_dir (Path): Diectory in which to store results such as predictions
            bags_dir (Path): Directory with feature bags
            verbose (bool, optional): Whether to print verbose messages. Defaults to True.
        """
        self.dataset = dataset
        self.vlog = get_vlog(verbose)

        # Path Setup
        self.model_dir = model_dir
        self.out_dir = out_dir
        self.bags_dir = bags_dir


    def load_predictions(
        self,
        model_path: Path
    ) -> pd.DataFrame:
        """Loads the predictions from a given model path (`model_path/predictions.parquet`) into a Dataframe and validates it by checking for required columns.

        Required:
        - Prediction probability columns starting with 'y_pred' (e.g., 'y_pred0', 'y_pred1', ...)
        - Base columns: 'slide' and 'y_true'


        Args:
            model_path (Path): Path to model directory

        Raises:
            FileNotFoundError: If `model_path/predictions.parquet` does not exist
            ValueError: If `model_path/predictions.parquet` does not contain any prediction probability columns
            ValueError: If `model_path/predictions.parquet` does not contain the required base columns

        Returns:
            pd.DataFrame: `model_path/predictions.parquet` loaded into a DataFrame
        """
        if not (predictions_path := model_path / "predictions.parquet").exists():
            raise FileNotFoundError(f"{model_path} does not contain a 'predictions.parquet' file")
        
        predictions = pd.read_parquet(predictions_path)

        all_columns = [column for column in predictions.columns]
        # We expect columns containing prediction probabilites to start with 'y_pred' (e.g 'y_pred0', 'y_pred1', ...)
        pred_columns = [column for column in all_columns if column.startswith("y_pred")]
        # Similarly, we expect predictions to contain 'slide' and 'y_true' columns
        base_columns = ["slide", "y_true"]

        if not pred_columns:
            raise ValueError("'predictions.parquet' does not contain the expected prediction columns")
        elif not all(base_column in all_columns for base_column in base_columns):
            raise ValueError("'predictions.parquet' does not contain the expected base columns")
        
        return predictions
    
    def calculate_metrics(
        self,
        predictions: pd.DataFrame | Path | str
    ) -> dict[str, float | np.ndarray]:
        """Calculate evaluation metrics from predictions DataFrame or path to predictions parquet file.

        Args:
            predictions (pd.DataFrame | PathIn): Predictions DataFrame or path to predictions parquet file
        Returns:
            dict[str, float | np.ndarray]: Dictionary containing evaluation metrics (AUC, AP, Accuracy, F1, ConfusionMatrix)
        """

        # Make sure we're working with a loaded DataFrame
        match predictions:
            case Path() | str():
                predictions = self.load_predictions(Path(predictions))
            case pd.DataFrame():
                pass
        
        # Extract true labels and calculate number of classes
        y_true = predictions["y_true"].astype(int)
        num_classes = len(y_true.unique())

        # We expect columns containing prediction probabilites to start with 'y_pred' (e.g 'y_pred0', 'y_pred1', ...)
        # Similarly, we may have ensemble predictions ending with '_ensemble' (e.g., 'y_pred0_ensemble', 'y_pred1_ensemble', ...)
        pred_columns = [column for column in predictions.columns if column.startswith("y_pred")]
        # Case 1: Ensemble predictions (priority)
        ensemble_columns = [col for col in pred_columns if col.endswith("_ensemble")]
        if ensemble_columns:
            # Use ensemble predictions
            prob_columns = [f"y_pred{i}_ensemble" for i in range(num_classes)]
            prediction_type = "ensemble"
        else:
            # Case 2: Single model predictions
            # Get regular y_pred columns (y_pred0, y_pred1, etc.)
            prob_columns = [f"y_pred{i}" for i in range(num_classes)]
            prediction_type = "single model"

        # Verify all expected probability columns exist
        missing_columns = [col for col in prob_columns if col not in predictions.columns]
        if missing_columns:
            raise ValueError(f"Missing probability columns for {prediction_type} predictions: {missing_columns}")
        
        # Get probability matrix
        prob_matrix = predictions[prob_columns].values

        # Get predicted classes
        if "y_pred_label" in predictions.columns:
            y_pred = predictions["y_pred_label"].astype(int)
        else:
            y_pred = np.argmax(prob_matrix, axis=1)

        # Calculate metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)

        # Binary classification
        if num_classes == 2:
            y_probs = prob_matrix[:, 1] # We really only need the prediction probabilities for label 1

            auc = float(roc_auc_score(y_true, y_probs))
            ap  = float(average_precision_score(y_true, y_probs))
            f1  = float(f1_score(y_true, y_pred))
        
        # Multiclass
        else:
            auc = float(roc_auc_score(y_true, prob_matrix, multi_class="ovr", average="macro"))

            ap_scores = []
            for class_idx in range(num_classes):
                # 0 if label is class_idx, 1 otherwise
                y_true_binary = (y_true == class_idx).astype(int)
                # Prediction probabilities for this class
                y_probs_class = prob_matrix[:, class_idx]

                if len(y_true_binary.unique()) > 1:
                    ap_class = average_precision_score(y_true_binary, y_probs_class)
                    ap_scores.append(ap_class)

            ap = float(np.mean(ap_scores)) if ap_scores else 0.0
            f1 = float(f1_score(y_true, y_pred, average="macro"))

        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

        return {
            "Accuracy": accuracy,
            "AUC": auc,
            "AP": ap,
            "F1": f1,
            "ConfusionMatrix": cm,
            "PerClassAccuracy": per_class_accuracy
        }

    def evaluate_models(
        self,
        model_dir: Path | None = None,
        bags_dir: Path | None = None,
        out_dir: Path | None = None,
        generate_attention_heatmaps: bool = False
    ) -> None:
        """Evaluates all models inside `self.model_dir` and writes the results to `self.out_dir`

        Args:
            model_dir (Path | None, optional): Directory containing model subdirectories. Defaults to `self.model_dir`.
            bags_dir (Path | None, optional): Directory containing bags. Defaults to `self.bags_dir`.
            out_dir (Path | None, optional): Output directory for predictions. Defaults to `self.out_dir`.
            generate_attention_heatmaps (bool, optional): Whether to generate attention heatmaps. Defaults to False.
        """
        # Default to instance variables if none provided
        model_dir = model_dir or self.model_dir
        bags_dir = bags_dir or self.bags_dir
        out_dir = out_dir or self.out_dir

        # Check if model_dir is a single model directory
        if is_model_directory(model_dir):
            model_paths = [model_dir]
            self.vlog(f"Single model directory detected: {model_dir}")
        # Else, collect all model subdirectories
        else:
            if not (model_paths := [subdir for subdir in model_dir.iterdir() if subdir.is_dir() and is_model_directory(subdir)]):
                self.vlog(f"No model directories found in {model_dir}", LogLevel.WARNING)
                return
        
        # Iterate over each model directory and evaluate
        for model_idx, model_path in enumerate(model_paths):
            self.vlog(f"Evaluating model {model_idx+1}/{len(model_paths)}: {model_path}")
            try:
                eval_mil(
                    weights=str(model_path),
                    bags=str(bags_dir),
                    dataset=self.dataset,
                    outcomes="label",
                    outdir=str(out_dir),
                    attention_heatmaps=generate_attention_heatmaps
                )
                self.vlog("Evaluation complete.\n")
            except Exception as e:
                self.vlog(f"Error evaluating model at {model_path}: {e}", LogLevel.ERROR)
                continue

    def generate_predictions(
        self,
        model_dir: Path | None = None,
        bags_dir: Path | None = None,
        out_dir: Path | None = None
    ) -> None:
        """Generates predictions for all models inside `self.model_dir` and writes the results to `self.out_dir`

        Args:
            model_dir (Path | None, optional): Directory containing model subdirectories. Defaults to `self.model_dir`.
            bags_dir (Path | None, optional): Directory containing bags. Defaults to `self.bags_dir`.
            out_dir (Path | None, optional): Output directory for predictions. Defaults to `self.out_dir`.
        """
        # Default to instance variables if none provided
        model_dir = model_dir or self.model_dir
        bags_dir = bags_dir or self.bags_dir
        out_dir = out_dir or self.out_dir

        # Check if model_dir is a single model directory
        if is_model_directory(model_dir):
            model_paths = [model_dir]
            self.vlog(f"Single model directory detected: {model_dir}")
        # Else, collect all model subdirectories
        else:
            if not (model_paths := [subdir for subdir in model_dir.iterdir() if subdir.is_dir() and is_model_directory(subdir)]):
                self.vlog(f"No model directories found in {model_dir}", LogLevel.WARNING)
                return
        
        # Iterate over each model directory and generate predictions
        for model_idx, model_path in enumerate(model_paths):
            self.vlog(f"Generating predictions with model {model_idx+1}/{len(model_paths)}: {model_path}")
            try:
                predictions = predict_mil(
                    model=str(model_path),
                    bags=str(bags_dir),
                    dataset=self.dataset,
                    outcomes="label",
                )
                # Cast to DataFrame
                # Can do this safely since predict_mil always returns a DataFrame if attention==False
                predictions = pd.DataFrame(predictions)

                # Save predictions to out_dir/model_name/predictions.parquet
                model_out_dir = out_dir / model_path.name
                model_out_dir.mkdir(parents=True, exist_ok=True)
                predictions_path = model_out_dir / "predictions.parquet"
                predictions.to_parquet(predictions_path, index=False)
                self.vlog(f"Predictions saved to {predictions_path}")

            except Exception as e:
                self.vlog(f"Error evaluating model at {model_path}: {e}", LogLevel.ERROR)
                continue

    def create_ensemble_predictions(
        self,
        model_dir: Path | None = None,
        output_path: Path | None = None,
        print_summary: bool = True
    ) -> tuple[pd.DataFrame, dict[str, float | np.ndarray]]:
        """Generate ensemble predictions from all models inside `model_dir` and saves the results to `output_path`

        Args:
            model_dir (Path | None, optional): Directory containing model subdirectories. Defaults to `self.model_dir`.
            output_path (Path | None, optional): Optional output file path to write results to. Supported formats: .csv, .parquet. Defaults to None.

        Raises:
            ValueError: If no predictions could be loaded from any model in `model_dir`
            ValueError: If no prediction columns are found for ensembling

        Returns:
            tuple[pd.DataFrame, dict[str, float | np.ndarray]]: A tuple of ensemble predictions as a DataFrame and a dictionary of evaluation metrics
        """
        model_dir = model_dir or self.model_dir
        output_path = output_path or (self.out_dir / "ensemble_predictions.parquet")

        # Check if model_dir is a single model directory
        if is_model_directory(model_dir):
            model_paths = [model_dir]
            self.vlog(f"Single model directory detected: {model_dir}")
        # Else, collect all model subdirectories
        else:
            if not (model_paths := [subdir for subdir in model_dir.iterdir() if subdir.is_dir() and is_model_directory(subdir)]):
                self.vlog(f"No model directories found in {model_dir}", LogLevel.WARNING)
                raise ValueError("No model directories found for ensembling")

        # Try to load predictions from each model that has been evaluated (should all be in model_dir)
        predictions_list: list[pd.DataFrame] = []
        for model_idx, submodel_dir in enumerate(model_paths):
            try:
                predictions = self.load_predictions(submodel_dir)

                # Add the model index to predictions columns so we can merge later
                pred_columns = [column for column in predictions.columns if column.startswith("y_pred")]
                rename_map = {pred_column: f"{pred_column}_model{model_idx}" for pred_column in pred_columns}
                predictions = predictions.rename(columns=rename_map)
                predictions_list.append(predictions)

                self.vlog(f"Loaded predictions from model {submodel_dir.name} ({model_idx+1}/{len(os.listdir(model_dir))})")
            except Exception as e:
                self.vlog(f"Error loading predictions from {submodel_dir}: {e}", LogLevel.WARNING)
                continue
        
        if not predictions_list:
            raise ValueError("Failed to load any predictions from model directory")

        # Merge predictions on the base columns
        merged = predictions_list[0].copy()

        for predictions in predictions_list[1:]:
            merged = merged.merge(
                predictions,
                on=["slide", "y_true"],
                how="inner"
            )
        
        # Get all prediction columns
        all_pred_columns = [
            column for column in merged.columns
            if column.startswith("y_pred")
        ]
        
        if not all_pred_columns:
            raise ValueError("No prediction columns found for ensembling")
        
        unique_classes = sorted(merged["y_true"].unique())
        n_classes = len(unique_classes)

        # Get prediction columns per class
        class_prediction_columns = {}
        for class_idx in range(n_classes):
            class_prediction_columns[class_idx] = [
                column for column in all_pred_columns
                if column.startswith(f"y_pred{class_idx}_")
            ]
        
        # Calculate ensemble (average) probabilities
        ensemble_probs = {}
        for class_idx in range(n_classes):
            if class_prediction_columns[class_idx]:
                ensemble_probs[f"y_pred{class_idx}_ensemble"] = merged[
                    class_prediction_columns[class_idx]
                ].mean(axis=1)
            else:
                self.vlog(f"No prediction columns found for class {class_idx}")
                ensemble_probs[f"y_pred{class_idx}_ensemble"] = 0.0

        # Add ensemble probabilities to DataFrame
        for column, probability in ensemble_probs.items():
            merged[column] = probability

        # Get probability matrix and make final predictions
        ensemble_probability_columns = [f"y_pred{class_idx}_ensemble" for class_idx in range(n_classes)]
        prob_matrix = merged[ensemble_probability_columns].values
        predicted_classes = np.argmax(prob_matrix, axis=1)
        merged["y_pred_label"] = predicted_classes

        # calculate metrics and print summary
        metrics = self.calculate_metrics(merged)

        # Optional summary
        if print_summary:
            summary = format_ensemble_summary(
                len(predictions_list),
                metrics["ConfusionMatrix"],  # type: ignore
                float(metrics["AUC"]),
                float(metrics["AP"]),
                float(metrics["Accuracy"]),
                float(metrics["F1"])
            )
            self.vlog(summary)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".csv":
            merged.to_csv(output_path, index=False)
        else:
            merged.to_parquet(output_path, index=False)
        self.vlog(f"Ensemble predictions saved to {output_path}")

        return merged, metrics

    def compare_models(
        self,
        model_dir: Path | None = None,
        metrics: list[str] = ["Accuracy", "AUC", "F1"]
    ) -> pd.DataFrame:
        """Compare metrics across multiple models
        
        Args:
            model_paths: List of model directories. If None, uses all in model_dir
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with model comparison
        """
        model_dir = model_dir or self.model_dir
        
        # Check if model_dir is a single model directory
        if is_model_directory(model_dir):
            model_paths = [model_dir]
            self.vlog(f"Single model directory detected: {model_dir}")
        # Else, collect all model subdirectories
        else:
            if not (model_paths := [subdir for subdir in model_dir.iterdir() if subdir.is_dir() and is_model_directory(subdir)]):
                self.vlog(f"No model directories found in {model_dir}", LogLevel.WARNING)
                raise ValueError("No model directories found for comparison")

        comparison_data = []
        for model_path in model_paths:
            try:
                predictions = self.load_predictions(model_path)
                model_metrics = self.calculate_metrics(predictions)
                
                row: dict[str, str | float] = {"model": model_path.name}
                for metric in metrics:
                    if metric in model_metrics:
                        value = model_metrics[metric]
                        # Convert numpy arrays and other types to string representation
                        if isinstance(value, np.ndarray):
                            row[metric] = round(float(value), 2)
                        else:
                            row[metric] = round(float(value), 2)
                    else:
                        row[metric] = "N/A"
                
                comparison_data.append(row)
                
            except Exception as e:
                self.vlog(f"Failed to evaluate {model_path.name}: {e}", LogLevel.WARNING)
                continue
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            self.vlog("Model Comparison:")
            self.vlog(comparison_df.to_string(index=False))
        
        return comparison_df
    
    # === Plotting === #
    def generate_plots(
        self,
        model_paths: list[Path] | None = None,
        save_path: Path | None = None,
        figsize: tuple[int, int] = (10, 10)
    ) -> None:
        """Generate all comparison plots and save them to `self.project_dir/figures`"""
        # Collect models from expected folder if not provided
        if model_paths is None:
            model_paths = sorted(
                [path for path in self.out_dir.iterdir() if path.is_dir()]
            )

        # Calculate and collect metrics for all models
        combined_metrics = {}
        for model_path in model_paths:
            try:
                predictions = self.load_predictions(model_path)
                model_metrics = self.calculate_metrics(predictions)
                combined_metrics[model_path.name] = model_metrics
            except Exception as e:
                self.vlog(f"Failed to load metrics for {model_path.name}: {e}")
                continue
        
        if not combined_metrics:
            self.vlog("No valid model data found for generating plots")
            return
        
        # Collect and execute all plotting methods
        plots = cast(
            dict[str, Figure], # Make sure the type annotation is correct
            {
                method_name.removeprefix('_plot_'): plot_method(
                    combined_metrics,
                    figsize=figsize,
                )
                for method_name in dir(self)
                if (
                    method_name.startswith('_plot_')
                    and callable((plot_method := getattr(self, method_name)))
                    and signature(plot_method).return_annotation == Figure
                )
            }
        )

        if not save_path:
            save_path = self.out_dir / "figures"
            save_path.mkdir(parents=True, exist_ok=True)

        # Save all generated plots
        for plot_name, fig in plots.items():
            plot_file = save_path / f"{plot_name}.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.vlog(f"Saved plot '{plot_name}' to {plot_file}")
        return

    def _plot_roc_curves(
        self,
        combined_metrics: dict[str, dict[str, float | np.ndarray]],
        figsize: tuple[int, int] = (10, 8)
    ) -> Figure:
        """Plot ROC curves for all models"""
        from sklearn.metrics import auc, roc_curve
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(combined_metrics)))
        
        for i, (model_name, _) in enumerate(combined_metrics.items()):
            try:
                # Load predictions for this model
                model_path = self.out_dir / model_name
                predictions = self.load_predictions(model_path)
                
                y_true = predictions["y_true"].astype(int)
                num_classes = len(y_true.unique())
                
                # Get prediction probabilities
                pred_columns = [column for column in predictions.columns if column.startswith("y_pred")]
                ensemble_columns = [col for col in pred_columns if col.endswith("_ensemble")]
                
                if ensemble_columns:
                    prob_columns = [f"y_pred{i}_ensemble" for i in range(num_classes)]
                else:
                    prob_columns = [f"y_pred{i}" for i in range(num_classes)]
                
                prob_matrix = predictions[prob_columns].values
                
                if num_classes == 2:
                    # Binary classification - single ROC curve
                    y_probs = prob_matrix[:, 1]  # Probabilities for positive class
                    
                    fpr, tpr, _ = roc_curve(y_true, y_probs)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(
                        fpr, tpr, 
                        color=colors[i], 
                        linewidth=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})'
                    )
                    
                else:
                    # Multiclass - plot ROC curve for each class
                    for class_idx in range(num_classes):
                        y_true_binary = (y_true == class_idx).astype(int)
                        y_probs_class = prob_matrix[:, class_idx]
                        
                        # Only plot if we have both classes
                        if len(y_true_binary.unique()) > 1:
                            fpr, tpr, _ = roc_curve(y_true_binary, y_probs_class)
                            roc_auc = auc(fpr, tpr)
                            
                            # Use different line styles for different classes
                            line_style = ['-', '--', '-.', ':'][class_idx % 4]
                            
                            plt.plot(
                                fpr, tpr,
                                color=colors[i],
                                linestyle=line_style,
                                linewidth=2,
                                label=f'{model_name} Class {class_idx} (AUC = {roc_auc:.3f})'
                            )
                            
            except Exception as e:
                self.vlog(f"Could not plot ROC curve for {model_name}: {e}", LogLevel.WARNING)
                continue
    
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        return plt.gcf()

    def _plot_model_comparison(
        self,
        combined_metrics: dict[str, dict[str, float | np.ndarray]],
        figsize: tuple[int, int] = (12, 8)
    ) -> Figure:
        data = pd.DataFrame(combined_metrics)
        metrics = [col for col in data.index if col != "ConfusionMatrix" and col != "PerClassAccuracy"]
        n_metrics = len(metrics)

        # Create subplots
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
        # n_metrics == 1 means only 1 subplot, cast to list for consistency
        if n_metrics == 1:
            axes = cast(
                list[Axes],
                [axes]
            )
        # Otherwise axes is a list of subplots
        else:
            axes = cast(
                list[Axes],
                axes
            )
        
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(data)))
        x_positions = np.arange(len(data.columns))
        model_names = list(data.columns)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            # Plot single metric
            bars = ax.bar(
                x_positions,
                data.loc[metric],
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5,
            )

            bar: Rectangle # Iterating over a BarContainer gives Rectangle objects
            for bar, value in zip(bars, data.loc[metric]):
                height = bar.get_height()
                # Place actual value above bar
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 0.005,
                    f'{value:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            ax.set_xticks(x_positions)
            # Set model names as x-tick labels
            # Since model names can be long, center them to the right
            # To avoid any offset issues
            ax.set_xticklabels(
                model_names,
                rotation=45,
                ha="right",
                rotation_mode="anchor"
            )
            
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_box_plots(
        self,
        combined_metrics: dict[str, dict[str, float | np.ndarray]],
        figsize: tuple[int, int] = (10, 8)
    ) -> Figure:
        # Collect data in long format for box plots
        plot_data = []
        for _, metrics in combined_metrics.items():
            for metric_name, metric_value in metrics.items():
                if metric_name in ["ConfusionMatrix", "PerClassAccuracy"]:
                    continue
                
                plot_data.append({
                    'Metric': metric_name,
                    'Value': float(metric_value)
                })

        df = pd.DataFrame(plot_data)

        # Create the plot
        plt.figure(figsize=figsize)
        
        sns.boxplot(
            data=df,
            x='Metric',
            y='Value',
            palette='Set2',
            width=0.5
        )

        sns.stripplot(
            data=df,
            x='Metric',
            y='Value',
            color='black',
            size=6,
            jitter=True,
            alpha=0.7
        )
        
        plt.title('Metric Distributions', fontsize=14, fontweight='bold')
        plt.ylabel('Value', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return plt.gcf()

    def _plot_per_class_accuracy(
        self,
        combined_metrics: dict[str, dict[str, float | np.ndarray]],
        figsize: tuple[int, int] = (12, 8)
    ) -> Figure:
        # Prepare data for plotting
        data = []
        for model_name, metrics in combined_metrics.items():
            per_class_acc = metrics.get("PerClassAccuracy")
            if isinstance(per_class_acc, np.ndarray):
                for class_idx, acc in enumerate(per_class_acc):
                    data.append({
                        "Model": model_name,
                        "Class": f"Class {class_idx}",
                        "Accuracy": acc
                    })
        
        df = pd.DataFrame(data)

        # Create the plot
        plt.figure(figsize=figsize)
        
        # Create a grouped bar plot
        ax = sns.barplot(data=df, x='Class', y='Accuracy', hue='Model', alpha=0.8)
        
        plt.title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt='%.2f', fontsize=9)
        
        plt.tight_layout()
        return plt.gcf()