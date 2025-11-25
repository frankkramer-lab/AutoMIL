import os
from pathlib import Path

import numpy as np
import pandas as pd
import slideflow as sf
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from slideflow.mil import eval_mil

from utils import LogLevel, format_ensemble_summary, get_vlog


class Evaluator:
    """loading predictions from .parquet and ensembling and evaluating MIL models"""    


    def __init__(self,
        project: sf.Project,
        dataset: sf.Dataset,
        model_dir: Path | None = None,
        ensemble_dir: Path | None = None,
        bags_dir: Path | None = None,
        verbose: bool = True
    ) -> None:
        
        self.project = project
        self.dataset = dataset
        self.vlog = get_vlog(verbose)

        # Path Setup
        self.project_path = Path(project.root)
        self.bags_path = bags_dir if bags_dir else self.project_path / "bags"
        self.model_dir = model_dir if model_dir else self.project_path / "models"
        self.ensemble_dir = ensemble_dir if ensemble_dir else self.project_path / "ensemble"
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)


    def load_predictions(self, model_path: Path) -> pd.DataFrame:
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
            predictions (pd.DataFrame | Path | str): Predictions DataFrame or path to predictions parquet file
        Returns:
            dict[str, float | np.ndarray]: Dictionary containing evaluation metrics (AUC, AP, Accuracy, F1, ConfusionMatrix)
        """

        # Make sure we're working with a loaded DataFrame
        match predictions:
            case Path() | str():
                predictions = self.load_predictions(Path(predictions))
            case pd.DataFrame():
                pass
        
        y_true = predictions["y_true"].astype(int)
        num_classes = len(y_true.unique())

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
        generate_attention_heatmaps: bool = False
    ) -> None:
        """Evaluates all models inside `self.model_dir` and writes the results to `self.ensemble_dir`

        Args:
            generate_attention_heatmaps (bool, optional): Whether to generate attention heatmaps. Defaults to False.
        """

        model_paths = [subdir for subdir in self.model_dir.iterdir() if subdir.is_dir()]

        if not model_paths:
            self.vlog(f"No model directories found in {self.model_dir}", LogLevel.WARNING)
            return
        
        for model_idx, model_path in enumerate(model_paths):
            self.vlog(f"Evaluating model {model_idx+1}/{len(model_paths)}: {model_path}")

            try:
                eval_mil(
                    weights=str(model_path),
                    bags=str(self.bags_path),
                    dataset=self.dataset,
                    outcomes="label",
                    outdir=str(self.ensemble_dir),
                    attention_heatmaps=generate_attention_heatmaps
                )
                self.vlog("Evaluation complete.\n")
            except Exception as e:
                self.vlog(f"Error evaluating model at {model_path}: {e}", LogLevel.ERROR)
                continue
        

    def create_ensemble_predictions(
        self,
        model_dir: Path | None = None,
        output_path: Path | None = None
    ) -> tuple[pd.DataFrame, dict[str, float | np.ndarray]]:

        if not model_dir:
            model_dir = self.ensemble_dir

        # Try to load predictions from each model that has been evaluated (should all be in ensemble_dir)
        predictions_list: list[pd.DataFrame] = []
        for model_idx, submodel_dir in enumerate([entry for entry in model_dir.iterdir() if entry.is_dir()]):
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
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == ".csv":
                merged.to_csv(output_path, index=False)
            else:
                merged.to_parquet(output_path, index=False)
            self.vlog(f"Ensemble predictions saved to {output_path}")

        return merged, metrics

        
    def compare_models(
        self,
        model_paths: list[Path] | None = None,
        metrics: list[str] = ["Accuracy", "AUC", "F1"]
    ) -> pd.DataFrame:
        """Compare metrics across multiple models
        
        Args:
            model_paths: List of model directories. If None, uses all in model_dir
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with model comparison
        """
        if model_paths is None:
            model_paths = sorted([p for p in self.ensemble_dir.iterdir() if p.is_dir()])
        
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