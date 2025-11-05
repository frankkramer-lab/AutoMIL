import os
from pathlib import Path

import numpy as np
import pandas as pd
import slideflow as sf
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from slideflow.mil import eval_mil

from utils import LogLevel, format_ensemble_summary, get_vlog


def load_and_validate_predictions(model_path: Path) -> pd.DataFrame:
    """Load the predictions from a given model path into a Dataframe and validates them

    Args:
        model_path (Path): Path to model directory

    Raises:
        FileNotFoundError: If the model directory does not contain a predictions table
        ValueError: If the predictions table does not contain columns for class prediction probabilities
        ValueError: If the predictions table does not contain the expected base columns (slide id, y_true)

    Returns:
        pd.DataFrame: Predictions table
    """
    # Make sure the file exists
    if not (predictions_path := model_path / "predictions.parquet").exists():
        raise FileNotFoundError(f"{model_path} does not contain a 'predictions.parquet' file")
    predictions = pd.read_parquet(predictions_path)
    
    # Make sure the required columns exist
    columns = [column for column in predictions.columns]
    pred_columns = [column for column in columns if column.startswith("y_pred")]
    if not pred_columns:
        raise ValueError("'predictions.parquet' does not contain the expected prediction columns")
    elif not all(column in columns for column in ["slide", "y_true"]):
        raise ValueError("'predictions.parquet' does not contain the expected base columns")

    return predictions

def evaluate(
    project: sf.Project,
    dataset: sf.Dataset,
    verbose: bool = True
):
    vlog = get_vlog(verbose)
    project_path = Path(project.root)
    bags_path    = project_path / "bags"
    model_dir    = project_path / "models"
    ensemble_dir = Path(project.root) / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    for model_idx, model_path in enumerate(model_dir.iterdir()):
        vlog(f"Predicting with model {model_idx+1} ({model_path})")
        vlog(f"Saving evaluation to {ensemble_dir / model_path.name}")
        eval_mil(
            weights=str(model_path),
            bags=str(bags_path),
            dataset=dataset,
            outcomes="label",
            outdir=str(ensemble_dir)
        )


def ensemble_predictions(
    model_dir: Path,
    output_path: Path | None = None,
    verbose: bool = True
):
    """
    Combine predictions from multiple trained MIL models (e.g., folds)
    using soft voting (average of predicted probabilities).

    Args:
        model_dir (Path): Directory containing subfolders for each trained model,
                          each with a `predictions.parquet` file.
        output_path (Path | None): Optional path to save the ensemble results as a .csv or .parquet file.
        verbose (bool): Whether to print progress messages.
    """
    vlog = get_vlog(verbose)
    predictions: list[pd.DataFrame] = []

    # === Loading Predictions ===
    # Iterate over subdirectories of the model dir (Ideally these should contain a model each)
    for idx, submodel_dir in enumerate([entry for entry in model_dir.iterdir() if entry.is_dir()]):
        try:
            # Try loading the predictions
            prediction = load_and_validate_predictions(submodel_dir)

            # Add model index to prediction columns for later merging
            pred_columns = [column for column in prediction.columns if column.startswith("y_pred")]
            rename_map = {pred_column: f"{pred_column}_model{idx}" for pred_column in pred_columns}
            prediction = prediction.rename(columns=rename_map)
            predictions.append(prediction)
            
            vlog(f"Loaded model {idx} predictions with columns: {list(prediction.columns)}")
            
        except Exception as e:
            vlog(f"{submodel_dir}: {e}", LogLevel.WARNING)
            continue

    if not predictions:
        raise ValueError("No valid prediction files found in model directory")

    # === Merging Predictions ===
    # Start with the first model's predictions
    merged = predictions[0].copy()

    for prediction in predictions[1:]:
        # y_pred'x' column holds the prediction probabiliy for class 'x'
        pred_columns = [column for column in prediction.columns if column.startswith("y_pred")]
        to_merge = prediction[["slide", "y_true"] + pred_columns].copy()

        # Merge on identifiers (slide id and label)
        merged = merged.merge(
            to_merge, 
            on=["slide", "y_true"], 
            how="inner",
        )
    
    # === Ensemble ===
    # Get all prediction columns from merged table
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
    for class_idx in unique_classes:
        class_prediction_columns[class_idx] = [
            column for column in all_pred_columns
            if column.startswith(f"y_pred{class_idx}_")
        ]
    
    # Calculate ensemble (average) probabilities
    ensemble_probs = {}
    for class_idx in unique_classes:
        # If we have predictions for this class
        if class_prediction_columns[class_idx]:
            # Calculate average of prediction probs
            ensemble_probs[f"y_pred{class_idx}_ensemble"] = merged[
                class_prediction_columns[class_idx]
            ].mean(axis=1)
        else:
            vlog(f"No prediction columns found for class {class_idx}")
            # Dummy column
            ensemble_probs[f"y_pred{class_idx}_ensemble"] = 0.0

    # Add ensemble probabilities to DataFrame
    for column, probability in ensemble_probs.items():
        merged[column] = probability

    # Retrieve probability matrix
    ensemble_probability_columns = [f"y_pred{class_idx}_ensemble" for class_idx in unique_classes]
    prob_matrix = merged[ensemble_probability_columns].values

    # Transform probabilities to distinct prediction (largest probability)
    predicted_classes = np.argmax(prob_matrix, axis=1)
    merged["y_pred_label"] = predicted_classes

    # Collect ground truth and prediction for metric calculation
    y_true = merged["y_true"].astype(int)
    y_pred = merged["y_pred_label"].astype(int)

    # === Calculating Metrics ===
    # Binary
    if n_classes == 2:
        y_probs = prob_matrix[:, 1] # label 1 probabilites

        auc = roc_auc_score(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred)

    # Multiclass
    else:
        auc = roc_auc_score(y_true, prob_matrix, multi_class="ovr", average="macro")

        # Calculate AP for each class
        ap_scores = []
        for class_idx in unique_classes:
            ap_class = average_precision_score(y_true, prob_matrix[:, class_idx])
            ap_scores.append(ap_class)
        ap = np.mean(ap_scores) if ap_scores else 0.0

        f1 = f1_score(y_true, y_pred, average="macro")
    
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # === Summary ===
    summary = format_ensemble_summary(
        len(predictions),
        cm,
        float(auc),
        float(ap),
        float(acc),
        float(f1)
    )
    vlog(summary)

    # === TODO | Plots ===

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".csv":
            merged.to_csv(output_path, index=False)
        else:
            merged.to_parquet(output_path, index=False)
        vlog(f"Ensemble predictions saved to {output_path}")

    metrics = {
        "AUC": auc, 
        "AP": ap, 
        "Accuracy": acc, 
        "F1": f1,
        "ConfusionMatrix": cm
    }
    return merged, metrics