import os
from pathlib import Path

import pandas as pd
import slideflow as sf
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, roc_auc_score)
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

    # Iterate over subdirectories of the model dir (Ideally these should contain a model each)
    for idx, submodel_dir in enumerate([entry for entry in model_dir.iterdir() if entry.is_dir()]):
        try:
            # Try loading the predictions
            prediction = load_and_validate_predictions(submodel_dir)

            # Add model index to prediction columns for later merging
            pred_columns = [column for column in prediction.columns if column.startswith("y_pred")]
            rename_map = {pred_column: f"{pred_column}_model{idx}" for pred_column in pred_columns}
            prediction = prediction.rename(columns=rename_map)  # Fix: assign back to prediction
            predictions.append(prediction)
            
            vlog(f"Loaded model {idx} predictions with columns: {list(prediction.columns)}")
            
        except Exception as e:
            vlog(f"{submodel_dir}: {e}", LogLevel.WARNING)
            continue

    if not predictions:
        raise ValueError("No valid prediction files found in model directory")

    # Start with the first model's predictions
    merged = predictions[0].copy()
    
    # Merge remaining predictions one by one
    for prediction in predictions[1:]:
        # Get only the prediction columns from this model (avoid merging duplicate slide/y_true)
        pred_cols_only = [column for column in prediction.columns if column.startswith("y_pred")]
        merge_df = prediction[["slide", "y_true"] + pred_cols_only].copy()
        
        # Merge with explicit suffixes to handle any remaining conflicts
        merged = merged.merge(
            merge_df, 
            on=["slide", "y_true"], 
            how="inner",
        )

    # Get all prediction columns for ensemble averaging
    all_pred_cols = [column for column in merged.columns if column.startswith("y_pred") and "model" in column]
    if not all_pred_cols:
        raise ValueError("No prediction columns found for ensembling")

    # For binary classification, assume we want to ensemble the positive class probabilities
    # If we have y_pred0 and y_pred1, use y_pred1 (positive class)
    positive_class_cols = []
    for model_idx in range(len(predictions)):
        # Look for positive class prediction column for this model
        pos_col = None
        for col in all_pred_cols:
            if f"model{model_idx}" in col:
                if "y_pred1" in col or (len([c for c in all_pred_cols if f"model{model_idx}" in c]) == 1):
                    pos_col = col
                    break
        if pos_col:
            positive_class_cols.append(pos_col)
    
    if not positive_class_cols:
        # Fallback: use all prediction columns
        positive_class_cols = all_pred_cols

    # Ensemble soft voting (average model prediction probabilities)
    merged["y_pred_ensemble"] = merged[positive_class_cols].mean(axis=1)
    merged["y_pred_label"] = (merged["y_pred_ensemble"] >= 0.5).astype(int)

    # Compute metrics
    y_true = merged["y_true"].astype(int)
    y_prob = merged["y_pred_ensemble"]
    y_pred = merged["y_pred_label"]

    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Verbose summary (ensure metrics are native Python floats for format_ensemble_summary)
    summary = format_ensemble_summary(
        len(predictions),
        cm,
        float(auc),
        float(ap),
        float(acc)
    )
    vlog(summary)

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".csv":
            merged.to_csv(output_path, index=False)
        else:
            merged.to_parquet(output_path, index=False)
        vlog(f"Ensemble predictions saved to {output_path}")

    return merged, {"AUC": auc, "AP": ap, "Accuracy": acc, "ConfusionMatrix": cm}
