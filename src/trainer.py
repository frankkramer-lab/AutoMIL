from __future__ import annotations

from functools import cached_property
from os.path import join
from pathlib import Path
from typing import cast

import pandas as pd
import slideflow as sf
import torch
import torch.nn as nn
from fastai.callback.core import Callback
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.learner import Learner
from slideflow.mil import _train_mil, build_fastai_learner, mil_config, utils
from slideflow.mil._params import TrainerConfig
from slideflow.mil.eval import generate_attention_heatmaps
from slideflow.mil.train import _fastai, _log_mil_params
from slideflow.util import path_to_name

from estimator import adjust_batch_size, estimate_model_size
from model import ModelManager
from utils import (BATCH_SIZE, EPOCHS, INFO_CLR, LEARNING_RATE, ModelType,
                   get_bag_avg_and_num_features, get_num_slides, get_vlog)


class Trainer:
    """Handles MIL model training with automatic batch size optimization and early stopping"""
    
    def __init__(
        self,
        bags_path: Path,
        project: sf.Project,
        train_dataset: sf.Dataset,
        val_dataset: sf.Dataset,
        model: ModelType,
        model_outdir: Path | None = None,
        lr: float = LEARNING_RATE,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        k: int = 3,
        enable_early_stopping: bool = True,
        early_stop_patience: int = 10,
        early_stop_monitor: str = "valid_loss",
        attention_heatmaps: bool = True,
        additional_callbacks: list[Callback] | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize a Trainer Instance. Sets up training configuration and optimizes hyperparameters.

        Args:
            bags_path (Path): Path to feature bags directory
            project (sf.Project): Slideflow project instance
            train_dataset (sf.Dataset): Training dataset
            val_dataset (sf.Dataset): Validation dataset
            model (ModelType): Model to use
            model_outdir (Path | None, optional): Output directory for trained models. If none, will use `project_root / models`. Defaults to None.
            lr (float, optional): Learning rate. Defaults to LEARNING_RATE.
            epochs (int, optional): (Maximum number of) Epochs to train for. Defaults to EPOCHS.
            batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
            k (int, optional): Number of folds to train. Defaults to 3.
            enable_early_stopping (bool, optional): Whether to use early stopping. Adds an ealry stopping callback to the FastAI Learner. Defaults to True.
            early_stop_patience (int, optional): Number of epochs without performance improvement before early stopping kicks in. Defaults to 10.
            early_stop_monitor (str, optional): Metric to monitor for early stopping. Defaults to "valid_loss".
            attention_heatmaps (bool, optional): Whether to generate attention heatmaps. Defaults to True.
            verbose (bool, optional): Whether to print verbose messages. Defaults to True.
        """
        self.bags_path = bags_path
        self.project = project
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.model_outdir = model_outdir or Path(self.project.root) / "models"
        self.lr = lr
        self.epochs = epochs
        self.initial_batch_size = batch_size
        self.attention_heatmaps = attention_heatmaps
        self.additional_callbacks = additional_callbacks
        self.k = k
        self.enable_early_stopping = enable_early_stopping
        self.early_stop_patience = early_stop_patience
        self.early_stop_monitor = early_stop_monitor

        self.vlog = get_vlog(verbose)

        # Hyperparameter validation
        self.model_manager = ModelManager(self.model)
        suggestions = self.model_manager.validate_hyperparameters(
            self.lr,
            self.initial_batch_size,
            self.bag_avg
        )

        for suggestion, value in suggestions.items():
            self.vlog(
                f"[yellow]Warning:[/] {suggestion} value out of bounds for model "
                f"[cyan]{self.model_manager.model_class.__name__}[/]. "
                f"Suggested value: [cyan]{value}[/]"
            )
            setattr(self, suggestion, value)

    @cached_property
    def num_classes(self) -> int:
        if self.train_dataset.annotations is not None:
            return self.train_dataset.annotations["label"].nunique()
        elif self.val_dataset.annotations is not None:
            return self.val_dataset.annotations["label"].nunique()
        else:
            return 2  # Assume binary classification as fallback

    @cached_property
    def num_slides(self) -> int:
        """Number of slides in training and validation dataset"""
        return get_num_slides(self.train_dataset) + get_num_slides(self.val_dataset)
    
    @cached_property
    def bag_avg(self) -> int:
        """Average number of tiles per bag"""
        return get_bag_avg_and_num_features(self.bags_path)[0]
    
    @cached_property
    def num_features(self) -> int:
        """Average number of features per tile"""
        return get_bag_avg_and_num_features(self.bags_path)[1]
    
    @cached_property
    def adjusted_batch_size(self) -> int:
        """Optimal batch size adjusted for VRAM constraints"""
        return self._compute_optimal_batch_size()

    @cached_property
    def estimated_size_mb(self) -> float:
        """Estimated model size in MB"""
        return self._estimate_model_size()

    @cached_property
    def config(self) -> TrainerConfig:
        """FastAI trainer configuration"""
        return self._build_config()

    @cached_property
    def device(self) -> torch.device:
        """The device to use for training"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def callbacks(self) -> list[Callback]:
        """List of FastAI Callbacks to use during training"""
        return self._setup_callbacks(self.additional_callbacks)

    # === Public Methods === #
    def train(
        self,
        model_label_override: str | None = None,
    ) -> Learner:
        """Fit a model of type `self.model` to `train_dataset` using `val_dataset` for validation.

        Note:
            This method closely emulates the internal training workflow of Slideflow's `slideflow.mil._train_mil_mode` method,
            With modifications that allow the passing of custom callbacks in order to enable early stopping,
            additional type safety checks and additonal logging functionality.

        Args:
            model_label_override (str | None, optional): String override for the directory name in which the model will be saved. If None
            slideflows default naming sheme will be used ('{index}_{model_type}_{label_column}'). Defaults to None.

        Returns:
            Learner: FastAI Learner object containing the trained model
        """
        self._debug_dataset_labels()

        # Determine output directory
        if model_label_override:
            outdir = self.model_outdir / model_label_override
        else:
            outdir = Path(self.config.prepare_training("label", exp_label=None, outdir=str(self.model_outdir)))
        self.vlog(f"Output directory: {outdir}")
        
        # Prepare validation bags
        val_bags = self._prepare_validation_bags()
        
        # Build learner with shape information
        result = build_fastai_learner(
            self.config,
            self.train_dataset,
            self.val_dataset,
            outcomes="label",
            bags=str(self.bags_path),
            outdir=str(outdir),
            device=self.device,
            return_shape=True
        )

        # with return_shape=True, result is a tuple
        if isinstance(result, tuple):
            learner, (n_in, n_out) = result
        else:
            learner = result
            n_in, n_out = 0, 0  # Shape info not available
        
        # Save MIL parameters (like Slideflow does)
        self._log_mil_params("label", learner, n_in, n_out, str(outdir))
        
        # Add custom callbacks if needed
        callbacks = self._setup_callbacks()
        for callback in callbacks:
            learner.add_cb(callback)
        
        # Train the model using Slideflow's method
        self.vlog(
            f"Starting training: {self.model.model_name} "
            f"(epochs={self.epochs}, batch_size={self.adjusted_batch_size})"
        )
        _fastai.train(learner, self.config)
        
        # Generate validation predictions with attention
        self.vlog("Generating validation predictions...")
        from slideflow.mil import predict_mil
        df, attention = predict_mil(
            learner.model,
            dataset=self.val_dataset,
            config=self.config,
            outcomes="label",
            bags=val_bags,
            attention=True
        )

        # Saving predictions, calculating metrics, exporting attention, and generating heatmaps
        # Really only feasible if we get a sensible return dataframe from `predict_mil`
        if isinstance(df, pd.DataFrame):
            # Save predictions
            pred_out = outdir / 'predictions.parquet'
            df.to_parquet(pred_out)
            self.vlog(f"Predictions saved to {pred_out}")
        
            # Calculate and display metrics
            self._run_metrics(df, "label", str(outdir))
            
            # Export attention arrays
            if attention and isinstance(attention, dict):
                self._export_attention(attention, val_bags, str(outdir))
            
            # Generate attention heatmaps
            if attention and isinstance(attention, dict) and self.attention_heatmaps:
                self._generate_heatmaps(val_bags, attention, str(outdir))
        else:
            self.vlog("Unable to generate predictions; skipping metrics and attention export.")

        # Measure actual memory usage during inference
        # This can later be compared to the estimated model size
        dummy_input = torch.randn(BATCH_SIZE, self.bag_avg, self.num_features).cuda()
        lens = torch.full((BATCH_SIZE,), self.bag_avg, dtype=torch.int32).cuda()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = learner.model(dummy_input, lens)
        self.actual_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        self.vlog(f"Training completed: {self.model.model_name}")
        return learner

    def train_k_fold(self, base_model_label_override: str | None = None) -> list[Learner]:
        """Performs k-fold training on `train_dataset`, creating and saving `self.k` trained MIL models

        Args:
            base_model_label_override (str | None, optional): Base string override for the directory name in which the model will be saved.
            The k-fold will be appended to the directory name. If None, slideflows default naming sheme will be used
            ('{index}_{model_type}_{label_column}'). Defaults to None.
        Returns:
            list[Learner]: List of trained FastAI Learners
        """
        outdir = self.model_outdir
        if base_model_label_override:
            outdir = outdir / base_model_label_override
        self.vlog(f"K-Fold output directory: {outdir}")
        
        learners = []
        for fold in range(self.k):
            self.vlog(f"=" * 50)
            self.vlog(f"Training fold {fold + 1}/{self.k}")
            self.vlog(f"=" * 50)
            
            # Create fold-specific paths and labels
            if base_model_label_override:
                fold_label = f"{base_model_label_override}_fold{fold}"
            else:
                fold_label = None
            
            # Train this fold
            learner = self.train(
                model_label_override=fold_label
            )
            learners.append(learner)
        
        self.vlog(f"Completed {self.k}-fold training")
        return learners

    def summary(self) -> None:
        """Print a summary of the trainer configuration
        
        Example:
            ```
            ╒═════════════════════════╤═══════════════╕                 
            │ Model Type              │ attention_mil │                 
            │ Learning Rate           │ 1e-04         │                 
            │ Epochs                  │ 300           │                 
            │ Initial Batch Size      │ 32            │                 
            │ Adjusted Batch Size     │ 100           │                 
            │ Estimated Model Size    │ 400.48 MB     │                 
            │ Actual Inference Memory │ 300.29 MB     │                 
            │ Number of Slides        │ 341           │                 
            │ Average Bag Size        │ 654           │                 
            │ Feature Dimensions      │ 768           │                 
            │ K-Fold                  │ 3             │                 
            │ Early Stopping          │ True          │                 
            │ Attention Heatmaps      │ True          │                 
            │ Device                  │ cuda          │                 
            ╘═════════════════════════╧═══════════════╛    
            ```
        """
        from tabulate import tabulate

        # Safe way to handle actual memory attribute
        # Since it may not have been set yet (Only after training)
        actual_mem_mb = getattr(self, 'actual_mem_mb', None)
        if actual_mem_mb is not None:
            actual_mem_str = f"{actual_mem_mb:.2f} MB"
        else:
            actual_mem_str = "N/A"

        table = [
            ("Model Type", self.model.model_name),
            ("Learning Rate", f"{self.lr:.0e}"),
            ("Epochs", self.epochs),
            ("Initial Batch Size", self.initial_batch_size),
            ("Adjusted Batch Size", self.adjusted_batch_size),
            ("Estimated Model Size", f"{self.estimated_size_mb:.2f} MB"),
            ("Actual Inference Memory", actual_mem_str),
            ("Number of Slides", self.num_slides),
            ("Average Bag Size", self.bag_avg),
            ("Feature Dimensions", self.num_features),
            ("K-Fold", self.k),
            ("Early Stopping", self.enable_early_stopping),
            ("Attention Heatmaps", self.attention_heatmaps),
            ("Device", str(self.device)),
        ]
        
        self.vlog("[bold underline]Trainer Summary:[/]")
        self.vlog(tabulate(table, tablefmt="fancy_outline"))

    # === Internals === #
    def _debug_dataset_labels(self) -> None:
        """Debug helper to inspect dataset labels"""
        train_ann = self.train_dataset.annotations
        val_ann = self.val_dataset.annotations
        
        if train_ann is not None and val_ann is not None:
            self.vlog(f"Train labels: {train_ann['label'].unique()}")
            self.vlog(f"Train label types: {[type(x) for x in train_ann['label'].unique()]}")
            self.vlog(f"Val labels: {val_ann['label'].unique()}")  
            self.vlog(f"Val label types: {[type(x) for x in val_ann['label'].unique()]}")
        else:
            self.vlog("WARNING: One or both datasets have no annotations")

    
    def _prepare_validation_bags(self) -> list:
        """Simple helper method that emulates how slideflow generates validation feature bags.

        Note:
            See `slideflow.mil._train_mil` for reference.
        Returns:
            list: list of validation bag paths
        """
        val_bags = self.val_dataset.get_bags(str(self.bags_path))
        self.vlog(f"Found {len(val_bags)} validation bags")
        return val_bags.tolist()
    
    def _log_mil_params(
        self,
        outcomes: str,
        learner: Learner,
        n_in: int,
        n_out: int,
        outdir: str
    ) -> None:
        """Simple helper method that emulates how slideflow logs and saves MIL parameters
        
        Note:
            See `slideflow.mil._train_mil` for reference.
        Args:
            outcomes (str): Label column name
            learner (Learner): FastAI Learner object
            n_in (int): input feature dimensions
            n_out (int): output dimensions / number of classes
            outdir (str): Output directory
        """
        # Attempt to read unique categories from learner
        if hasattr(learner.dls.train_ds, 'encoder'):
            encoder = learner.dls.train_ds.encoder
            if encoder is not None:
                unique = encoder.categories_[0].tolist()
            else:
                unique = None
        else:
            unique = None
        
        # Use Slideflow's internal logging function
        _log_mil_params(self.config, outcomes, unique, str(self.bags_path), n_in, n_out, outdir)
    
    def _run_metrics(self, df: pd.DataFrame, outcomes: str, outdir: str) -> None:
        """Simple helper method that emulates how slideflow caculates and logs metrics.
        
        Note:
            See `slideflow.mil._train_mil` for reference.

        Args:
            df (pd.DataFrame): DataFrame containing predictions
            outcomes (str): Label column name
            outdir (str): Output directory
        """
        # Rename columns for metrics calculation
        utils.rename_df_cols(df, outcomes, categorical=self.config.is_classification(), inplace=True)
        
        # Run metrics using Slideflow's method
        self.config.run_metrics(df, level='slide', outdir=outdir)
    
    def _export_attention(self, attention: dict, val_bags: list, outdir: str) -> None:
        """Simple helper method that emulates how slideflow exports attention arrays.
        
        Note:
            See `slideflow.mil._train_mil` for reference.
        Args:
            attention (dict): Dictionary mapping bag paths to attention arrays
            val_bags (list): List of validation bag paths
            outdir (str): Output directory
        """
        attention_dir = join(outdir, 'attention')
        bag_names = [path_to_name(b) for b in val_bags]
        
        # Convert attention dict to list of arrays
        attention_arrays = list(attention.values())
        utils._export_attention(attention_dir, attention_arrays, bag_names)
        self.vlog(f"Attention arrays exported to {attention_dir}")
    
    def _generate_heatmaps(self, val_bags: list, attention: dict, outdir: str) -> None:
        """Generate a heatmap using slideflow

        Args:
            val_bags (list): List of validation bag paths
            attention (dict): Dictionary mapping bag paths to attention arrays
            outdir (str): Output directory
        """
        heatmap_dir = join(outdir, 'heatmaps')
        generate_attention_heatmaps(
            outdir=heatmap_dir,
            dataset=self.val_dataset,
            bags=val_bags,
            attention=list(attention.values()),
        )
        self.vlog(f"Attention heatmaps generated in {heatmap_dir}")
    
    def _compute_optimal_batch_size(self) -> int:
        """Compute VRAM-Optimal batch size based on estimated model memory usage and free memory

        Returns:
            int: adjusted batch size
        """
        adjusted_batch_size = adjust_batch_size(
            self.model,
            self.initial_batch_size,
            self.num_slides,
            self.num_features,
            self.bag_avg,
        )

        # Ensure we do not exceed model-specific maximum batch size
        adjusted_batch_size = min(
            self.model_manager.config.max_batch_size,
            adjusted_batch_size
        )
        
        self.vlog(
            f"Adjusted batch size to [{INFO_CLR}]{adjusted_batch_size}[/] "
            f"(tiles/bag={self.bag_avg}, dim={self.num_features})"
        )
        
        return adjusted_batch_size
    
    def _estimate_model_size(self) -> float:
        """Estimate model size (reserved memory) in MB

        Returns:
            float: Estimated model size in MB
        """
        return estimate_model_size(
            model_type=self.model,
            batch_size=self.adjusted_batch_size,
            bag_size=self.bag_avg,
            input_dim=self.num_features,
            num_classes=self.num_classes
        )
    
    def _build_config(self) -> TrainerConfig:
        """Builds a MIL model configuration using slideflow's `mil_config` method

        Returns:
            TrainerConfig: TrainerConfig object
        """
        cfg = mil_config(
            model=self.model.model_name,
            trainer="fastai",
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.adjusted_batch_size,
        )
        
        # Casting because mil_config should always return a TrainerConfig
        return cast(TrainerConfig, cfg)
    
    def _setup_callbacks(self, additional_callbacks: list[Callback] | None = None) -> list[Callback]:
        """Sets up callbacks for the FastAI Learner, including an EarlyStopping Callback

        Args:
            additional_callbacks (list[Callback]): A list of additional callbacks to include

        Returns:
            list: List of callbacks
        """
        callbacks = []
        
        # Early stopping
        if self.enable_early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=self.early_stop_monitor,
                    patience=self.early_stop_patience,
                )
            )
        callbacks.extend(additional_callbacks if additional_callbacks else [])
        
        return callbacks