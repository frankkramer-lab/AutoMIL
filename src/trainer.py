from __future__ import annotations

from functools import cached_property
from os.path import isdir, join
from pathlib import Path

import pandas as pd
import slideflow as sf
import torch
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.learner import Learner
from slideflow.mil import build_fastai_learner, mil_config
from slideflow.mil._params import TrainerConfig
from slideflow.mil.eval import generate_attention_heatmaps
from slideflow.mil.train import _fastai, _log_mil_params
from slideflow.util import path_to_name

from estimator import adjust_batch_size, estimate_model_size
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
        lr: float = LEARNING_RATE,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        k: int = 3,
        enable_early_stopping: bool = True,
        early_stop_patience: int = 10,
        early_stop_monitor: str = "valid_loss",
        early_stop_mode: str = "min",
        attention_heatmaps: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize trainer with configuration and compute optimal settings
        
        Args:
            bags_path: Path to feature bags directory
            project: Slideflow project instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model: Model type to train
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Initial batch size (will be optimized)
            k: Number of folds for cross-validation
            fit_one_cycle: Whether to use one-cycle learning rate policy
            enable_early_stopping: Whether to enable early stopping
            early_stop_patience: Patience for early stopping
            early_stop_monitor: Metric to monitor for early stopping
            early_stop_mode: Mode for early stopping ('min' or 'max')
            verbose: Whether to print verbose messages
        """
        self.bags_path = bags_path
        self.project = project
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.initial_batch_size = batch_size
        self.attention_heatmaps = attention_heatmaps
        self.k = k
        self.enable_early_stopping = enable_early_stopping
        self.early_stop_patience = early_stop_patience
        self.early_stop_monitor = early_stop_monitor
        self.early_stop_mode = early_stop_mode

        self.vlog = get_vlog(verbose)

    @cached_property
    def num_slides(self) -> int:
        return get_num_slides(self.train_dataset) + get_num_slides(self.val_dataset)
    
    @cached_property
    def bag_avg(self) -> int:
        return get_bag_avg_and_num_features(self.bags_path)[0]
    
    @cached_property
    def num_features(self) -> int:
        return get_bag_avg_and_num_features(self.bags_path)[1]
    
    @cached_property
    def adjusted_batch_size(self) -> int:
        """Get the adjusted batch size"""
        return self._compute_optimal_batch_size()

    @cached_property
    def estimated_size_mb(self) -> float:
        """Get the estimated model size in MB"""
        return self._estimate_model_size()

    @cached_property
    def config(self) -> TrainerConfig:
        """Get the training configuration"""
        return self._build_config()

    @cached_property
    def device(self) -> torch.device:
        """Get the device used for training"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(
        self,
        outcomes: str = "label",
        exp_label: str | None = None,
        outdir: Path | str | None = None,
    ) -> Learner:
        """Train MIL model with comprehensive evaluation (Slideflow-style)
        
        This method mimics Slideflow's _train_mil function, providing:
        - Model training with FastAI
        - Automatic validation predictions
        - Metrics calculation and export
        - Attention export
        - Optional heatmap generation
        
        Args:
            outcomes: Column name for outcomes in annotations
            exp_label: Experiment label for this training run
            outdir: Output directory for saving models and results
            
        Returns:
            Trained FastAI Learner instance
        """
        # Set defaults
        exp_label = exp_label or f"{self.model.model_name}_training"
        outdir = Path(outdir or Path(self.project.root) / "models" / exp_label)
        outdir.mkdir(parents=True, exist_ok=True)
        
        self.vlog(f"Starting MIL training: {exp_label}")
        self.vlog(f"Output directory: {outdir}")
        
        # Prepare validation bags
        val_bags = self._prepare_validation_bags()
        
        # Build learner with shape information
        result = build_fastai_learner(
            self.config,
            self.train_dataset,
            self.val_dataset,
            outcomes,
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
        self._log_mil_params(outcomes, learner, n_in, n_out, str(outdir))
        
        # Add custom callbacks if needed
        callbacks = self._setup_callbacks()
        for callback in callbacks:
            learner.add_cb(callback)
        
        # Train the model using Slideflow's method
        self.vlog(
            f"Starting training: {exp_label} "
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
            outcomes=outcomes,
            bags=val_bags,
            attention=True
        )

        if isinstance(df, pd.DataFrame):
            # Save predictions
            pred_out = outdir / 'predictions.parquet'
            df.to_parquet(pred_out)
            self.vlog(f"Predictions saved to {pred_out}")
        
            # Calculate and display metrics
            self._run_metrics(df, outcomes, str(outdir))
            
            # Export attention arrays
            if attention and isinstance(attention, dict):
                self._export_attention(attention, val_bags, str(outdir))
            
            # Generate attention heatmaps
            if attention and isinstance(attention, dict) and self.attention_heatmaps:
                self._generate_heatmaps(val_bags, attention, str(outdir))
        else:
            self.vlog("Unable to generate predictions; skipping metrics and attention export.")

        model = learner.model
        # Actual inference memory tracking
        # Create dummy input
        dummy_input = torch.randn(BATCH_SIZE, self.bag_avg, self.num_features).cuda()
        lens = torch.full((BATCH_SIZE,), self.bag_avg, dtype=torch.int32).cuda()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input, lens)
        self.actual_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        self.vlog(f"Training completed: {exp_label}")
        return learner

    def train_k_fold(
        self,
        outcomes: str = "label",
        base_exp_label: str | None = None,
        base_outdir: Path | str | None = None
    ) -> list[Learner]:
        """Train k-fold cross validation with comprehensive evaluation
        
        Args:
            outcomes: Column name for outcomes in annotations
            base_exp_label: Base experiment label (will append fold number)
            base_outdir: Base output directory
            
        Returns:
            List of trained learners, one per fold
        """
        base_outdir = Path(base_outdir or Path(self.project.root) / "models")
        num_models = len([dir for dir in base_outdir.iterdir() if isdir(dir)])
        base_exp_label = base_exp_label or f"{self.model.model_name}_{num_models + 1}"
        
        learners = []
        
        for fold in range(self.k):
            self.vlog(f"=" * 50)
            self.vlog(f"Training fold {fold + 1}/{self.k}")
            self.vlog(f"=" * 50)
            
            # Create fold-specific paths and labels
            fold_label = f"{base_exp_label}_fold{fold}"
            fold_outdir = base_outdir / fold_label
            
            # Train this fold
            learner = self.train(
                outcomes=outcomes,
                outdir=fold_outdir
            )
            
            learners.append(learner)
        
        self.vlog(f"Completed {self.k}-fold training")
        return learners

    def summary(self) -> None:
        """Print a summary of the trainer configuration"""
        from tabulate import tabulate

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
        
        self.vlog("[bold underline]Trainer Configuration:[/]")
        self.vlog(tabulate(table, tablefmt="fancy_outline"))

    # === Private Methods === #
    
    def _prepare_validation_bags(self) -> list:
        """Prepare validation bags like Slideflow does"""
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
        """Log MIL parameters like Slideflow does"""
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
        """Calculate and display metrics like Slideflow does"""
        from slideflow.mil import utils

        # Rename columns for metrics calculation
        utils.rename_df_cols(df, outcomes, categorical=self.config.is_classification(), inplace=True)
        
        # Run metrics using Slideflow's method
        self.config.run_metrics(df, level='slide', outdir=outdir)
    
    def _export_attention(self, attention: dict, val_bags: list, outdir: str) -> None:
        """Export attention arrays like Slideflow does"""
        from slideflow.mil import utils
        
        attention_dir = join(outdir, 'attention')
        bag_names = [path_to_name(b) for b in val_bags]
        
        # Convert attention dict to list of arrays
        attention_arrays = list(attention.values())
        utils._export_attention(attention_dir, attention_arrays, bag_names)
        self.vlog(f"Attention arrays exported to {attention_dir}")
    
    def _generate_heatmaps(self, val_bags: list, attention: dict, outdir: str) -> None:
        """Generate attention heatmaps like Slideflow does"""
        heatmap_dir = join(outdir, 'heatmaps')
        generate_attention_heatmaps(
            outdir=heatmap_dir,
            dataset=self.val_dataset,
            bags=val_bags,
            attention=list(attention.values()),
        )
        self.vlog(f"Attention heatmaps generated in {heatmap_dir}")
    
    def _compute_optimal_batch_size(self) -> int:
        """Compute VRAM-optimal batch size from existing bags"""
        adjusted_batch_size = adjust_batch_size(
            self.model,
            self.initial_batch_size,
            self.num_slides,
            self.num_features,
            self.bag_avg,
        )
        
        self.vlog(
            f"Adjusted batch size to [{INFO_CLR}]{adjusted_batch_size}[/] "
            f"(tiles/bag={self.bag_avg}, dim={self.num_features})"
        )
        
        return adjusted_batch_size
    
    def _estimate_model_size(self) -> float:
        """Estimate model memory usage in MB"""
        return estimate_model_size(
            model=self.model,
            batch_size=self.adjusted_batch_size,
            bag_size=self.bag_avg,
            input_dim=self.num_features,
        )
    
    def _build_config(self) -> TrainerConfig:
        """Build the training configuration"""
        cfg = mil_config(
            model=self.model.model_name,
            trainer="fastai",
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.adjusted_batch_size,
        )
        
        if not isinstance(cfg, TrainerConfig):
            raise ValueError("Got unexpected return type from mil_config")
            
        return cfg
    
    def _setup_callbacks(self) -> list:
        """Setup training callbacks"""
        callbacks = []
        
        # Early stopping
        if self.enable_early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=self.early_stop_monitor,
                    patience=self.early_stop_patience,
                )
            )
        
        return callbacks