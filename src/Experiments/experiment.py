"""
Experiment Base Class

Generic experiment base class from which all specific experiment types inherit.
"""
import inspect
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slideflow as sf
import torch
from fastai.learner import Learner
from matplotlib.figure import Figure
from slideflow.mil import mil_config, train_mil

from estimator import estimate_dynamic_vram_usage
from utils import (BATCH_SIZE, EPOCHS, ERROR_CLR, INFO_CLR, LEARNING_RATE,
                   SUCCESS_CLR, ModelType, get_bag_avg_and_num_features,
                   get_num_slides, get_vlog)


class Experiment(ABC):
    """Abstract experiment base class"""

    def __init__(
        self, 
        project_dir: Path, 
        experiment_name: str,
        results_dir: Optional[Path] = None,
        model_type: ModelType = ModelType.Attention_MIL
    ):
        """
        Initialize the experiment.
        
        Args:
            project_dir: Path to the project directory
            experiment_name: Name of the experiment (used for results subdirectory)
            results_dir: Directory to save results and plots (defaults to project_dir/experiment_name)
        """
        self.project_dir = project_dir
        self.experiment_name = experiment_name
        self.results_dir = results_dir or project_dir / experiment_name
        self.results_dir.mkdir(exist_ok=True)
        self.model_type = model_type
        
        # Storage for experiment results
        self.performance_metrics: Dict[Any, Dict] = {}

    @abstractmethod
    def get_parameter_grid(self) -> List[Dict[str, float | int]]:
        """Define a parameter grid for the experiment.

        This grid will later be used to run model training with different parameters.

        Example:
            [
                {'batch_size': 2},
                {'batch_size': 4},
                {'batch_size': 8, 'learning_rate': 0.001},
                {'batch_size': 16, 'learning_rate': 0.01, epochs: 50},
                ...
            ]

        Returns:
            Grid of parameters to test.
        """
        pass

    @abstractmethod
    def create_model_config(self, model_type: ModelType, **params) -> Any:
        """
        Create model configuration for given parameters.
        
        Args:
            **params: Parameter values for this experiment run
            
        Returns:
            Model configuration object
        """
        pass

    def extract_fold_metrics(
        self,
        learner: Learner,
        fold_idx: int,
        **context
    ) -> dict[str, int | float]:
        """
        Extract metrics from the trained model for a single fold.
        
        Args:
            learner: Trained model/learner object
            fold_idx: Current fold index
            **context: Additional context (training_time, actual_memory_mb, batch_size, etc.)
            
        Returns:
            Dictionary containing fold-specific metrics
        """
        # Extract validation accuracy and loss if available
        val_accuracy = None
        val_loss = None
        
        if hasattr(learner, 'validate'):
            val_metrics = learner.validate()
            if val_metrics is not None and len(val_metrics) >= 2:
                val_loss = val_metrics[0].item() if isinstance(val_metrics[0], torch.Tensor) else val_metrics[0]
                val_accuracy = val_metrics[1].item() if isinstance(val_metrics[1], torch.Tensor) else val_metrics[1]
        
        fold_result = {
            'fold': fold_idx + 1,
            'batch_size': context.get('batch_size'),
            'training_time': context.get('training_time'),
            'estimated_memory_mb': context.get('estimated_memory_mb'),
            'actual_memory_mb': context.get('actual_memory_mb'),
            'val_accuracy': val_accuracy,
            'val_loss': val_loss
        }
        
        return fold_result

    def create_experiment_plots(self, return_figures: bool = True) -> list[Figure] | None:
        """
        Create experiment-specific plots.

        Raises:
            ValueError: If no performance metrics are available (i.e., experiments have not been run yet)
        
        Returns:
            List of matplotlib figures
        """
        if not self.performance_metrics:
            raise ValueError("No performance metrics available. Run experiments first.")

        def get_return_type(func: Callable) -> Any:
            """Get the return type annotation (as class) of a function"""
            signature = inspect.signature(func)
            return signature.return_annotation

        # Collecting and executing all class methods that:
        # 1. Start with '_plot_'
        # 2. Are callable
        # 3. return a matplotlib Figure
        from typing import cast

        # Dirty trick to satisfy the type checker
        figures = cast(
            list[Figure],
            [
                plot_method()
                for method_name in dir(self)
                if (
                    method_name.startswith('_plot_')
                    and callable((plot_method := getattr(self, method_name)))
                    and get_return_type(plot_method) == Figure
                )
            ]
        )

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Save figures
        for index, figure in enumerate(figures):
            plot_path = self.results_dir / f"{self.experiment_name}_plot_{index + 1}.png"
            figure.savefig(
                plot_path,
                bbox_inches='tight',
                dpi=300
            )
        
        if return_figures: return figures

    def _get_parameter_key(self, params: dict[str, int | float]) -> str:
        """
        Generate a string key from parameter dictionary.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            String representation of parameters
        """
        param_strs = [f"{k}_{v}" for k, v in sorted(params.items())]
        return "_".join(param_strs)
    
    def run_experiment(
        self,
        project: sf.Project,
        dataset: sf.Dataset,
        model_type: ModelType,
        k_folds: int = 3,
        verbose: bool = True,
        **kwargs
    ) -> dict[str, dict]:
        
        vlog = get_vlog(verbose)
        parameter_grid = self.get_parameter_grid()

        vlog(f"Starting {self.experiment_name} with [{INFO_CLR}]{len(parameter_grid)}[/] parameter combinations")

        for index, params in enumerate(parameter_grid):
            param_key = self._get_parameter_key(params)
            vlog(f"\n--- Experiment {index+1}/{len(parameter_grid)}: {param_key} ---")

            try:
                # Single parameter experiment
                metrics = self._run_single_experiment(
                    project=project,
                    dataset=dataset,
                    model_type=model_type,
                    k_folds=k_folds,
                    verbose=verbose,
                    params=params,
                    **kwargs
                )
                
                self.performance_metrics[param_key] = metrics
                vlog(f"[{SUCCESS_CLR}]Completed experiment with {param_key}[/]")
            
            except Exception as e:
                # Check for 'failed' key later
                vlog(f"[{ERROR_CLR}]Failed experiment with {param_key}: {e}[/]")
                self.performance_metrics[param_key] = {
                    'error': str(e),
                    'failed': True,
                    'params': params
                }
        
        self._save_results()
        return self.performance_metrics
    
    def _run_single_experiment(
        self,
        project: sf.Project,
        dataset: sf.Dataset,
        model_type: ModelType,
        k_folds: int,
        verbose: bool,
        params: dict[str, int | float],
        **kwargs
    ) -> dict[str, int | float]:
        """
        Run a single experiment with specific parameters.
        
        Args:
            project: Slideflow project
            dataset: Dataset to train on
            model_type: Model type
            k_folds: Number of folds
            verbose: Verbose logging
            params: Parameter dictionary for this experiment
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing aggregated performance metrics
        """
        vlog = get_vlog(verbose)
        project_path = Path(project.root)
        bags_path = project_path / "bags"
        
        # Create model-specific output directory
        param_key = self._get_parameter_key(params)
        model_path = project_path / "models" / f"{self.experiment_name}_{param_key}"
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create model configuration
        config = self.create_model_config(model_type, **params)
        
        fold_metrics = []
        total_training_time = 0
        
        # Create k-fold splits
        folds = list(dataset.kfold_split(k_folds, labels="label"))
        
        for fold_idx, (train_data, val_data) in enumerate(folds):
            vlog(f"Training fold [{INFO_CLR}]{fold_idx + 1}/{k_folds}[/] with {param_key}")
            
            # Clear CUDA cache before each fold
            self._clear_gpu_memory()
            
            # Get dataset characteristics
            project_path = Path(project.root)
            bags_path = project_path / "bags"
            tiles_per_bag, input_dim = get_bag_avg_and_num_features(bags_path)
            batch_size = config.batch_size if hasattr(config, 'batch_size') else BATCH_SIZE
            
            # Estimate required memory
            estimated_memory = estimate_dynamic_vram_usage(
                model_cls=model_type.value,
                batch_size=batch_size,
                num_classes=2,
                input_dim=input_dim,
                tiles_per_bag=tiles_per_bag
            )
            
            # Train the model
            start_time = time.time()
            
            learner = self._train_model(
                config=config,
                train_data=train_data,
                val_data=val_data,
                project=project,
                bags_path=str(bags_path),
                model_path=str(model_path)
            )
            
            fold_training_time = time.time() - start_time
            total_training_time += fold_training_time
            
            # Get memory usage
            actual_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            # Extract fold-specific metrics
            fold_result = self.extract_fold_metrics(
                learner=learner,
                fold_idx=fold_idx,
                training_time=fold_training_time,
                actual_memory_mb=actual_memory,
                estimated_memory_mb=estimated_memory,
                **params,
                **kwargs
            )
            
            fold_metrics.append(fold_result)
            
            # Cleanup
            del learner
            self._clear_gpu_memory()
        
        # Aggregate metrics across folds
        aggregated_metrics = self._aggregate_fold_metrics(
            fold_metrics=fold_metrics,
            total_time=total_training_time,
            params=params
        )
        
        return aggregated_metrics

    def _train_model(
        self,
        config: Any,
        train_data: sf.Dataset,
        val_data: sf.Dataset,
        project: sf.Project,
        bags_path: str,
        model_path: str
    ) -> Learner:
        """
        Train a single model with given configuration.
        
        Args:
            config: Model configuration
            train_data: Training dataset
            val_data: Validation dataset
            project: Slideflow project
            bags_path: Path to bags directory
            model_path: Output path for model
            
        Returns:
            Trained learner object
        """
        return train_mil(
            config=config,
            train_dataset=train_data,
            val_dataset=val_data,
            outcomes="label",
            project=project,
            bags=bags_path,
            outdir=model_path
        )

    def _aggregate_fold_metrics(
        self,
        fold_metrics: List[Dict[str, Any]],
        total_time: float,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across all folds, including performance metrics.
        
        Args:
            fold_metrics: List of per-fold metrics
            total_time: Total training time
            params: Parameter dictionary used for this experiment
            
        Returns:
            Aggregated metrics dictionary
        """
        def _aggregate_single_metric(metrics_list: List[Dict], key: str) -> List[Union[int, float]]:
            """Helper method to aggregate specific metric and filter out None values"""
            return [key_metric for metrics in metrics_list if (key_metric := metrics.get(key)) is not None]
        
        # Extract common metrics
        val_accuracies = _aggregate_single_metric(fold_metrics, "val_accuracy")
        val_losses     = _aggregate_single_metric(fold_metrics, "val_loss")
        training_times = _aggregate_single_metric(fold_metrics, "training_time")
        memory_usage   = _aggregate_single_metric(fold_metrics, "actual_memory_mb")
        estimated_memory = _aggregate_single_metric(fold_metrics, "estimated_memory_mb")
        
        # Base aggregated metrics
        aggregated = {
            'params': params,
            'total_training_time': total_time,
            'avg_training_time_per_fold': np.mean(training_times) if training_times else None,
            'std_training_time_per_fold': np.std(training_times) if training_times else None,
            'avg_memory_usage_mb': np.mean(memory_usage) if memory_usage else None,
            'std_memory_usage_mb': np.std(memory_usage) if memory_usage else None,
            'max_memory_usage_mb': np.max(memory_usage) if memory_usage else None,
            'estimated_memory_mb': np.mean(estimated_memory) if estimated_memory else None,
            'num_folds': len(fold_metrics),
            'fold_details': fold_metrics
        }
        
        # Add performance metrics if available
        if val_losses:
            aggregated.update({
                'avg_val_loss': np.mean(val_losses),
                'std_val_loss': np.std(val_losses),
                'min_val_loss': np.min(val_losses),
            })
        
        if val_accuracies:
            aggregated.update({
                'avg_val_accuracy': np.mean(val_accuracies),
                'std_val_accuracy': np.std(val_accuracies),
                'max_val_accuracy': np.max(val_accuracies),
            })
        
        return aggregated

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _save_results(self) -> None:
        """Save experiment results to JSON file."""
        results_path = self.results_dir / f"{self.experiment_name}_results.json"
        
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_for_json(self.performance_metrics)
        
        mode = 'w' if not results_path.exists() else 'a'
        with open(results_path, mode) as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to: {results_path}")


class BatchSizeExperiment(Experiment):
    """
    Experiment to analyze the impact of different batch sizes on model performance
    and resource usage.
    """
    def __init__(
        self, 
        project_dir: Path,
        model_type: ModelType = ModelType.Attention_MIL,
        batch_sizes: Optional[List[int]] = None,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize the BatchSizeExperiment.
        
        Args:
            project_dir: Path to the project directory
            batch_sizes: List of batch sizes to test (defaults to [2, 4, 8, 16, 32])
            results_dir: Directory to save results and plots (defaults to project_dir/batch_size_analysis)
        """
        super().__init__(
            project_dir=project_dir, 
            experiment_name="batch_size_analysis",
            results_dir=results_dir,
            model_type=model_type
        )
        self.batch_sizes = batch_sizes or [2, 4, 8, 16, 32]
    
    def get_parameter_grid(self) -> list[dict[str, int | float]]:
        """Define the parameter grid for batch sizes."""
        return [{'batch_size': bs} for bs in self.batch_sizes]
    
    def create_model_config(self, model_type: ModelType, **params) -> Any:
        """
        Create model configuration for given batch size.
        """
        batch_size = params.get('batch_size', BATCH_SIZE)
        
        return mil_config(
            model=model_type.value,
            batch_size=batch_size,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
        )
        
    def _plot_performance_vs_batch_size(self) -> Figure:
        """Plot model performance metrics against batch size."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Model Performance vs Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Performance Metric")

        # Plotting each performance metric
        for metric, values in self.performance_metrics.items():
            if not isinstance(values, list):
                continue
            ax.plot(self.batch_sizes, values, label=metric)

        ax.legend()
        return fig
    
    def _plot_memory_vs_batch_size(self) -> Figure:
        """Plot memory usage against batch size."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Memory Usage vs Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Memory Usage (MB)")

        memory_usages = [
            metrics.get('avg_memory_usage_mb', 0) 
            for metrics in self.performance_metrics.values()
        ]
        
        ax.plot(self.batch_sizes, memory_usages, marker='o', color='orange', label='Avg Memory Usage (MB)')
        ax.legend()
        return fig
    
    def _plot_training_time_vs_batch_size(self) -> Figure:
        """Plot training time against batch size."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Training Time vs Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Total Training Time (s)")

        training_times = [
            metrics.get('total_training_time', 0) 
            for metrics in self.performance_metrics.values()
        ]
        
        ax.plot(self.batch_sizes, training_times, marker='o', color='green', label='Total Training Time (s)')
        ax.legend()
        return fig

class LearningRateExperiment(Experiment):
    """
    Experiment to analyze the impact of different learning rates on model performance
    and resource usage.
    """
    def __init__(
        self, 
        project_dir: Path,
        model_type: ModelType = ModelType.Attention_MIL,
        learning_rates: Optional[List[float]] = None,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize the LearningRateExperiment.
        
        Args:
            project_dir: Path to the project directory
            learning_rates: List of learning rates to test (defaults to [0.0001, 0.001, 0.01, 0.1])
            results_dir: Directory to save results and plots (defaults to project_dir/learning_rate_analysis)
        """
        super().__init__(
            project_dir=project_dir, 
            experiment_name="learning_rate_analysis",
            results_dir=results_dir,
            model_type=model_type
        )
        self.learning_rates = learning_rates or [0.0001, 0.001, 0.01, 0.1]
    
    def get_parameter_grid(self) -> list[dict[str, int | float]]:
        """Define the parameter grid for learning rates."""
        return [{'learning_rate': lr} for lr in self.learning_rates]
    
    def create_model_config(self, model_type: ModelType, **params) -> Any:
        """
        Create model configuration for given learning rate.
        """
        learning_rate = params.get('learning_rate', LEARNING_RATE)
        
        return mil_config(
            model=model_type.value,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=learning_rate,
        )
        
    def _plot_performance_vs_learning_rate(self) -> Figure:
        """Plot model performance metrics against learning rate."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Model Performance vs Learning Rate")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Performance Metric")

        # Plotting each performance metric
        for metric, values in self.performance_metrics.items():
            if not isinstance(values, list):
                continue
            ax.plot(self.learning_rates, values, label=metric)

        ax.legend()
        return fig
    
    def _plot_memory_vs_learning_rate(self) -> Figure:
        """Plot memory usage against learning rate."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Memory Usage vs Learning Rate")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Memory Usage (MB)")

        memory_usages = [
            metrics.get('avg_memory_usage_mb', 0) 
            for metrics in self.performance_metrics.values()
        ]
        ax.plot(self.learning_rates, memory_usages, marker='o', color='orange', label='Avg Memory Usage (MB)')
        ax.legend()
        return fig