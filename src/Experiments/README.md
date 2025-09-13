# Experiment Framework

This directory contains a generic experiment framework for running automated machine learning experiments with slideflow. The framework provides a structured way to test different hyperparameters and compare model performance.

## Quick Start

### Using the Framework

```python
from src.Experiments import BatchSizeExperiment
from pathlib import Path

# Create experiment
experiment = BatchSizeExperiment(
    project_dir=Path("path/to/project"),
    batch_sizes=[2, 4, 8, 16, 32]
)

# Run experiment
experiment.run_experiment(
    project=project,
    dataset=dataset,
    k_folds=3,
    verbose=True
)

# Create plots and summary
experiment.create_plots()
experiment.print_summary()
```

### Backward Compatibility

The framework maintains backward compatibility with your existing code:

```python
from src.Experiments import run_batch_size_analysis

# This works exactly like the original function
experiment = run_batch_size_analysis(
    slide_dir=Path("path/to/slides"),
    annotation_file=Path("path/to/annotations.csv"),
    project_dir=Path("path/to/project"),
    patient_column="patient",
    label_column="label",
    batch_sizes=[2, 4, 8, 16],
    verbose=True
)
```

## Architecture

### BaseExperiment Class

The `BaseExperiment` class provides the core functionality:

- **Cross-validation training**: Automated k-fold cross-validation
- **Metrics collection**: Standardized metric extraction and aggregation  
- **Result persistence**: JSON serialization of results
- **Visualization**: Automated plot generation
- **Progress tracking**: Comprehensive logging

### BatchSizeExperiment Implementation

The `BatchSizeExperiment` class demonstrates how to extend the base framework:

- Tests different batch sizes (e.g., [2, 4, 8, 16, 32])
- Measures training time, memory usage, and model performance
- Creates specialized plots for batch size analysis
- Includes memory estimation accuracy analysis

## Benefits

### Code Reuse
- Common experiment functionality is written once
- No duplication of cross-validation logic, metrics aggregation, etc.
- Consistent error handling and logging

### Maintainability  
- Clear separation of concerns
- Easy to modify base functionality for all experiments
- Standardized interfaces

### Extensibility
- Adding new experiment types requires only implementing 4 methods
- Framework handles all the heavy lifting automatically
- Supports any type of hyperparameter

### Consistency
- All experiments produce results in the same format
- Standardized metrics and output structure
- Consistent plotting and reporting

## Creating New Experiments

To create a new experiment type, inherit from `BaseExperiment` and implement 4 abstract methods:

```python
from src.Experiments.experiment import BaseExperiment

class MyExperiment(BaseExperiment):
    def get_parameter_grid(self):
        """Define what parameters to test"""
        return [
            {'learning_rate': 0.001, 'dropout': 0.1},
            {'learning_rate': 0.01, 'dropout': 0.2},
            # ... more combinations
        ]
    
    def create_model_config(self, **params):
        """Create model configuration for given parameters"""
        return mil_config(
            lr=params['learning_rate'],
            dropout=params['dropout'],
            # ... other config
        )
    
    def extract_fold_metrics(self, learner, fold_idx, **context):
        """Extract metrics from trained model"""
        return {
            'fold': fold_idx + 1,
            'learning_rate': context['learning_rate'],
            'dropout': context['dropout'],
            'val_loss': extracted_loss,
            # ... other metrics
        }
    
    def create_experiment_plots(self):
        """Create experiment-specific plots"""
        # Create matplotlib figures
        return [fig1, fig2, fig3]
```

## Output

Each experiment creates:

1. **Results directory**: `project_dir/batch_analysis/`
2. **JSON results**: `batch_analysis_results.json` 
3. **Plots**: `batch_analysis_plot_1.png`, `batch_analysis_plot_2.png`, etc.
4. **Console summary**: Performance metrics and recommendations

## Migration from Original Code

The original `BatchSizeExperiment` class functionality has been preserved but restructured:

- **Same results**: Produces identical output to the original implementation
- **Better structure**: Code is now more modular and reusable
- **Easier maintenance**: Common functionality is centralized
- **Future-proof**: Easy to extend for new experiment types

### Key Changes

1. **Parameter definition** moved to `get_parameter_grid()`
2. **Model configuration** moved to `create_model_config()`  
3. **Metric extraction** moved to `extract_fold_metrics()`
4. **Plotting** moved to `create_experiment_plots()`
5. **Common functionality** inherited from `BaseExperiment`

This makes the code much easier to maintain and extend while preserving all existing functionality.
