"""
Experiments Package

This package contains the base experiment framework and the batch size experiment
implementation for automated machine learning experiments with slideflow.
"""

from .experiment import Experiment
from .experiment import BatchSizeExperiment

__all__ = [
    'Experiment', 
    'BatchSizeExperiment', 
]
