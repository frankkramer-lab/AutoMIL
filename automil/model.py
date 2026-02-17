#==============================================================================#
#  AutoMIL - Automated Machine Learning for Image Classification in            #
#  Whole-Slide Imaging with Multiple Instance Learning                         #
#                                                                              #
#  Copyright (C) 2026 Jonas Waibel                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.      #
#==============================================================================#
"""
Model management utilities for AutoMIL.

This module provides the :class:`automil.model.ModelManager` class, which is responsible for instantiating MIL models and
validating hyperparameters against model-specific contraints/limits
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from slideflow.mil.models import Attention_MIL, TransMIL
from slideflow.mil.models.bistro.transformer import \
    Attention as BistroTransformer
from slideflow.util import log as slideflow_log

from .util import MAX_BATCH_SIZE, ModelType


def create_model_instance(
    model_type: ModelType,
    input_dim: int,
    n_out: int = 2
) -> nn.Module:
    """Safely creates a model instance with the correct parameters.

    This method instantiates a model corresponding to the provided
    :class:`ModelType` with the specified input and output dimensions
    
    Args:
        model_type: The ModelType enum
        input_dim: Input feature dimension
        n_out: Number of output classes

    Returns:
        Instantiated model
    """
    try:
        match model_type:

            case ModelType.Attention_MIL:
                model_cls = Attention_MIL
                return model_cls(n_feats=input_dim, n_out=n_out)
            
            case ModelType.TransMIL:
                model_cls = TransMIL
                return model_cls(n_feats=input_dim, n_out=n_out)
            
            case ModelType.BistroTransformer:
                model_cls = BistroTransformer
                return model_cls(dim=input_dim)

            case _:
                return model_cls()
    except Exception as e:
        slideflow_log.error(f"Error while creating model instance: {e}")
        raise e


@dataclass
class ModelConfig:
    """
    A configuration class for MIL models.

    Defines model-specific constraints and parameter mappings
    used by :class:`ModelManager` for model instantiation within specified parameter ranges
    """

    model_cls: type[nn.Module]
    """Corresponding python class implementing the model."""

    """
    Mapping from standardized parameter names (``input_dim``, ``num_classes``)
    to model-specific constructor arguments.
    """
    slideflow_model_name: str
    """Slideflow internal name of the model"""

    input_params: dict[str, str]
    """
    Mapping from standardized parameter names (``input_dim``, ``num_classes``)
    to model-specific constructor arguments.
    """

    min_lr: float
    """Minimum recommended learning rate."""

    max_lr: float
    """Maximum recommended learning rate."""

    max_batch_size: int
    """Maximum safe batch size."""

    max_tiles_per_bag: int
    """Maximum number of tiles allowed per bag."""


class ModelManager:
    """
    Manages the instantiation and configuration of MIL models.

    This class provides:
        - An interface for creating automil supported MIL models
        - Model-specific hyperparameter validation and adjustments should they be outside of recommended model-limits
        - dummy input generation for debugging and validation
    """
    # Baseline internal model configurations
    _MODEL_CONFIGS: dict[ModelType, ModelConfig] = {
        ModelType.Attention_MIL: ModelConfig(
            model_cls=Attention_MIL,
            slideflow_model_name="attention_mil",
            input_params={"input_dim": "n_feats", "num_classes": "n_out"},
            min_lr=1e-5,
            max_lr=1e-4,
            max_batch_size=MAX_BATCH_SIZE,
            max_tiles_per_bag=1000, # Can be quite large for Attention_MIL
        ),
        # We use smaller batch sizes for TransMIL due to memory constraints
        ModelType.TransMIL: ModelConfig(
            model_cls=TransMIL,
            slideflow_model_name="transmil",
            input_params={"input_dim": "n_feats", "num_classes": "n_out"},
            min_lr=1e-5,
            max_lr=1e-4,
            max_batch_size=32,
            max_tiles_per_bag=500,
        ),
        ModelType.BistroTransformer: ModelConfig(
            model_cls=BistroTransformer,
            slideflow_model_name="bistro_transformer",
            input_params={"input_dim": "dim", "num_classes": "heads"},
            min_lr=1e-5,
            max_lr=1e-4,
            max_batch_size=MAX_BATCH_SIZE,
            max_tiles_per_bag=1000,
        )
    }

    def __init__(self, model_type: ModelType) -> None:
        f"""Instantiates a ModelManager object

        Args:
            model_type (ModelType): Type of model to instantiate. Can be one of: {
                [model.name for model in ModelType]
            }
        """
        self.model_type = model_type
        self.config = self._MODEL_CONFIGS[model_type]
    
    @property
    def slideflow_name(self) -> str:
        """
        Slideflow-internal identifier for the managed model.

        Returns:
            str:
                Slideflow model name.
        """
        return self.config.slideflow_model_name

    @property
    def model_class(self) -> type[nn.Module]:
        """
        Corresponding python class implementing the model.

        Returns:
            type[nn.Module]:
                Model class.
        """
        return self.config.model_cls

    def create_model(self, input_dim: int = 1024, num_classes: int = 2, **kwargs) -> nn.Module:
        """Instantiates the model with validated hyperparameters.

        Args:
            input_dim (int, optional): Feature dimensions. Defaults to 1024.
            num_classes (int, optional): Number of classes. Defaults to 2.

        Returns:
            nn.Module: Instantiated model
        """
        # Map standardized parameter names to model-specific names
        model_params = {
            self.config.input_params["input_dim"]: input_dim,
            self.config.input_params["num_classes"]: num_classes,
        }
        # Update with remaining kwargs
        model_params.update(kwargs)

        # Fallback: If instantiation fails, try with defaults
        try:
            return self.model_class(**model_params)
        except TypeError as e:
            return self.model_class()

    def create_dummy_input(
        self, 
        batch_size: int, 
        tiles_per_bag: int, 
        input_dim: int
    ) -> tuple:
        """Creates an appropriate dummy input for the model
        
        Dummy input tensors can be used for a variety of tasks. Primarily they are used
        to perform `dry runs`, for example to measure the memory reservation of a model instance

        Args:
            batch_size: Number of samples in batch
            tiles_per_bag: Number of tiles per bag
            input_dim: Feature dimension
            
        Returns:
            Tuple of tensors to pass to model forward()
        """
        match self.model_type:

            case ModelType.Attention_MIL:
                # Both expect a lens tensor in addition to input
                dummy_input = torch.randn(batch_size, tiles_per_bag, input_dim).cuda()
                lens = torch.tensor([tiles_per_bag] * batch_size).cuda()
                return (dummy_input, lens)

            case ModelType.TransMIL | ModelType.BistroTransformer:
                # BistroTransformer expects only input (no lens)
                dummy_input = torch.randn(batch_size, tiles_per_bag, input_dim).cuda()
                return (dummy_input,)
            

    def validate_hyperparameters(self, lr: float, batch_size: int, max_tiles_per_bag: int) -> dict[str, float | int]:
        """
        Validates a set of hyperparameters against model-specific constraints.

        Args:
            lr (float):
                Learning rate.
            batch_size (int):
                Batch size.
            max_tiles_per_bag (int):
                Maximum tiles per bag.

        Returns:
            dict[str, float | int]:
                Suggested parameter adjustments for out-of-range values.
        """
        suggestions = {}

        # TODO | Better tuning logic / strategy (probably for all but definitely for lr)
        if not (self.config.min_lr <= lr <= self.config.max_lr):
            suggestions["lr"] = (self.config.min_lr + self.config.max_lr) / 2

        if batch_size > self.config.max_batch_size:
            suggestions["batch_size"] = self.config.max_batch_size

        if max_tiles_per_bag > self.config.max_tiles_per_bag:
            suggestions["max_tiles_per_bag"] = self.config.max_tiles_per_bag
        
        return suggestions

    @classmethod
    def compare_models(cls) -> str:
        """Generates a comparison table for all available models
        
        Returns:
            str: A comparison table as string
        """
        from tabulate import tabulate
        
        table = []
        for model_type, config in cls._MODEL_CONFIGS.items():
            table.append([
                model_type.name,
                config.slideflow_model_name,
                config.max_batch_size,
                config.max_tiles_per_bag,
                f"{config.min_lr:.0e}-{config.max_lr:.0e}",
            ])
        
        headers = [
            "Model Type", "Slideflow Name", "Max Batch Size", 
            "Max Tiles Per Bag", "LR Range"
        ]
        
        return tabulate(table, headers=headers, tablefmt="fancy_outline")