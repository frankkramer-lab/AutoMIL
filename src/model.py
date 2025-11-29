from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from slideflow.mil.models import Attention_MIL, TransMIL
from slideflow.mil.models.bistro.transformer import \
    Attention as BistroTransformer

from utils import MAX_BATCH_SIZE, ModelType


@dataclass
class ModelConfig:
    model_cls: type[nn.Module]
    slideflow_model_name: str # slideflow internal model identifier
    input_params: dict[str, str] # A mappping between standardized param namesand model-specific names
    min_lr: float
    max_lr: float
    max_batch_size: int
    max_tiles_per_bag: int


class ModelManager:

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
            input_params={"input_dim": "dim", "num_classes": "num_classes"},
            min_lr=1e-5,
            max_lr=1e-4,
            max_batch_size=MAX_BATCH_SIZE,
            max_tiles_per_bag=1000,
        )
    }

    def __init__(self, model_type: ModelType) -> None:
        self.model_type = model_type
        self.config = self._MODEL_CONFIGS[model_type]
    
    @property
    def slideflow_name(self) -> str:
        return self.config.slideflow_model_name

    @property
    def model_class(self) -> type[nn.Module]:
        return self.config.model_cls

    def create_model(self, input_dim: int = 1024, num_classes: int = 2, **kwargs) -> nn.Module:
        """Instantiates a model with appropriate parameters

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

    def validate_hyperparameters(self, lr: float, batch_size: int, max_tiles_per_bag: int) -> dict[str, float | int]:
        """Validates hyperparameters against model constraints

        Args:
            lr (float): Learning rate
            batch_size (int): Batch size
            max_tiles_per_bag (int): Maximum tiles per bag

        Returns:
            dict[str, float | int]: Suggested adjustments for out-of-bounds hyperparameters (e.g., {"lr": 1e-4, "batch_size": 32})
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
        """Generate a comparison table of all models"""
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