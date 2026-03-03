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
Model memory estimation for AutoMIL.

This module provides the :class:`automil.memory.MemoryEstimator` class, which is
responsible for empirically probing training memory usage.
"""

import torch

from .model import ModelManager
from .runtime import RuntimeContext
from .util import ModelType, get_vlog


class MemoryEstimator:
    """
    Empirical GPU memory estimator for MIL models.

    Uses forward + backward passes to measure peak training memory.
    Supports AMP and safe OOM handling.
    """

    def __init__(
        self,
        model_type: ModelType,
        runtime: RuntimeContext,
        model_manager: ModelManager,
    ) -> None:
        self.model_type = model_type
        self.runtime = runtime
        self.model_manager = model_manager
        # Used for caching previously measured memory peaks
        self._memory_cache: dict[tuple, float] = {}

    # === Public Methods === #
    def estimate_peak_memory_mb(
        self,
        batch_size: int,
        bag_size: int,
        input_dim: int,
        num_classes: int,
    ) -> float:
        """Measure peak memory usage for a single training step

        Args:
            batch_size (int): Batch size
            bag_size (int): Average number of instances per bag
            input_dim (int): Input feature dimensions
            num_classes (int): Number of classes

        Returns:
            float: Peak memory usage in MB.
        """
        vlog = get_vlog(True)
        # Tuple of relevant parameters to use in populating peak memory cache
        key_vector = (self.model_type, batch_size, bag_size, input_dim, num_classes)

        if key_vector in self._memory_cache.keys():
            return self._memory_cache[key_vector]

        model = self._build_model(input_dim, num_classes)

        dummy_input = self.model_manager.create_dummy_input(
            batch_size=batch_size,
            tiles_per_bag=bag_size,
            input_dim=input_dim,
            runtime=self.runtime,
        )
        peak = self._measure(model, dummy_input)

        self._memory_cache[key_vector] = peak
        return peak

    # === Internals === #
    def _build_model(self, input_dim: int, num_classes: int) -> torch.nn.Module:
        """Builds a model instance using the associated ModelManager.

        Args:
            input_dim (int): Input feature dimensions
            num_classes (int): Number of classes

        Returns:
            torch.nn.Module: Model Instance
        """
        model = self.model_manager.create_model(
            input_dim=input_dim,
            num_classes=num_classes,
        )
        model = self.runtime.move(model)
        return model

    def _measure(self, model: torch.nn.Module, dummy_input: tuple) -> float:
        """Performs a training step (forward and backwards pass) to capture peak memory usage.

        Args:
            model (torch.nn.Module): Model Instance
            dummy_input (tuple): Dummy Input tensor or tuple

        Returns:
            float: Peak memory in MB during training step
        """
        # Set model to train mode
        # TODO | Check if .eval mode also suitable
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        self.runtime.reset_peak_memory()

        # Single training step
        try:
            optimizer.zero_grad()

            with self.runtime.autocast():
                output = model(*dummy_input)

                # Synthetic loss
                if isinstance(output, tuple):
                    output = output[0]

                loss = output.sum()

            if self.runtime.mixed_precision:
                scaler = torch.GradScaler(self.runtime.device.type)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Capture peak memory (reserved)
            peak_mem = self.runtime.peak_memory_mb()

        finally:
            del model
            del optimizer
            torch.cuda.empty_cache()

        return peak_mem