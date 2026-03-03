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
Memory aware hyperparameter tuning utility for AutoMIL

This module provides the :class:`automil.resource_optimizer.ResourceOptimizer` class,
which is currently responsible for adjusting the batch size to the maximum feasible bound with respect
to the available GPU memoryand other constraints such as the dataset size.
"""
import torch

from .feasibility import is_feasible_cheap as is_feasible
from .memory import MemoryEstimator
from .model import ModelManager
from .runtime import RuntimeContext


class ResourceOptimizer:

    def __init__(
        self,
        runtime: RuntimeContext,
        memory_estimator: MemoryEstimator,
        model_manager: ModelManager,
    ):
        self.runtime = runtime
        self.memory_estimator = memory_estimator
        self.model_manager = model_manager

    def find_max_batch_size(
        self,
        initial_batch_size: int,
        bag_size: int,
        input_dim: int,
        num_classes: int,
        num_slides: int,
        safety_margin: float = 0.9,
        max_binary_steps: int = 100,
    ) -> int:
        """Finds the maximum feasible batch size via expansion and subsequent binary search.
        Starting from an initial batch size, this method determines the maximum batch size that is possible
        with regards to memory and dataset constraints such as the estimated memory usage of the model and
        the available number of slides in the dataset.

        Args:
            initial_batch_size (int): Initial batch size
            bag_size (int): Average number of instances (tiles) per bag
            input_dim (int): Input feature dimensions
            num_classes (int): Number of classes present in the dataset
            num_slides (int): Number of slides in the dataset
            safety_margin (float, optional): Safety margin for how much of the free memory is allowed to be reserved. Defaults to 0.9.
            max_binary_steps (int, optional): Maximum number of steps for the binary search. Defaults to 100.

        Returns:
            int: Maximum feasible batch size found
        """
        if self.runtime.device.type != "cuda":
            return min(
                initial_batch_size,
                num_slides,
                self.model_manager.config.max_batch_size,
            )

        free_mem_mb = self.runtime.free_memory_mb()
        target_mem_mb = free_mem_mb * safety_margin

        # === Loop 1: Determine upper bound for batch size === #
        batch_size = max(initial_batch_size, 1)
        last_safe = 0

        while is_feasible(
            model_manager=self.model_manager,
            batch_size=batch_size,
            dataset_size=num_slides,
        ):
            try:
                peak = self.memory_estimator.estimate_peak_memory_mb(
                    batch_size=batch_size,
                    bag_size=bag_size,
                    input_dim=input_dim,
                    num_classes=num_classes,
                )

                if peak >= target_mem_mb:
                    break

                last_safe = batch_size
                batch_size *= 2

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    break
                else:
                    raise

        if last_safe == 0:
            return 1

        # === Loop 2: Determine maximum feaisble batch size via binary search === #
        low = last_safe
        high = batch_size
        best = last_safe

        for _ in range(max_binary_steps):

            if low >= high:
                break

            mid = (low + high) // 2

            if not is_feasible(
                model_manager=self.model_manager,
                batch_size=mid,
                dataset_size=num_slides,
            ):
                high = mid - 1
                continue
            
            try:
                peak = self.memory_estimator.estimate_peak_memory_mb(
                    batch_size=mid,
                    bag_size=bag_size,
                    input_dim=input_dim,
                    num_classes=num_classes,
                )

                if peak < target_mem_mb:
                    best = mid
                    low = mid
                else:
                    high = mid - 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    break
                else:
                    raise

        return max(best, 1)