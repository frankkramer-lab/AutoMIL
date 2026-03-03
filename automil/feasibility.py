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
Simple feasibility checker for AutoMIL

This module provides the :func:`automil.feasibility.is_feasible` method,
which is responsible for checking a given set of parameters against sensible limits to see if
a configuration is feasible
"""
from slideflow.util import log

from .memory import MemoryEstimator
from .model import ModelManager
from .runtime import RuntimeContext


def is_feasible(
    runtime: RuntimeContext,
    memory_estimator: MemoryEstimator,
    model_manager: ModelManager,
    batch_size: int,
    dataset_size: int,
    bag_size: int,
    input_dim: int,
    num_classes: int,
    safety_margin: float = 0.9,
    min_steps_per_epoch: int = 1,
) -> tuple[bool, list[str]]:
    """Checks the given input arguments against sensible upper and lower limits and determines whether they are feasible.
    This method also performs memory estimation to see if the given cofiguration would exceed the free memory with respect to a given safety margin.

    Args:
        runtime (RuntimeContext): RuntimeContext instance
        memory_estimator (MemoryEstimator): MemoryEstimator instance
        model_manager (ModelManager): ModelManager instance
        dataset_size (int): Number of instances (whole-slide images) in the dataset
        bag_size (int): Average number of instances (tiles) üer bag
        input_dim (int): Input feature dimensions
        num_classes (int): Number of classes
        safety_margin (float, optional): Safety margin for memory usage. Target memory usage will be `free memory` * `safety_margin`. Defaults to 0.9.
        min_steps_per_epoch (int, optional): The minimum number of training steps per epoch. Defaults to 1.

    Returns:
        tuple[bool, list[str]]: tuple containing whether the given parameters are feasible and a list of violations if not (empty list if feasible).
    """
    violations = []

    # === Batch size contraints === #
    if batch_size > dataset_size:
        violations.append("batch_size_exceeds_dataset_size")

    if batch_size > model_manager.config.max_batch_size:
        violations.append("batch_size_exceeds_model_limit")
    
    # === Training constraints === #
    steps = dataset_size // batch_size
    if steps < min_steps_per_epoch:
        violations.append("too_few_steps_per_epoch")

    # === Memory constraints === #
    if runtime.device.type == "cuda":
        free_mem = runtime.free_memory_mb()
        target = free_mem * safety_margin

        peak = memory_estimator.estimate_peak_memory_mb(
            batch_size=batch_size,
            bag_size=bag_size,
            input_dim=input_dim,
            num_classes=num_classes,
        )

        if peak >= target:
            violations.append("exceeds_memory_limit")
    
    feasible = (len(violations) == 0)
    return feasible, violations

def is_feasible_cheap(
    model_manager: ModelManager,
    batch_size: int,
    dataset_size: int,
    min_steps_per_epoch: int = 1,
) -> bool:
    """Checks the given input arguments against sensible upper and lower limits and determines whether they are feasible.
    Contrary to is_feasible, this method does not perform memory estimation.

    Args:
        model_manager (ModelManager): ModelManager instance
        batch_size: Batch size to test for feasibility.
        dataset_size (int): Number of instances (whole-slide images) in the dataset
        min_steps_per_epoch (int, optional): The minimum number of training steps per epoch. Defaults to 1.

    Returns:
        bool: Whether the input is feasible or not
    """
    violations = []

    # === Batch size contraints === #
    if batch_size > dataset_size:
        violations.append("batch_size_exceeds_dataset_size")

    if batch_size > model_manager.config.max_batch_size:
        violations.append("batch_size_exceeds_model_limit")
    
    # === Training constraints === #
    steps = dataset_size // batch_size
    if steps < min_steps_per_epoch:
        violations.append("too_few_steps_per_epoch")

    feasible = (len(violations) == 0)
    return feasible