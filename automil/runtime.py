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
Runtime management for AutoMIL

This module provides the :class:`automil.runtime.RuntimeContext` class, which is
responsible for handling and managing runtime variables, such as the pytorch device and automated mixed precision (amp).
It is also responsible for providing memory management utilities for the  given device, including measuring and resetting memory stats.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class RuntimeContext:
    """
    Encapsulates runtime context management for AutoMIL.

    This class handles device configuration, precision configuration,
    and memory profiling.
    """
    device: torch.device
    mixed_precision: bool = False
    deterministic: bool = False

    @classmethod
    def auto(
        cls,
        mixed_precision: bool = False,
        deterministic: bool = False
    ) -> "RuntimeContext":
        """
        Automatically determine runtime device and create a corresponding RuntimeContext.

        Args:
            mixed_precision: Whether to enable AMP.
            deterministic: Whether to enforce deterministic CUDA behavior.

        Returns:
            RuntimeContext instance.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        return cls(
            device,
            mixed_precision and device.type == "cuda",
            deterministic
        )

    def move(self, obj):
        """
        Moves a model or tensor to the runtime device.

        Args:
            obj: Object to move to runtime device
        """
        return obj.to(self.device)
    
    @contextmanager
    def autocast(self) -> Iterator[None]:
        """
        Context manager for AMP (automatic mixed precision).
        Does nothinng if mixed precision is disabled.
        """
        if self.mixed_precision:
            with torch.autocast(self.device.type):
                yield
        else:
            yield

    def reset_peak_memory(self) -> None:
        """
        Reset CUDA peak memory stats.
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> float:
        """
        Return peak reserved memory in MB.

        Returns:
            float: Peak reserved memory (since last reset) in MB.
        """
        if self.device.type == "cuda":
            return torch.cuda.max_memory_reserved() / (1024 ** 2)
        return 0.0

    def free_memory_mb(self) -> float:
        """
        Returns available GPU memory in MB.

        Returns:
            float: Available memory on the GPU in MB.
        """
        if self.device.type == "cuda":
            free_mem, _ = torch.cuda.mem_get_info()
            return free_mem / (1024 ** 2)
        return 0.0