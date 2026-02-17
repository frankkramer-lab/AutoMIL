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
GPU utility functions.
"""
import subprocess

import torch


def get_free_memory() -> float:
    """Return the amount of free memory on the current GPU in MB."""
    free_mem, _ = torch.cuda.mem_get_info()
    return free_mem / (1024 ** 2)  # Convert to MB

def get_cuda_gpu_memory_used() -> int:
    """Retrieves the total memory the cuda driver has reserved using nvidia-smi.

    Returns:
        int: Memory in MB
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    )
    return int(result.decode().strip().split('\n')[0])  # memory in MB of GPU 0

def reserve_tensor_memory() -> float:
    """Gets the amount of memory overhead reserved when allocating a minimal tensor (small as possible).

    Returns:
        float: Memory overhead for tensor allocation in MB
    """
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()

    # We can get the memory overhead by measuring
    # the total memory reserved before and after allocating a minimal tensor
    before = get_cuda_gpu_memory_used()
    a = torch.FloatTensor(1).cuda()
    torch.cuda.synchronize() # Ensure the allocation is complete
    after = get_cuda_gpu_memory_used()
    return after - before