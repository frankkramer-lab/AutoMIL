from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.hooks import RemovableHandle

from utils import MAX_BATCH_SIZE, get_gpu_memory


def estimate_dynamic_vram_usage(
    model_cls: Type[nn.Module],
    input_dim: int = 1024,
    tiles_per_bag: int = 100,
    batch_size: int = 4,
    num_classes: int = 2,
    return_rounded: bool = True
) -> float:
    """Estimate the approximate VRAM usage dynamically for a given torch module.

    Args:
        model_cls: The MIL model class (e.g., TransMIL)
        input_dim: Feature vector size per tile
        tiles_per_bag: Number of tiles per bag/sample
        batch_size: Number of samples per batch
        num_classes: Number of output classes for the model
        return_rounded: Whether to round the result to 2 decimal places

    Returns:
        Estimated VRAM usage in megabytes.
    """
    # Initiate Model and set it to evaluation mode
    try:
        model: nn.Module = model_cls(n_feats=input_dim, n_out=num_classes).cuda()
    except TypeError:
        model: nn.Module = model_cls()
    model.eval()

    # Dummy input tensor for dry run
    dummy_input: torch.Tensor = torch.randn(batch_size, tiles_per_bag, input_dim).cuda()
    lens = torch.tensor([tiles_per_bag] * batch_size).cuda()

    output_sizes: list[int] = []

    def output_size_hook(module: nn.Module, input: Any, output: Any) -> None:
        """Forward hook to count the number of elements in a feature map

        Args:
            module (nn.Module): Module to count the number of elements of intermediate feature maps of
            input (Any): input
            output (Any): output (should be a tensor or a iterable)
        """
        if isinstance(output, torch.Tensor):
            output_sizes.append(output.numel())
        elif isinstance(output, (list, tuple)):
            output_sizes.extend(o.numel() for o in output if isinstance(o, torch.Tensor))

    # Register hooks to all modules
    hooks = [m.register_forward_hook(output_size_hook)
             for m in model.modules() if not isinstance(m, nn.Sequential)]

    # Dry run
    # TODO | Check time requirements
    with torch.no_grad():
        model(dummy_input, lens)

    # Removing hooks
    # TODO | Check if necessary
    for h in hooks:
        h.remove()

    # sum of output sizes in Mb
    dry_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    output_sum = sum(output_sizes) * 4 / (1024 ** 2)
    param_mem = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    buffer_mem = sum(b.numel() for b in model.buffers()) * 4 / (1024**2)
    input_mem = dummy_input.numel() * 4 / (1024**2) + lens.numel() * 4 / (1024**2)
    sum_mem = dry_mem + output_sum + param_mem + buffer_mem + input_mem
    return round(sum_mem, 2) if return_rounded else sum_mem

def measure_vram_usage(model_cls, input_dim=1024, tiles_per_bag=100, batch_size=4, num_classes=2):
    """Measure the actual VRAM usage during a forward and backward pass."""
        # Initiate Model and set it to evaluation mode
    try:
        model: nn.Module = model_cls(n_feats=input_dim, n_out=num_classes).cuda()
    except TypeError:
        model: nn.Module = model_cls()
    model.eval()
    dummy_input = torch.randn(batch_size, tiles_per_bag, input_dim, device="cuda")
    lens = torch.tensor([tiles_per_bag] * batch_size, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    output = model(dummy_input, lens)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return round(max_mem, 2)

def adjust_batch_size(
    model_cls: Type[nn.Module],
    initial_batch_size: int,
    num_slides: int,
    input_dim: int,
    tiles_per_bag: int
) -> int:
    """Adjust batch size based on the available memory.

    Args:
        model_cls: The MIL model class (e.g., TransMIL)
        initial_batch_size: Initial batch size
        num_slides: Number of slides in the dataset (used to compute upper bound for batch size)
        input_dim: Feature vector size (usually 1024)
        tiles_per_bag: Number of tiles per bag

    Returns:
        Batch size adjusted to available memory.
    """
    global MAX_BATCH_SIZE
    
    # Get estimated memory usage of model and free memory (both in Mb)
    estimated_mem_usage = estimate_dynamic_vram_usage(
        model_cls,
        input_dim,
        tiles_per_bag,
        initial_batch_size
    )
    free_mem = get_gpu_memory()["free_MB"]

    # Adjust batch size according to free memory (with upper bound in mind)
    batch_size_limit = min(
        num_slides // 2,
        MAX_BATCH_SIZE
    )
    adjusted_batch_size = min(
        round(free_mem / estimated_mem_usage) * initial_batch_size, 
        batch_size_limit
    )

    return int(adjusted_batch_size)

class SizeEstimator():
    """Size estimator for evaluating a Pytorch model's size in memory.
    This is a slightly modernized version of the original code on: https://github.com/jacobkimmel/pytorch_modelsize
    """
    def __init__(
            self,
            model: nn.Module,
            input_size: tuple = (1,1,32,32),
            bits: int = 32
        ) -> None:
        """Estimates a PyTorch model's size in memory

        Args:
            model (nn.Module): The model to estimate size for
            input_size (tuple, optional): Input dimensions as a tuple. Defaults to (1,1,32,32).
            bits (int, optional): Bits. Defaults to 32.
        """
        self.model = model
        self.input_size: tuple = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self) -> list[np.ndarray]:
        """Return sizes of all model parameters

        Returns:
            list[np.ndarray]: List of parameter sizes as numpy arrays
        """
        return [
            np.array(param.size())
            for param in self.model.parameters()
        ]

    def get_output_sizes(self) -> list[np.ndarray]:
        """Return sizes of all model outputs after a forward pass

        Returns:
            list[np.ndarray]: List of output sizes as numpy arrays
        """
        # Adding _ to avoid pylint warnings (clashes with 'input' keyword)
        _input = Variable(
            torch.FloatTensor(*self.input_size),
            volatile=True
        )
        # TODO | The module list should probably be a class attribute
        modules = [
            module for module in self.model.modules()
            if module.parameters() is not None
        ]
        output_sizes: list[np.ndarray] = []

        for index, module in enumerate(modules):
            if index == 0: continue
            output: torch.Tensor = module(_input)
            output_sizes.append(np.array(output.size()))
            _input = output

        return output_sizes

    def calculate_param_bits(self, parameter_sizes: list[np.ndarray]) -> np.signedinteger:
        """Calculate bits to store model parameters

        Args:
            parameter_sizes (list[np.ndarray]): List of parameter sizes as numpy arrays

        Returns:
            np.signedinteger: Total bits to store model parameters
        """
        total_bits = 0

        for parameter_size in parameter_sizes:
            total_bits += np.prod(
                parameter_size
            ) * self.bits

        return np.int64(total_bits)

    def calculate_forward_backward_bits(self, output_sizes: list[np.ndarray]) -> np.signedinteger:
        """Calculate bits to store forward and backward pass

        Args:
            output_sizes (list[np.ndarray]): List of output sizes as numpy arrays

        Returns:
            np.signedinteger: Total bits to store forward and backward pass
        """
        total_bits = 0

        for output_size in output_sizes:
            total_bits += np.prod(
                output_size
            ) * self.bits

        # Multiply by 2 to account for both forward and backward pass        
        return np.int64(total_bits * 2)

    def calculate_input_bits(self) -> np.signedinteger:
        """Calculate bits to store input

        Returns:
            np.signedinteger: Total bits to store input
        """
        return np.prod(
            np.array(self.input_size)
        ) * self.bits

    def estimate_size(self) -> tuple[float, int]:
        """Calculate total model size in megabytes and bits

        Returns:
            tuple[float, int]: tuple of (size in megabytes, size in bits)
        """
        parameter_sizes         = self.get_parameter_sizes()
        output_sizes            = self.get_output_sizes()
        total_parameters_bits   = self.calculate_param_bits(parameter_sizes)
        forward_backward_bits   = self.calculate_forward_backward_bits(output_sizes)
        input_bits              = self.calculate_input_bits()

        total = total_parameters_bits + forward_backward_bits + input_bits

        total_megabytes = (total/8)/(1024**2)
        return float(total_megabytes), int(total)

class SizeEstimatorHooks:
    """Size estimator for evaluating a Pytorch model's size in memory using hooks.
    This is a slightly modernized version of the original code on: https://github.com/jacobkimmel/pytorch_modelsize
    The usage of hooks enables capturing all intermediate output sizes, even those not carried out by modules with parameters (Pooling etc.).
    For an overview of the original discussion and code see:
        - Github Repository: https://github.com/jacobkimmel/pytorch_modelsize
        - Original Discussion: https://discuss.pytorch.org/t/gpu-memory-estimation-given-a-network/1713
    """
    def __init__(
        self,
        model_cls: Type[nn.Module],
        input_size: tuple = (1, 1, 32, 32),
        num_classes: int = 2,
        batch_size: int = 32,
        tiles_per_bag: int = 100,
        bits: int = 32
    ) -> None:
        """Estimates a PyTorch model's size in memory

        Args:
            model (nn.Module): The model to estimate size for
            input_size (tuple, optional): Input dimensions as a tuple. Defaults to (1,1,32,32).
            bits (int, optional): Bits. Defaults to 32.
        """
        try:
            model: nn.Module = model_cls(n_feats=input_size[-1], n_out=num_classes).cuda()
        except TypeError:
            model: nn.Module = model_cls()
        model.eval()
        self.model = model
        self.input_size: tuple = input_size
        self.batch_size = batch_size
        self.tiles_per_bag = tiles_per_bag
        self.bits = bits
        self.activation_sizes: list[np.ndarray] = []  # stores outputs
        self.handles = []

    def get_parameter_sizes(self) -> list[np.ndarray]:
        """Return sizes of all model parameters

        Returns:
            list[np.ndarray]: List of parameter sizes as numpy arrays
        """
        return [
            np.array(param.size())
            for param in self.model.parameters()
        ]

    def _register_hooks(self) -> tuple[list[int], list[RemovableHandle]]:
        """Registers forward hooks to capture module outputs."""
        output_sizes: list[int] = []
        def output_size_hook(module: nn.Module, input: Any, output: Any) -> None:
            """Forward hook to count the number of elements in a feature map

            Args:
                module (nn.Module): Module to count the number of elements of intermediate feature maps of
                input (Any): input
                output (Any): output (should be a tensor or a iterable)
            """
            if isinstance(output, torch.Tensor):
                output_sizes.append(output.numel())
            elif isinstance(output, (list, tuple)):
                output_sizes.extend(o.numel() for o in output if isinstance(o, torch.Tensor))

        handles: list[RemovableHandle] = []
        for module in self.model.modules():
            # Register only on leaf modules to avoid duplicates
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(output_size_hook)
                handles.append(handle)
        
        return output_sizes, handles

    def get_output_sizes(self) -> list[int]:
        """Run forward pass with hooks to capture all output sizes."""
        output_sizes, handles = self._register_hooks()

        # Dummy input tensor for dry run
        dummy_input: torch.Tensor = torch.randn(self.batch_size, self.tiles_per_bag, self.input_size[-1]).cuda()
        lens = torch.tensor([self.tiles_per_bag] * self.batch_size).cuda()

        with torch.no_grad():
            _ = self.model(dummy_input, lens)

        for handle in handles: handle.remove()
        return output_sizes

    def calculate_param_bits(self, parameter_sizes: list[np.ndarray]) -> np.signedinteger:
        """Calculate bits to store model parameters"""
        total_bits = 0
        for parameter_size in parameter_sizes:
            total_bits += np.prod(parameter_size) * self.bits
        return np.int64(total_bits)

    def calculate_forward_backward_bits(self, output_sizes: list[int]) -> np.signedinteger:
        """Calculate bits to store forward and backward pass"""
        total_bits = 0
        for output_size in output_sizes:
            total_bits += np.prod(output_size) * self.bits
        # Multiply by 2 to account for both forward and backward pass
        return np.int64(total_bits * 2)

    def calculate_input_bits(self) -> np.signedinteger:
        """Calculate bits to store input"""
        return np.prod(np.array(self.input_size)) * self.bits

    def estimate_size(self) -> tuple[float, int]:
        """Calculate total model size in megabytes and bits

        Returns:
            tuple[float, int]: tuple of (size in megabytes, size in bits)
        """
        parameter_sizes = self.get_parameter_sizes()
        output_sizes = self.get_output_sizes()
        total_parameters_bits = self.calculate_param_bits(parameter_sizes)
        forward_backward_bits = self.calculate_forward_backward_bits(output_sizes)
        input_bits = self.calculate_input_bits()

        total = total_parameters_bits + forward_backward_bits + input_bits
        total_megabytes = (total / 8) / (1024 ** 2)

        return float(total_megabytes), int(total)
