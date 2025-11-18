import inspect
from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.hooks import RemovableHandle

from utils import MAX_BATCH_SIZE, get_free_memory, reserve_tensor_memory


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
    free_mem = get_free_memory()

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

class MILSizeEstimator:
    """Modified size estimator for evaluating a Slideflow MIL model's size in memory.
    This is a slightly modernized version of the original code on: https://github.com/jacobkimmel/pytorch_modelsize and specially adapted for Slideflow MIL models.
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

    def _register_hooks(self) -> list:
        """Registers forward hooks to capture module outputs."""
        def output_size_hook(module: nn.Module, input: Any, output: Any) -> None:
            """Forward hook to count the number of elements in a feature map

            Args:
                module (nn.Module): Module to count the number of elements of intermediate feature maps of
                input (Any): input
                output (Any): output (should be a tensor or a iterable)
            """
            if isinstance(output, torch.Tensor):
                self.output_sizes.append(output.numel())
            elif isinstance(output, (list, tuple)):
                self.output_sizes.extend(o.numel() for o in output if isinstance(o, torch.Tensor))

        for module in self.model.modules():
            # Register only on leaf modules to avoid duplicates
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(output_size_hook)
                self.handles.append(handle)
        
        return self.handles

    def get_output_sizes(self) -> list[int]:
        """Run forward pass with hooks to capture all output sizes.

        Returns:
            list[int]: List of output sizes
        """
        self.output_sizes = []  # Initialize output sizes list
        self._register_hooks()

        # Dummy input tensor for dry run
        dummy_input: torch.Tensor = torch.randn(
            self.batch_size,
            self.tiles_per_bag,
            *self.input_size[1:]
        ).cuda()
        # Slideflow specific: Some models require lens input
        if hasattr(self.model, "use_lens"):
            lens = torch.tensor([self.tiles_per_bag] * self.batch_size).cuda()
            with torch.no_grad():
                _ = self.model(dummy_input, lens)
        else:
            with torch.no_grad():
                _ = self.model(dummy_input)

        for handle in self.handles: handle.remove()
        return self.output_sizes

    # ------------------------------------------------------------------
    # Bit calculations
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Final estimate
    # ------------------------------------------------------------------
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


class UnifiedSizeEstimator:
    def __init__(
        self,
        model: nn.Module,
        input_size: tuple = (1, 3, 224, 224),  # Default for regular models
        mil_config: dict | None = None,  # Optional MIL configuration
        bits: int = 32
    ) -> None:
        """Estimates a PyTorch model's size in memory for both regular and Slideflow MIL models.
        This is a slightly modernized version of the original code on: https://github.com/jacobkimmel/pytorch_modelsize and specially adapted for Slideflow MIL models.
        The usage of hooks enables capturing all intermediate output sizes, even those not carried out by modules with parameters (Pooling etc.).
        For an overview of the original discussion and code see:
            - Github Repository: https://github.com/jacobkimmel/pytorch_modelsize
            - Original Discussion: https://discuss.pytorch.org/t/gpu-memory-estimation-given-a-network/1713
        Args:
            model (nn.Module): The model to estimate size for
            input_size (tuple): Input dimensions. For regular models: (batch_size, channels, height, width)
                                For MIL: (batch_size, tiles_per_bag, feature_dim) or use mil_config
            mil_config (dict, optional): MIL-specific configuration with keys:
                - batch_size: Number of bags per batch
                - tiles_per_bag: Number of tiles per bag
                - feature_dim: Feature dimension per tile
            bits (int, optional): Bits precision. Defaults to 32.
        """
        self.model: nn.Module = model
        self.input_size: tuple = input_size
        self.mil_config: dict = mil_config or {}
        self.bits: int = bits
        self.handles: list[RemovableHandle] = []
        
        # Determine if this is a MIL model
        self.is_mil_model = self._detect_mil_model()
        
    def _detect_mil_model(self) -> bool:
        """Analyzes the passed model to determine whether it is a Slideflow MIL model or not.
        Inpects the forward method signature and checks for MIL-specific attributes.

        Returns:
            bool: Whether the model is a MIL model or not.
        """
        # Check forward method signature
        forward_sig = inspect.signature(self.model.forward)
        params = list(forward_sig.parameters.keys())
        
        # MIL models typically have 'lens' parameter or specific naming
        if 'lens' in params or 'lengths' in params:
            return True
            
        # Check if model has mil-specific attributes
        mil_attributes = ['use_lens', 'mil', 'attention', 'aggregation']
        if any(hasattr(self.model, attr) for attr in mil_attributes):
            return True
            
        # Check if input_size suggests MIL format (3D instead of 4D for images)
        if len(self.input_size) == 3 and self.mil_config:
            return True
            
        return False
    
    def _create_dummy_input(self) -> tuple:
        """Create a dummy input tensor thats appropriate for the passed model type.

        Returns:
            tuple: tuple containing the dummy input (Either (input,) or (input, lens))
        """
        if self.is_mil_model:
            # MIL model input
            batch_size = self.mil_config.get('batch_size', self.input_size[0])
            tiles_per_bag = self.mil_config.get('tiles_per_bag', 100)
            feature_dim = self.mil_config.get('feature_dim', self.input_size[-1])
            
            dummy_input = torch.randn(batch_size, tiles_per_bag, feature_dim)
            
            # Check if model needs lens parameter
            import inspect
            forward_sig = inspect.signature(self.model.forward)
            if 'lens' in forward_sig.parameters or 'lengths' in forward_sig.parameters:
                lens = torch.tensor([tiles_per_bag] * batch_size)
                return (dummy_input, lens)
            else:
                return (dummy_input,)
        else:
            # Regular model input
            dummy_input = torch.randn(*self.input_size)
            return (dummy_input,)
    
    def get_parameter_sizes(self) -> list[np.ndarray]:
        """Returns the sizes of all model parameters.

        Returns:
            list[np.ndarray]: list of parameter sizes
        """
        return [np.array(param.size()) for param in self.model.parameters()]
    
    def _register_hooks(self) -> list[int]:
        """Registers forward hooks to collect intermediate output sizes.

        Returns:
            list[int]: List in which the output sizes will be stored upon the next forward pass
        """
        output_sizes: list[int] = []
        
        def output_size_hook(module: nn.Module, input: Any, output: Any) -> None:
            """Forward hook to count the number of elements in a feature map."""
            if isinstance(output, torch.Tensor):
                output_sizes.append(output.numel())
            elif isinstance(output, (list, tuple)):
                output_sizes.extend(o.numel() for o in output if isinstance(o, torch.Tensor))

        for module in self.model.modules():
            # Register only on leaf modules to avoid duplicates
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(output_size_hook)
                self.handles.append(handle)
        
        return output_sizes
    
    def get_output_sizes(self) -> list[int]:
        """Run forward pass with hooks to capture all output sizes.

        Returns:
            list[int]: Filled list of output sizes
        """
        output_sizes = self._register_hooks()
        
        # Create appropriate dummy input
        dummy_inputs = self._create_dummy_input()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            dummy_inputs_tuple = tuple(inp.cuda() if isinstance(inp, torch.Tensor) else inp 
                                       for inp in dummy_inputs)
        else:
            dummy_inputs_tuple = dummy_inputs
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(*dummy_inputs_tuple)
        
        # Clean up hooks
        for handle in self.handles:
            handle.remove()
        
        return output_sizes
    
    def calculate_param_bits(self, parameter_sizes: list[np.ndarray]) -> np.signedinteger:
        """Calculate bits to store model parameters.

        Args:
            parameter_sizes (list[np.ndarray]): list of parameter sizes

        Returns:
            np.signedinteger: Total bits to store model parameters
        """
        total_bits = 0
        for parameter_size in parameter_sizes:
            total_bits += np.prod(parameter_size) * self.bits
        return np.int64(total_bits)
    
    def calculate_forward_backward_bits(self, output_sizes: list[int]) -> np.signedinteger:
        """Calculate bits to store forward and backward pass.

        Args:
            output_sizes (list[int]): List of output sizes

        Returns:
            np.signedinteger: Total bits to store forward and backward pass
        """
        total_bits = sum(output_size * self.bits for output_size in output_sizes)
        # Multiply by 2 to account for both forward and backward pass
        return np.int64(total_bits * 2)
    
    def calculate_input_bits(self) -> np.signedinteger:
        """Calculate bits to store input.

        Returns:
            np.signedinteger: Total bits to store input
        """
        return np.prod(np.array(self.input_size)) * self.bits
    
    def estimate_size(self, include_memory_overhead: bool = True) -> tuple[float, int]:
        """Calculate total model size in megabytes and bits.

        Returns:
            tuple[float, int]: tuple of (size in megabytes, size in bits)
        """
        if include_memory_overhead and torch.cuda.is_available():
            overhead_mb = reserve_tensor_memory()
            overhead_bits = int(overhead_mb * 8 * 1024**2)  # Convert MB to bits
        else:
            overhead_bits = 0
        parameter_sizes = self.get_parameter_sizes()
        output_sizes = self.get_output_sizes()
        total_parameters_bits = self.calculate_param_bits(parameter_sizes)
        forward_backward_bits = self.calculate_forward_backward_bits(output_sizes)
        input_bits = self.calculate_input_bits()

        total = total_parameters_bits + forward_backward_bits + input_bits + overhead_bits 
        total_megabytes = (total / 8) / (1024 ** 2)

        return float(total_megabytes), int(total)
    
def estimate_model_size(model: type[nn.Module], batch_size: int, bag_size: int, input_dim: int, include_memory_overhead: bool = True) -> float:
    """Estimates the size of a MIL pytorch module in MB

    Args:
        model (nn.Module): Pytorch module
        batch_size (int): Batch size
        bag_size (int): Bag size (number of tiles in bag)
        input_dim (int): Input dimensions
        include_memory_overhead (bool, optional): Whether to include the memory overhead of tensor allocation in the estimate. Defaults to True.

    Returns:
        float: Estimated memory in MB
    """
    estimated_mem_mb, _ = UnifiedSizeEstimator(
        model=model(n_feats=input_dim, n_out=2),
        input_size=(batch_size, bag_size, input_dim),
        bits=16
    ).estimate_size(include_memory_overhead=False)
    return estimated_mem_mb
