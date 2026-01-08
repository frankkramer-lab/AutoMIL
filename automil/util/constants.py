"""
Project level constants for AutoMIL.
"""
# Random seed for reproducibility
RANDOM_SEED = 42

# === Default Hyperparameters === #
# Default feature extractor
FEATURE_EXTRACTOR = "ctranspath"

# Learning Rate
LEARNING_RATE: float = 1e-4

# Batch Size
BATCH_SIZE: int = 32

# Maximum Batch size for training, used for estimating VRAM usage
# Choice of 100 is inspired by https://arxiv.org/abs/2503.10510v1
MAX_BATCH_SIZE: int = 100

# Number of Epochs
EPOCHS: int = 40

# Based on commonly cited microns per pixel (mpp) values for different default magnifications
COMMON_MPP_VALUES = {
    "20x": 0.5,
    "40x": 0.25,
    "10x": 1.0,
    "5x": 2.0,
}

# Colors for variables in log messages
INFO_CLR:       str = "cyan"       # General purpose  | Variable names and parameters
SUCCESS_CLR:    str = "green"      # Success Messages | Completed operations
ERROR_CLR:      str = "red"        # Error Messages   | Warnings
HIGHLIGHT_CLR:  str = "yellow"     # Highlighting     | Important information