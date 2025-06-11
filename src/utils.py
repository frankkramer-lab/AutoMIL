from enum import Enum

"""
Variables and utility methods used throughout the project.
"""

# ------------------------- #
# --- Project Variables --- #
# ------------------------- #

# --- General ---

# Random seed for reproducing results
RANDOM_SEED: int = 42

# --- Paths ---



# --- Hyperparameters ---

# Resolution Presets for extracting dataset tiles (specifies tile size and magnification level)
class RESOLUTION_PRESETS(Enum):
    Low = (1_000, "10x")
    High = (299, "20x")

# --- Bags ---

# Feature Extractor to use
FEATURE_EXTRACTOR: str = "plip"

# --- Training ---

# Learning Rate
LEARNING_RATE: float = 1e-4

# Batch Size
BATCH_SIZE: int = 32

# Number of Epochs
EPOCHS: int = 40

# ----------------------- #
# --- Utility methods --- #
# ----------------------- #