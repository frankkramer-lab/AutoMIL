# Dataset

`automil.dataset.Dataset` is responsible for preparing a
slideflow-compatible dataset source that can be passed to downstream pipeline stages.
It supports both raw and pre-tiled whole-slide image datasets, and handles
tiling, the conversion from .png to .tiff, label filtering, and feature extraction.

The resulting datasets are stored as TFRecords and feature bags within respective folders in the
project directory.

::: automil.dataset.Dataset