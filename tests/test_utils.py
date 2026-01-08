from pathlib import Path

import pandas as pd
import pytest
from slideflow.mil.models import Attention_MIL, TransMIL
from slideflow.mil.models.bistro.transformer import \
    Attention as BistroTransformer

from automil.dataset import get_unique_labels
from automil.util import RESOLUTION_PRESETS, ModelType
from automil.util.pretiled import is_input_pretiled
from automil.util.tiff_conversion import batch_generator


# === Helper === #
def create_pretiled_structure(
    root: Path,
    slides: dict[str, int],
    ext: str = ".tiff",
):
    """
    Create a pretiled directory structure.

    Example:
    ```
        slides = {
            "slide1": 3,  # slide1 has 3 tiles
            "slide2": 2,  # slide2 has 2 tiles
        }
        create_pretiled_structure(tmp_path / "pretiled", slides)
        --> Creates:
        pretiled/
            slide1/
                tile_0.tiff
                tile_1.tiff
                tile_2.tiff
            slide2/
                tile_0.tiff
                tile_1.tiff
    ```

    Args:
        root (Path): Root directory where the pretiled structure will be created.
        slides (dict[str, int]): Dictionary mapping slide names to number of tiles.
        ext (str): File extension for the tile files.
    """
    root.mkdir(parents=True, exist_ok=True)

    for slide_name, num_tiles in slides.items():
        slide_dir = root / slide_name
        slide_dir.mkdir()
        for i in range(num_tiles):
            (slide_dir / f"tile_{i}{ext}").touch()


def test_model_type_to_model_names():
    """Test that ModelType enum values map to the correct model names."""
    assert ModelType.Attention_MIL.model_name == "attention_mil"
    assert ModelType.BistroTransformer.model_name == "bistro.transformer"
    assert ModelType.TransMIL.model_name == "transmil"

def test_model_type_to_model_class():
    """Test that ModelType enum values map to the correct model classes."""
    assert ModelType.Attention_MIL.model_class == Attention_MIL
    assert ModelType.BistroTransformer.model_class == BistroTransformer
    assert ModelType.TransMIL.model_class == TransMIL

def test_resolution_presets():
    """Test that RESOLUTION_PRESETS have correct attributes."""
    low = RESOLUTION_PRESETS.Low
    assert low.magnification == "10x"
    assert low.tile_px == 1_000

    high = RESOLUTION_PRESETS.High
    assert high.magnification == "20x"
    assert high.tile_px == 299

    ultra_low = RESOLUTION_PRESETS.Ultra_Low
    assert ultra_low.magnification == "5x"
    assert ultra_low.tile_px == 2_000

    ultra = RESOLUTION_PRESETS.Ultra
    assert ultra.magnification == "40x"
    assert ultra.tile_px == 224

def test_batch_generator_exact_multiple():
    """Test batch_generator with exact multiples."""
    data = [1, 2, 3, 4]
    batches = [batch for batch in batch_generator(data, batch_size=2)]

    assert batches == [
        [1, 2],
        [3, 4]
    ]

def test_batch_generator_with_remainder():
    """Test batch_generator with remainder."""
    data = [1, 2, 3, 4, 5]
    batches = [batch for batch in batch_generator(data, batch_size=2)]

    assert batches == [
        [1, 2],
        [3, 4],
        [5]
    ]

def test_batch_generator_batch_size_larger_than_list():
    """Test batch_generator when batch size is larger than the list."""
    data = [1, 2, 3]
    batches = list(batch_generator(data, batch_size=10))

    assert batches == [
        [1, 2, 3],
    ]

def test_batch_generator_with_batch_size_one():
    """Test batch_generator with batch size of one."""
    data = [1, 2, 3]
    batches = list(batch_generator(data, batch_size=1))

    assert batches == [
        [1],
        [2],
        [3],
    ]

def test_is_input_pretiled_true(tmp_path):
    """Test is_input_pretiled returns True for pretiled input."""
    pretiled_dir = tmp_path / "pretiled"

    create_pretiled_structure(
        root=pretiled_dir,
        slides={
            "slide1": 3,
            "slide2": 2,
        },
        ext=".tiff",
    )

    assert is_input_pretiled(pretiled_dir) is True

def test_is_input_pretiled_false_no_subdirs(tmp_path):
    """Test is_input_pretiled returns False when there are no subdirectories."""
    pretiled_dir = tmp_path / "slides"
    pretiled_dir.mkdir()

    # Create some files directly in the root directory, not in subdirectories
    for i in range(3):
        (pretiled_dir / f"tile_{i}.tiff").touch()

    assert is_input_pretiled(pretiled_dir) is False

def test_is_input_pretiled_invalid_extension(tmp_path):
    pretiled_dir = tmp_path / "slides"

    create_pretiled_structure(
        pretiled_dir,
        slides={"slideA": 1},
        ext=".txt",
    )

    assert is_input_pretiled(pretiled_dir) is False

def test_is_input_pretiled_empty_directory(tmp_path):
    """Test is_input_pretiled returns False for an empty directory."""
    pretiled_dir = tmp_path / "empty_slides"
    pretiled_dir.mkdir()

    assert is_input_pretiled(pretiled_dir) is False

def test_is_input_pretiled_with_slide_ids(tmp_path):
    """Test is_input_pretiled with provided slide IDs."""
    pretiled_dir = tmp_path / "pretiled"

    create_pretiled_structure(
        root=pretiled_dir,
        slides={
            "slide1": 2,
            "slide2": 2,
            "slide3": 2,
        },
        ext=".tiff",
    )

    slide_ids = ["slide1", "slide2", "slide3"]
    assert is_input_pretiled(pretiled_dir, slide_ids=slide_ids) is True

def test_get_unique_labels(tmp_path):
    """Test get_unique_labels returns correct unique labels."""
    df = pd.DataFrame(
        {
            "patient": [1, 2, 3, 4],
            "label": ["A", "B", "A", "C"],
        }
    )
    df.to_csv(tmp_path / "annotations.csv", index=False)
    unique_labels = get_unique_labels(tmp_path / "annotations.csv", "label")

    assert unique_labels == ["A", "B", "C"]
