from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from automil.dataset import Dataset
from automil.util import COMMON_MPP_VALUES, RESOLUTION_PRESETS


def test_dataset_initialization(mock_project):
    """Test basic Dataset initialization."""
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"tumor": 1, "normal": 0},
    )

    assert dataset.project is mock_project
    assert dataset.resolution is RESOLUTION_PRESETS.Low
    assert dataset.label_map == {"tumor": 1, "normal": 0}
    assert dataset.is_pretiled is False


def test_cached_properties(mock_project):
    """Test that cached properties return expected values."""
    with patch.object(Dataset, "_compute_mpp", return_value=0.5):
        dataset = Dataset(
            project=mock_project,
            resolution=RESOLUTION_PRESETS.High,
            label_map={"A": 0}
        )

        assert dataset.tile_px == RESOLUTION_PRESETS.High.tile_px
        assert dataset.magnification == RESOLUTION_PRESETS.High.magnification
        assert dataset.mpp == 0.5
        assert dataset.tile_um == int(dataset.tile_px * 0.5)


def test_tfrecords_directory_paths(mock_project):
    """Test tfrecords_dir returns correct paths for different configurations."""
    base = Path(mock_project.root)

    # Standard configuration
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0}
    )
    assert dataset.tfrecords_dir == base / "tfrecords"
    
    # Pretiled configuration
    dataset_pretiled = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0},
        is_pretiled=True
    )
    assert dataset_pretiled.tfrecords_dir == base / "tfrecords" / "pretiled"
    
    # TIFF conversion configuration
    dataset_tiff = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0},
        tiff_conversion=True
    )
    assert dataset_tiff.tfrecords_dir == base / "tfrecords" / "tiff_buffer"

def test_mpp_falls_back_to_common_values_when_no_slide_dir(mock_project):
    """
    Test that Dataset.mpp falls back to COMMON_MPP_VALUES when no slide_dir is provided.
    """
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.High,
        label_map={"A": 0},
    )

    dataset.slide_dir = None

    mpp = dataset.mpp
    expected = COMMON_MPP_VALUES.get(dataset.magnification, 0.5)

    assert mpp == expected


def test_mpp_uses_calculated_average_when_available(mock_project):
    """
    Test that Dataset.mpp uses the averaged MPP when calculate_average_mpp returns a value.
    """
    with (
        patch("automil.dataset.calculate_average_mpp", return_value=0.75),
        patch.object(Path, "exists", lambda self: True),
        patch.object(Path, "iterdir", lambda self: iter(()))
    ):
        dataset = Dataset(
            project=mock_project,
            resolution=RESOLUTION_PRESETS.High,
            label_map={"A": 0},
            slide_dir=Path("/fake/slides"),
        )

        assert dataset.mpp == 0.75


def test_tile_um_is_derived_from_mpp_and_tile_px(mock_project):
    """
    Test that tile_um are computed consistently and correctly from tile_px and mpp.
    """
    with (
        patch("automil.util.slide.calculate_average_mpp", return_value=0.5),
        patch.object(Path, "exists", lambda self: True),
        patch.object(Path, "iterdir", lambda self: iter(()))
    ):
        dataset = Dataset(
            project=mock_project,
            resolution=RESOLUTION_PRESETS.High,
            label_map={"A": 0},
            slide_dir=Path("/fake/slides"),
        )

        assert dataset.tile_um == int(dataset.tile_px * dataset.mpp)


def test_pretiled_without_slide_dir_raises_error(mock_project):
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0},
        is_pretiled=True,
        slide_dir=None,
    )

    with pytest.raises(ValueError, match="slide_dir must be provided"):
        dataset.prepare_dataset_source()


def test_prepare_dataset_source_calls_convert_pretiled_if_is_pretiled(mock_project):
    fake_dataset = MagicMock()

    with (
        patch.object(Dataset, "_convert_pretiled", return_value=fake_dataset) as mock_convert,
        patch.object(Dataset, "_apply_label_filter", return_value=fake_dataset),
        patch.object(Dataset, "_extract_features"),
    ):
        dataset = Dataset(
            project=mock_project,
            resolution=RESOLUTION_PRESETS.Low,
            label_map={"A": 0},
            is_pretiled=True,
            slide_dir=Path("/fake/slides"),
        )

        result = dataset.prepare_dataset_source()

        mock_convert.assert_called_once()
        assert result is fake_dataset

def test_prepare_dataset_source_calls_extract_tiles_if_not_pretiled(mock_project):
    fake_dataset = MagicMock()

    mock_project.dataset = MagicMock(return_value=fake_dataset)

    with (
        patch.object(Dataset, "_apply_label_filter", return_value=fake_dataset),
        patch.object(Dataset, "_extract_tiles") as mock_extract,
        patch.object(Dataset, "_extract_features"),
    ):
        dataset = Dataset(
            project=mock_project,
            resolution=RESOLUTION_PRESETS.Low,
            label_map={"A": 0},
            is_pretiled=False,
        )

        result = dataset.prepare_dataset_source()

        mock_extract.assert_called_once_with(fake_dataset)
        assert result is fake_dataset


