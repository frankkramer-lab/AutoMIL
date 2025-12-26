from pathlib import Path
from unittest.mock import patch

from automil.dataset import Dataset
from automil.utils import RESOLUTION_PRESETS


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


def test_tfrecords_dir_paths(mock_project):
    """Test tfrecords_dir returns correct paths for different configurations."""
    # Standard configuration
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0}
    )
    assert dataset.tfrecords_dir == Path("/tmp/project/tfrecords")
    
    # Pretiled configuration
    dataset_pretiled = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0},
        is_pretiled=True
    )
    assert dataset_pretiled.tfrecords_dir == Path("/tmp/project/tfrecords/pretiled")
    
    # TIFF conversion configuration
    dataset_tiff = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0},
        tiff_conversion=True
    )
    assert dataset_tiff.tfrecords_dir == Path("/tmp/project/tfrecords/tiff_buffer")


def test_compute_mpp_fallback_when_no_slide_dir(mock_project):
    """Test MPP computation fallback to common values when slide_dir is None."""
    from automil.utils import COMMON_MPP_VALUES

    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.High,
        label_map={"A": 0}
    )

    dataset.slide_dir = None
    mpp = dataset._compute_mpp()

    expected = COMMON_MPP_VALUES.get(dataset.magnification, 0.5)
    assert mpp == expected


def test_compute_mpp_by_average(mock_project):
    """Test MPP computation by averaging all slides."""
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.High,
        label_map={"A": 0},
        slide_dir=Path("/fake/slides")
    )

    with (
        patch("automil.dataset.calculate_average_mpp", return_value=0.75),
        patch.object(Path, "exists", lambda self: True),
    ):
        mpp = dataset._compute_mpp(by_average=True)

    assert mpp == 0.75

def test_compute_mpp_from_first_slide(mock_project):
    """Test MPP computation from the first slide."""
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.High,
        label_map={"A": 0},
        slide_dir=Path("/fake/slides")
    )

    with (
        patch("automil.dataset.get_mpp_from_slide", return_value=0.6),
        patch.object(Path, "exists", lambda self: True),
        patch.object(Path, "glob", lambda self, _: iter([Path("slide.svs")])),
    ):
        mpp = dataset._compute_mpp(by_average=False)

    assert mpp == 0.6

