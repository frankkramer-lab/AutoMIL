import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from automil.dataset import Dataset
from automil.tiling import SlideflowBackend, SlideforgeBackend, get_tile_backend, write_index
from automil.util import RESOLUTION_PRESETS, TileBackendType


@pytest.fixture
def fake_slideforge(monkeypatch):
    """Stand-in for the slideforge binding, which writes an empty tfrecord on extract."""
    module = MagicMock()
    module.extract.side_effect = lambda path, tfrecord, **kwargs: (open(tfrecord, "wb").close(), (1, 2))[1]
    monkeypatch.setitem(sys.modules, "slideforge", module)
    return module


@pytest.fixture
def tiling_dataset(tmp_path):
    """Mock sf.Dataset with two slides and a tfrecords folder."""
    dataset = MagicMock()
    dataset.tile_px = 256
    dataset.tile_um = 128
    dataset.tfrecords_folders.return_value = [str(tmp_path / "tfrecords" / "256px_128um")]
    dataset.slide_paths.return_value = [str(tmp_path / "a.svs"), str(tmp_path / "b.svs")]
    return dataset


def test_get_tile_backend(fake_slideforge):
    assert isinstance(get_tile_backend(TileBackendType.Slideflow), SlideflowBackend)
    assert isinstance(get_tile_backend(TileBackendType.Slideforge), SlideforgeBackend)
    with pytest.raises(ValueError):
        get_tile_backend("unknown")  # type: ignore[arg-type]  # deliberately invalid


def test_slideforge_backend_missing_binding(monkeypatch):
    monkeypatch.setitem(sys.modules, "slideforge", None)
    with pytest.raises(ImportError, match="automil\\[slideforge\\]"):
        SlideforgeBackend()


def test_slideflow_backend_forwards_arguments(mock_sf_dataset):
    SlideflowBackend(report=False, num_threads=4).extract(
        mock_sf_dataset, tissue_detection="otsu", stain_normalizer="reinhard"
    )
    mock_sf_dataset.extract_tiles.assert_called_once_with(
        qc="otsu", normalizer="reinhard", report=False, num_threads=4
    )


@patch("automil.tiling.write_index")
def test_slideforge_backend_extracts_in_grid_mode(mock_write_index, fake_slideforge, tiling_dataset, tmp_path):
    SlideforgeBackend(min_tissue_fraction=0.3, threads=2).extract(
        tiling_dataset, tissue_detection="otsu", stain_normalizer="reinhard"
    )

    tfrecords_dir = tmp_path / "tfrecords" / "256px_128um"
    assert fake_slideforge.extract.call_count == 2
    fake_slideforge.extract.assert_any_call(
        str(tmp_path / "a.svs"),
        tfrecord=str(tfrecords_dir / "a.tfrecords"),
        target_mpp=0.5,
        tile_px=256,
        tile_um=128.0,
        min_tissue_fraction=0.3,
        stain_normalize=True,
        jpeg_quality=95,
        threads=2,
    )
    assert mock_write_index.call_count == 2
    tiling_dataset.update_manifest.assert_called_once_with(force_update=True)


@patch("automil.tiling.write_index")
def test_slideforge_backend_without_qc_and_normalization(mock_write_index, fake_slideforge, tiling_dataset):
    SlideforgeBackend().extract(tiling_dataset, tissue_detection=None, stain_normalizer=None)

    kwargs = fake_slideforge.extract.call_args.kwargs
    assert kwargs["min_tissue_fraction"] is None
    assert kwargs["stain_normalize"] is False


@patch("automil.tiling.write_index")
def test_slideforge_backend_skips_existing(mock_write_index, fake_slideforge, tiling_dataset, tmp_path):
    tfrecords_dir = tmp_path / "tfrecords" / "256px_128um"
    tfrecords_dir.mkdir(parents=True)
    (tfrecords_dir / "a.tfrecords").touch()

    SlideforgeBackend().extract(tiling_dataset, tissue_detection="otsu", stain_normalizer=None)

    assert fake_slideforge.extract.call_count == 1
    assert fake_slideforge.extract.call_args.args[0] == str(tmp_path / "b.svs")


def test_slideforge_backend_rejects_unsupported_normalizer(fake_slideforge, tiling_dataset):
    with pytest.raises(ValueError, match="reinhard"):
        SlideforgeBackend().extract(tiling_dataset, tissue_detection="otsu", stain_normalizer="macenko")


def _write_tfrecord(path, locations):
    """Writes a tfrecord with slideflow's own writer, one record per location"""
    from slideflow.io import TFRecordWriter
    from slideflow.io.torch import serialized_record

    writer = TFRecordWriter(str(path))
    for i, (loc_x, loc_y) in enumerate(locations):
        # Image sizes vary so records have different lengths
        writer.write(serialized_record(b"slide", bytes(range(256)) * (1 + i % 7), loc_x=loc_x, loc_y=loc_y))
    writer.close()


def test_write_index_matches_slideflow(tmp_path):
    from slideflow.util.tfrecord2idx import _build_index_from_tfrecord, find_index, index_has_locations, load_index

    locations = [(0, 0), (149, 447), (2**40, 7), (-5, 300), (127, 128), (16383, 16384)]
    tfrecord = tmp_path / "slide.tfrecords"
    _write_tfrecord(tfrecord, locations)

    write_index(tfrecord)

    expected_offsets, expected_locations = _build_index_from_tfrecord(str(tfrecord))
    index = find_index(str(tfrecord))
    assert index is not None and index == str(tmp_path / "slide.index.npz")
    assert index_has_locations(index)  # otherwise slideflow's build_index() would rebuild it
    np.testing.assert_array_equal(load_index(str(tfrecord)), expected_offsets)
    np.testing.assert_array_equal(np.load(index)["locations"], expected_locations)


def test_write_index_replaces_stale_index(tmp_path):
    from slideflow.util.tfrecord2idx import find_index

    tfrecord = tmp_path / "slide.tfrecords"
    _write_tfrecord(tfrecord, [(1, 2)])
    (tmp_path / "slide.index").write_text("0 1\n")  # stale plain-text index, which slideflow would prefer

    write_index(tfrecord)

    assert not (tmp_path / "slide.index").exists()
    assert find_index(str(tfrecord)) == str(tmp_path / "slide.index.npz")


def test_write_index_empty_tfrecord(tmp_path):
    from slideflow.util.tfrecord2idx import get_tfrecord_length

    tfrecord = tmp_path / "slide.tfrecords"
    tfrecord.touch()

    write_index(tfrecord)

    assert get_tfrecord_length(str(tfrecord)) == 0


@patch("automil.tiling.create_index")
def test_write_index_falls_back_to_slideflow(mock_create_index, tmp_path):
    tfrecord = tmp_path / "slide.tfrecords"
    # Valid framing, but the payload is not a tf.train.Example
    tfrecord.write_bytes((3).to_bytes(8, "little") + b"\0" * 4 + b"\xff\xff\xff" + b"\0" * 4)

    write_index(tfrecord)

    mock_create_index.assert_called_once_with(str(tfrecord))


@patch("automil.dataset.get_tile_backend")
def test_dataset_uses_configured_tiler(mock_get_backend, mock_project, mock_sf_dataset, tmp_path):
    dataset = Dataset(
        project=mock_project,
        resolution=RESOLUTION_PRESETS.Low,
        label_map={"A": 0},
        slide_dir=tmp_path,
        tiler=TileBackendType.Slideforge,
        verbose=False,
    )
    with patch.object(Dataset, "_compute_mpp", return_value=0.5):
        dataset._extract_tiles(mock_sf_dataset)

    mock_get_backend.assert_called_once_with(TileBackendType.Slideforge)
    mock_get_backend.return_value.extract.assert_called_once_with(
        mock_sf_dataset, tissue_detection="otsu", stain_normalizer="reinhard"
    )
