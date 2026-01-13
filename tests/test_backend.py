import os
from pathlib import Path
from unittest.mock import patch

import pytest

from automil.util.backend import (configure_image_backend, has_png_slides,
                                  is_ome_tiff)


# === Helper Methods === #
def create_file(dir: Path, name: str) -> Path:
    """Helper method to create a dummy file."""
    path = dir / name
    path.touch()
    return path

def test_is_ome_tiff_detects_file(tmp_path):
    """Test that is_ome_tiff correctly identifies OME-TIFF files."""
    file_path = create_file(tmp_path, "slide.ome.tiff")
    assert is_ome_tiff(file_path) is True

def test_has_png_slides_detects_png(tmp_path):
    """Test that has_png_slides correctly identifies presence of PNG slides."""
    create_file(tmp_path, "slide1.png")
    create_file(tmp_path, "slide2.tiff")
    assert has_png_slides(tmp_path) is True

def test_backend_uses_cucim(tmp_path):
    """Test that the backend is default (cucim) when no special conditions are met."""
    result = configure_image_backend(
        tmp_path,
        needs_png_conversion=False,
        verbose=False
    )

    assert result is False
    assert "SF_SLIDE_BACKEND" not in os.environ or os.environ["SF_SLIDE_BACKEND"] != "cucim"

def test_backend_switches_to_libvips_for_png(tmp_path):
    """Test that the backend switches to libvips when PNG slides are present."""
    create_file(tmp_path, "slide1.png")

    with patch("automil.util.backend.libvips_available", return_value=True):
        result = configure_image_backend(
            tmp_path,
            needs_png_conversion=True,
            verbose=False,
        )

    assert result is True
    assert os.environ["SF_SLIDE_BACKEND"] == "libvips"

def test_backend_raises_error_when_png_present(tmp_path):
    """Test that an error is raised when png slides are present but libvips is unavailable."""
    create_file(tmp_path, "slide1.png")

    with patch("automil.util.backend.libvips_available", return_value=False):
        with pytest.raises(RuntimeError):
            configure_image_backend(
                tmp_path,
                needs_png_conversion=True,
                verbose=False,
            )

def test_backend_switches_to_libvips_for_ome_tiff(tmp_path):
    """Test that the backend switches to libvips when OME-TIFF slides are present."""
    create_file(tmp_path, "slide1.ome.tiff")

    with patch("automil.util.backend.libvips_available", return_value=True):
        result = configure_image_backend(
            tmp_path,
            needs_png_conversion=False,
            verbose=False,
        )

    assert result is False
    assert os.environ["SF_SLIDE_BACKEND"] == "libvips"

def test_backend_raises_error_when_ome_tiff_present(tmp_path):
    """Test that an error is raised when OME-TIFF slides are present but libvips is unavailable."""
    create_file(tmp_path, "slide1.ome.tiff")

    with patch("automil.util.backend.libvips_available", return_value=False):
        with pytest.raises(RuntimeError):
            configure_image_backend(
                tmp_path,
                needs_png_conversion=False,
                verbose=False,
            )
    


