# automil/utils/backend.py

import os
import platform
from pathlib import Path

from .constants import INFO_CLR
from .enums import LogLevel
from .logging import get_vlog


# === Helpers === #
def libvips_available() -> bool:
    """Checks if libvips (or more specifically the python API pyvips) is available

    Returns:
        bool: Whether libvips is available
    """
    try:
        import pyvips  # noqa
        return True
    except Exception:
        return False

def is_ome_tiff(file_path: Path) -> bool:
    """Checks if the given file path is an OME-TIFF file.
    OME-TIFF is a popular file format for Whole Slide Images: https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/

    Args:
        file_path (Path): Path to a file

    Returns:
        bool: Whether `file_path` is an OME-TIFF file
    """
    suffixes = [suffix.lower() for suffix in file_path.suffixes]
    return suffixes in ([".ome", ".tiff"], [".ome", ".tif"])

def has_png_slides(slide_dir: Path) -> bool:
    """Checks if the given slide directory contains any .png slides

    Args:
        slide_dir (Path): The slide directory to check

    Returns:
        bool: Whether the slide directory contains any .png slides
    """
    return any(p.suffix.lower() == ".png" for p in slide_dir.iterdir())


def configure_image_backend(
    slide_dir: Path,
    *,
    needs_png_conversion: bool,
    verbose: bool = True,
) -> bool:
    """Configures the slideflow image backend.

    By default slideflow uses the cucim library for Whole Slide Image reading: https://docs.rapids.ai/api/cucim/stable/
    However, cucim is not available on Windows, and has limited support for certain file formats (e.g. PNG, OME-TIFF).

    In the following cases, the backend is configured to use libvips instead:
    - The operating system is Windows
    - The slide directory contains any PNG slides | TIFF conversion is needed
    - The slide directory contains any OME-TIFF slides (which cucim does not support)   

    Args:
        slide_dir (Path): The slide directory
        needs_png_conversion (bool): Whether PNG -> TIFF conversion is needed
        verbose (bool, optional): Whether to print verbose messages. Defaults to True.

    Raises:
        RuntimeError: If libvips is required but not available

    Returns:
        bool: Whether PNG -> TIFF conversion is needed
    """
    vlog = get_vlog(verbose)

    ome_tiff_present = any(is_ome_tiff(p) for p in slide_dir.iterdir())

    requires_libvips = (
        needs_png_conversion
        or ome_tiff_present
    )

    if not requires_libvips:
        vlog(f"Using default image backend [{INFO_CLR}]cucim[/]")
        return False

    if not libvips_available():

        error_message = ""
        if needs_png_conversion:
            error_message += "- PNG slides detected (requires PNG -> TIFF conversion)\n"
        if ome_tiff_present:
            error_message += "- OME-TIFF slides detected\n"

        vlog(
            "libvips required (but not installed) due to the following reasons:\n"
            f"{error_message}",
            LogLevel.ERROR,
        )
        raise RuntimeError("libvips backend required but unavailable")

    os.environ["SF_SLIDE_BACKEND"] = "libvips"
    vlog(f"Using [{INFO_CLR}]libvips[/] backend")
    return needs_png_conversion
