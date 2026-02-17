#==============================================================================#
#  AutoMIL - Automated Machine Learning for Image Classification in            #
#  Whole-Slide Imaging with Multiple Instance Learning                         #
#                                                                              #
#  Copyright (C) 2026 Jonas Waibel                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.      #
#==============================================================================#
"""
Uility functions for converting pretiled images to TFRecords.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from slideflow.io import TFRecordWriter, write_tfrecords_multi
from slideflow.io.torch import serialized_record
from slideflow.util.tfrecord2idx import create_index

from .constants import INFO_CLR
from .logging import get_vlog


def is_input_pretiled(slide_dir: Path, slide_ids: list[str] | None = None) -> bool:
    """Check if the input slide directory contains pretiled slides.

    `slide_dir` is considered pretiled, if the following conditions are met:
        1. `slide_dir` contains a subdirectory for each slide (Ideally named after the slide ID)
        2. Each slide subdirectory contains loose image tiles (.tif, .tiff, .png, .svs)
    
    Example structure:
    ```
        slide_dir/
        |-- slide1/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
        |-- slide2/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
    ```

    Args:
        slide_dir: Path to the slide directory
        slide_ids: Optional list of expected slide IDs (will check if subdirectory names match these IDs)
    Returns:
        True if the slide directory contains pretiled slides, False otherwise.
    """
    tile_formats = [".png", ".svs", ".tiff", ".tif"]
    if not slide_dir.is_dir() or not slide_dir.exists():
        return False
    
    # Collect all subdirectories
    slide_subdirs = [entry for entry in slide_dir.iterdir() if entry.is_dir()]
    
    # Check if there even are subdirectories
    if not slide_subdirs:
        return False

    # If slide IDs are provided, check subdirectory names against them
    if slide_ids:
        if not all(
            slide_subdir.name in slide_ids
            for slide_subdir in slide_subdirs
        ):
            return False

    # Check that subdirectories contain loose image files
    if not all(
        file.suffix in tile_formats for slide_subdir in slide_subdirs
        for file in slide_subdir.iterdir() if file.is_file()
    ):
        return False

    # If all checks were passed, this is (likely) a pretiled input
    return True

def pretiled_to_tfrecords(
    pretiled_root: Path,
    outdir: Path,
    overwrite: bool = False,
    verbose: bool = True
):
    """
    Convert loose tiles organized by slide subfolders into TFRecords,
    assigning sequential indices compatible with Slideflow.

    Expected structure:
        pretiled_root/
            slideA/
                tile_00001.tif
                tile_00002.tif
                ...
            slideB/
                tile_00001.tif
                ...
    or:
        pretiled_root/
            slideA/
                tile_00001.tiff
                tile_00002.tiff
                ...
            slideB/
                tile_00001.tiff
                ...

    Args:
        pretiled_root: Root folder with one subfolder per slide.
        outdir: Destination directory for .tfrecord files.
        overwrite: Whether to overwrite existing TFRecord files.
        verbose: Whether to print progress information.
    """
    vlog = get_vlog(verbose)
    outdir.mkdir(parents=True, exist_ok=True)

    slide_dirs = sorted([entry for entry in pretiled_root.iterdir() if entry.is_dir()])
    if not slide_dirs:
        raise ValueError(f"No slide subdirectories found in {pretiled_root}")

    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        tfrecord_path = outdir / f"{slide_id}.tfrecords"
        if tfrecord_path.exists() and not overwrite:
            vlog(f"[{INFO_CLR}]{tfrecord_path}[/] already exists. Skipping.")
            continue
        
        extensions = [".png", ".svs", ".tif", ".tiff"]
        tile_paths = sorted([tile for tile in slide_dir.iterdir() if tile.suffix in extensions])
        if not tile_paths:
            vlog(f"No tiles found in [{INFO_CLR}]{slide_dir}[/], skipping.")
            continue

        writer = TFRecordWriter(str(tfrecord_path))
        vlog(f"Writing [{INFO_CLR}]{len(tile_paths)}[/] tiles for [{INFO_CLR}]{slide_id}[/]")

        for idx, tile_path in enumerate(tile_paths):
            # Aritrary grid location
            grid_width = len(tile_paths) if len(tile_paths) < 32 else 32
            loc_x = int(idx % grid_width)
            loc_y = int(idx // grid_width)

            # Convert tile to bytes
            label = bytes(slide_id, 'utf-8')
            img = np.array(Image.open(tile_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_string = cv2.imencode(".png", img)[1].tobytes()

            # Serializze and write record
            record = serialized_record(label, image_string, loc_x=loc_x, loc_y=loc_y)
            writer.write(record)

        writer.close()

        # Create index file for the TFRecords
        create_index(str(tfrecord_path))

def pretiled_to_tfrecords_multi(
    pretiled_root: Path,
    outdir: Path,
    verbose: bool = True
):
    """
    Convert loose .tif/.tiff tiles organized by slide subfolders into TFRecords

    Expected structure:
        pretiled_root/
            slideA/
                tile_00001.tif
                tile_00002.tif
                ...
            slideB/
                tile_00001.tif
                ...
    or:
        pretiled_root/
            slideA/
                tile_00001.tiff
                tile_00002.tiff
                ...
            slideB/
                tile_00001.tiff
                ...

    Args:
        pretiled_root: Root folder with one subfolder per slide.
        outdir: Destination directory for .tfrecord files.
    """
    vlog = get_vlog(verbose)
    outdir.mkdir(parents=True, exist_ok=True)

    vlog(f"Converting pretiled images in [{INFO_CLR}]{pretiled_root}[/] to TFRecords in [{INFO_CLR}]{outdir}[/]")
    # Use write_tfrecords_multi to process multiple slide subdirectories
    # Each subdirectory name is taken as a slide name
    write_tfrecords_multi(
        str(pretiled_root),
        str(outdir)
    )