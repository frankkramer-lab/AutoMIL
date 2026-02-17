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
Utility funtions for exttracting slide metadata using OpenSlide: https://openslide.org/
"""
from pathlib import Path
from typing import Optional

import openslide
from slideflow.dataset import Dataset as sfDataset


def get_num_slides(dataset: sfDataset) -> int:
    """Return the number of slides in a given dataset source.

    Args:
        dataset: Dataset object

    Returns:
        Number of slides in the dataset.
    """
    return len(dataset.slides())

def get_slide_magnification(slide_path: str | Path) -> Optional[str]:
    """Retrieve magnification from slide properties or estimate using microns per pixel.

    Args:
        slide_path: Path to slide as string or Path object

    Returns:
        Magnification as string (e.g. '20') or None if unable to determine.
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        # Typical property for objective magnification
        if (mag := slide.properties.get("openslide.objective-power")):
            return mag
        # Estimate magnification from mpp_x
        if (mpp_x := slide.properties.get("openslide.mpp-x")):
            estimated_mag = 10 / float(mpp_x)
            return str(estimated_mag)
    # Estimation (or opening slide) failed
    except Exception:
        return None

def get_lowest_magnification(slide_dir: Path) -> Optional[str]:
    """Return the lowest magnification from slides in a directory.

    Args:
        slide_dir: Path to directory containing slide files

    Returns:
        Lowest magnification as string or None if no valid magnification found.
    """
    magnifications = []
    for slide_path in slide_dir.iterdir():
        if (mag := get_slide_magnification(slide_path)):
            try:
                magnifications.append(float(mag))
            except ValueError:
                continue  # Skip invalid magnifications
    return str(min(magnifications)) if magnifications else None

def get_mpp_from_slide(slide_path: Path) -> Optional[float]:
    """Extract microns per pixel (MPP) from a slide file using OpenSlide.

    Args:
        slide_path: Path to the slide file (.svs, .tiff, etc.)

    Returns:
        MPP value if available, else None.
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        if (mpp_x := slide.properties.get("openslide.mpp-x")) is None:
            return None
        else:
            return float(mpp_x)
    except (openslide.OpenSlideError, ValueError, TypeError, KeyError):
        return None
    
def calculate_average_mpp(slide_dir: Path, return_rounded: bool = True) -> Optional[float]:
    """Calculate the average microns per pixel (MPP) from all slides in a directory.

    Args:
        slide_dir: Path to the directory containing slide files
        return_rounded: Whether to round the result to 2 decimal places

    Returns:
        Average MPP if available, else None.
    """
    mpp_values = []
    for slide_path in slide_dir.iterdir():
        if (mpp := get_mpp_from_slide(slide_path)):
            mpp_values.append(mpp)
    if not mpp_values: return None
    average_mpp = sum(mpp_values) / len(mpp_values)
    if return_rounded and average_mpp is not None:
        return round(average_mpp, 2)
    return average_mpp

def get_slide_properties(slide_path: str) -> Optional[dict]:
    """Retrieve slide properties as a dictionary.

    Args:
        slide_path: Path to slide as string

    Returns:
        Dictionary with slide properties or None if unable to retrieve.
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        return {key: val for key, val in slide.properties.items()}
    # Estimation (or opening slide) failed
    except Exception:
        return None