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
"""Core utilities and decorators for the AutoMIL CLI"""
from pathlib import Path
from typing import Any, Callable

# === External libraries === #
import click

# === Internal libraries === #
from .constants import MODEL_CHOICES, RESOLUTION_CHOICES
from .params import LazyChoice


# === Arguments === #
def train_arguments(f: Callable[..., Any]) -> Callable[..., Any]:
    """Training-related CLI arguments."""

    f = click.argument(
        "project_dir",
        type=click.Path(file_okay=False, path_type=Path)
    )(f)

    f = click.argument(
        "annotation_file",
        type=click.Path(exists=True, path_type=Path)
    )(f)

    f = click.argument(
        "slide_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path)
    )(f)

    return f

def predict_arguments(f: Callable[..., Any]) -> Callable[..., Any]:
    """The set of arguments for prediction related CLI commands such as `automil predict` and `automil evaluate`."""
    f = click.argument(
        "model_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path)
    )(f)

    f = click.argument(
        "bags_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path)
    )(f)

    f = click.argument(
        "annotation_file",
        type=click.Path(exists=True, file_okay=True, path_type=Path)
    )(f)

    f = click.argument(
        "slide_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path)
    )(f)

    return f

# === Options === #
def column_overwrite_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """All column overwrite options."""
    f = click.option(
        "-sc", "--slide_column",
        default=None,
        help="Name of the column containing slide names"
    )(f)

    f = click.option(
        "-lc", "--label_column",
        default="label",
        help="Name of the column containing labels"
    )(f)

    f = click.option(
        "-pc", "--patient_column",
        default="patient",
        help="Name of the column containing patient IDs"
    )(f)

    return f

def train_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """All training-specific options."""
    model_choice = click.Choice(MODEL_CHOICES)

    f = click.option(
        "-k",
        type=int,
        default=3,
        help="Number of folds"
    )(f)

    f = click.option(
        "-m", "--model",
        type=model_choice,
        default=model_choice.choices[0],
        help="Model type to train and evaluate"
    )(f)

    f = click.option(
        "-r", "--resolutions",
        default="Low",
        help=f"Comma-separated resolutions. Available: {', '.join(RESOLUTION_CHOICES)}"
    )(f)

    return f

def verbose_option(f: Callable[..., Any]) -> Callable[..., Any]:
    """Verbose flag."""
    return click.option(
        "-v", "--verbose",
        is_flag=True,
        help="Enables additional logging messages"
    )(f)

def output_dir_option(*, default: str):
    """Output directory option"""
    def decorator(f):
        return click.option(
            "-o", "--output-dir",
            type=click.Path(file_okay=True), default=default,
            help="Directory to which to save evaluation results"
        )(f)
    return decorator


def preprocessing_options(f):
    """All preprocessing (and processing) specific options."""
    td_choice = click.Choice(["otsu", "blur", "both"])

    f = click.option(
        "--tissue-detection",
        type=td_choice,
        default=td_choice.choices[0],
        help="Tissue detection method to utilize (e.g otsu)"
    )(f)

    f = click.option(
        "--stain-normalizer",
        type=LazyChoice(
            "slideflow.norm",
            attribute="StainNormalizer",
            transform=lambda cls: cls.normalizers.keys()
        ),
        default="reinhard",
        help="Stain normalization method to utilize (e.g macenko, reinhard etc.)"
    )(f)

    return f

def dataset_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Dataset related and specific options."""
    f = click.option(
        "-p", 
        "--is-pretiled",
        is_flag=True,
        help="Indicates that the input format is pretiled slides"
    )(f)
    f = click.option(
        "-t",
        "--transform_labels",
        is_flag=True,
        help="Transforms labels to float values (0.0, 1.0, ...)"
    )(f)

    return f

def run_pipeline_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Options specific to `automil run-pipeline`."""
    f = click.option(
        "--split-file",
        type=click.Path(file_okay=True),
        default="split.json",
        help="Path to a .json file defining train-test splits"
    )(f)

    return f