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
"""Train/test split subcommand for AutoMIL CLI"""
import sys
import traceback
from pathlib import Path
import click

from ..help import CREATE_SPLIT_HELP
from ..constants import CONTEXT_SETTINGS

@click.command(
    "create-split",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    help=CREATE_SPLIT_HELP
)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.option(
    "-o", "--output-file", type=click.Path(file_okay=True), default="split.json",
    help="Path to which to save the split .json file"
)
@click.option("-f", "--test-fraction", type=float, default=0.2, help="Fraction of slides to include in the test set")
@click.option("-r", "--read-only", is_flag=True, help="If set, existing split file will not be overwritten")
@click.option("-v", "--verbose", is_flag=True, help="Enables additional logging messages")
def create_split(
    slide_dir:       str | Path,
    annotation_file: str | Path,
    output_file:     str | Path,
    test_fraction:   float,
    read_only:       bool,
    verbose:         bool
):
    """
    Create a train–test split file from dataset annotations.

    This command reads the provided annotation file and generates a train–test
    split, which is saved as a JSON file. The resulting split can be reused
    for reproducible training and evaluation.

    Args:
        slide_dir (str | Path):
            Directory containing whole-slide images.

        annotation_file (str | Path):
            CSV file containing slide- or patient-level annotations and labels.

        output_file (str | Path):
            Path to which the split JSON file will be written.

        test_fraction (float):
            Fraction of samples to assign to the test set.

        read_only (bool):
            If enabled, an existing split file will not be overwritten.

        verbose (bool):
            Enables verbose logging output.

    ### Examples

    Create a split with default settings:

        automil create-split /data/slides /data/annotations.csv -o split.json

    Create a split without overwriting an existing file:

        automil create-split /data/slides /data/annotations.csv -o split.json --read-only

    ### Output file format

    The output JSON file contains slide identifiers grouped by split name.

    Example structure:

        {
        "train": ["slide1", "slide2", ...],
        "test":  ["slide3", "slide4", ...]
        }

    Depending on the configuration, a `validation` split may be generated
    instead of or in addition to a `test` split.
    """

    import slideflow as sf

    from util import INFO_CLR, LogLevel, get_vlog

    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: [{INFO_CLR}]{command}[/]")

    # Some type coercion
    slide_dir = Path(slide_dir)
    annotation_file = Path(annotation_file)
    output_file = Path(output_file)

    try:
        # Minimal dataset for splitting
        dataset = sf.Dataset(
            slides=str(slide_dir),
            annotations=str(annotation_file)
        )
        # Create the split and save it
        _, _ = dataset.split(
            labels="label",
            val_fraction=test_fraction,
            splits=str(output_file),
            read_only=read_only
        )
    
    except Exception as e:
        tb = traceback.format_exc()
        vlog(tb, LogLevel.ERROR)
        vlog(f"Error: {e}", LogLevel.ERROR)
        return