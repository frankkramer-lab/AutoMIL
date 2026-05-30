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
"""Prediction subcommand for AutoMIL CLI"""
import sys
import traceback
from pathlib import Path

import click

from ..constants import CONTEXT_SETTINGS
from ..core import (column_overwrite_options, output_dir_option,
                    predict_arguments, verbose_option)
from ..help import PREDICT_HELP


@click.command(
    name="predict",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    help=PREDICT_HELP
)
@predict_arguments
@output_dir_option(default="predictions")
@column_overwrite_options
@verbose_option
def predict(
    slide_dir:   str | Path,
    annotation_file: str | Path,
    bags_dir:    str | Path,
    model_dir:   str | Path,
    output_dir: str | Path,
    patient_column:  str,
    label_column:    str,
    slide_column:    str | None,
    verbose:     bool
):
    """
    Generate predictions using one or more trained MIL models.

    This command loads trained model checkpoints and generates predictions
    for the slides in `SLIDE_DIR` using precomputed tile feature bags from
    `BAGS_DIR`. Predictions are written to the specified output directory.

    Args:
        slide_dir (str | Path):
            Directory containing whole-slide images.

        annotation_file (str | Path):
            CSV file containing slide- or patient-level annotations and labels.

        bags_dir (str | Path):
            Directory containing extracted tile feature bags.

        model_dir (str | Path):
            Directory containing trained model checkpoints.

        output_dir (str | Path):
            Directory to which prediction files will be written.

        patient_column (str):
            Name of the column containing patient identifiers.

        label_column (str):
            Name of the column containing class labels.

        slide_column (str | None):
            Name of the column containing slide identifiers.

        verbose (bool):
            Enables verbose logging output.

    ### Examples

    Basic usage with multiple models:

        automil predict /data/slides /data/annotations.csv /data/bags /data/models -o ./predictions

    Generate predictions with a single model:

        automil predict /data/slides /data/annotations.csv /data/bags /data/models/model_1 -v

    Override annotation column names:

        automil predict -pc "patient_id" -lc "outcome" -sc "slide_id" \
            /data/slides /data/annotations.csv /data/bags /data/models/model_1 \
            -o ./predictions

    ### Expected model directory structure

    `MODEL_DIR` may either point to a single model directory or to a parent
    directory containing multiple model subdirectories.

    Single model example:

        /data/models/model_1/
        |-- best_valid.pth
        |-- ...

    Multiple models example:

        /data/models/
        |-- model_1/
        |    |-- best_valid.pth
        |-- model_2/
        |    |-- best_valid.pth
        |    |-- ...

    ??? Note "Multiple models"
        When multiple models are provided, AutoMIL generates a separate
        prediction file for each model.

    ### Annotation file requirements

    The annotation file must be a CSV file containing at least the following columns:

    - Patient identifiers (default column name: `patient`)
    - Slide identifiers (default column name: `slide`; optional)
    - Class labels (default column name: `label`)

    By default, AutoMIL looks for columns named `patient`, `slide`, and `label`.
    These defaults can be overridden using the `--patient_column`,
    `--slide_column`, and `--label_column` options.

    ### Minimal annotation file example

        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1

    ### Output directory format

    `OUTPUT_DIR` must be a directory path. Prediction results are saved as
    separate `.csv` or `.parquet` files inside this directory.

    When multiple models are used, output files include a suffix indicating
    the corresponding model.
    """

    import slideflow as sf
    from evaluation import Evaluator
    from project import Project
    from util import INFO_CLR, LogLevel, get_vlog

    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: [{INFO_CLR}]{command}[/]")

    # Some type coercion
    slide_dir = Path(slide_dir)
    bags_dir =  Path(bags_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    # Setup output folder as project (modifies annotation file)
    project = Project(
        Path(output_dir),
        Path(annotation_file),
        Path(slide_dir),
        patient_column,
        label_column,
        slide_column,
        transform_labels=False,
        verbose=verbose,
    )
    project.setup_project_scaffold()
    annotation_file = project.modified_annotations_file
    
    # Create a minimal dataset (needed for prediction)
    dataset = sf.Dataset(
        slides=str(slide_dir),
        annotations=str(annotation_file)
    )

    # Generate predictions
    try:
        evaluator = Evaluator(
            dataset,
            model_dir,
            output_dir,
            bags_dir,
            verbose=verbose
        )
        evaluator.generate_predictions()

    except Exception as e:
        tb = traceback.format_exc()
        vlog(tb, LogLevel.ERROR)
        vlog(f"Error: {e}", LogLevel.ERROR)
        return