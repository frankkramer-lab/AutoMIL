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
"""Training subcommand for AutoMIL CLI"""
import sys
import traceback
from pathlib import Path
import click

from ..help import TRAIN_HELP
from ..constants import CONTEXT_SETTINGS

from ..core import (train_arguments, column_overwrite_options, train_options,
                    preprocessing_options, dataset_options, verbose_option
)

@click.command(
    name="train",
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    help=TRAIN_HELP
)
@train_arguments
@column_overwrite_options
@train_options
@preprocessing_options
@dataset_options
@verbose_option
def train(
    slide_dir:       Path,
    annotation_file: Path,
    project_dir:     Path,
    patient_column:  str,
    label_column:    str,
    slide_column:    str | None,
    resolutions:     str,
    model:           str,
    tissue_detection: str,
    stain_normalizer: str,
    k:               int,
    is_pretiled:      bool,
    transform_labels: bool,
    verbose:          bool
):
    """
    Train one or more MIL models on a given dataset.

    This command initializes an AutoMIL project, prepares the dataset,
    and trains MIL models using k-fold cross-validation. Training can be
    performed at one or multiple resolution presets.

    Pipeline stages:

    1. Project setup and configuration
    2. Dataset preparation and tile extraction
    3. Model training with k-fold cross-validation

    Args:
        slide_dir (str | Path):
            Directory containing whole-slide images or pre-extracted tiles.

        annotation_file (str | Path):
            CSV file containing slide- or patient-level annotations and labels.

        project_dir (str | Path):
            Output directory where trained models and intermediate files
            will be written.

        patient_column (str):
            Name of the column containing patient identifiers.

        label_column (str):
            Name of the column containing class labels.

        slide_column (str | None):
            Name of the column containing slide identifiers.

        resolutions (str):
            Comma-separated list of resolution presets to train on.

        model (str):
            Model architecture to train.

        k (int):
            Number of folds used for k-fold cross-validation.

        is_pretiled (bool):
            Indicates that the input slides are already tiled.

        transform_labels (bool):
            If enabled, transforms labels to floating-point values.

        verbose (bool):
            Enables verbose logging output.

    ### Examples
      Basic usage with default settings:

        automil train /data/slides /data/annotations.csv ./results

      Multi-resolution training with verbose output::

        automil train -r "Low,High" -v /data/slides /data/annotations.csv ./results

      Custom model and 5-fold configuration:

        automil train -m TransMIL -k 5 /data/slides /data/annotations.csv ./results

      Using pre-tiled slides::
        
        automil train -p /data/slides /data/annotations.csv ./results

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

    ### Expected slide directory structure
      `SLIDE_DIR` should contain whole slide images in supported formats
      such as .svs, .tiff, or .png.
      Example structure:

        /data/slides/
        |-- slide1.svs
        |-- slide2.tiff
        |-- slide3.tiff
    
    ??? Note "PNG Slide Handling"
        If slides are in PNG, AutoMIL will first convert them to TIFF for easier processing.

    ### Using pretiled data
      If tiles have already been extracted from the slides, use the `--is_pretiled` flag.
      In the case of pretiled data, AutoMIL expects the following directory structure for `SLIDE_DIR`:

        /data/slides/
        |-- slide1/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
        |-- slide2/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
    
    ??? Note "Slide name matching"
        Tile names are arbitrary but slide subdirectories must match the slide names in ANNOTATION_FILE.

    ### Providing a train-test split
      Use the `--split-file` option to provide a JSON file defining train-test splits.
      The JSON file will have the following structure:

            {
            "train": ["slide1", "slide2", ...],
            "test":  ["slide3", "slide4", ...]
            }

      or:

            {
            "train": ["slide1", "slide2", ...],
            "validation":  ["slide3", "slide4", ...]
            }

    ### Output structure

        project_dir/
        ├── bags/           # Extracted tile features
        ├── models/         # Trained model checkpoints  
        ├── ensemble/       # Ensemble predictions
        ├── annotations.csv # Processed annotations
        └── results.json    # Performance metrics
    
    """
    
    import slideflow as sf

    from dataset import Dataset
    from project import Project
    from trainer import Trainer
    from util import (INFO_CLR, RESOLUTION_PRESETS, LogLevel, ModelType,
                       get_vlog)
    from util.backend import configure_image_backend, has_png_slides
    from util.pretiled import is_input_pretiled

    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: [{INFO_CLR}]{command}[/]")

    # Define some paths
    bags_dir = Path(project_dir) / "bags"

    # Some type coercion
    slide_dir = Path(slide_dir)
    annotation_file = Path(annotation_file)
    project_dir = Path(project_dir)

    try:

        # === 1. Parsing === #
        # Parse given string resolutions into list of RESOLUTION_PRESETS
        resolution_presets: list[RESOLUTION_PRESETS] = []
        for res in [r.strip() for r in resolutions.split(',')]: resolution_presets.append(RESOLUTION_PRESETS[res])
        vlog(f"Using resolution presets: [{INFO_CLR}]{[preset.name for preset in resolution_presets]}[/]")

        # Parse the model type
        model_type = ModelType[model]
        vlog(f"Using model type: [{INFO_CLR}]{model_type.name}[/]")

        # === 2. Image Backend Configuration === #
        png_slides_present = has_png_slides(slide_dir)

        tiff_conversion = configure_image_backend(
            slide_dir=slide_dir,
            needs_png_conversion=png_slides_present,
            verbose=verbose,
        )

        # === 3. Project Creation And Setup === #
        project_setup = Project(
            Path(project_dir),
            Path(annotation_file),
            Path(slide_dir),
            patient_column,
            label_column,
            slide_column,
            transform_labels=transform_labels,
            verbose=verbose,
        )
        # Prepare slideflow project object
        project = project_setup.prepare_project()
        # We'll need the label map and slide ids for the dataset setup
        label_map = project_setup.label_map
        slide_ids = project_setup.slide_ids

        project_setup.summary()
        
        # === 4. Setup Dataset Sources ===
        # Determine if the slide_dir has pretiled slides
        if not is_pretiled: # is_pretiled == False means the flag was not set
            is_pretiled = is_input_pretiled(
                slide_dir,
                slide_ids
            )

        datasets: dict[str, sf.Dataset] = {}
        for resolution in resolution_presets:
            vlog(f"Setting up dataset for resolution preset: [{INFO_CLR}]{resolution.name}[/]")

            dataset = Dataset(
                project,
                resolution,
                label_map,
                slide_dir=Path(slide_dir),
                tissue_detection=tissue_detection,
                stain_normalizer=stain_normalizer,
                bags_dir=Path(project_dir) / "bags",
                is_pretiled=is_pretiled,
                tiff_conversion=tiff_conversion,
                verbose=verbose
            )
            dataset.summary()
            datasets[resolution.name] = dataset.prepare_dataset_source()
            vlog(f"Dataset setup complete for resolution preset: [{INFO_CLR}]{resolution.name}[/]")
        
        # === 5. Model Training === #
        for resolution in resolution_presets:
            dataset = datasets[resolution.name]
            vlog(f"Train/Test split for resolution preset '[{INFO_CLR}]{resolution.name}[/]': "
                 f"[{INFO_CLR}]{len(dataset.slides())}[/] train slides"
            )

            train, val = dataset.split(
                labels="label",
                val_fraction=0.2
            )

            trainer = Trainer(
                bags_dir,
                project,
                train,
                val,
                model=model_type,
                k=k,
                epochs=300
            )
            trainer.train_k_fold()
            trainer.summary()
    
    except Exception as e:
        tb = traceback.format_exc()
        vlog(tb, LogLevel.ERROR)
        vlog(f"Error: {e}", LogLevel.ERROR)
        return