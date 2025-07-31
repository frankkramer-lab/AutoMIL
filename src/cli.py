import warnings

# Suppressing warnings related to pkg_ressources and timm
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import click
import slideflow as sf
from slideflow import Dataset, Project

from pipeline import (configure_image_backend, create_project_scaffold,
                      run_dataset_setup_loop, setup_dataset, setup_project,
                      train_with_estimate_comparison)
from utils import RESOLUTION_PRESETS, ModelType

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(name="AutoMIL",      context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.argument("project_dir",      type=click.Path(file_okay=False))
@click.option("-pc", "--patient_column", type=str, default="patient", help="Name of the column containing patient IDs")
@click.option("-lc", "--label_column",   type=str, default="label",   help="Name of the column containing labels")
@click.option("-sc", "--slide_column",   type=str, default=None,      help="Name of the column containing slide names")
@click.option("-k",                      type=int, default=3,         help="number of folds to train per resolution level")
@click.option("-t", "--transform_labels", is_flag=True,                 help="Transforms labels to float values (0.0, 1.0, ...)")
@click.option("-v", "--verbose",          is_flag=True,                 help="Enables additional logging messages")
@click.option("-c", "--cleanup",          is_flag=True,                 help="Deletes the created project structure")
def AutoMIL(
    slide_dir:       str,
    annotation_file: str,
    project_dir:     str,
    patient_column:  str,
    label_column:    str,
    slide_column:    str,
    k:               int,
    transform_labels: bool,
    verbose:          bool,
    cleanup:          bool
    ):
    sf.setLoggingLevel(10) # INFO: 20, DEBUG: 10

    # --- 1. Image Backend Configuration ---
    png_slides_present: bool = any(
        [slide.suffix.lower() == ".png" for slide in Path(slide_dir).iterdir()]
    )
    # If no PNG slides are present, we configure the image backend
    if not png_slides_present:
        configure_image_backend(verbose=verbose)
    tiff_conversion = png_slides_present

    # --- 2. Project Creation And Setup ---
    modified_annotation_file, label_map = create_project_scaffold(
        Path(project_dir),
        Path(annotation_file),
        patient_column,
        label_column,
        slide_column,
        verbose=verbose,
        transform_labels=transform_labels,
    )
    project: sf.Project = setup_project(
        Path(slide_dir),
        Path(project_dir),
        modified_annotation_file,
        verbose=verbose,
    )

    # --- 3. Setup Dataset Sources ---
    dataset = setup_dataset(
        project,
        RESOLUTION_PRESETS.Low,
        label_map,
        Path(slide_dir),
        verbose=verbose,
        tiff_conversion=tiff_conversion,
    )

    # --- 4. Run Training ---
    train_with_estimate_comparison(
        ModelType.Attention_MIL,
        project,
        dataset,
        k=3,
        verbose=verbose,
    )


if __name__ == '__main__':
    AutoMIL()





