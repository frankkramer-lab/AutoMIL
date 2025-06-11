import os
import platform
import shutil
import sys
from pathlib import Path

import click
import slideflow as sf
from slideflow.mil import mil_config, train_mil
from slideflow.slide import qc
from slideflow.util import is_project, log

from utils import (BATCH_SIZE, EPOCHS, FEATURE_EXTRACTOR, LEARNING_RATE,
                   RESOLUTION_PRESETS)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(name="AutoMIL",      context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("project_dir",      type=click.Path(file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.option("-v", "--verbose",    is_flag=True, help="Enables additional logging messages")
@click.option("-c", "--cleanup",    is_flag=True, help="Deletes the created project structure")
def AutoMIL(slide_dir: str, project_dir: str, annotation_file: str, verbose: bool, cleanup: bool):

    # Shortening flag
    v = verbose
    # Logging colours/fonts
    c_info = "cyan"
    c_path = "bold"
    c_error = "red"

    # --- Image Backend Selection ---

    # During tiling, cucim and openslide create new processes/threads through 'forking'
    # 'fork' is an unsupported system call on windows, thus we use libvips
    # Make sure libvips is installed and in your PATH if you're using windows
    match (system := platform.system().lower()):
        case "windows":
            log.info(f"[{c_info}]Windows[/] detected")
            os.environ["SF_SLIDE_BACKEND"] = "libvips"
        case "linux":
            log.info(f"[{c_info}]Linux[/] detected")
            os.environ["SF_SLIDE_BACKEND"] = "openslide"
        case _:
            log.error(f"Invalid OS: [{c_error}]{system}[/] | Use either Windows or Linux")
            sys.exit(1)
    
    # --- Project Configuration ---

    # Checks if directory already exists and has correct structure
    if is_project(project_dir):
        log.info(f"[{c_path}]{project_dir}[/] is already a project. Loaded")
        project = sf.load_project(project_dir)
    else:
        # Returns a Project instance from which to build dataset
        log.info(f"Project structure created at [{c_path}]{project_dir}[/]")
        project = sf.create_project(
            name = "AutoMIL",
            root = project_dir,
            slides      = slide_dir,
            annotations = annotation_file
        )
    
    # By default, slideflow creates a blank annotation file slate
    # Thus we overwrite the blank slate with the given annotation file
    given_annotations = Path(annotation_file).name
    project_annotations = Path(project_dir) / given_annotations
    shutil.copy(annotation_file, project_annotations)

    # Setting up datasets with various resolutions
    for Preset in RESOLUTION_PRESETS:
        tile_size, magnification = Preset.value
        bags_path  = project_dir + "/bags"
        model_path = project_dir + "/models"

        # -- Tile Extraction & Bags ---
        dataset = project.dataset(
            tile_size,
            magnification
        )

        # extract tiles into datasets tile folder using normalizer
        dataset.extract_tiles(
            qc=qc.Otsu(),
            normalizer="reinhard_mask",
            report=False
        )

        project.generate_feature_bags(
            obj     = project,  # Somehow you have to pass the project as an argument
            model   = FEATURE_EXTRACTOR,
            dataset = dataset,
            outdir  = Path(project_dir) / "bags"
        )

        # --- Training ---

        # Define the model configuration
        config = mil_config(
            model      = "attention_mil",
            trainer    = "fastai",
            lr         = LEARNING_RATE,
            epochs     = EPOCHS,
            batch_size = BATCH_SIZE
        )

        # 5-fold split is default
        for train, val in dataset.kfold_split(k=5, labels="subtype"):
            
            # trained model will be stored in data/models/...
            train_mil(
                    config        = config,
                    train_dataset = train,
                    val_dataset   = val,
                    outcomes = "subtype",
                    project  = project,
                    bags     = bags_path,
                    outdir   = model_path
            )

        # --- Evaluation | TODO ---
        pass

    # --- Deleting Project Structure | OPTIONAL ---
    if cleanup:
        if v:
            log.info(f"Deleting project structure at [{c_path}]{project_dir}[/]")
        shutil.rmtree(project_dir)


if __name__ == '__main__':
    AutoMIL()



