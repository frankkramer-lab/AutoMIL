import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import slideflow as sf
from slideflow.mil import mil_config, train_mil
from slideflow.mil.models import TransMIL
from slideflow.slide import qc
from slideflow.util import is_project, log

from estimator import adjust_batch_size
from utils import (BATCH_SIZE, EPOCHS, FEATURE_EXTRACTOR, LEARNING_RATE,
                   RESOLUTION_PRESETS, get_num_slides)


def configure_image_backend(png_slides_present: bool, verbose: bool = True):
    """Selects the image backend based on the systems operating system

    During tiling, cucim and openslide create new processes/threads through 'forking'. \\
    'fork' is an unsupported system call on windows, thus we use libvips. \\
    
    **NOTE**: Make sure libvips is installed and in your PATH if you're using windows. \\
    **NOTE**: This method terminates the current script upon encountering a invalid OS.
    
    Args:
        verbose (bool, optional): Whether to log info messages. Defaults to True.
    """
    # neither cucim nor openslide are capable of working with .png slides
    if png_slides_present:
        if verbose:
            log.info("Using [cyan]openslide[/] as some or all slides are in .png format")
        os.environ["SF_SLIDE_BACKEND"] = "openslide"
        return
    
    system = platform.system().lower()
    if system == "windows":
        if verbose:
            log.info("Using [cyan]libvips[/] backend on Windows")
        os.environ["SF_SLIDE_BACKEND"] = "libvips"
    elif system == "linux":
        if verbose:
            log.info("Using [cyan]cucim[/] backend on Linux")
        os.environ["SF_SLIDE_BACKEND"] = "cucim"
    else:
        if verbose:
            log.error(f"Unsupported OS: [red]{system}[/]")
        sys.exit(1)

def setup_annotations(
        annotations_file: Path,
        patient_column:   str,
        label_column:     str,
        project_dir:      Path,
        slide_column:     Optional[str] = None,
    ) -> Path:
    """Modifies a given annotations file to conform to slideflows expected format

    NOTE:
        The annotations file may contain any number of columns, but slideflow expects at least the 'label' and 'patient' column. \n
        This method renames the given columns to 'patient' and 'label' respectively and saves the modified .csv to the project folder.

    Args:
        annotations_file (Path): Path to the annotations file in .csv format
        patient_column (str): Name of the column containing patient identifiers (Will be renamed to 'patient')
        label_column (str): Name of the column containing labels (Will be renamed to 'label')
        project_dir (Path): Directory in which the project structure will be created (modified annotations will be stored here)

    Raises:
        Exception: If unable to create the modified annotation file

    Returns:
        Path: Path to modified annotations file (project_dir/annotations.csv)
    """
    annotations = pd.read_csv(annotations_file, index_col=patient_column)
    annotations.index.name = "patient"
    annotations.rename(columns={label_column: "label"}, inplace=True)
    if slide_column:
        annotations.rename(columns={slide_column: "slide"}, inplace=True)
    annotation_file = project_dir / "annotations.csv"
    annotations.to_csv(annotation_file, index=True)
    if not annotation_file.exists():
        raise Exception(f"Annotation file [bold]{annotation_file}[/] could not be created.")
    if not annotations.empty:
        log.info(f"Annotations saved to [bold]{annotation_file}[/]")

    return annotation_file  

def setup_project(
        slide_dir:       Path,
        annotation_file: Path,
        project_dir:     Path,
        patient_column:  str,
        label_column:    str,
        slide_column:    Optional[str] = None,
        verbose:         bool = True
    ) -> sf.Project:
    """Sets up the project structure in *project_dir* or loads it into a Project Instance, if it already exists

    Args:
        slide_dir (Path): directory containing slides
        project_dir (Path): directory in which the project structure should be created or is already present
        annotation_file (Path): .csv file with annotations
        verbose (bool, optional): Whether to log info messages. Defaults to True.

    Returns:
        Project: Created or loaded project instance
    """
    # Many slideflow methods only accept paths in the form of strings
    project_dir_str = str(project_dir)

    if is_project(project_dir_str):
        if verbose:
            log.info(f"[bold]{project_dir}[/] is already a project. Loaded")
        return sf.load_project(project_dir_str)
    elif not project_dir.exists():
        project_dir.mkdir()

    if not (annotations := Path(annotation_file)).suffix == ".csv":
        raise Exception(f"[red]{annotation_file}[/] is not a valid annotations file (not in .csv format)")

    # Might raise another Exception
    modified_annotations = setup_annotations(
        annotations,
        patient_column,
        label_column,
        Path(project_dir),
        slide_column
    )

    if verbose:
        log.info(f"Creating project at [bold]{project_dir}[/]")
    project = sf.create_project(
        name="AutoMIL",
        root=project_dir_str,
        slides=str(slide_dir),
        annotations=str(modified_annotations)
    )

    # Overwrite placeholder annotations
    shutil.copy(annotation_file, project_dir / annotation_file.name)
    return project


def setup_dataset(
        project: sf.Project,
        preset:  RESOLUTION_PRESETS,
        verbose: bool = True
    ) -> sf.Dataset:
    """Sets up a comprehensive dataset source for a given tile size and magnification by extracting tiles and generating feature bags

    Args:
        project (sf.Project): Project instance from which to build dataset source
        preset (RESOLUTION_PRESETS): Resolution preset containing tile size and magnification
        verbose (bool, optional): Whether to log info messages. Defaults to True.
    """
    # the .value of a preset is a tuple of tile size (int) and a magnification (str)
    tile_size, magnification = preset.value
    dataset = project.dataset(tile_size, magnification)
    project_path = Path(project.root)
    bags_path    = project_path / "bags"

    # tiles will be stored in project/tfrecords/{tile_size}px{magnification}x/...
    if verbose:
        log.info(f"Extracting tiles at {magnification} | Tile size: {tile_size}")
    dataset.extract_tiles(
        qc=qc.Otsu(),
        normalizer="reinhard_mask",
        report=True
    )

    # bags will be stored in project/bags/...
    if verbose:
        log.info("Generating feature bags...")
    extractor = sf.build_feature_extractor(
        name=FEATURE_EXTRACTOR,
        resize=224
    )
    dataset.generate_feature_bags(
        model=extractor,
        outdir=str(bags_path)
    )

    return dataset

def train(
        project: sf.Project,
        dataset: sf.Dataset,
        k:       int  = 3,
        verbose: bool = True
    ):
    """Trains models on a k-fold split with using a specific tile size and magnification preset

    Args:
        project (sf.Project): Project instance
        dataset (sf.Dataset): dataset source on whicht to train
        k (int, optional): number of folds. Defaults to 3.
        verbose (bool, optional): Whether to log info messages. Defaults to True.
    """
    global BATCH_SIZE

    # Coniguring some paths
    project_path = Path(project.root)
    bags_path  = project_path / "bags"
    model_path = project_path / "models"

    # Initial batch size will be adjusted to accomodate memory usage
    initial_batch_size = BATCH_SIZE
    # Indicator of dataset size
    num_slides = get_num_slides(dataset)

    # TODO | We need ways to determine/estimate these parameters dynamically
    input_dim = 1024 # 1024 seems like a sensible default
    tiles_per_bag = 100 # Maybe take average over all bags?

    adjusted_batch_size = adjust_batch_size(
        TransMIL,
        initial_batch_size,
        num_slides,
        input_dim,
        tiles_per_bag
    )

    config = mil_config(
        model="transmil",
        trainer="fastai",
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=adjusted_batch_size
    )

    for i, (train, val) in enumerate(dataset.kfold_split(k, labels="label")):
        if verbose:
            log.info(f"Training fold {i+1}/5")
        train_mil(
            config=config,
            train_dataset=train,
            val_dataset=val,
            outcomes="label",
            project=project,
            bags=str(bags_path),
            outdir=str(model_path)
        )

def run_pipeline(
        slide_dir:       Path,
        annotation_file: Path,
        project_dir:     Path,
        patient_column:  str,
        label_column:    str,
        k:               int, 
        verbose:         bool = True,
        cleanup:         bool = False
    ):
    """Runs the entire AutoMIL pipeline.

    Args:
        slide_dir (Path): Directory in which the slides are located
        project_dir (Path): Directory in which the project structure will be created
        annotation_file (Path): Path to the datas annotation file
        verbose (bool, optional): Verbose mode enables additional info messages. Defaults to False.
        cleanup (bool, optional): Whether to delete the entire project structure afterwards. Defaults to False.
    """
    sf.setLoggingLevel(20)

    png_slides_present = any([
        ".png" == slide.suffix for slide in slide_dir.iterdir()
    ])
    configure_image_backend(png_slides_present)
    
    # --- Project Setup ---
    try:
        project = setup_project(slide_dir, annotation_file, project_dir, patient_column, label_column)
    except Exception as err:
        log.error(err)
        sys.exit(1)

    # --- Dataset Preparation and Training Loop
    for preset in RESOLUTION_PRESETS:
        start_time = time.time()
        dataset = setup_dataset(project, preset, verbose=verbose)
        log.info(f"[green]Finished creating a {preset.name} resolution dataset source[/] in {time.time() - start_time:.1f} seconds\n")

        """
        start_time = time.time()
        train(project, dataset, k, verbose=verbose)
        log.info(f"[green]Finished training {k} models on {preset.name} resolution dataset source[/] in {time.time() - start_time:.1f} seconds\n")
        """

    # --- Cleanup ---
    if cleanup:
        log.info(f"Cleaning up [bold]{project_dir}[/]")
        shutil.rmtree(project_dir)
