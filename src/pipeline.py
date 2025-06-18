import os
import platform
import shutil
import sys
import time
from pathlib import Path

import slideflow as sf
from slideflow.mil import mil_config, train_mil
from slideflow.slide import qc
from slideflow.util import is_project, log

from utils import (BATCH_SIZE, EPOCHS, FEATURE_EXTRACTOR, LEARNING_RATE,
                   RESOLUTION_PRESETS)


def configure_image_backend(verbose: bool = True):
    """Selects the image backend based on the systems operating system

    During tiling, cucim and openslide create new processes/threads through 'forking'. \\
    'fork' is an unsupported system call on windows, thus we use libvips. \\
    
    **NOTE**: Make sure libvips is installed and in your PATH if you're using windows. \\
    **NOTE**: This method terminates the current script upon encountering a invalid OS.
    
    Args:
        verbose (bool, optional): Whether to log info messages. Defaults to True.
    """
    system = platform.system().lower()
    match system:
        case "windows":
            if verbose:
                log.info("Using [cyan]libvips[/] backend on Windows")
            os.environ["SF_SLIDE_BACKEND"] = "libvips"
        case "linux":
            if verbose:
                log.info("Using [cyan]openslide[/] backend on Linux")
            os.environ["SF_SLIDE_BACKEND"] = "openslide"
        case _:
            if verbose:
                log.error(f"Unsupported OS: [red]{system}[/]")
            sys.exit(1)


def setup_project(slide_dir: Path, project_dir: Path, annotation_file: Path, verbose: bool = True) -> sf.Project:
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

    if verbose:
        log.info(f"Creating project at [bold]{project_dir}[/]")
    project = sf.create_project(
        name="AutoMIL",
        root=project_dir_str,
        slides=slide_dir,
        annotations=annotation_file
    )

    # Overwrite placeholder annotations
    shutil.copy(annotation_file, project_dir / annotation_file.name)
    return project


def setup_dataset(project: sf.Project, preset: RESOLUTION_PRESETS, verbose: bool = True) -> sf.Dataset:
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
        report=False
    )

    # bags will be stored in project/bags/...
    if verbose:
        log.info("Generating feature bags...")
    project.generate_feature_bags(
        obj=project,
        model=FEATURE_EXTRACTOR,
        dataset=dataset,
        outdir=bags_path
    )

    return dataset

def train(project: sf.Project, dataset: sf.Dataset, k: int = 3, verbose: bool = True):
    """Trains models on a k-fold split with using a specific tile size and magnification preset

    Args:
        project (sf.Project): Project instance
        dataset (sf.Dataset): dataset source on whicht to train
        k (int, optional): number of folds. Defaults to 3.
        verbose (bool, optional): Whether to log info messages. Defaults to True.
    """
    # Coniguring some paths
    project_path = Path(project.root)
    bags_path  = project_path / "bags"
    model_path = project_path / "models"

    config = mil_config(
        model="attention_mil",
        trainer="fastai",
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    for i, (train, val) in enumerate(dataset.kfold_split(k=5, labels="subtype")):
        if verbose:
            log.info(f"Training fold {i+1}/5")
        train_mil(
            config=config,
            train_dataset=train,
            val_dataset=val,
            outcomes="subtype",
            project=project,
            bags=str(bags_path),
            outdir=str(model_path)
        )


def run_pipeline(slide_dir: Path, project_dir: Path, annotation_file: Path, verbose: bool = True, cleanup: bool = False):
    """Runs the entire AutoMIL pipeline.

    Args:
        slide_dir (_type_): Directory in which the slides are located
        project_dir (_type_): Directory in which the project structure will be created
        annotation_file (_type_): Path to the datas annoation file
        verbose (bool, optional): Verbose mode enables additional info messages. Defaults to False.
        cleanup (bool, optional): Whether to delete the entire project structure afterwards. Defaults to False.
    """
    # TODO: Make k configurable (cli argument, config ?)
    k: int = 3

    configure_image_backend()
    project = setup_project(slide_dir, project_dir, annotation_file)

    for preset in RESOLUTION_PRESETS:
        start_time = time.time()
        dataset = setup_dataset(project, preset, verbose=verbose)
        log.info(f"[green]Finished creating a {preset.name} resolution dataset source[/] in {time.time() - start_time:.1f} seconds\n")

        start_time = time.time()
        train(project, dataset, k, verbose=verbose)
        log.info(f"[green]Finished training {k} models on {preset.name} resolution dataset source[/] in {time.time() - start_time:.1f} seconds\n")

    if cleanup:
        log.info(f"Cleaning up [bold]{project_dir}[/]")
        shutil.rmtree(project_dir)
