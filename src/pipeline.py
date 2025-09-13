import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import slideflow as sf
import torch
from slideflow.mil import mil_config, train_mil
from slideflow.mil.models import TransMIL
from slideflow.slide import qc
from slideflow.util import is_project, log

from estimator import adjust_batch_size, estimate_dynamic_vram_usage
from utils import (BATCH_SIZE, COMMON_MPP_VALUES, EPOCHS, FEATURE_EXTRACTOR,
                   LEARNING_RATE, RESOLUTION_PRESETS, ModelType,
                   INFO_CLR, SUCCESS_CLR, ERROR_CLR,
                   batch_conversion_concurrent, batch_generator,
                   calculate_average_mpp, get_bag_avg_and_num_features,
                   get_num_slides, get_unique_labels, get_vlog)

def configure_image_backend(verbose: bool = True):
    """Select the image backend based on the system's operating system.

    During tiling, cucim and openslide create new processes/threads through 'forking'.
    'fork' is an unsupported system call on Windows, thus we use libvips.
    
    Args:
        verbose: Whether to log info messages

    Note:
        Make sure libvips is installed and in your PATH if you're using Windows.
        This method terminates the current script upon encountering an invalid OS.
    """
    vlog = get_vlog(verbose)
    system = platform.system().lower()
    if system == "windows":
        vlog(f"Using [{INFO_CLR}]libvips[/] backend on Windows")
        os.environ["SF_SLIDE_BACKEND"] = "libvips"
    elif system == "linux":
        vlog(f"Using [{INFO_CLR}]cucim[/] backend on Linux")
        os.environ["SF_SLIDE_BACKEND"] = "cucim"
    else:
        vlog(f"Unsupported OS: [{ERROR_CLR}]{system}[/]")
        sys.exit(1)

def setup_annotations(
        annotations_file: Path,
        patient_column:   str,
        label_column:     str,
        project_dir:      Path,
        slide_column:     Optional[str] = None,
        transform_labels: bool = True,
        verbose:          bool = True
    ) -> tuple[Path, dict | list[str]]:
    """Modify annotations file to conform to slideflow's expected format.

    The annotations file may contain any number of columns, but slideflow expects
    at least the 'label' and 'patient' columns. This method renames the given
    columns to 'patient' and 'label' respectively and saves the modified CSV
    to the project folder.

    Args:
        annotations_file: Path to the annotations file in CSV format
        patient_column: Name of the column containing patient identifiers (renamed to 'patient')
        label_column: Name of the column containing labels (renamed to 'label')
        project_dir: Directory where the project structure will be created
        slide_column: Name of the column containing slide identifiers (renamed to 'slide')
        transform_labels: Whether to transform labels to float values
        verbose: Whether to log info messages

    Returns:
        A tuple containing the path to the modified annotations file and the label mapping.

    Raises:
        Exception: If unable to create the modified annotation file
    """
    vlog = get_vlog(verbose)
    annotations = pd.read_csv(annotations_file, index_col=patient_column)

    # Loaded patients as index so we have to change the name attribute
    annotations.index.name = "patient"
    # If no slide column is provided, we copy the index to a new column
    if not slide_column:
        annotations["slide"] = annotations.index
    # Rename columns to conform to slideflow's expectations
    column_renames = {
        label_column: "label",
        **({slide_column: "slide"} if slide_column else {})
    }
    annotations.rename(columns=column_renames, inplace=True)
    # Transform labels to float values
    if transform_labels:
        label_map = {
            label: float(index) for index, label in enumerate(annotations["label"].unique())
        }
        annotations["label"] = annotations["label"].map(label_map)
        vlog(f"Transformed labels to float values: [{INFO_CLR}]{', '.join([f'{k}: {v}' for k, v in label_map.items()])}[/]")
    else:
        label_map = [label for label in annotations["label"].dropna().unique()]
    
    
    # Save the modified annotations to the project directory
    annotation_file = project_dir / "annotations.csv"
    annotations.to_csv(annotation_file, index=True)
    # Error handling
    if not annotation_file.exists():
        raise Exception(f"Annotation file [bold]{annotation_file}[/] could not be created.")
    if not annotations.empty:
        log.info(f"Annotations saved to [{SUCCESS_CLR}]{annotation_file}[/]")

    return annotation_file, label_map

def create_project_scaffold(
        project_dir: Path,
        annotations_file: Path,
        patient_column: str,
        label_column: str,
        slide_column: Optional[str] = None,
        transform_labels: bool = True,
        verbose: bool = True
    ) -> tuple[Path, dict | list[str]]:
    """Set up a simple project directory structure with an annotations file.

    Creates the project directory if it doesn't exist and saves a modified
    version of the annotations file to 'project_dir/annotations.csv'.

    Args:
        project_dir: Path to the project directory
        annotations_file: Path to the annotations file in CSV format
        patient_column: Column name containing patient identifiers (renamed to 'patient')
        label_column: Column name containing labels (renamed to 'label')
        slide_column: Column name containing slide identifiers (renamed to 'slide')
        transform_labels: Whether to transform labels to float values
        verbose: Whether to log info messages

    Returns:
        A tuple containing the path to the modified annotations file and the label mapping.
    """
    vlog = get_vlog(verbose)
    # Simple project directory creation
    if not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        vlog(f"Created project directory at [{SUCCESS_CLR}]{project_dir}[/]")
    else:
        vlog(f"Project directory [{INFO_CLR}]{project_dir}[/] already exists")
    
    # Annotations file setup
    if (annotations := project_dir / "annotations.csv") in project_dir.iterdir():
        vlog(f"Annotations file [{INFO_CLR}]{annotations}[/] already exists")
        return annotations, get_unique_labels(annotations, label_column)
    else:
        modified_annotations, label_map = setup_annotations(
            annotations_file,
            patient_column,
            label_column,
            project_dir,
            slide_column,
            transform_labels
        )
        vlog(f"Modified annotations saved to [{SUCCESS_CLR}]{modified_annotations}[/]")
        return modified_annotations, label_map

def setup_project(
        slide_dir:       Path,
        project_dir:     Path,
        annotation_file: Path,
        verbose:         bool = True
    ) -> sf.Project:
    """Set up the project structure or load existing project.

    Sets up the project structure in project_dir or loads it into a Project
    instance if it already exists.

    Args:
        slide_dir: Directory containing slides
        project_dir: Directory where the project structure should be created or already exists
        annotation_file: CSV file with annotations
        verbose: Whether to log info messages

    Returns:
        Created or loaded project instance.
    """
    vlog = get_vlog(verbose)
    # Many slideflow methods only accept paths in the form of strings
    project_dir_str = str(project_dir)

    if is_project(project_dir_str):
        vlog(f"[{INFO_CLR}]{project_dir}[/] is already a project. Loaded")
        return sf.load_project(project_dir_str)

    vlog(f"Creating project at [{INFO_CLR}]{project_dir}[/]")
    project = sf.create_project(
        name="AutoMIL",
        root=project_dir_str,
        slides=str(slide_dir),
        annotations=str(annotation_file)
    )

    return project

def setup_dataset(
        project: sf.Project,
        preset:  RESOLUTION_PRESETS,
        label_map: dict | list[str],
        slide_dir: Optional[Path] = None,
        verbose: bool = True,
        tiff_conversion: bool = False
    ) -> sf.Dataset:
    """Set up a comprehensive dataset source for a given resolution preset.

    Creates dataset source by extracting tiles and generating feature bags for
    the specified tile size and magnification.

    Args:
        project: Project instance from which to build dataset source
        preset: Resolution preset containing tile size and magnification
        label_map: Dictionary or list containing label mappings
        slide_dir: Directory containing slide files
        verbose: Whether to log info messages
        tiff_conversion: Whether TIFF conversion is required

    Returns:
        Configured dataset instance.
    """
    global COMMON_MPP_VALUES

    vlog = get_vlog(verbose)
    # the .value of a preset is a tuple of tile size (int) and a magnification (str)
    tile_size, magnification = preset.value

    # Calulcate microns per tile (pixels per tile * microns per pixel)
    mpp = None
    if slide_dir:
        mpp = calculate_average_mpp(slide_dir)
        vlog(f"Calculated average microns per pixel (MPP) of [{INFO_CLR}]{mpp}[/] for slides in [{INFO_CLR}]{slide_dir}[/]")
    if mpp is None:
        vlog(f"Using default MPP value of [{INFO_CLR}]{COMMON_MPP_VALUES.get(magnification, 0.5)}[/] for magnification [{INFO_CLR}]{magnification}[/]")
        mpp = COMMON_MPP_VALUES.get(magnification, 0.5)
    tile_um = int(tile_size * mpp)

    # adjust filters to labels
    match label_map:
        # dict with (original label: float transformed label) key value pairs
        case dict():
            unique_labels = list(label_map.values())
        # plain list of unique labels
        case list():
            unique_labels = label_map
        # None or empty list (should not happen but is here for completeness)
        case _:
            unique_labels = None

    filters = {"label": unique_labels} if unique_labels else None
    vlog(f"Using filter [{INFO_CLR}]{filters}[/] for dataset")
    dataset = project.dataset(
        tile_size,
        tile_um,
        # slideflow needs this to recognize these values as classes
        filters= {"label": unique_labels} if unique_labels else None,
    )
    project_path = Path(project.root)
    bags_path    = project_path / "bags"

    # tiles will be stored in project/tfrecords/{tile_size}px{magnification}x/...
    vlog(f"Extracting tiles at [{INFO_CLR}]{magnification}[/] | Tile size: [{INFO_CLR}]{tile_size}[/]")
    extract_tiles(
        dataset,
        tiff_conversion=tiff_conversion,
        clear_buffer=True
    )

    # bags will be stored in project/bags/...
    vlog("Generating feature bags...")
    extractor = sf.build_feature_extractor(
        name=FEATURE_EXTRACTOR,
        resize=224
    )
    dataset.generate_feature_bags(
        model=extractor,
        outdir=str(bags_path)
    )

    return dataset

def extract_tiles(
        dataset: sf.Dataset,
        tiff_conversion: bool = False,
        clear_buffer:    bool = True,
        verbose: bool = True
) -> bool:
    """Perform tile extraction for a given dataset source.

    Extracts tiles from slides in the dataset. Optionally performs batchwise
    TIFF conversion beforehand if slides are in unsuitable format.

    Args:
        dataset: Dataset instance from which to extract tiles
        tiff_conversion: Whether to perform batchwise TIFF conversion beforehand
        clear_buffer: Whether to clear the buffer after each batch (only valid with TIFF conversion)
        verbose: Whether to log progress messages

    Returns:
        True if tile extraction was successful, False otherwise.
    """
    vlog = get_vlog(verbose)
    # Default case: no tiff conversion required, slides are already in a suitable format
    if not tiff_conversion:
        try:
            dataset.extract_tiles(
                qc=qc.Otsu(),
                normalizer="reinhard_mask",
                report=True
            )
            return True
        except Exception as e:
            log.error(f"Error extracting tiles: {e}")
            return False
    # Exception: tiff conversion required, slides are in unsuitable format (e.g .png)
    else:
        slide_list = [Path(slide) for slide in dataset.slide_paths()]
        tfrecords_path = Path(dataset.tfrecords_folders()[0])  # Should be project_dir / tfrecords / {tile_size}px{magnification}x
        batch_size = 10
        
        vlog(f"Converting slides to TIFF format in batches of [{INFO_CLR}]{batch_size}[/] slides")

        for batch_idx, batch in enumerate(batch_generator(slide_list, batch_size=batch_size)):
            vlog(f"Processing batch [{INFO_CLR}]{batch_idx + 1}/{len(slide_list) // batch_size + 1}[/] with [{INFO_CLR}]{len(batch)}[/] slides")
            # Multithreaded .png -> .tiff conversion
            batch_conversion_concurrent(
                batch,
                Path(dataset.tfrecords_folders()[0]) # Should be project_dir / tfrecords / {tile_size}px{magnification}x
            )
            vlog(f"Converted [{SUCCESS_CLR}]{len(batch)}[/] slides to TIFF format")

            # Tile extraction for batch
            try:
                dataset.extract_tiles(
                    slides=batch,
                    qc=qc.Otsu(),   
                    normalizer="reinhard_mask", 
                )
            except Exception as e:
                log.error(f"Error extracting tiles for batch {batch_idx}: {e}")
                return False
            
            # Cleans up the current .tiff buffer after each batch
            if clear_buffer:
                for tiff in [file for file in tfrecords_path.iterdir() if file.suffix == ".tiff"]:
                    try:
                        tiff.unlink()
                    except Exception as e:
                        log.error(f"Error deleting TIFF file {tiff}: {e}")
        return True

def run_dataset_setup_loop(
        project: sf.Project,
        label_map: dict | list[str],
        verbose: bool = True
    ) -> dict[RESOLUTION_PRESETS, sf.Dataset]:
    """Run the dataset setup loop for all resolution presets.

    Creates dataset sources for all resolution presets defined in RESOLUTION_PRESETS.

    Args:
        project: Project instance from which to build dataset sources
        label_map: Dictionary or list containing label mappings
        verbose: Whether to log info messages

    Returns:
        Dictionary containing dataset sources for each resolution preset.
    """
    dataset_sources = {}
    for preset in RESOLUTION_PRESETS:
        start_time = time.time()
        dataset = setup_dataset(project, preset, label_map, verbose=verbose)
        if verbose:
            log.info(f"[{SUCCESS_CLR}]Finished creating a [{INFO_CLR}]{preset.name}[/] resolution dataset source[/] in [{INFO_CLR}]{time.time() - start_time:.1f}[/] seconds\n")
        dataset_sources[preset] = dataset
    
    return dataset_sources

def train(
        project: sf.Project,
        dataset: sf.Dataset,
        k:       int  = 3,
        verbose: bool = True
    ):
    """Train models on a k-fold split using a specific tile size and magnification preset.

    Args:
        project: Project instance
        dataset: Dataset source on which to train
        k: Number of folds for cross-validation
        verbose: Whether to log info messages
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
            log.info(f"Training fold [{INFO_CLR}]{i+1}/5[/]")
        learner = train_mil(
            config=config,
            train_dataset=train,
            val_dataset=val,
            outcomes="label",
            project=project,
            bags=str(bags_path),
            outdir=str(model_path)
        )

def train_with_estimate_comparison(
        model_type:  ModelType,
        project:     sf.Project,
        dataset:     sf.Dataset,
        k:           int  = 3,
        verbose: bool = True
    ):
    """Train models on a k-fold split and compare estimated vs actual VRAM usage.

    Trains models using a specific tile size and magnification preset, while
    also comparing an estimate of the VRAM usage with the actual VRAM usage during training.

    Args:
        model_type: Type of model to train
        project: Project instance
        dataset: Dataset source on which to train
        k: Number of folds for cross-validation
        verbose: Whether to log info messages
    """
    global BATCH_SIZE
    model_cls = model_type.value
    vlog = get_vlog(verbose)

    # Coniguring some paths
    project_path = Path(project.root)
    bags_path  = project_path / "bags"
    model_path = project_path / "models"

    # Get average number of tiles per bag and number of features per tile
    tiles_per_bag, input_dim = get_bag_avg_and_num_features(bags_path)
    adjusted_batch_size = adjust_batch_size(
        model_cls,
        BATCH_SIZE,
        get_num_slides(dataset),
        input_dim,
        tiles_per_bag
    )
    vlog(f"Adjusted batch size to [{SUCCESS_CLR}]{adjusted_batch_size}[/] for model [{INFO_CLR}]{model_cls.__name__}[/]")
    # Estimate memory
    estimated_mem_mb = estimate_dynamic_vram_usage(
        model_cls=model_cls,
        input_dim=input_dim,
        tiles_per_bag=tiles_per_bag,
        batch_size=adjusted_batch_size,
        num_classes=2,
        return_rounded=False
    )

    config = mil_config(
        model="attention_mil",
        trainer="fastai",
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=adjusted_batch_size,
        fit_one_cycle=True,
    )

    if k == 1:
        train, val = dataset.split(
            labels="label",
            val_fraction=0.2
        )
        folds = [(train, val)]
    else:
        folds = [(train, val) for train, val in dataset.kfold_split(k, labels="label")]

    for i, (train, val) in enumerate(folds):
        vlog(f"Training fold [{INFO_CLR}]{i+1}/5[/]")

        # Clear cuda cache befor training
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        learner = train_mil(
            config=config,
            train_dataset=train,
            val_dataset=val,
            outcomes="label",
            project=project,
            bags=str(bags_path),
            outdir=str(model_path)
        )
        
        # Examine entire learner object
        for attr in dir(learner):
            log.info(f"learner.{attr} = {getattr(learner, attr)}")
        
        # Extract trained model
        model = learner.model
        if isinstance(model, torch.nn.Module):
            model = model.eval().cuda()
            model_cls = type(model)
        else:
            raise TypeError(f"Recived an invalid model {type(model)}")

        # Actual inference memory tracking
        # Create dummy input
        dummy_input = torch.randn(BATCH_SIZE, tiles_per_bag, input_dim).cuda()
        lens = torch.full((BATCH_SIZE,), tiles_per_bag, dtype=torch.int32).cuda()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input, lens)
        actual_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        log.info(
            f"Fold [{INFO_CLR}]{i+1}[/]: Estimated VRAM = [{INFO_CLR}]{estimated_mem_mb:.2f}[/] MB | "
            f"Actual Peak VRAM = [{INFO_CLR}]{actual_mem_mb:.2f}[/] MB"
        )

        # Cleanup
        # TODO | Check if this is necessary
        del model, dummy_input, learner
        torch.cuda.empty_cache()

def run_pipeline(
        slide_dir:       Path,
        annotation_file: Path,
        project_dir:     Path,
        patient_column:  str,
        label_column:    str,
        k:               int, 
        verbose:         bool = True,
        cleanup: bool = False
    ):
    """Run the entire AutoMIL pipeline.

    Executes the complete pipeline including project setup, dataset preparation,
    and training loop for all resolution presets.

    Args:
        slide_dir: Directory containing the slides
        annotation_file: Path to the data annotation file
        project_dir: Directory where the project structure will be created
        patient_column: Name of the patient column in annotations
        label_column: Name of the label column in annotations
        k: Number of folds for cross-validation
        verbose: Whether to enable additional info messages
        cleanup: Whether to delete the entire project structure afterwards
    """
    sf.setLoggingLevel(20)

    png_slides_present = any([
        ".png" == slide.suffix for slide in slide_dir.iterdir()
    ])
    configure_image_backend(png_slides_present)
    
    # --- Project Setup ---
    try:
        project = setup_project(slide_dir, project_dir, annotation_file, verbose=verbose)
    except Exception as err:
        log.error(err)
        sys.exit(1)

    # --- Dataset Preparation and Training Loop
    for preset in RESOLUTION_PRESETS:
        start_time = time.time()
        #dataset = setup_dataset(project, preset, verbose=verbose)
        log.info(f"[{SUCCESS_CLR}]Finished creating a [{INFO_CLR}]{preset.name}[/] resolution dataset source[/] in [{INFO_CLR}]{time.time() - start_time:.1f}[/] seconds\n")

        """
        start_time = time.time()
        train(project, dataset, k, verbose=verbose)
        log.info(f"[{SUCCESS_CLR}]Finished training [{INFO_CLR}]{k}[/] models on [{INFO_CLR}]{preset.name}[/] resolution dataset source[/] in [{INFO_CLR}]{time.time() - start_time:.1f}[/] seconds\n")
        """

    # --- Cleanup ---
    if cleanup:
        log.info(f"Cleaning up [{INFO_CLR}]{project_dir}[/]")
        shutil.rmtree(project_dir)
