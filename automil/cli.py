"""The entry point CLI for running AutoMIL"""
# === External libraries === #
import sys
import traceback
import warnings

# Suppressing warnings related to pkg_ressources and timm
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import click
import slideflow as sf

# === Internal modules === #
from .dataset import Dataset
from .evaluation import Evaluator
from .pipeline import configure_image_backend
from .project import Project
from .trainer import Trainer
from .utils import (RESOLUTION_PRESETS, LogLevel, ModelType, get_vlog,
                    is_input_pretiled)

# === Setup === #
CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 120,
    "show_default": True,
}

# === CLI === #
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="1.0.0", prog_name="AutoMIL")
def AutoMIL():
    """AutoMIL: Automated Multiple Instance Learning for Whole Slide Images."""
    pass

@AutoMIL.command(
    name="run-pipeline", 
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True
)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.argument("project_dir",      type=click.Path(file_okay=False))
@click.option(
    "-pc", "--patient_column", type=str, default="patient",
    help="Name of the column containing patient IDs"
)
@click.option(
    "-lc", "--label_column", type=str, default="label",
    help="Name of the column containing labels"
)
@click.option(
    "-sc", "--slide_column", type=str, default=None,
    help="Name of the column containing slide names"
)
@click.option(
    "-r", "--resolutions",
    type=str,
    default="Low",
    help=f"Comma-separated list of resolution presets to train on. "
         f"Available: {', '.join([res.name for res in RESOLUTION_PRESETS])} "
         f"(e.g., 'Low,High')"
)
@click.option(
    "-m", "--model",
    type=(model_choice := click.Choice([model.name for  model in ModelType])),
    default=model_choice.choices[0],
    help=f"Model type to train and evaluate"
)
@click.option(
    "-k", type=int, default=3,
    help="number of folds to train per resolution level"
)
@click.option(
    "--split-file", type=click.Path(file_okay=True), default="split.json",
    help="Path to a .json file defining train-test splits"
)
@click.option("-t", "--transform_labels", is_flag=True, help="Transforms labels to float values (0.0, 1.0, ...)")
@click.option("-p", "--is-pretiled",      is_flag=True, help="Indicated that the input format is pretiled slides")
@click.option("-v", "--verbose",          is_flag=True, help="Enables additional logging messages")
def run_pipeline(
    slide_dir:       str | Path,
    annotation_file: str | Path,
    project_dir:     str | Path,
    patient_column:  str,
    label_column:    str,
    slide_column:    str | None,
    resolutions:     str,
    model:           str,
    k:               int,
    split_file:      str | None,
    transform_labels: bool,
    is_pretiled:      bool,
    verbose:          bool
    ):
    """
    Execute the complete AutoMIL pipeline for whole slide image analysis.
    
    \b
    `run-pipeline` executes the entire AutoMIL workflow, encompassing:
      [1] Project setup and configuration
      [2] Dataset preparation and tile extraction  
      [3] Model training with k-fold cross-validation
      [4] Model evaluation and ensemble creation
      [5] Results comparison and visualization

    \b
    ARGUMENTS:
      SLIDE_DIR        Directory containing whole slide images (.svs, .tiff, etc.)
      ANNOTATION_FILE  .csv file with slide/patient annotations and labels  
      PROJECT_DIR      Output directory for all results and models
    
    \b
    EXAMPLES:
      # Basic usage with default settings
      automil run-pipeline /data/slides /data/annotations.csv ./results
      
      # Multi-resolution training with verbose output
      automil run-pipeline -r "Low,High" -v /data/slides /data/annotations.csv ./results
      
      # Custom model and k-fold settings
      automil run-pipeline -m TransMIL -k 5 /data/slides /data/annotations.csv ./results
      
      # Skip tiling if tiles are pre-extracted
      automil run-pipeline -p /data/slides /data/annotations.csv ./results
      
      # Custom column names in the annotation file
      automil run-pipeline -pc "patient_name" -lc "diagnosis" -sc "slide_name" /data/slides /data/annotations.csv ./results
      
      # Provide a predefined train-test split
      automil run-pipeline --split-file /data/split.json /data/slides /data/annotations.csv ./results

    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1

    \b
    EXPECTED SLIDE DIRECTORY STRUCTURE:
      SLIDE_DIR should contain whole slide images in supported formats
      such as .svs, .tiff, or .png.
      Example structure:
        /data/slides/
        |-- slide1.svs
        |-- slide2.tiff
        |-- slide3.tiff
      If slides are in PNG, AutoMIL will first convert them to TIFF for easier processing.

    \b
    USING PRETILED DATA:
      If tiles have already been extracted from the slides, use the `--is_pretiled` flag.
      In the case of pretiled data, AutoMIL expects the following directory structure for SLIDE_DIR:
        /data/slides/
        |-- slide1/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
        |-- slide2/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
      Tile names are arbitrary but slide subdirectories must match the slide names in ANNOTATION_FILE.

    \b
    PROVIDING A TRAIN TEST SPLIT:
        Use the `--split-file` option to provide a JSON file defining train-test splits.
        The JSON file should have the following structure:
            {
            "train": ["slide1", "slide2", ...],
            "test":  ["slide3", "slide4", ...]
            }
        or:
            {
            "train": ["slide1", "slide2", ...],
            "validation":  ["slide3", "slide4", ...]
            }

    \b
    OUTPUT STRUCTURE:
      project_dir/
      ├── bags/           # Extracted tile features
      ├── models/         # Trained model checkpoints  
      ├── ensemble/       # Ensemble predictions
      ├── annotations.csv # Processed annotations
      └── results.json    # Performance metrics
    """
    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: {command}")

    # Define some paths
    bags_dir = Path(project_dir) / "bags"
    models_dir = Path(project_dir) / "models"
    ensemble_dir = Path(project_dir) / "ensemble"

    # Some type coercion
    slide_dir = Path(slide_dir)
    annotation_file = Path(annotation_file)
    project_dir = Path(project_dir)

    try:

        # === 1. Parsing === #
        # Parse given string resolutions into list of RESOLUTION_PRESETS
        resolution_presets: list[RESOLUTION_PRESETS] = []

        for res in [r.strip() for r in resolutions.split(',')]:
            if res not in RESOLUTION_PRESETS.__members__:
                vlog(f"Invalid resolution preset '{res}'. Available presets: {[preset.name for preset in RESOLUTION_PRESETS]}", LogLevel.ERROR)
                return
            else:
                resolution_presets.append(RESOLUTION_PRESETS[res])
        
        vlog(f"Using resolution presets: {[preset.name for preset in resolution_presets]}")

        # Parse the model type
        if model in ModelType.__members__:
            model_type = ModelType[model]
            vlog(f"Using model type: {model_type.name}")
        # This should never happen since we use click.Choice but it is here for completeness
        else:
            vlog(f"Invalid model type '{model}'. Available models: {[m.name for m in ModelType]}", LogLevel.ERROR)
            return

        # === 2. Image Backend Configuration === #
        # TODO | Check if this is necessary or if slideflow handles it automatically
        png_slides_present: bool = any(
            [slide.suffix.lower() == ".png" for slide in Path(slide_dir).iterdir()]
        )
        # If no PNG slides are present, we configure the image backend
        if not png_slides_present:
            configure_image_backend(verbose=verbose)
        tiff_conversion = png_slides_present

        # === 3. Project Creation And Setup === #
        project_setup = Project(
            Path(project_dir),
            Path(annotation_file),
            Path(slide_dir),
            patient_column,
            label_column,
            slide_column,
            transform_labels,
            verbose,
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
        for preset in resolution_presets:
            vlog(f"Setting up dataset for resolution preset: {preset.name}")

            dataset = Dataset(
                project,
                preset,
                label_map,
                slide_dir=Path(slide_dir),
                bags_dir=Path(project_dir) / "bags",
                is_pretiled=is_pretiled,
                tiff_conversion=tiff_conversion,
                verbose=verbose
            )
            dataset.summary()
            datasets[preset.name] = dataset.prepare_dataset_source()
            vlog(f"Dataset setup complete for resolution preset: {preset.name}")

        # === 5. Prepare (or Load) Train/Test Split === #
        dataset = datasets[resolution_presets[0].name]
        train, test = dataset.split(
            labels="label",
            val_fraction=0.2,
            splits=split_file
        )
        
        # === 6. Model Training === #
        for resolution in resolution_presets:
            vlog(f"Train/Test split for resolution preset '{resolution.name}': "
                 f"{len(train.slides())} train slides"
            )

            train, val = train.split(
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

        # === 7. Model Evaluation === #
        evaluator = Evaluator(
            test,
            models_dir,
            ensemble_dir,
            bags_dir,
            verbose=verbose
        )

        evaluator.evaluate_models(generate_attention_heatmaps=True)
        evaluator.create_ensemble_predictions(
            output_path=Path(project.root) / "ensemble_predictions.csv"
        )

        evaluator.compare_models()
        evaluator.generate_plots(
            save_path=Path(project.root) / "figures",
            model_paths=None
        )
    
    except Exception as e:
        tb = traceback.format_exc()
        vlog(tb, LogLevel.ERROR)
        vlog(f"Error: {e}", LogLevel.ERROR)
        return

@AutoMIL.command(name="train", context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.argument("project_dir",      type=click.Path(file_okay=False))
@click.option(
    "-pc", "--patient_column", type=str, default="patient",
    help="Name of the column containing patient IDs"
)
@click.option(
    "-lc", "--label_column", type=str, default="label",
    help="Name of the column containing labels"
)
@click.option(
    "-sc", "--slide_column", type=str, default=None,
    help="Name of the column containing slide names"
)
@click.option(
    "-r", "--resolutions",
    type=str,
    default="Low",
    help=f"Comma-separated list of resolution presets to train on. "
         f"Available: {', '.join([res.name for res in RESOLUTION_PRESETS])} "
         f"(e.g., 'Low,High')"
)
@click.option(
    "-m", "--model",
    type=(model_choice := click.Choice([model.name for  model in ModelType])),
    default=model_choice.choices[0],
    help=f"Model type to train and evaluate"
)
@click.option(
    "-k", type=int, default=3,
    help="number of folds to train per resolution level"
)
@click.option("-p", "--is-pretiled",      is_flag=True, help="Indicated that the input format is pretiled slides")
@click.option("-t", "--transform_labels", is_flag=True, help="Transforms labels to float values (0.0, 1.0, ...)")
@click.option("-v", "--verbose",          is_flag=True, help="Enables additional logging messages")
def train(
    slide_dir:       str | Path,
    annotation_file: str | Path,
    project_dir:     str | Path,
    patient_column:  str,
    label_column:    str,
    slide_column:    str | None,
    resolutions:     str,
    model:           str,
    k:               int,
    is_pretiled:      bool,
    transform_labels: bool,
    verbose:          bool
):
    """
    Train a single or multiple MIL models on a given dataset.
    
    \b
    `train` sets up a project and dataset source, then trains MIL models using k-fold cross-validation:
      [1] Project setup and configuration
      [2] Dataset preparation and tile extraction  
      [3] Model training with k-fold cross-validation

    \b
    ARGUMENTS:
      SLIDE_DIR        Directory containing whole slide images (.svs, .tiff, etc.)
      ANNOTATION_FILE  CSV file with slide/patient annotations and labels  
      PROJECT_DIR      Output directory for all results and models
    
    \b
    EXAMPLES:
      # Basic usage with default settings
      automil train /data/slides /data/annotations.csv ./results
      
      # Multi-resolution training with verbose output
      automil train -r "Low,High" -v /data/slides /data/annotations.csv ./results
      
      # Custom model and k-fold settings
      automil train -m TransMIL -k 5 /data/slides /data/annotations.csv ./results
      
      # Skip tiling if tiles are pre-extracted
      automil train -p /data/slides /data/annotations.csv ./results
      
      # Custom column names in the annotation file
      automil train -pc "patient_name" -lc "diagnosis" -sc "slide_name" /data/slides /data/annotations.csv ./results

    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1

    \b
    EXPECTED SLIDE DIRECTORY STRUCTURE:
      SLIDE_DIR should contain whole slide images in supported formats
      such as .svs, .tiff, or .png.
      Example structure:
        /data/slides/
        |-- slide1.svs
        |-- slide2.tiff
        |-- slide3.tiff
      If slides are in PNG, AutoMIL will first convert them to TIFF for easier processing.

    \b
    USING PRETILED DATA:
      If tiles have already been extracted from the slides, use the `--is_pretiled` flag.
      In the case of pretiled data, AutoMIL expects the following directory structure for SLIDE_DIR:
        /data/slides/
        |-- slide1/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
        |-- slide2/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
      Tile names are arbitrary but slide subdirectories must match the slide names in ANNOTATION_FILE.

    \b
    OUTPUT STRUCTURE:
      project_dir/
      ├── bags/           # Extracted tile features
      ├── models/         # Trained model checkpoints  
      ├── ensemble/       # Ensemble predictions
      ├── annotations.csv # Processed annotations
      └── results.json    # Performance metrics
    """
    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: {command}")

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

        for res in [r.strip() for r in resolutions.split(',')]:
            if res not in RESOLUTION_PRESETS.__members__:
                vlog(f"Invalid resolution preset '{res}'. Available presets: {[preset.name for preset in RESOLUTION_PRESETS]}", LogLevel.ERROR)
                return
            else:
                resolution_presets.append(RESOLUTION_PRESETS[res])
        
        vlog(f"Using resolution presets: {[preset.name for preset in resolution_presets]}")

        # Parse the model type
        if model in ModelType.__members__:
            model_type = ModelType[model]
            vlog(f"Using model type: {model_type.name}")
        # This should never happen since we use click.Choice but it is here for completeness
        else:
            vlog(f"Invalid model type '{model}'. Available models: {[m.name for m in ModelType]}", LogLevel.ERROR)
            return

        # === 2. Image Backend Configuration === #
        png_slides_present: bool = any(
            [slide.suffix.lower() == ".png" for slide in Path(slide_dir).iterdir()]
        )
        # If no PNG slides are present, we configure the image backend
        if not png_slides_present:
            configure_image_backend(verbose=verbose)
        tiff_conversion = png_slides_present

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
            vlog(f"Setting up dataset for resolution preset: {resolution.name}")

            dataset = Dataset(
                project,
                resolution,
                label_map,
                slide_dir=Path(slide_dir),
                bags_dir=Path(project_dir) / "bags",
                is_pretiled=is_pretiled,
                tiff_conversion=tiff_conversion,
                verbose=verbose
            )
            dataset.summary()
            datasets[resolution.name] = dataset.prepare_dataset_source()
            vlog(f"Dataset setup complete for resolution preset: {resolution.name}")
        
        # === 5. Model Training === #
        for resolution in resolution_presets:
            dataset = datasets[resolution.name]
            vlog(f"Train/Test split for resolution preset '{resolution.name}': "
                 f"{len(dataset.slides())} train slides"
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

@AutoMIL.command(name="predict", context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",    type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.argument("bags_dir",     type=click.Path(exists=True, file_okay=False))
@click.argument("model_dir",    type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o", "--output-dir", 
    type=click.Path(file_okay=True), default="predictions",
    help="Directory to which to save predictions (should either be .csv or .parquet)"
)
@click.option(
    "-pc", "--patient_column", type=str, default="patient",
    help="Name of the column containing patient IDs"
)
@click.option(
    "-lc", "--label_column", type=str, default="label",
    help="Name of the column containing labels"
)
@click.option(
    "-sc", "--slide_column", type=str, default=None,
    help="Name of the column containing slide names"
)
@click.option("-v", "--verbose", is_flag=True, help="Enables additional logging messages")
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
    Generate predictions using a single or multiple trained MIL models.

    \b
    `predict` loads all available model checkpoints from MODEL_DIR and generates predictions
    on the slides in SLIDE_DIR using the corresponding tile features in BAGS_DIR.
    The results are saved to OUTPUT_FILE.

    \b
    ARGUMENTS:
        SLIDE_DIR     Directory containing whole slide images (.svs, .tiff, ...)
        ANNOTATION_FILE  .csv file with slide/patient annotations and labels
        BAGS_DIR      Directory containing tile feature bags (.pt files)
        MODEL_DIR     Directory containing trained model checkpoints (.pth files)

    \b
    EXAMPLES:
      # Generate predictions with multiple models (generates one output file per model)
      automil predict /data/slides /data/annotations.csv /data/bags /data/models/ -o ./predictions

      # Generate predictions with a single model
      automil predict /data/slides /data/annotations.csv /data/bags /data/models/model_1 -v
    
      # Generate predictions with a single model (override column names)
      automil predict -pc "patient_id" -lc "outcome" -sc "slide_id" /data/slides /data/annotations.csv /data/bags /data/models/model_1 -o ./predictions
    
    \b
    EXPECTED MODEL DIRECTORY STRUCTURE:
        MODEL_DIR can refer to a single model directory (containing one .pth file) or
        a parent directory containing multiple model subdirectories with .pth files.
        EXAMPLE STRUCTURE FOR A SINGLE MODEL:
            /data/models/model_1/
                |-- best_valid.pth
                |....
        EXAMPLE STRUCTURE FOR MULTIPLE MODELS:
          /data/models/
            |-- model_1/
            |    |-- best_valid.pth
            |-- model_2/
            |    |-- best_valid.pth
            |    |...
    
    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1
    
    \b
    OUTPUT DIRECTORY FORMAT:
        OUTPUT_DIR should be a directory path.
        Predictions will be saved in separate .csv or .parquet files within this directory.
        If multiple models are used, separate output files will be created for each model,
        adding a suffix with the model name to the specified OUTPUT_DIR path.
    """
    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: {command}")

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

@AutoMIL.command(name="evaluate", context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",    type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.argument("bags_dir",     type=click.Path(exists=True, file_okay=False))
@click.argument("model_dir",    type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o", "--output-dir", 
    type=click.Path(file_okay=True), default="evaluation",
    help="Directory to which to save evaluation results"
)
@click.option(
    "-pc", "--patient_column", type=str, default="patient",
    help="Name of the column containing patient IDs"
)
@click.option(
    "-lc", "--label_column", type=str, default="label",
    help="Name of the column containing labels"
)
@click.option(
    "-sc", "--slide_column", type=str, default=None,
    help="Name of the column containing slide names"
)
@click.option("-v", "--verbose", is_flag=True, help="Enables additional logging messages")
def evaluate(
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
    Evaluate a single or multiple trained MIL models.

    \b
    `evaluate` loads all available model checkpoints from MODEL_DIR, generates predictions
    on the slides in SLIDE_DIR using the corresponding tile features in BAGS_DIR and evaluates them.
    The results are saved to OUTPUT_FILE.

    \b
    ARGUMENTS:
        SLIDE_DIR     Directory containing whole slide images (.svs, .tiff, ...)
        ANNOTATION_FILE  .csv file with slide/patient annotations and labels
        BAGS_DIR      Directory containing tile feature bags (.pt files)
        MODEL_DIR     Directory containing trained model checkpoints (.pth files)

    \b
    EXAMPLES:
      # Evaluate a single model
      automil evaluate /data/slides /data/annotations.csv /data/bags /data/models/model_1 -o ./results

      # Evaluate multiple models (generates one output file per model)
      automil evaluate /data/slides /data/annotations.csv /data/bags /data/models/ -v
    
      # Evaluate a single model (override column names)
      automil evaluate -pc "patient_id" -lc "outcome" -sc "slide_id" /data/slides /data/annotations.csv /data/bags /data/models/model_1 -o ./results      
    
    \b
    EXPECTED MODEL DIRECTORY STRUCTURE:
        MODEL_DIR can refer to a single model directory (containing one .pth file) or
        a parent directory containing multiple model subdirectories with .pth files.
        EXAMPLE STRUCTURE FOR A SINGLE MODEL:
            /data/models/model_1/
                |-- best_valid.pth
                |....
        EXAMPLE STRUCTURE FOR MULTIPLE MODELS:
          /data/models/
            |-- model_1/
            |    |-- best_valid.pth
            |-- model_2/
            |    |-- best_valid.pth
            |    |...
    
    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1

    \b
    OUTPUT DIRECTORY FORMAT:
        OUTPUT_DIR should be a directory path.
        Predictions will be saved in separate .csv or .parquet files within this directory.
        If multiple models are used, separate output files will be created for each model,
        adding a suffix with the model name to the specified OUTPUT_DIR path.
    """
    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: {command}")

    # Some type coercion
    slide_dir =  Path(slide_dir)
    bags_dir =   Path(bags_dir)
    model_dir =  Path(model_dir)
    output_dir = Path(output_dir)

    vlog(f"Evaluating models in: {model_dir}")

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

    # Evaluate models
    try:
        evaluator = Evaluator(
            dataset,
            model_dir,
            output_dir,
            bags_dir,
            verbose=verbose
        )
        evaluator.evaluate_models(generate_attention_heatmaps=True)
        evaluator.compare_models()
        evaluator.generate_plots()

    except Exception as e:
        tb = traceback.format_exc()
        vlog(tb, LogLevel.ERROR)
        vlog(f"Error: {e}", LogLevel.ERROR)
        return

@AutoMIL.command("create-split", context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
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
    Create a train-test split .json file based on the provided annotations.

    \b
    `create-split` reads the annotation file and generates a train-test split,
    saving it to OUTPUT_FILE in .json format.

    \b
    ARGUMENTS:
      SLIDE_DIR        Directory containing whole slide images (.svs, .tiff, etc.)
      ANNOTATION_FILE  .csv file with slide/patient annotations and labels  

    \b
    EXAMPLES:
      # Basic usage with default settings
      automil create-split /data/slides /data/annotations.csv -o split.json

    \b
    OUTPUT FILE FORMAT:
      The output JSON file will have the following structure:
        {
          "train": ["slide1", "slide2", ...],
          "test":  ["slide3", "slide4", ...]
        }
    """
    # Getting a verbose logger
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(20) # INFO: 20, DEBUG: 10

    # Logging the executed command
    command = " ".join(sys.argv)
    vlog(f"Executing command: {command}")

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


def main():
    """Entry point for the automil package"""
    AutoMIL()


if __name__ == '__main__':
    main()






