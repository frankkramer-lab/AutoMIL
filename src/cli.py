import sys
import traceback
import warnings

# Suppressing warnings related to pkg_ressources and timm
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import click
import slideflow as sf

from dataset import Dataset
from evaluation import Evaluator
from Experiments.experiment import BatchSizeExperiment
from pipeline import (configure_image_backend, create_project_scaffold,
                      setup_dataset, setup_project)
from project import Project
from trainer import Trainer
from utils import RESOLUTION_PRESETS, LogLevel, ModelType, get_vlog

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 120,
    "show_default": True,
}

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
@click.option("-s", "--skip_tiling",      is_flag=True, help="Skips the tiling step (assumes tiles are already extracted)")
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
    skip_tiling:      bool,
    verbose:          bool
    ):
    """Executes the full AutoMIL pipeline
    
    1. Image backend configuration
    2. Project setup
    3. Dataset configuration
    4. Model training
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
        project = project_setup.prepare_project()
        label_map = project_setup.get_label_map()

        project_setup.summary()
        
        # === 4. Setup Dataset Sources ===
        datasets: dict[str, sf.Dataset] = {}
        for preset in resolution_presets:
            vlog(f"Setting up dataset for resolution preset: {preset.name}")

            dataset = Dataset(
                project,
                preset,
                label_map,
                slide_dir=Path(slide_dir),
                bags_dir=Path(project_dir) / "bags",
                pretiled=skip_tiling,
                tiff_conversion=tiff_conversion,
                verbose=verbose
            )
            dataset.summary()
            datasets[preset.name] = dataset.prepare_dataset_source()
            vlog(f"Dataset setup complete for resolution preset: {preset.name}")

        # === 5. Prepare or Load Train/Test Split === #
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
            project,
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
    
    except Exception as e:
        tb = traceback.format_exc()
        vlog(tb, LogLevel.ERROR)
        vlog(f"Error: {e}", LogLevel.ERROR)
        return

@AutoMIL.command(name="batch-analysis", context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.argument("project_dir",      type=click.Path(file_okay=False))
@click.option("-pc", "--patient_column",  type=str, default="patient",     help="Name of the column containing patient IDs")
@click.option("-lc", "--label_column",    type=str, default="label",       help="Name of the column containing labels")
@click.option("-sc", "--slide_column",    type=str, default=None,          help="Name of the column containing slide names")
@click.option("-bs", "--batch_sizes",     type=str, default="2,4,8,16,32", help="Comma-separated list of batch sizes to test")
@click.option("-k",                       type=int, default=3,             help="Number of folds to train per batch size")
@click.option("-t", "--transform_labels", is_flag=True,                   help="Transforms labels to float values (0.0, 1.0, ...)")
@click.option("-s", "--skip_tiling",      is_flag=True,                   help="Skips the tiling step (assumes tiles are already extracted)")
@click.option("-v", "--verbose",          is_flag=True,                   help="Enables additional logging messages")
def batch_analysis(
    slide_dir:       str,
    annotation_file: str,
    project_dir:     str,
    patient_column:  str,
    label_column:    str,
    slide_column:    str,
    batch_sizes:     str,
    k:               int,
    transform_labels: bool,
    skip_tiling:      bool,
    verbose:          bool
    ):
    """Runs a batch size analysis
    
    Runs a batch size analysis by training models with different batch sizes
    and comparing their performance and resource usage.
    """
    vlog = get_vlog(verbose)
    sf.setLoggingLevel(10) # INFO: 20, DEBUG: 10
    
    # --- 1. Parse Batch Sizes ---
    try:
        batch_size_list = [int(bs.strip()) for bs in batch_sizes.split(',')]
    except ValueError:
        vlog(f"Error: Invalid batch sizes. Please provide comma-separated integers.", LogLevel.ERROR)
        return

    try:
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
            tiff_conversion=False,
            skip_tiling=skip_tiling
        )

        # --- 4. Run Batch Size Analysis ---
        experiment = BatchSizeExperiment(
            project_dir=Path(project_dir),
            model_type=ModelType.Attention_MIL,
            batch_sizes=batch_size_list,
            results_dir=Path(project_dir) / "batch_size_analysis",
        )
        experiment.run_experiment(
            project,
            dataset,
            model_type=ModelType.Attention_MIL,
            k=k,
            verbose=verbose
        )
        experiment.create_experiment_plots(return_figures=False)
    
    except Exception as e:
        tb = traceback.format_exc()
        vlog(tb, LogLevel.ERROR)
        vlog(f"Error: {e}", LogLevel.ERROR)
        return

if __name__ == '__main__':
    AutoMIL()





