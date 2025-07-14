from pathlib import Path

import click

from pipeline import run_pipeline, setup_annotations

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(name="AutoMIL",      context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.argument("patient_column",   type=str)
@click.argument("label_column",     type=str)
@click.argument("project_dir",      type=click.Path(file_okay=False))
@click.option("-k",                 type=int, default=3, help="number of folds to train per resolution level")
@click.option("-v", "--verbose",    is_flag=True, help="Enables additional logging messages")
@click.option("-c", "--cleanup",    is_flag=True, help="Deletes the created project structure")
def AutoMIL(slide_dir: str, annotation_file: str, patient_column: str, label_column: str, project_dir: str, k: int, verbose: bool, cleanup: bool):
    run_pipeline(
        Path(slide_dir),
        Path(annotation_file),
        Path(project_dir),
        patient_column,
        label_column,
        k,
        verbose=verbose,
        cleanup=cleanup
    )

if __name__ == '__main__':
    AutoMIL()





