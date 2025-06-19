from pathlib import Path

import click

from src.pipeline import run_pipeline

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(name="AutoMIL",      context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.argument("slide_dir",        type=click.Path(exists=True, file_okay=False))
@click.argument("project_dir",      type=click.Path(file_okay=False))
@click.argument("annotation_file",  type=click.Path(exists=True, file_okay=True))
@click.option("-v", "--verbose",    is_flag=True, help="Enables additional logging messages")
@click.option("-c", "--cleanup",    is_flag=True, help="Deletes the created project structure")
def AutoMIL(slide_dir: str, project_dir: str, annotation_file: str, verbose: bool, cleanup: bool):
    run_pipeline(
        Path(slide_dir),
        Path(project_dir),
        Path(annotation_file),
        verbose=verbose,
        cleanup=cleanup
    )

if __name__ == '__main__':
    AutoMIL()





