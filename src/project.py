from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import slideflow as sf
import torch
from slideflow.slide import qc
from slideflow.util import is_project
from tabulate import tabulate

from utils import INFO_CLR, SUCCESS_CLR, get_unique_labels, get_vlog


# === Helpers === #
def contains_columns(data: pd.DataFrame | Path, columns: Iterable[str], return_missing: bool = False) -> bool | set[str]:
    """Checks whether a given table (as Dataframe or path to file) contains all `columns`

    Args:
        data (pd.DataFrame | Path): data table to check columns of
        columns (list[str]): columns to check
        return_missing (bool, optional): If true, returns the subset of missing columns. Defaults to False.

    Returns:
        bool | set[str]: Whether all columns are present, or the set of missing columns
    """
    if isinstance(data, Path):
        match data.suffix:
            case ".parquet":
                data = pd.read_parquet(data)
            case ".csv":
                data = pd.read_csv(data)
            case _:
                data = pd.read_csv(data)

    if return_missing:
        return set(data.columns) - set(columns)
    else:
        return set(columns).issubset(set(data.columns))   

class Project:
    """
    A helper class for creating and managing an AutoMIL Slideflow project instance

    Given a directory for the project, an annotations file, and a slide directory,
    this class sets up the necessary project structure, modifies the annotations file to conform to Slideflow's expected format,
    and creates or loads a Slideflow project instance.
    """
    def __init__(
        self,
        project_dir: Path | str,
        annotations_file: Path | str,
        slide_dir: Path | str,
        patient_column: str,
        label_column: str,
        slide_column: str | None = None,
        transform_labels: bool = False,
        verbose: bool = True
    ) -> None:
        """Initializes a Project instance.

        Args:
            project_dir (Path | str): Directory in which to set up project
            annotations_file (Path | str): annotations file
            slide_dir (Path | str): Slide directory
            patient_column (str): column containing patient identifiers
            label_column (str): column containing labels
            slide_column (str | None, optional): column containing slide identifiers. Defaults to None.
            transform_labels (bool, optional): Whether to transform labels to a float mapping. Defaults to False.
            verbose (bool, optional): Whether to log verbose messages. Defaults to True.
        """
        self.project_dir: Path = Path(project_dir)
        self.annotations_file: Path = Path(annotations_file)
        self.slide_dir: Path = Path(slide_dir)

        self.patient_column = patient_column
        self.label_column = label_column
        self.slide_column = slide_column

        self.transform_labels = transform_labels
        self.verbose = verbose
        self.vlog = get_vlog(verbose)

        self.project: sf.Project | None = None
        self.modified_annotations: Path | None = None
        self.label_map: dict | list[str] | None = None

    # === Public Methods === #
    def prepare_project(self) -> sf.Project:
        """
        Sets up the project directory structure, modifies and stores annotations, and creates or loads
        a Slideflow project.

        This method:
            1. Creates the project folder if necessary.
            2. Normalizes and saves annotations to project_dir/annotations.csv.
            3. Creates a new Slideflow project or loads an existing one.

        Returns:
            sf.Project: A slideflow project instance
        """
        # Create project scaffold
        self._create_project_scaffold(
            self.annotations_file,
            self.patient_column,
            self.label_column,
            self.slide_column,
            self.transform_labels
        )

        # Load or create project
        if is_project(str(self.project_dir)):
            self.vlog(f"Loading existing project at {self.project_dir}")
            self.project = sf.load_project(str(self.project_dir))
        else:
            self.vlog(f"Creating new project at {self.project_dir}")
            self.project = sf.create_project(
                name="AutoMIL",
                root=str(self.project_dir),
                slides=str(self.slide_dir),
                annotations=str(self.modified_annotations),
            )

        return self.project
    
    def get_label_map(self) -> dict | list[str]:
        """
        Returns the label map generated during annotation setup.

        Returns:
            dict | list[str]: The label map (dict if transform_labels=True, else list of unique labels).
        """
        if self.label_map is None:
            raise ValueError("Label map has not been set up yet. Call prepare_project() first.")
        return self.label_map
    
    def summary(self) -> None:
        """Prints a simple summary of the Project Instance in a tabular format

        Example:
            ```
            ╒═══════════════════════╤═══════════════════════════════╕                                                                                                                                                                       
            │ Project Directory:    │ project                       │                                                                                                                                                                       
            │ Slide Directory:      │ data/slides                   │                                                                                                                                                                       
            │ Annotations File:     │ data/annotations.csv          │                                                                                                                                                                       
            │ Patient Column:       │ patient                       │                                                                                                                                                                       
            │ Label Column:         │ label                         │                                                                                                                                                                       
            │ Slide Column:         │ slide                         │                                                                                                                                                                       
            │ Transform Labels:     │ False                         │                                                                                                                                                                       
            │ Modified Annotations: │ data/annotations.csv          │                                                                                                                                                                       
            │ Slideflow Project:    │ Loaded                        │                                                                                                                                                                       
            ╘═══════════════════════╧═══════════════════════════════╛
            ```   
        """
        vlog = self.vlog

        self.vlog("[bold underline]Project Summary[/]")
        table = [
            ("Project Directory:", str(self.project_dir)),
            ("Slide Directory:", str(self.slide_dir)),
            ("Annotations File:", str(self.annotations_file)),
            ("Patient Column:", self.patient_column),
            ("Label Column:", self.label_column),
            ("Slide Column:", self.slide_column or "None (using patient ID)"),
            ("Transform Labels:", str(self.transform_labels)),
            ("Modified Annotations:", str(self.modified_annotations or "Not yet created")),
            ("Slideflow Project:", "Loaded" if self.project else "Not initialized"),
        ]
        vlog(tabulate(table, tablefmt="fancy_outline"))

    # === Internals === #
    def _setup_annotations(
        self,
        annotations_file: Path,
        patient_column: str,
        label_column: str,
        slide_column: str | None,
        transform_labels: bool
    ) -> tuple[Path, dict | list[str]]:
        """
        Normalize the input annotations file to the required format.

        This includes:
            - Validating the presence of required columns.
            - Renaming the patient and label columns to `patient` and `label`.
            - Creating or renaming the `slide` column.
            - Optionally transforming labels to float encodings.
            - Saving the normalized file to project_dir/annotations.csv.

        AutoMIL requires the annotations file to have the following columns:
            - patient | contains patient identifiers
            - slide   | contains slide identifiers
            - label   | contains labels

        Returns:
            tuple[Path, dict | list[str]]:
                - Path to the saved normalized annotations CSV.
                - Label map (dict if transform_labels=True, else list of unique labels).

        Raises:
            ValueError:
                If required columns are missing.
            IOError:
                If the output annotations file cannot be written.
        """
        # Make sure given columns exist
        if not contains_columns(annotations_file, self._get_required_columns()):
            raise ValueError(f"Annotations file is missing required columns.")

        # Load annotations
        annotations = pd.read_csv(annotations_file, index_col=patient_column)
        annotations.index.name = "patient"

        # Renaming the slide column if provided, otherwise just use the patient column as slide identifier
        if not slide_column:
            annotations["slide"] = annotations.index
        else:
            annotations.rename(columns={slide_column: "slide"}, inplace=True)
        # Rename label column
        annotations.rename(columns={label_column: "label"}, inplace=True)

        labels = annotations["label"].unique()
        # Transform labels to float values
        if transform_labels:
            label_map = {label: float(i) for i, label in enumerate(labels)}
            annotations["label"] = annotations["label"].map(label_map)
            pretty = ", ".join(f"{k}: {v}" for k, v in label_map.items())
            self.vlog(f"Transformed labels to float values: [{INFO_CLR}]{pretty}[/]")
        else:
            label_map = annotations["label"].dropna().unique().tolist()
        
        # Save modified annotations
        out_path = self.project_dir / "annotations.csv"
        annotations.to_csv(out_path, index=True)

        if not out_path.exists():
            raise IOError(f"Failed to write annotations file: {out_path}")

        if annotations.empty:
            self.vlog("Warning: annotation file written but is empty.")

        self.vlog(f"Annotations saved to [{INFO_CLR}]{out_path}[/]")

        return out_path, label_map

    def _create_project_scaffold(
        self,
        annotations_file: Path,
        patient_column: str,
        label_column: str,
        slide_column: str | None,
        transform_labels: bool
        ) -> None:
        """Create the project directory (if needed) and generate modified
        annotations inside it.

        This method ensures that:
            - The project directory exists.
            - Normalized annotations are written to project_dir/annotations.csv.
            - label_map and modified_annotations are stored on the instance.

        Args:
            annotations_file (Path): Annotations file
            patient_column (str): column with patient identifiers
            label_column (str): column with label
            slide_column (str | None): column with slide identifiers. Optional
            transform_labels (bool): Whether to tranform labels to floats

        Raises:
            ValueError: If existing annotations file in project directory is invalid.
        """
        # Simple project directory creation
        if not self.project_dir.exists():
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self.vlog(f"Created project directory at [{INFO_CLR}]{self.project_dir}[/]")
        else:
            self.vlog(f"Project directory [{INFO_CLR}]{self.project_dir}[/] already exists")
        
        # Default: Checl for existing annotations file in project directory
        if (out_path := self.project_dir / "annotations.csv").exists():
            if not contains_columns(out_path, self._get_required_columns()):
                raise ValueError(f"Existing annotations file in project directory is invalid.")
            self.vlog(f"Using existing annotations file in project directory.")
            self.modified_annotations = out_path
            self.label_map = get_unique_labels(self.modified_annotations, self.label_column)
        # Fallback: create modified annotations file
        else:
            modified_annotations, label_map = self._setup_annotations(
                annotations_file,
                patient_column,
                label_column,
                slide_column,
                transform_labels
            )
            self.modified_annotations = modified_annotations
            self.label_map = label_map
        self.vlog(f"[{SUCCESS_CLR}]Project scaffold setup complete.[/]")
    
    def _get_required_columns(self) -> set[str]:
        """
        Returns the set of required columns for the annotations file.

        Returns:
            set[str]: Set of required column names.
        """
        required_columns = {self.patient_column, self.label_column}
        if self.slide_column:
            required_columns.add(self.slide_column)
        return required_columns
