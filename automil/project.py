"""
Module for ``automil.Project``, which assists with setting up and managing an AutoMIL project.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Iterable

import pandas as pd
import slideflow as sf
from slideflow.util import is_project
from tabulate import tabulate

from .utils import INFO_CLR, SUCCESS_CLR, get_vlog


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
    Assists with setting up and managing a slideflow project instance.

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
        self.modified_annotations_file: Path = self.project_dir / "annotations.csv"

        self.patient_column = patient_column
        self.label_column = label_column
        self.slide_column = slide_column

        self.transform_labels = transform_labels
        self.vlog = get_vlog(verbose)

    # === Properties === #
    @cached_property
    def required_columns(self) -> set[str]:
        """Required columns for annotations file"""
        required = {self.patient_column, self.label_column}
        if self.slide_column:
            required.add(self.slide_column)
        return required

    @property
    def label_map(self) -> dict | list[str]:
        """
        Label mapping created during annotations setup.
        
        Returns:
            dict | list[str]: The label map (dict if transform_labels=True, else list of unique labels).
            
        Raises:
            AttributeError: If the label map has not been set up yet. Call setup_project_scaffold() first.
        """
        if not hasattr(self, '_label_map'):
            raise AttributeError(
                "Label map has not been set up yet. Call setup_project_scaffold() first."
            )
        return self._label_map
    
    @property
    def slide_ids(self) -> list[str]:
        """List of unique slide identifiers from the modified annotations file.

        Returns:
            list[str]: List of unique slide IDs.
        """
        if not hasattr(self, 'modified_annotations'):
            raise AttributeError(
                "Modified annotations have not been set up yet. Call setup_project_scaffold() first."
            )
        return self.modified_annotations["slide"].astype(str).unique().tolist()

    # === Public Methods === #
    def setup_project_scaffold(self) -> None:
        """Sets up the project directory structure and modifies annotations file."""
        self._setup_project_folder()
        self.modified_annotations = self._setup_annotations()
        self._label_map = self._setup_label_map()
        self.vlog(f"[{SUCCESS_CLR}]Project scaffold setup complete[/]")

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
        # Setup project folder and annotations
        self.setup_project_scaffold()

        # Load or create project
        if is_project(str(self.project_dir)):
            self.vlog(f"Loading existing project at [{INFO_CLR}]{self.project_dir}[/]")
            self.project = sf.load_project(str(self.project_dir))
        else:
            self.vlog(f"Creating new project at [{INFO_CLR}]{self.project_dir}[/]")
            self.project = sf.create_project(
                name="AutoMIL",
                root=str(self.project_dir),
                slides=str(self.slide_dir),
                annotations=str(self.modified_annotations_file),
            )
        return self.project
    
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

        vlog("[bold underline]Project Summary[/]")
        table = [
            ("Project Directory:", str(self.project_dir)),
            ("Slide Directory:", str(self.slide_dir)),
            ("Annotations File:", str(self.annotations_file)),
            ("Patient Column:", self.patient_column),
            ("Label Column:", self.label_column),
            ("Slide Column:", self.slide_column or "None (using patient ID)"),
            ("Transform Labels:", str(self.transform_labels)),
            ("Modified Annotations:", str(self.modified_annotations_file) or "Not yet created"),
            ("Slideflow Project:", "Loaded" if self.project else "Not initialized"),
        ]
        vlog(tabulate(table, tablefmt="fancy_outline"))

    # === Internals === #
    def _setup_project_folder(self) -> None:
        """Creates the project directory if it does not exist."""
        if not self.project_dir.exists():
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self.vlog(f"Created project directory at [{INFO_CLR}]{self.project_dir}[/]")
        else:
            self.vlog(f"Project directory [{INFO_CLR}]{self.project_dir}[/] already exists")

    def _setup_annotations(self) -> pd.DataFrame:
        """
        Normalize the input annotations file to the required format and set up label map.

        This includes:
            - Validating the presence of required columns.
            - Renaming the patient and label columns to `patient` and `label`.
            - Creating or renaming the `slide` column.
            - Optionally transforming labels to float encodings.
            - Creating and storing the label map for later use.
            - Saving the normalized file to project_dir/annotations.csv.

        AutoMIL requires the annotations file to have the following columns:
            - patient | contains patient identifiers
            - slide   | contains slide identifiers
            - label   | contains labels

        Raises:
            ValueError:
                If required columns are missing.
            IOError:
                If the output annotations file cannot be written.
        """
        # Make sure given columns exist
        if not (missing := contains_columns(self.annotations_file, self.required_columns, return_missing=True)):
            raise ValueError(f"Annotations file is missing required columns: {missing}")

        # Load annotations
        annotations = pd.read_csv(self.annotations_file, index_col=self.patient_column)
        annotations.index.name = "patient"

        # Renaming the slide column if provided, otherwise just use the patient column as slide identifier
        if not self.slide_column:
            annotations["slide"] = annotations.index
        else:
            annotations.rename(columns={self.slide_column: "slide"}, inplace=True)
        # Rename label column
        annotations.rename(columns={self.label_column: "label"}, inplace=True)

        # Save modified annotations
        out_path = self.modified_annotations_file
        annotations.to_csv(out_path, index=True)

        if not out_path.exists():
            raise IOError(f"Failed to write annotations file: {out_path}")

        if annotations.empty:
            self.vlog("Warning: annotation file written but is empty.")

        self.vlog(f"Annotations saved to [{INFO_CLR}]{out_path}[/]")
        return annotations

    def _setup_label_map(self) -> dict | list[str]:
        """Sets up the label map based on the modified annotations file.

        Returns:
            dict | list[str]: The label map (dict if transform_labels=True, else list of unique labels).
        """
        annotations = self.modified_annotations
        labels = annotations["label"].unique()
        
        # Transform labels to float values and store the mapping
        if self.transform_labels:
            label_map = {label: float(i) for i, label in enumerate(sorted(labels))}
            pretty = ", ".join(f"{k}: {v}" for k, v in label_map.items())
            self.vlog(f"Transformed labels to float values: [{INFO_CLR}]{pretty}[/]")
        else:
            # Store unique labels as sorted list
            label_map = sorted(labels.astype(str).tolist())
        
        return label_map