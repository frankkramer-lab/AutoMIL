#==============================================================================#
#  AutoMIL - Automated Machine Learning for Image Classification in            #
#  Whole-Slide Imaging with Multiple Instance Learning                         #
#                                                                              #
#  Copyright (C) 2026 Jonas Waibel                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.      #
#==============================================================================#
"""
Project management utilities for AutoMIL.

This module provides the :class:`automil.project.Project` class, which is
responsible for initializing, validating, and managing an AutoMIL project
along with an underlying Slideflow project instance.
"""


from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Iterable

import pandas as pd
import slideflow as sf
from slideflow.util import is_project

from .util import INFO_CLR, SUCCESS_CLR, get_vlog
from .util.logging import render_kv_table


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
        return set(columns) - set(data.columns)
    else:
        return set(columns).issubset(set(data.columns))   

class Project:
    """
    Manages the setup of an AutoMIL project.

    The Project class is responsible for:
        - Modifying the annotation file to conform to the expected slideflow format
        - Creating the project directory structure
        - Creating or loading a Slideflow project instance
        - Exposing project attributes to downstream processes

    A Project instance must be prepared before training, evaluation,
    or prediction can be performed.
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

        This metod itself does not create or modify files or directories. To prepare a directory to house
        a project, call :meth:`prepare_project`

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
        """
        Set of required columns expected in the annotation file.

        Includes:
            - Patient identifier column
            - Label column
            - Slide identifier column (if provided)

        Returns:
            Set of required columns
        """
        required = {self.patient_column, self.label_column}
        if self.slide_column:
            required.add(self.slide_column)
        return required

    @property
    def label_map(self) -> dict | list[str]:
        """
        Mapping between original labels and model-ready labels.

        The mapping is created during project scaffold setup.

        Returns:
            dict:
                Mapping from label to float if ``transform_labels=True`` or a list of unique labels otherwise.

        Raises:
            AttributeError:
                If the project scaffold has not been set up yet.
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
            List of unique slide IDs.
        """
        if not hasattr(self, 'modified_annotations'):
            raise AttributeError(
                "Modified annotations have not been set up yet. Call setup_project_scaffold() first."
            )
        return self.modified_annotations["slide"].astype(str).unique().tolist()

    # === Public Methods === #
    def setup_project_scaffold(self) -> None:
        """
        Creates the project directory and normalizes annotations.

        This method:
            - Creates the project directory if it does not exist
            - Normalizes the annotation file to Slideflow format
            - Generates and stores the label mapping
        """
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
        """Prints a simple summary of the Project Instance in a tabular format"""
        vlog = self.vlog
        rows = [
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


        vlog("[bold underline]Project Summary[/]")
        vlog(render_kv_table(rows, width=256))

    # === Internals === #
    def _setup_project_folder(self) -> None:
        """
        Ensures the project directory exists.

        Creates the directory and parent directories if necessary.
        """
        if not self.project_dir.exists():
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self.vlog(f"Created project directory at [{INFO_CLR}]{self.project_dir}[/]")
        else:
            self.vlog(f"Project directory [{INFO_CLR}]{self.project_dir}[/] already exists")

    def _setup_annotations(self) -> pd.DataFrame:
        """
        Normalizes the input annotations file to the required format and set up label map.

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
        if (missing := contains_columns(self.annotations_file, self.required_columns, return_missing=True)):
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