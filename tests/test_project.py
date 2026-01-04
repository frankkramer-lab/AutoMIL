from pathlib import Path
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest

from automil.project import Project, contains_columns


def test_project_initialization():
    """Test basic Project initialization."""
    project = Project(
        project_dir=Path("/tmp/project"),
        annotations_file=Path("/tmp/project/annotations.csv"),
        slide_dir=Path("/tmp/project/slides"),
        patient_column="patient_id",
        label_column="label",
    )

    assert project.project_dir == Path("/tmp/project")
    assert project.annotations_file == Path("/tmp/project/annotations.csv")
    assert project.slide_dir == Path("/tmp/project/slides")
    assert project.patient_column == "patient_id"
    assert project.label_column == "label"

def test_contains_columns_df_all_present():
    """Test contains_columns returns True when all columns are present."""
    df = pd.DataFrame({
        "patient": [1],
        "label": ["A"],
    })

    result = contains_columns(df, ["patient", "label"])
    assert result is True

def test_contains_columns_df_missing():
    """Test contains_columns returns False when some columns are missing."""
    df = pd.DataFrame({
        "patient": [1],
        "slide": ["s1"],
    })

    result = contains_columns(df, ["patient", "label"])
    assert result is False

def test_contains_columns_returns_missing_set():
    """Test contains_columns returns the correct set of missing columns."""
    df = pd.DataFrame({
        "patient": [1],
        "slide": ["s1"],
    })

    result = contains_columns(
        df,
        ["patient", "label", "slide"],
        return_missing=True
    )
    assert result == {"label"}

def test_contains_columns_csv(tmp_path):
    """Test contains_columns with a CSV file input."""
    df = pd.DataFrame({
        "patient": [1],
        "label": ["A"],
    })

    path = tmp_path / "annotations.csv"
    df.to_csv(path, index=False)

    result = contains_columns(path, ["patient", "label"])
    assert result is True

def test_contains_columns_parquet_missing(tmp_path):
    """Test contains_columns with a Parquet file input and missing columns."""
    df = pd.DataFrame({
        "patient": [1],
        "label": ["A"],
    })

    path = tmp_path / "annotations.parquet"
    df.to_parquet(path)

    missing = contains_columns(
        path,
        ["patient", "label", "slide"],
        return_missing=True
    )

    assert missing == {"slide"}

def test_required_columns(base_project):

    expected = {"patient_id", "label"}
    assert base_project.required_columns == expected

def test_required_columns_with_slide_column(project_factory):
    base_project = project_factory(
        kind="base",
        slide_column="slide_name"
    )

    expected = {"patient_id", "label", "slide_name"}
    assert base_project.required_columns == expected

def test_label_map_before_setup_raises_error(base_project):

    with pytest.raises(AttributeError):
        _ = base_project.label_map

def test_slide_ids_before_setup_raises_error(base_project):

    with pytest.raises(AttributeError):
        _ = base_project.slide_ids

def test_setup_annotations_with_valid_csv(base_project, tmp_path):
    """Test that _setup_annotations processes and saves the annotations correctly by renaming the given columns"""
    df = pd.DataFrame({
        "patient_id": [1, 2],
        "label": ["A", "B"],
        "slide_name": ["s1", "s2"],
    })

    path = tmp_path / "annotations.csv"
    df.to_csv(path, index=False)
    base_project.annotations_file = path

    # _setup_annnotations expects the project directory to already exist
    base_project.project_dir.mkdir(parents=True, exist_ok=True)

    base_project._setup_annotations()

    assert base_project.modified_annotations_file.exists()
    modified_ann = pd.read_csv(
        base_project.modified_annotations_file
    )
    assert [col in modified_ann.columns for col in ["patient", "label", "slide"]]

def test_setup_annotations_missing_columns_raises_error(base_project, tmp_path):
    df = pd.DataFrame({
        "patient_id": [1, 2],
        "slide_name": ["s1", "s2"],
    })

    path = tmp_path / "annotations.csv"
    df.to_csv(path, index=False)

    base_project.annotations_file = path

    # _setup_annnotations expects the project directory to already exist
    base_project.project_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="Annotations file is missing required columns"):
        base_project._setup_annotations()

def test_setup_label_map_without_transform(base_project, tmp_path):
    """Test that _setup_label_map creates the correct label map when called without transform_labels"""
    df = pd.DataFrame({
        "patient_id": [1, 2, 3],
        "label": ["A", "B", "A"],
        "slide_name": ["s1", "s2", "s3"],
    })

    path = tmp_path / "annotations.csv"
    df.to_csv(path, index=False)
    base_project.annotations_file = path

    base_project.setup_project_scaffold()

    expected_list = ["A", "B"]
    assert base_project.label_map == expected_list

def test_setup_label_map_with_transform(project_factory, tmp_path):
    """Test that _setup_label_map creates the correct label map when called with transform_labels"""
    base_project = project_factory(
        kind="base",
        transform_labels=True
    )

    df = pd.DataFrame({
        "patient_id": [1, 2, 3],
        "label": ["A", "B", "C"],
        "slide_name": ["s1", "s2", "s3"],
    })

    path = tmp_path / "annotations.csv"
    df.to_csv(path, index=False)
    base_project.annotations_file = path

    base_project.setup_project_scaffold()

    expected_list = {"A": 0, "B": 1, "C": 2}
    assert base_project.label_map == expected_list

def test_prepare_project_calls_load_project(base_project):
    """Test that prepare_project calls slideflow.load_project if project exists at project_dir"""

    with (
        patch("automil.project.is_project", return_value=True),
        patch("automil.project.Project.setup_project_scaffold"),
        patch("automil.project.sf.load_project") as mock_load_project
    ):
        base_project.prepare_project()

        mock_load_project.assert_called_once_with(str(base_project.project_dir))

def test_prepare_project_calls_create_project(base_project):
    """Test that prepare_project calls Project.create_project if project does not exist at project_dir"""

    with (
        patch("automil.project.is_project", return_value=False),
        patch("automil.project.Project.setup_project_scaffold"),
        patch("automil.project.sf.create_project") as mock_create_project
    ):
        base_project.prepare_project()

        mock_create_project.assert_called_once()