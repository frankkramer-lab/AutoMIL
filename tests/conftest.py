from unittest.mock import MagicMock

import pytest
from slideflow import Project as sf_Project

from automil.project import Project as AutoMIL_Project


@pytest.fixture
def project_factory(tmp_path):
    """Basic factory fixture for creating a project.
    Returns a function that can create either a Mockup or a real AutoMIL Project instance.
    """
    def _create(kind: str = "base", **kwargs) -> MagicMock | AutoMIL_Project:
        if kind == "mock":
            mock_project = MagicMock(spec=sf_Project)
            mock_project.root = tmp_path / "project"
            mock_project.sources = {}
            mock_project.annotations = None
            return mock_project
        
        elif kind == "base":
            return AutoMIL_Project(
                project_dir=tmp_path / "project",
                annotations_file=tmp_path / "project" / "annotations.csv",
                slide_dir=tmp_path / "project" / "slides",
                patient_column="patient_id",
                label_column="label",
                **kwargs
            )
        
        raise ValueError(f"Unknown project kind: {kind}")
    
    return _create

@pytest.fixture
def mock_project(project_factory) -> MagicMock:
    """Fixture that provides a mock Project instance."""
    return project_factory(kind="mock")

@pytest.fixture
def base_project(project_factory) -> AutoMIL_Project:
    """Fixture that provides a base Project instance."""
    return project_factory(kind="base")