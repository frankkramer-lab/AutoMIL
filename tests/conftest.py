from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_project():
    """A simple project mockup for testing."""
    project = MagicMock(spec=["root", "sources", "annotations"])
    project.root = "/tmp/project"
    project.sources = {}
    project.annotations = None
    return project