from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from automil.model import ModelManager
from automil.trainer import Trainer
from automil.utils import BATCH_SIZE, ModelType


def test_trainer_initialization(mock_sf_dataset, mock_project):
    """Test basic Trainer initialization."""
    with patch("automil.trainer.get_bag_avg_and_num_features", return_value=(100, 768)):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=mock_sf_dataset,
            val_dataset=mock_sf_dataset,
            model=ModelType.Attention_MIL,
        )

        assert trainer.initial_batch_size == BATCH_SIZE

def test_trainer_init_applies_hyperparameter_suggestions(
    mock_project,
    mock_sf_dataset,
):
    """Test that Trainer applies suggested hyperparameters."""
    with (
        patch("automil.trainer.get_bag_avg_and_num_features", return_value=(100, 768)),
        patch.object(ModelManager, "validate_hyperparameters", return_value={
            "lr": 1e-3,
            "initial_batch_size": 16,
        }),
    ):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=mock_sf_dataset,
            val_dataset=mock_sf_dataset,
            model=ModelType.Attention_MIL,
            lr=1e-4,
            batch_size=32,
        )

        assert trainer.lr == 1e-3
        assert trainer.initial_batch_size == 16

def test_num_classes_uses_train_dataset_annotations(
    mock_project,
):
    """Test that Trainer.num_classes uses train_dataset annotations if available."""
    train_dataset = MagicMock()
    train_dataset.annotations = pd.DataFrame({"label": [0, 1, 1, 2]})

    val_dataset = MagicMock()
    val_dataset.annotations = None

    with patch(
        "automil.trainer.get_bag_avg_and_num_features",
        return_value=(50, 512),
    ):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=ModelType.Attention_MIL,
        )

        assert trainer.num_classes == 3

def test_num_slides_sums_up_train_and_val(mock_project):
    """Test that Trainer.num_slides sums up slide counts from both train and val datasets."""

    with (
        patch("automil.trainer.get_num_slides", side_effect=[10, 15]),
        patch("automil.trainer.get_bag_avg_and_num_features", return_value=(50, 512)),
    ):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=MagicMock(),
            val_dataset=MagicMock(),
            model=ModelType.Attention_MIL,
        )

        assert trainer.num_slides == 25

def test_bag_avg_and_num_features_are_derived(
    mock_project,
    mock_sf_dataset,
):
    """Test that Trainer.bag_avg and Trainer.num_features are correctly derived."""
    with patch(
        "automil.trainer.get_bag_avg_and_num_features",
        return_value=(123, 768),
    ):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=mock_sf_dataset,
            val_dataset=mock_sf_dataset,
            model=ModelType.Attention_MIL,
        )

        assert trainer.bag_avg == 123
        assert trainer.num_features == 768

def test_estimated_size_mb_delegates_to_estimator(
    mock_project,
    mock_sf_dataset,
):
    with (
        patch("automil.trainer.get_bag_avg_and_num_features", return_value=(64, 512)),
        patch("automil.trainer.estimate_model_size", return_value=321.5) as mock_estimate,
    ):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=mock_sf_dataset,
            val_dataset=mock_sf_dataset,
            model=ModelType.Attention_MIL,
        )

        size = trainer.estimated_size_mb

        assert size == 321.5
        mock_estimate.assert_called_once()

def test_device_cpu_when_cuda_unavailable(
    mock_project,
    mock_sf_dataset,
):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("automil.trainer.get_bag_avg_and_num_features", return_value=(50, 512)),
    ):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=mock_sf_dataset,
            val_dataset=mock_sf_dataset,
            model=ModelType.Attention_MIL,
        )

        assert trainer.device.type == "cpu"

def test_config_is_built_via_mil_config(
    mock_project,
    mock_sf_dataset,
):
    fake_config = MagicMock()

    with (
        patch("automil.trainer.get_bag_avg_and_num_features", return_value=(32, 256)),
        patch("automil.trainer.mil_config", return_value=fake_config) as mock_mil_config,
    ):
        trainer = Trainer(
            bags_path=Path("/fake/bags"),
            project=mock_project,
            train_dataset=mock_sf_dataset,
            val_dataset=mock_sf_dataset,
            model=ModelType.Attention_MIL,
            lr=1e-4,
            epochs=10,
        )

        cfg = trainer.config

        assert cfg is fake_config
        mock_mil_config.assert_called_once()
