"""Tests for the AutoMIL command line interface.

The CLI commands import their heavy dependencies (slideflow, Project, Dataset, Trainer, Evaluator)
inside the command body. These tests replace those classes with mocks before invoking a command,
so they exercise argument parsing, option wiring, control flow and exit codes without
touching slides, GPUs or model training.
"""
import sys
import types
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from automil.cli.main import AutoMIL
from automil.cli.params import LazyChoice
from automil.util import RESOLUTION_PRESETS, LogLevel, ModelType

COMMANDS = ["run-pipeline", "train", "evaluate", "predict", "create-split"]


# === Helpers & Fixtures === #
def make_sf_dataset(num_slides: int = 4) -> MagicMock:
    """Creates a mock slideflow dataset whose `split` returns two new mock datasets."""
    dataset = MagicMock(name="sf_dataset")
    dataset.slides.return_value = [f"slide_{i}" for i in range(num_slides)]
    dataset.split.side_effect = lambda *args, **kwargs: (make_sf_dataset(), make_sf_dataset())
    return dataset


@dataclass
class CliMocks:
    """Mocks replacing heavy import classes"""
    project_cls: MagicMock
    dataset_cls: MagicMock
    trainer_cls: MagicMock
    evaluator_cls: MagicMock
    sf_dataset_cls: MagicMock
    configure_backend: MagicMock
    is_input_pretiled: MagicMock
    logs: list[tuple[LogLevel, str]] = field(default_factory=list)

    def logged_errors(self) -> list[str]:
        return [message for level, message in self.logs if level == LogLevel.ERROR]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cli_mocks(monkeypatch) -> CliMocks:
    """Patches every heavy dependency the CLI commands import at call time."""
    import slideflow

    import automil.dataset
    import automil.evaluation
    import automil.project
    import automil.trainer
    import automil.util
    import automil.util.backend
    import automil.util.pretiled

    project_cls = MagicMock(name="Project")
    project_setup = project_cls.return_value
    project_setup.label_map = ["A", "B"]
    project_setup.slide_ids = ["slide_0", "slide_1"]
    project_setup.prepare_project.return_value.root = "project_root"

    dataset_cls = MagicMock(name="Dataset")
    dataset_cls.return_value.prepare_dataset_source.side_effect = lambda: make_sf_dataset()

    sf_dataset_cls = MagicMock(name="sf.Dataset", side_effect=lambda *args, **kwargs: make_sf_dataset())

    mocks = CliMocks(
        project_cls=project_cls,
        dataset_cls=dataset_cls,
        trainer_cls=MagicMock(name="Trainer"),
        evaluator_cls=MagicMock(name="Evaluator"),
        sf_dataset_cls=sf_dataset_cls,
        configure_backend=MagicMock(name="configure_image_backend", return_value=False),
        is_input_pretiled=MagicMock(name="is_input_pretiled", return_value=False),
    )

    def fake_get_vlog(verbose: bool):
        def _vlog(message: str, level: LogLevel = LogLevel.INFO):
            mocks.logs.append((level, str(message)))
        return _vlog

    monkeypatch.setattr(automil.project, "Project", mocks.project_cls)
    monkeypatch.setattr(automil.dataset, "Dataset", mocks.dataset_cls)
    monkeypatch.setattr(automil.trainer, "Trainer", mocks.trainer_cls)
    monkeypatch.setattr(automil.evaluation, "Evaluator", mocks.evaluator_cls)
    monkeypatch.setattr(slideflow, "Dataset", mocks.sf_dataset_cls)
    monkeypatch.setattr(automil.util.backend, "configure_image_backend", mocks.configure_backend)
    monkeypatch.setattr(automil.util.pretiled, "is_input_pretiled", mocks.is_input_pretiled)
    monkeypatch.setattr(automil.util, "get_vlog", fake_get_vlog)
    return mocks


@pytest.fixture
def paths(tmp_path) -> types.SimpleNamespace:
    """Creates minimal inputs for CLI path arguments."""
    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()
    (slide_dir / "slide_0.svs").touch()

    annotations = tmp_path / "annotations.csv"
    annotations.write_text("patient,slide,label\np0,slide_0,A\n")

    bags_dir = tmp_path / "bags"
    bags_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    return types.SimpleNamespace(
        slide_dir=slide_dir,
        annotations=annotations,
        project_dir=tmp_path / "project",
        bags_dir=bags_dir,
        model_dir=model_dir,
        tmp=tmp_path,
    )


def train_args(paths, *extra: str) -> list[str]:
    return ["train", str(paths.slide_dir), str(paths.annotations), str(paths.project_dir), *extra]


def predict_args(command: str, paths, *extra: str) -> list[str]:
    return [command, str(paths.slide_dir), str(paths.annotations), str(paths.bags_dir), str(paths.model_dir), *extra]


# === CLI group === #
def test_version_matches_package_metadata(runner):
    """Test `automil --version` reports the installed package version."""
    result = runner.invoke(AutoMIL, ["--version"])

    assert result.exit_code == 0
    assert version("automil") in result.output


def test_help_lists_all_commands(runner):
    """Test `automil --help` lists every registered command."""
    result = runner.invoke(AutoMIL, ["--help"])

    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_renders(runner, command):
    """Test every command's help page renders without errors."""
    result = runner.invoke(AutoMIL, [command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_without_arguments_shows_help(runner, cli_mocks, command):
    """Test invoking a command without arguments shows its usage instead of running it."""
    result = runner.invoke(AutoMIL, [command])

    assert "Usage:" in result.output
    cli_mocks.project_cls.assert_not_called()
    cli_mocks.sf_dataset_cls.assert_not_called()


# === Argument validation === #
def test_train_rejects_missing_slide_dir(runner, cli_mocks, paths):
    """Test a non-existent slide directory is rejected before any work is done."""
    result = runner.invoke(
        AutoMIL,
        ["train", str(paths.tmp / "missing"), str(paths.annotations), str(paths.project_dir)],
    )

    assert result.exit_code == 2
    assert "does not exist" in result.output
    cli_mocks.project_cls.assert_not_called()


def test_train_rejects_missing_annotation_file(runner, cli_mocks, paths):
    """Test a non-existent annotation file is rejected before any work is done."""
    result = runner.invoke(
        AutoMIL,
        ["train", str(paths.slide_dir), str(paths.tmp / "missing.csv"), str(paths.project_dir)],
    )

    assert result.exit_code == 2
    cli_mocks.project_cls.assert_not_called()


@pytest.mark.parametrize("option, value", [
    ("--model", "ResNet"),
    ("--tissue-detection", "magic"),
    ("-k", "three"),
])
def test_train_rejects_invalid_option_values(runner, cli_mocks, paths, option, value):
    """Test invalid option values are rejected with a usage error."""
    result = runner.invoke(AutoMIL, train_args(paths, option, value))

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    cli_mocks.project_cls.assert_not_called()


def test_train_rejects_unknown_stain_normalizer(runner, cli_mocks, paths):
    """Test the lazily loaded stain normalizer choices reject unknown methods."""
    result = runner.invoke(AutoMIL, train_args(paths, "--stain-normalizer", "not-a-normalizer"))

    assert result.exit_code == 2
    assert "not a valid choice" in result.output
    cli_mocks.project_cls.assert_not_called()


def test_evaluate_rejects_missing_model_dir(runner, cli_mocks, paths):
    """Test a non-existent model directory is rejected before any work is done."""
    result = runner.invoke(
        AutoMIL,
        ["evaluate", str(paths.slide_dir), str(paths.annotations), str(paths.bags_dir), str(paths.tmp / "missing")],
    )

    assert result.exit_code == 2
    cli_mocks.evaluator_cls.assert_not_called()


# === train === #
def test_train_uses_defaults(runner, cli_mocks, paths):
    """Test `automil train` wires its default options into Project, Dataset and Trainer."""
    result = runner.invoke(AutoMIL, train_args(paths))

    assert result.exit_code == 0, result.output

    project_args = cli_mocks.project_cls.call_args
    assert project_args.args == (
        paths.project_dir, paths.annotations, paths.slide_dir, "patient", "label", None,
    )
    assert project_args.kwargs["transform_labels"] is False

    dataset_args = cli_mocks.dataset_cls.call_args
    assert dataset_args.args[1] is RESOLUTION_PRESETS.Low
    assert dataset_args.kwargs["tissue_detection"] == "otsu"
    assert dataset_args.kwargs["stain_normalizer"] == "reinhard"
    assert dataset_args.kwargs["bags_dir"] == paths.project_dir / "bags"

    trainer_args = cli_mocks.trainer_cls.call_args
    assert trainer_args.kwargs["model"] is ModelType.Attention_MIL
    assert trainer_args.kwargs["k"] == 3
    cli_mocks.trainer_cls.return_value.train_k_fold.assert_called_once()


def test_train_passes_column_options_to_project(runner, cli_mocks, paths):
    """Test the column overwrite options reach the Project setup."""
    result = runner.invoke(
        AutoMIL,
        train_args(paths, "-pc", "case_id", "-lc", "diagnosis", "-sc", "slide_name", "--transform_labels"),
    )

    assert result.exit_code == 0, result.output
    project_args = cli_mocks.project_cls.call_args
    assert project_args.args[3:6] == ("case_id", "diagnosis", "slide_name")
    assert project_args.kwargs["transform_labels"] is True


def test_train_passes_model_and_folds_to_trainer(runner, cli_mocks, paths):
    """Test `-m` and `-k` reach the Trainer."""
    result = runner.invoke(AutoMIL, train_args(paths, "-m", "TransMIL", "-k", "5"))

    assert result.exit_code == 0, result.output
    trainer_args = cli_mocks.trainer_cls.call_args
    assert trainer_args.kwargs["model"] is ModelType.TransMIL
    assert trainer_args.kwargs["k"] == 5


def test_train_passes_preprocessing_options_to_dataset(runner, cli_mocks, paths):
    """Test the tissue detection and stain normalization options reach the Dataset."""
    result = runner.invoke(
        AutoMIL,
        train_args(paths, "--tissue-detection", "both", "--stain-normalizer", "macenko"),
    )

    assert result.exit_code == 0, result.output
    dataset_args = cli_mocks.dataset_cls.call_args
    assert dataset_args.kwargs["tissue_detection"] == "both"
    assert dataset_args.kwargs["stain_normalizer"] == "macenko"


def test_train_runs_once_per_resolution(runner, cli_mocks, paths):
    """Test multiple resolutions create one dataset and one training run per preset."""
    result = runner.invoke(AutoMIL, train_args(paths, "-r", "Low, High"))

    assert result.exit_code == 0, result.output
    presets = [call.args[1] for call in cli_mocks.dataset_cls.call_args_list]
    assert presets == [RESOLUTION_PRESETS.Low, RESOLUTION_PRESETS.High]
    assert cli_mocks.trainer_cls.call_count == 2
    assert cli_mocks.trainer_cls.return_value.train_k_fold.call_count == 2


def test_train_detects_pretiled_input_when_flag_not_set(runner, cli_mocks, paths):
    """Test pretiled input is auto-detected when `--is-pretiled` is not given."""
    cli_mocks.is_input_pretiled.return_value = True

    result = runner.invoke(AutoMIL, train_args(paths))

    assert result.exit_code == 0, result.output
    cli_mocks.is_input_pretiled.assert_called_once_with(paths.slide_dir, ["slide_0", "slide_1"])
    assert cli_mocks.dataset_cls.call_args.kwargs["is_pretiled"] is True


def test_train_skips_pretiled_detection_when_flag_set(runner, cli_mocks, paths):
    """Test `--is-pretiled` bypasses the auto-detection."""
    result = runner.invoke(AutoMIL, train_args(paths, "--is-pretiled"))

    assert result.exit_code == 0, result.output
    cli_mocks.is_input_pretiled.assert_not_called()
    assert cli_mocks.dataset_cls.call_args.kwargs["is_pretiled"] is True


def test_train_passes_tiff_conversion_from_backend_configuration(runner, cli_mocks, paths):
    """Test the result of the image backend configuration reaches the Dataset."""
    cli_mocks.configure_backend.return_value = True

    result = runner.invoke(AutoMIL, train_args(paths))

    assert result.exit_code == 0, result.output
    assert cli_mocks.dataset_cls.call_args.kwargs["tiff_conversion"] is True


# === Exit codes === #
def test_train_exits_with_error_on_unknown_resolution(runner, cli_mocks, paths):
    """Test an unknown resolution preset fails the command with exit code 1."""
    result = runner.invoke(AutoMIL, train_args(paths, "-r", "Medium"))

    assert result.exit_code == 1
    assert any("Medium" in message for message in cli_mocks.logged_errors())
    cli_mocks.trainer_cls.assert_not_called()


def test_train_exits_with_error_when_training_fails(runner, cli_mocks, paths):
    """Test an exception during training fails the command with exit code 1 and logs the error."""
    cli_mocks.trainer_cls.return_value.train_k_fold.side_effect = RuntimeError("CUDA out of memory")

    result = runner.invoke(AutoMIL, train_args(paths))

    assert result.exit_code == 1
    assert "Error: CUDA out of memory" in cli_mocks.logged_errors()


def test_run_pipeline_exits_with_error_when_evaluation_fails(runner, cli_mocks, paths):
    """Test an exception during evaluation fails `run-pipeline` with exit code 1."""
    cli_mocks.evaluator_cls.return_value.evaluate_models.side_effect = RuntimeError("no predictions")

    result = runner.invoke(
        AutoMIL,
        ["run-pipeline", str(paths.slide_dir), str(paths.annotations), str(paths.project_dir)],
    )

    assert result.exit_code == 1
    assert "Error: no predictions" in cli_mocks.logged_errors()


@pytest.mark.parametrize("command", ["evaluate", "predict"])
def test_prediction_commands_exit_with_error_when_project_setup_fails(runner, cli_mocks, paths, command):
    """Test errors during project setup are handled like any other error (logged, exit code 1)."""
    cli_mocks.project_cls.return_value.setup_project_scaffold.side_effect = ValueError("missing column 'label'")

    result = runner.invoke(AutoMIL, predict_args(command, paths))

    assert result.exit_code == 1
    assert "Error: missing column 'label'" in cli_mocks.logged_errors()
    cli_mocks.evaluator_cls.assert_not_called()


def test_create_split_exits_with_error_when_split_fails(runner, cli_mocks, paths):
    """Test an exception while splitting fails `create-split` with exit code 1."""
    cli_mocks.sf_dataset_cls.side_effect = None
    cli_mocks.sf_dataset_cls.return_value.split.side_effect = ValueError("not enough slides")

    result = runner.invoke(AutoMIL, ["create-split", str(paths.slide_dir), str(paths.annotations)])

    assert result.exit_code == 1
    assert "Error: not enough slides" in cli_mocks.logged_errors()


# === run-pipeline === #
def test_run_pipeline_trains_then_evaluates(runner, cli_mocks, paths):
    """Test `run-pipeline` trains on the train split and evaluates on the held-out test split."""
    result = runner.invoke(
        AutoMIL,
        ["run-pipeline", str(paths.slide_dir), str(paths.annotations), str(paths.project_dir)],
    )

    assert result.exit_code == 0, result.output
    cli_mocks.trainer_cls.return_value.train_k_fold.assert_called_once()

    # The evaluator receives the test split of the first resolution's dataset
    source = cli_mocks.dataset_cls.return_value.prepare_dataset_source
    assert source.call_count == 1
    evaluator_args = cli_mocks.evaluator_cls.call_args.args
    assert evaluator_args[1:] == (
        paths.project_dir / "models",
        paths.project_dir / "ensemble",
        paths.project_dir / "bags",
    )

    evaluator = cli_mocks.evaluator_cls.return_value
    evaluator.evaluate_models.assert_called_once_with(generate_attention_heatmaps=True)
    evaluator.create_ensemble_predictions.assert_called_once()
    evaluator.compare_predictions.assert_called_once()
    evaluator.generate_plots.assert_called_once()


def test_run_pipeline_passes_split_file(runner, cli_mocks, paths):
    """Test `--split-file` is used for the initial train/test split."""
    split_file = paths.tmp / "my_split.json"
    datasets = []
    cli_mocks.dataset_cls.return_value.prepare_dataset_source.side_effect = (
        lambda: datasets.append(make_sf_dataset()) or datasets[-1]
    )

    result = runner.invoke(
        AutoMIL,
        ["run-pipeline", str(paths.slide_dir), str(paths.annotations), str(paths.project_dir),
         "--split-file", str(split_file)],
    )

    assert result.exit_code == 0, result.output
    assert datasets[0].split.call_args.kwargs["splits"] == str(split_file)


# === evaluate / predict === #
def test_evaluate_runs_full_evaluation(runner, cli_mocks, paths):
    """Test `evaluate` sets up the output project and runs evaluation, comparison and plotting."""
    output_dir = paths.tmp / "evaluation"
    cli_mocks.project_cls.return_value.modified_annotations_file = output_dir / "annotations.csv"

    result = runner.invoke(AutoMIL, predict_args("evaluate", paths, "-o", str(output_dir), "-lc", "diagnosis"))

    assert result.exit_code == 0, result.output

    project_args = cli_mocks.project_cls.call_args
    assert project_args.args[0] == output_dir
    assert project_args.args[4] == "diagnosis"
    cli_mocks.sf_dataset_cls.assert_called_once_with(
        slides=str(paths.slide_dir),
        annotations=str(output_dir / "annotations.csv"),
    )

    evaluator_args = cli_mocks.evaluator_cls.call_args.args
    assert evaluator_args[1:] == (paths.model_dir, output_dir, paths.bags_dir)
    evaluator = cli_mocks.evaluator_cls.return_value
    evaluator.evaluate_models.assert_called_once_with(generate_attention_heatmaps=True)
    evaluator.compare_predictions.assert_called_once()
    evaluator.generate_plots.assert_called_once()


def test_predict_generates_predictions(runner, cli_mocks, paths):
    """Test `predict` only generates predictions and does not run the evaluation."""
    result = runner.invoke(AutoMIL, predict_args("predict", paths))

    assert result.exit_code == 0, result.output
    evaluator = cli_mocks.evaluator_cls.return_value
    evaluator.generate_predictions.assert_called_once()
    evaluator.evaluate_models.assert_not_called()


@pytest.mark.parametrize("command, default_dir", [("evaluate", "evaluation"), ("predict", "predictions")])
def test_prediction_commands_default_output_dir(runner, cli_mocks, paths, command, default_dir):
    """Test the default output directory of `evaluate` and `predict`."""
    result = runner.invoke(AutoMIL, predict_args(command, paths))

    assert result.exit_code == 0, result.output
    assert cli_mocks.project_cls.call_args.args[0] == Path(default_dir)


# === create-split === #
def test_create_split_uses_defaults(runner, cli_mocks, paths):
    """Test `create-split` defaults: 20% test fraction, `split.json`, overwrite allowed."""
    cli_mocks.sf_dataset_cls.side_effect = None

    result = runner.invoke(AutoMIL, ["create-split", str(paths.slide_dir), str(paths.annotations)])

    assert result.exit_code == 0, result.output
    cli_mocks.sf_dataset_cls.return_value.split.assert_called_once_with(
        labels="label", val_fraction=0.2, splits="split.json", read_only=False,
    )


def test_create_split_passes_options(runner, cli_mocks, paths):
    """Test `create-split` passes the output file, test fraction and read-only flag to slideflow."""
    cli_mocks.sf_dataset_cls.side_effect = None
    output_file = paths.tmp / "splits" / "fold.json"

    result = runner.invoke(
        AutoMIL,
        ["create-split", str(paths.slide_dir), str(paths.annotations),
         "-o", str(output_file), "-f", "0.3", "--read-only"],
    )

    assert result.exit_code == 0, result.output
    cli_mocks.sf_dataset_cls.assert_called_once_with(
        slides=str(paths.slide_dir), annotations=str(paths.annotations),
    )
    cli_mocks.sf_dataset_cls.return_value.split.assert_called_once_with(
        labels="label", val_fraction=0.3, splits=str(output_file), read_only=True,
    )


# === LazyChoice === #
@pytest.fixture
def fake_module(monkeypatch) -> types.ModuleType:
    """Registers a fake importable module exposing a registry of choices."""
    module = types.ModuleType("fake_registry")
    module.Registry = types.SimpleNamespace(options={"alpha": 1, "beta": 2, "gamma": 3})
    monkeypatch.setitem(sys.modules, "fake_registry", module)
    return module


def make_lazy_choice(transform=None) -> LazyChoice:
    return LazyChoice(
        "fake_registry",
        attribute="Registry",
        transform=transform or (lambda registry: registry.options.keys()),
    )


def test_lazy_choice_does_not_import_on_creation():
    """Test the module is only imported when the choices are needed."""
    choice = LazyChoice("module.that.does.not.exist", attribute="Anything")

    assert choice.get_metavar(None, None) == "NORMALIZER"


def test_lazy_choice_accepts_valid_value(fake_module):
    """Test a value from the loaded choices is accepted unchanged."""
    choice = make_lazy_choice()

    assert choice.convert("beta", None, None) == "beta"
    assert tuple(choice.choices) == ("alpha", "beta", "gamma")


def test_lazy_choice_rejects_invalid_value(fake_module):
    """Test a value outside the loaded choices fails with a helpful message."""
    choice = make_lazy_choice()

    with pytest.raises(click.BadParameter, match="Choose from: alpha, beta, gamma"):
        choice.convert("delta", None, None)


def test_lazy_choice_caches_choices(fake_module):
    """Test the choices are loaded once and reused."""
    transform = MagicMock(return_value=["alpha"])
    choice = make_lazy_choice(transform)

    choice.convert("alpha", None, None)
    choice.convert("alpha", None, None)
    _ = choice.choices

    transform.assert_called_once()


def test_lazy_choice_shell_completion_filters_by_prefix(fake_module):
    """Test shell completion only suggests choices starting with the typed prefix."""
    choice = make_lazy_choice(lambda registry: ["alpha", "alphabet", "beta"])

    completions = [item.value for item in choice.shell_complete(None, None, "alp")]

    assert completions == ["alpha", "alphabet"]
