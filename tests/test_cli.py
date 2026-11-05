from pathlib import Path

import pytest
from typer.testing import CliRunner

from xai_inv.cli import app
from xai_inv.config import load_config

CONFIGS = sorted(Path(__file__).parents[1].joinpath("configs").glob("*.yaml"))
runner = CliRunner()


def test_configs_exist():
    assert {p.stem for p in CONFIGS} == {"smoke", "small", "full", "prototype"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path):
    cfg = load_config(path)
    assert cfg.name == path.stem
    assert cfg.data.n_texts > 0


@pytest.mark.parametrize("command", ["filter", "metrics", "report"])
def test_command_runs(command):
    result = runner.invoke(app, [command, "--config", str(CONFIGS[0])])
    assert result.exit_code == 0, result.output
