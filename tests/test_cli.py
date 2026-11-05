from pathlib import Path

import pytest

from xai_inv.config import load_config

CONFIGS = sorted(Path(__file__).parents[1].joinpath("configs").glob("*.yaml"))


def test_configs_exist():
    assert {p.stem for p in CONFIGS} == {"smoke", "small", "full", "prototype"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path):
    cfg = load_config(path)
    assert cfg.name == path.stem
    assert cfg.data.n_texts > 0
