import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv import tracking


@pytest.fixture
def results_dir(tmp_path):
    pd.DataFrame(
        {
            "transformation": ["synonyms", "synonyms", "paraphrase"],
            "filter_status": ["accepted", "rejected", "accepted"],
            "filter_reason": ["passed", "low_semantic_similarity", "passed"],
        }
    ).to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pd.DataFrame(
        {"method": ["ig", "ig"], "device": ["cuda", "cuda"], "seconds": [0.5, 1.5]}
    ).to_parquet(tmp_path / "explanations.parquet", index=False)
    pd.DataFrame(
        {
            "transformation": ["synonyms", "paraphrase", "paraphrase"],
            "method": ["ig", "ig", "ig"],
            "alignment_coverage": [1.0, 0.8, 0.9],
            "spearman": [0.9, 0.4, 0.6],
            "top_k_jaccard": [0.5, 0.2, 0.4],
        }
    ).to_parquet(tmp_path / "metrics.parquet", index=False)
    return tmp_path


def make_cfg(path, mode):
    return OmegaConf.create(
        {"name": "test", "paths": {"interim": str(path)}, "wandb": {"project": "p", "mode": mode}}
    )


def test_summarize_results(results_dir):
    scalars, tables = tracking.summarize_results(
        pd.read_parquet(results_dir / "filtered_pairs.parquet"),
        pd.read_parquet(results_dir / "explanations.parquet"),
        pd.read_parquet(results_dir / "metrics.parquet"),
    )
    assert scalars["filter/synonyms/accepted_rate"] == 0.5
    assert scalars["time/ig/mean_seconds"] == 1.0
    assert scalars["invariance/paraphrase/ig/spearman"] == pytest.approx(0.5)
    assert scalars["invariance/synonyms/ig/alignment_coverage"] == 1.0
    assert set(tables) == {"filtering", "timing", "invariance"}
    assert tables["filtering"].pairs.sum() == 3


def test_log_results_skips_disabled_and_missing_config(results_dir, monkeypatch):
    monkeypatch.setattr(tracking.wandb, "init", lambda **kwargs: pytest.fail("init called"))
    assert not tracking.log_results(make_cfg(results_dir, "disabled"))
    assert not tracking.log_results(OmegaConf.create({"paths": {"interim": str(results_dir)}}))


def test_log_results_sends_summary_and_tables(results_dir, monkeypatch):
    class FakeRun:
        def __init__(self):
            self.summary, self.logged, self.finished = {}, {}, False

        def log(self, data):
            self.logged.update(data)

        def finish(self):
            self.finished = True

    run = FakeRun()
    init_kwargs = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return run

    monkeypatch.setattr(tracking.wandb, "init", fake_init)

    assert tracking.log_results(make_cfg(results_dir, "offline"))
    assert (init_kwargs["project"], init_kwargs["mode"], init_kwargs["name"]) == (
        "p",
        "offline",
        "test",
    )
    assert init_kwargs["config"]["wandb"]["mode"] == "offline"
    assert run.summary["time/ig/mean_seconds"] == 1.0
    assert set(run.logged) == {"filtering", "timing", "invariance"}
    assert run.finished
