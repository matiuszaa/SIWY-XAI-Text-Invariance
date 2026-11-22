import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv import explanations
from xai_inv.data import text_key
from xai_inv.explainers import WordAttribution


@pytest.fixture
def pairs_dir(tmp_path):
    pd.DataFrame(
        {
            "pair_id": ["a-synonyms", "a-paraphrase", "b-synonyms"],
            "original_text": ["human original", "human original", "rejected original"],
            "transformed_text": ["human transformed", "human rewritten", "rejected transformed"],
            "original_predicted_label": [0, 0, 1],
            "transformed_predicted_label": [0, 0, 0],
            "original_p_chatgpt": [0.2, 0.2, 0.9],
            "transformed_p_chatgpt": [0.3, 0.1, 0.4],
            "filter_status": ["accepted", "accepted", "rejected"],
        }
    ).to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    return tmp_path


def make_cfg(path, n_steps=3):
    return OmegaConf.create(
        {
            "paths": {"interim": str(path)},
            "detector": {"model": "test-model", "device": "cpu"},
            "explainers": {
                "ig": {"n_steps": n_steps},
                "lime": {"n_samples": 10, "seed": 42, "rerun_seed": 43},
            },
        }
    )


@pytest.fixture
def calls(monkeypatch):
    calls = []

    class FakeDetector:
        def __init__(self, model, device):
            assert (model, device) == ("test-model", "cpu")

    def fake_run_explainer(detector, method, params, text, target, seed):
        assert isinstance(detector, FakeDetector)
        calls.append((method, text, seed))
        words = text.split()
        return WordAttribution(words, np.full(len(words), target + (seed or 0) / 100), 0.0)

    monkeypatch.setattr(explanations, "Detector", FakeDetector)
    monkeypatch.setattr(explanations, "run_explainer", fake_run_explainer)
    return calls


def test_explain_texts_explains_each_accepted_text_once(pairs_dir, calls):
    out_path = explanations.explain_texts(make_cfg(pairs_dir))
    result = pd.read_parquet(out_path)

    assert out_path == pairs_dir / "explanations.parquet"
    assert sorted(calls, key=str) == sorted(
        [
            ("ig", "human original", None),
            ("ig", "human transformed", None),
            ("ig", "human rewritten", None),
            ("lime", "human original", 42),
            ("lime", "human original", 43),
            ("lime", "human transformed", 42),
            ("lime", "human rewritten", 42),
        ],
        key=str,
    )
    rerun = result[result.run.eq("rerun")]
    assert rerun.text_key.tolist() == [text_key("human original")]
    assert rerun.method.tolist() == ["lime"]
    original = result[result.text_key.eq(text_key("human original")) & result.run.eq("main")]
    assert original.target.tolist() == [0, 0]
    assert original.p_chatgpt.tolist() == [0.2, 0.2]
    assert [w.tolist() for w in original.words] == [["human", "original"]] * 2


def test_explain_texts_reuses_cached_explanations(pairs_dir, calls):
    explanations.explain_texts(make_cfg(pairs_dir))
    n_first = len(calls)
    explanations.explain_texts(make_cfg(pairs_dir))
    assert len(calls) == n_first

    explanations.explain_texts(make_cfg(pairs_dir, n_steps=5))
    assert [method for method, _, _ in calls[n_first:]] == ["ig", "ig", "ig"]


def test_run_explainer_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown explainer"):
        explanations.run_explainer(None, "occlusion", OmegaConf.create({}), "a", 0, None)
