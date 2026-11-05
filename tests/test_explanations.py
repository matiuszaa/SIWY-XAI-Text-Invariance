import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from xai_inv import explanations
from xai_inv.explainers import WordAttribution


def test_explain_texts_writes_both_versions_of_accepted_pairs(tmp_path, monkeypatch):
    pairs = pd.DataFrame(
        {
            "pair_id": ["a-synonyms", "b-synonyms"],
            "text_id": ["a", "b"],
            "label": [0, 1],
            "transformation": ["synonyms", "synonyms"],
            "original_text": ["human original", "rejected original"],
            "transformed_text": ["human transformed", "rejected transformed"],
            "original_predicted_label": [0, 1],
            "transformed_predicted_label": [0, 0],
            "original_p_chatgpt": [0.2, 0.9],
            "transformed_p_chatgpt": [0.3, 0.4],
            "filter_status": ["accepted", "rejected"],
        }
    )
    pairs.to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    cfg = OmegaConf.create(
        {
            "paths": {"interim": str(tmp_path)},
            "detector": {"model": "test-model", "device": "cpu"},
            "explainers": {"ig": {"n_steps": 3}},
        }
    )

    class FakeDetector:
        def __init__(self, model, device):
            assert (model, device) == ("test-model", "cpu")

    def fake_explain_ig(detector, text, target, n_steps):
        assert isinstance(detector, FakeDetector)
        assert n_steps == 3
        return WordAttribution(text.split(), np.array([target + 0.1] * len(text.split())), 0.0)

    monkeypatch.setattr(explanations, "Detector", FakeDetector)
    monkeypatch.setattr(explanations, "explain_ig", fake_explain_ig)

    out_path = explanations.explain_texts(cfg)

    result = pd.read_parquet(out_path)
    assert out_path == tmp_path / "explanations.parquet"
    assert result.pair_id.tolist() == ["a-synonyms", "a-synonyms"]
    assert result.version.tolist() == ["original", "transformed"]
    assert result.text.tolist() == ["human original", "human transformed"]
    assert result.true_label.tolist() == [0, 0]
    assert result.predicted_label.tolist() == [0, 0]
    assert result.p_chatgpt.tolist() == [0.2, 0.3]
    assert [words.tolist() for words in result.words] == [
        ["human", "original"],
        ["human", "transformed"],
    ]
    assert [scores.tolist() for scores in result.ig_scores] == [[0.1, 0.1], [0.1, 0.1]]
