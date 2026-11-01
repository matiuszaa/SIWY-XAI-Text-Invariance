import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from xai_inv import explanations
from xai_inv.detector import Predictions
from xai_inv.explainers import WordAttribution


def test_explain_texts_writes_predictions_and_attributions(tmp_path, monkeypatch):
    texts = pd.DataFrame(
        {
            "text_id": ["a", "b"],
            "label": [0, 1],
            "text": ["human text", "chatgpt text"],
        }
    )
    texts.to_parquet(tmp_path / "texts.parquet", index=False)
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

        def predict(self, text_values):
            assert text_values == ["human text", "chatgpt text"]
            return Predictions(np.array([0, 1]), np.array([0.2, 0.8]))

    def fake_explain_ig(detector, text, target, n_steps):
        assert isinstance(detector, FakeDetector)
        assert n_steps == 3
        return WordAttribution(text.split(), np.array([target + 0.1, target + 0.2]), 0.0)

    monkeypatch.setattr(explanations, "Detector", FakeDetector)
    monkeypatch.setattr(explanations, "explain_ig", fake_explain_ig)

    out_path = explanations.explain_texts(cfg)

    result = pd.read_parquet(out_path)
    assert out_path == tmp_path / "explanations.parquet"
    assert result.text_id.tolist() == ["a", "b"]
    assert result.true_label.tolist() == [0, 1]
    assert result.predicted_label.tolist() == [0, 1]
    assert result.p_chatgpt.tolist() == [0.2, 0.8]
    assert [words.tolist() for words in result.words] == [["human", "text"], ["chatgpt", "text"]]
    assert [scores.tolist() for scores in result.ig_scores] == [[0.1, 0.2], [1.1, 1.2]]
