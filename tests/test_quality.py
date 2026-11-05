import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from xai_inv import quality
from xai_inv.detector import Predictions


def test_filter_by_detector_scores_both_versions_and_keeps_stable_pairs(tmp_path, monkeypatch):
    pd.DataFrame(
        {
            "pair_id": ["stable", "changed", "invalid"],
            "original_text": ["original stable", "original changed", "bad"],
            "transformed_text": ["transformed stable", "transformed changed", None],
            "transformation_status": ["ok", "ok", "rejected"],
        }
    ).to_parquet(tmp_path / "pairs.parquet", index=False)
    cfg = OmegaConf.create(
        {"paths": {"interim": str(tmp_path)}, "detector": {"model": "test", "device": "cpu"}}
    )

    class FakeDetector:
        def __init__(self, model, device):
            assert (model, device) == ("test", "cpu")
            self.calls = 0

        def predict(self, texts):
            self.calls += 1
            assert len(texts) == 2
            if self.calls == 1:
                return Predictions(np.array([0, 1]), np.array([0.1, 0.9]))
            return Predictions(np.array([0, 0]), np.array([0.2, 0.4]))

    monkeypatch.setattr(quality, "Detector", FakeDetector)

    out_path = quality.filter_by_detector(cfg)
    result = pd.read_parquet(out_path)

    assert out_path == tmp_path / "filtered_pairs.parquet"
    assert result.filter_status.tolist() == ["accepted", "rejected", "rejected"]
    assert result.filter_reason.tolist() == [
        "same_detector_decision",
        "detector_decision_changed",
        "transformation_rejected",
    ]
    assert result.original_predicted_label.tolist()[:2] == [0, 1]
    assert result.transformed_predicted_label.tolist()[:2] == [0, 0]
    assert result.original_p_chatgpt.tolist()[:2] == [0.1, 0.9]
    assert result.transformed_p_chatgpt.tolist()[:2] == [0.2, 0.4]
    assert pd.isna(result.original_p_chatgpt.iloc[2])
