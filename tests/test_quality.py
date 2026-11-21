import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv import quality
from xai_inv.detector import Predictions


def make_cfg(path, min_ratio=0.7, max_ratio=1.3):
    return OmegaConf.create(
        {
            "paths": {"interim": str(path)},
            "detector": {"model": "test", "device": "cpu"},
            "filter": {"min_length_ratio": min_ratio, "max_length_ratio": max_ratio},
        }
    )


class FakeDetector:
    def __init__(self, model, device):
        assert (model, device) == ("test", "cpu")
        self.calls = 0

    def predict(self, texts):
        self.calls += 1
        assert len(texts) == 3
        if self.calls == 1:
            return Predictions(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8]))
        return Predictions(np.array([0, 0, 0]), np.array([0.2, 0.4, 0.3]))


def test_length_ratio():
    assert quality.length_ratio("a b c d", "a b c") == 0.75


def test_apply_filters_records_first_failed_check(tmp_path, monkeypatch):
    pd.DataFrame(
        {
            "pair_id": ["stable", "changed", "short", "invalid"],
            "original_text": ["a b c d", "a b c d", "a b c d", "bad"],
            "transformed_text": ["a b x d", "a x c d", "a b", None],
            "transformation_status": ["ok", "ok", "ok", "rejected"],
        }
    ).to_parquet(tmp_path / "pairs.parquet", index=False)
    monkeypatch.setattr(quality, "Detector", FakeDetector)

    out_path = quality.apply_filters(make_cfg(tmp_path))
    result = pd.read_parquet(out_path)

    assert out_path == tmp_path / "filtered_pairs.parquet"
    assert result.filter_status.tolist() == ["accepted", "rejected", "rejected", "rejected"]
    assert result.filter_reason.tolist() == [
        "passed",
        "detector_decision_changed",
        "length_ratio_out_of_range",
        "transformation_rejected",
    ]
    assert result.length_ratio.tolist()[:3] == [1.0, 1.0, 0.5]
    assert pd.isna(result.length_ratio.iloc[3])
    assert result.original_predicted_label.tolist()[:3] == [0, 1, 1]
    assert result.transformed_predicted_label.tolist()[:3] == [0, 0, 0]
    assert result.original_p_chatgpt.tolist()[:3] == [0.1, 0.9, 0.8]
    assert result.transformed_p_chatgpt.tolist()[:3] == [0.2, 0.4, 0.3]
    assert pd.isna(result.original_p_chatgpt.iloc[3])


@pytest.mark.parametrize(("min_ratio", "max_ratio"), [(0.0, 1.3), (1.2, 1.3), (0.7, 0.9)])
def test_apply_filters_rejects_invalid_ratio_range(tmp_path, min_ratio, max_ratio):
    pd.DataFrame(
        {"original_text": ["a"], "transformed_text": ["b"], "transformation_status": ["ok"]}
    ).to_parquet(tmp_path / "pairs.parquet", index=False)
    with pytest.raises(ValueError, match="length ratio"):
        quality.apply_filters(make_cfg(tmp_path, min_ratio, max_ratio))
