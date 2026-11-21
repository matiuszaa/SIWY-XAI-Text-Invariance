import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from xai_inv import quality
from xai_inv.detector import Predictions


def make_cfg(path, min_ratio=0.7, max_ratio=1.3, min_similarity=0.85):
    return OmegaConf.create(
        {
            "paths": {"interim": str(path)},
            "detector": {"model": "test", "device": "cpu"},
            "filter": {
                "min_length_ratio": min_ratio,
                "max_length_ratio": max_ratio,
                "sbert_model": "sbert",
                "min_similarity": min_similarity,
                "device": "cpu",
            },
        }
    )


class FakeDetector:
    def __init__(self, model, device):
        assert (model, device) == ("test", "cpu")
        self.calls = 0

    def predict(self, texts):
        self.calls += 1
        assert len(texts) == 4
        if self.calls == 1:
            return Predictions(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.3]))
        return Predictions(np.array([0, 0, 0, 1]), np.array([0.2, 0.4, 0.3, 0.6]))


def fake_similarity(originals, transformed, model_name, device):
    assert (model_name, device) == ("sbert", "cpu")
    assert len(originals) == len(transformed) == 4
    return np.array([0.95, 0.9, 0.9, 0.5])


def test_length_ratio():
    assert quality.length_ratio("a b c d", "a b c") == 0.75


def test_semantic_similarity_is_cosine_of_embeddings(monkeypatch):
    embeddings = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "c": [0.6, 0.8],
    }

    class FakeSentenceTransformer:
        def __init__(self, model_name, device):
            assert (model_name, device) == ("sbert", "cpu")

        def encode(self, texts, convert_to_tensor, normalize_embeddings):
            assert convert_to_tensor and normalize_embeddings
            return torch.tensor([embeddings[t] for t in texts])

    monkeypatch.setattr(quality, "SentenceTransformer", FakeSentenceTransformer)

    result = quality.semantic_similarity(["a", "a", "c"], ["a", "b", "a"], "sbert", "cpu")

    np.testing.assert_allclose(result, [1.0, 0.0, 0.6])


def test_apply_filters_records_first_failed_check(tmp_path, monkeypatch):
    pd.DataFrame(
        {
            "pair_id": ["stable", "changed", "short", "different", "invalid"],
            "original_text": ["a b c d", "a b c d", "a b c d", "a b c d", "bad"],
            "transformed_text": ["a b x d", "a x c d", "a b", "w x y z", None],
            "transformation_status": ["ok", "ok", "ok", "ok", "rejected"],
        }
    ).to_parquet(tmp_path / "pairs.parquet", index=False)
    monkeypatch.setattr(quality, "Detector", FakeDetector)
    monkeypatch.setattr(quality, "semantic_similarity", fake_similarity)

    out_path = quality.apply_filters(make_cfg(tmp_path))
    result = pd.read_parquet(out_path)

    assert out_path == tmp_path / "filtered_pairs.parquet"
    assert result.filter_status.tolist() == ["accepted"] + ["rejected"] * 4
    assert result.filter_reason.tolist() == [
        "passed",
        "detector_decision_changed",
        "length_ratio_out_of_range",
        "low_semantic_similarity",
        "transformation_rejected",
    ]
    assert result.length_ratio.tolist()[:4] == [1.0, 1.0, 0.5, 1.0]
    assert result.semantic_similarity.tolist()[:4] == [0.95, 0.9, 0.9, 0.5]
    assert (
        result.iloc[4][["length_ratio", "semantic_similarity", "original_p_chatgpt"]].isna().all()
    )
    assert result.original_predicted_label.tolist()[:4] == [0, 1, 1, 0]
    assert result.transformed_predicted_label.tolist()[:4] == [0, 0, 0, 1]
    assert result.original_p_chatgpt.tolist()[:4] == [0.1, 0.9, 0.8, 0.3]
    assert result.transformed_p_chatgpt.tolist()[:4] == [0.2, 0.4, 0.3, 0.6]


@pytest.mark.parametrize(
    ("min_ratio", "max_ratio", "min_similarity", "message"),
    [
        (0.0, 1.3, 0.85, "length ratio"),
        (1.2, 1.3, 0.85, "length ratio"),
        (0.7, 0.9, 0.85, "length ratio"),
        (0.7, 1.3, 1.5, "min_similarity"),
    ],
)
def test_apply_filters_rejects_invalid_thresholds(
    tmp_path, min_ratio, max_ratio, min_similarity, message
):
    pd.DataFrame(
        {"original_text": ["a"], "transformed_text": ["b"], "transformation_status": ["ok"]}
    ).to_parquet(tmp_path / "pairs.parquet", index=False)
    with pytest.raises(ValueError, match=message):
        quality.apply_filters(make_cfg(tmp_path, min_ratio, max_ratio, min_similarity))
