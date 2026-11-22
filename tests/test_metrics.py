import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv import metrics
from xai_inv.data import text_key
from xai_inv.metrics import (
    align_attributions,
    as_targets,
    compute_metrics,
    spearman_similarity,
    top_k_jaccard,
)


def test_as_targets_accepts_identity_and_simalign_mappings():
    assert as_targets(np.array([1, 0])) == [[1], [0]]
    assert as_targets([np.array([0, 2]), np.array([], dtype=int)]) == [[0, 2], []]


def test_align_attributions_uses_saved_word_mapping():
    original, transformed = align_attributions([1.0, 2.0], [20.0, 10.0], [[1], [0]])
    np.testing.assert_allclose(original, [1.0, 2.0])
    np.testing.assert_allclose(transformed, [10.0, 20.0])


def test_align_attributions_sums_targets_and_drops_unaligned_words():
    original, transformed = align_attributions([1.0, 2.0, 3.0], [0.5, 0.25, 4.0], [[0, 1], [], [2]])
    np.testing.assert_allclose(original, [1.0, 3.0])
    np.testing.assert_allclose(transformed, [0.75, 4.0])


def test_align_attributions_rejects_invalid_mapping():
    with pytest.raises(ValueError, match="out-of-range"):
        align_attributions([1.0], [2.0], [[1]])


def test_identical_attributions_have_perfect_metrics():
    scores = np.array([0.1, -0.8, 0.3, 0.6])
    assert spearman_similarity(scores, scores) == 1.0
    k, jaccard = top_k_jaccard(scores, scores, fraction=0.5)
    assert (k, jaccard) == (2, 1.0)


def test_compute_metrics_aligns_pairs_without_saved_mapping(tmp_path, monkeypatch):
    pd.DataFrame(
        {
            "pair_id": ["accepted", "rejected", "unaligned"],
            "text_id": ["a", "b", "a"],
            "transformation": ["synonyms", "synonyms", "backtranslation"],
            "original_text": ["x y z", "x y", "x y z"],
            "transformed_text": ["x y w", "x w", "z x"],
            "word_alignment": [[0, 1, 2], [0, 1], None],
            "filter_status": ["accepted", "rejected", "accepted"],
        }
    ).to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pd.DataFrame(
        {
            "text_key": [text_key(t) for t in ["x y z", "x y w", "z x", "x y z"]],
            "method": ["ig", "ig", "ig", "ig"],
            "run": ["main", "main", "main", "rerun"],
            "scores": [[0.1, 0.9, -0.2], [0.1, 0.9, -0.2], [-0.2, 0.1], [0.0, 0.0, 1.0]],
        }
    ).to_parquet(tmp_path / "explanations.parquet", index=False)
    cfg = OmegaConf.create(
        {
            "paths": {"interim": str(tmp_path)},
            "metrics": {"top_k_fraction": 0.5},
            "alignment": {"model": "mbert", "method": "itermax", "device": "cpu"},
        }
    )

    class FakeAligner:
        def __init__(self, model, method, device):
            assert (model, method, device) == ("mbert", "itermax", "cpu")

        def align(self, original, transformed):
            assert (original, transformed) == ("x y z", "z x")
            return [[1], [], [0]]

    monkeypatch.setattr(metrics, "WordAligner", FakeAligner)

    out_path = compute_metrics(cfg)
    result = pd.read_parquet(out_path)

    assert out_path == tmp_path / "metrics.parquet"
    assert result.pair_id.tolist() == ["accepted", "unaligned"]
    assert result.method.tolist() == ["ig", "ig"]
    assert [[t.tolist() for t in a] for a in result.word_alignment] == [
        [[0], [1], [2]],
        [[1], [], [0]],
    ]
    np.testing.assert_allclose(result.alignment_coverage, [1.0, 2 / 3])
    assert result.n_aligned_words.tolist() == [3, 2]
    np.testing.assert_allclose(result.spearman, [1.0, 1.0])
    assert result.top_k.tolist() == [2, 1]
    assert result.top_k_jaccard.tolist() == [1.0, 1.0]
