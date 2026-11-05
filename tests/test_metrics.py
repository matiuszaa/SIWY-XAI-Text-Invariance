import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv.metrics import align_attributions, compute_metrics, spearman_similarity, top_k_jaccard


def test_align_attributions_uses_saved_word_mapping():
    original, transformed = align_attributions([1.0, 2.0], [20.0, 10.0], [1, 0])
    np.testing.assert_allclose(original, [1.0, 2.0])
    np.testing.assert_allclose(transformed, [10.0, 20.0])


def test_align_attributions_rejects_invalid_mapping():
    with pytest.raises(ValueError, match="out-of-range"):
        align_attributions([1.0], [2.0], [1])


def test_identical_attributions_have_perfect_metrics():
    scores = np.array([0.1, -0.8, 0.3, 0.6])
    assert spearman_similarity(scores, scores) == 1.0
    k, jaccard = top_k_jaccard(scores, scores, fraction=0.5)
    assert (k, jaccard) == (2, 1.0)


def test_compute_metrics_writes_one_row_per_accepted_pair(tmp_path):
    pd.DataFrame(
        {
            "pair_id": ["accepted", "rejected"],
            "text_id": ["a", "b"],
            "transformation": ["synonyms", "synonyms"],
            "word_alignment": [[0, 1, 2], [0, 1]],
            "filter_status": ["accepted", "rejected"],
        }
    ).to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["accepted", "accepted"],
            "version": ["original", "transformed"],
            "ig_scores": [[0.1, 0.9, -0.2], [0.1, 0.9, -0.2]],
        }
    ).to_parquet(tmp_path / "explanations.parquet", index=False)
    cfg = OmegaConf.create(
        {"paths": {"interim": str(tmp_path)}, "metrics": {"top_k_fraction": 0.5}}
    )

    out_path = compute_metrics(cfg)
    result = pd.read_parquet(out_path)

    assert out_path == tmp_path / "metrics.parquet"
    assert result.pair_id.tolist() == ["accepted"]
    assert result.n_aligned_words.tolist() == [3]
    assert result.spearman.tolist() == [1.0]
    assert result.top_k.tolist() == [2]
    assert result.top_k_jaccard.tolist() == [1.0]
