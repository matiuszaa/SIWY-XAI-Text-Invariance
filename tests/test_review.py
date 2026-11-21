import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv.review import REVIEW_COLUMNS, export_review_sample, sample_for_review


@pytest.fixture
def pairs():
    rows = []
    for name, n in [("backtranslation", 10), ("paraphrase", 10), ("synonyms", 2)]:
        for i in range(n):
            rows.append(
                {
                    "pair_id": f"t{i}-{name}",
                    "transformation": name,
                    "original_text": f"original {i}",
                    "transformed_text": f"{name} {i}",
                    "semantic_similarity": 0.9,
                    "filter_status": "accepted",
                }
            )
    rows.append({**rows[0], "pair_id": "bad-paraphrase", "semantic_similarity": None})
    return pd.DataFrame(rows)


def test_sample_for_review_is_stratified_and_blind(pairs):
    sample = sample_for_review(pairs, n_pairs=9, seed=0)

    assert sample.columns.tolist() == REVIEW_COLUMNS
    counts = sample.pair_id.str.split("-").str[-1].value_counts().to_dict()
    assert counts == {"backtranslation": 3, "paraphrase": 3, "synonyms": 2}
    assert "bad-paraphrase" not in sample.pair_id.tolist()
    assert (sample.meaning_preserved == "").all()


def test_sample_for_review_is_deterministic(pairs):
    a = sample_for_review(pairs, n_pairs=6, seed=1)
    b = sample_for_review(pairs, n_pairs=6, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_export_review_sample_does_not_overwrite_annotations(tmp_path, pairs):
    pairs.to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    cfg = OmegaConf.create(
        {"seed": 42, "paths": {"interim": str(tmp_path)}, "review": {"n_pairs": 4}}
    )

    out_path = export_review_sample(cfg)
    assert len(pd.read_csv(out_path)) == 4

    with pytest.raises(FileExistsError):
        export_review_sample(cfg)
    export_review_sample(cfg, force=True)
