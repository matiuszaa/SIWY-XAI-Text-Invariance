import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv.review import (
    REVIEW_COLUMNS,
    export_review_sample,
    parse_answer,
    sample_for_review,
    summarize_review,
    threshold_table,
)


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


@pytest.mark.parametrize(
    ("value", "expected"), [("Tak", True), (" y ", True), ("0", False), ("nie", False), ("", None)]
)
def test_parse_answer(value, expected):
    assert parse_answer(value) is expected


def test_parse_answer_rejects_unknown_value():
    with pytest.raises(ValueError, match="maybe"):
        parse_answer("maybe")


def test_threshold_table_counts_kept_and_lost_pairs():
    annotated = pd.DataFrame(
        {"semantic_similarity": [0.95, 0.9, 0.85, 0.8], "meaning_preserved": [1, 0, 1, 0]}
    )

    table = threshold_table(annotated, [0.8, 0.9, 0.99])

    assert table.kept.tolist() == [4, 2, 0]
    assert table.good_kept.tolist() == [2, 1, 0]
    assert table.bad_kept.tolist() == [2, 1, 0]
    assert table.good_lost.tolist() == [0, 1, 2]
    assert table.precision.tolist()[:2] == [0.5, 0.5]
    assert pd.isna(table.precision.iloc[2])
    assert table.recall.tolist() == [1.0, 0.5, 0.0]


def test_summarize_review_joins_hidden_scores(tmp_path, pairs):
    pairs.loc[pairs.pair_id.eq("t1-paraphrase"), "semantic_similarity"] = 0.8
    pairs.to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["t0-paraphrase", "t1-paraphrase", "t0-synonyms"],
            "original_text": ["a", "b", "c"],
            "transformed_text": ["a", "b", "c"],
            "meaning_preserved": ["tak", "nie", ""],
            "notes": ["", "lost the | negation", ""],
        }
    ).to_csv(tmp_path / "review.csv", index=False)
    cfg = OmegaConf.create(
        {"name": "test", "paths": {"interim": str(tmp_path)}, "review": {"thresholds": [0.85]}}
    )

    report = summarize_review(cfg).read_text(encoding="utf-8")

    assert "Annotated pairs: 2 / 3" in report
    assert "| paraphrase | 2 | 1 |" in report
    assert "| 0.85 | 1 | 1 | 0 | 0 | 1.000 | 1.000 |" in report
    assert "| t1-paraphrase | 0.800 | lost the \\| negation |" in report


def test_summarize_review_requires_annotations(tmp_path, pairs):
    pairs.to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pairs.assign(meaning_preserved="", notes="")[REVIEW_COLUMNS].to_csv(
        tmp_path / "review.csv", index=False
    )
    cfg = OmegaConf.create({"name": "test", "paths": {"interim": str(tmp_path)}})

    with pytest.raises(ValueError, match="No annotated pairs"):
        summarize_review(cfg)
