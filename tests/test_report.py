import pandas as pd
from omegaconf import OmegaConf

from xai_inv.report import create_report


def test_create_report_summarizes_filtering_and_metrics(tmp_path):
    pd.DataFrame(
        {
            "pair_id": ["accepted", "rejected", "invalid", "a-paraphrase"],
            "text_id": ["a", "b", "c", "a"],
            "label": [0, 1, 1, 0],
            "transformation": ["synonyms", "synonyms", "synonyms", "paraphrase"],
            "transformation_status": ["ok", "ok", "rejected", "ok"],
            "filter_status": ["accepted", "rejected", "rejected", "rejected"],
            "filter_reason": [
                "passed",
                "detector_decision_changed",
                "transformation_rejected",
                "low_semantic_similarity",
            ],
            "original_text": ["A small cat.", "A calm dog.", "Unchanged text.", "A small cat."],
            "transformed_text": ["A little cat.", "A quiet dog.", None, "Cats are small."],
            "original_predicted_label": [0, 1, None, 0],
            "transformed_predicted_label": [0, 0, None, 0],
            "original_p_chatgpt": [0.2, 0.9, None, 0.2],
            "transformed_p_chatgpt": [0.3, 0.4, None, 0.2],
        }
    ).to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["accepted", "rejected", "accepted"],
            "text_id": ["a", "b", "a"],
            "transformation": ["synonyms", "synonyms", "synonyms"],
            "method": ["ig", "ig", "lime"],
            "alignment_coverage": [1.0, 1.0, 1.0],
            "spearman": [0.2, 0.6, 0.1],
            "top_k_jaccard": [0.5, 1.0, 0.0],
        }
    ).to_parquet(tmp_path / "metrics.parquet", index=False)
    pd.DataFrame(
        {
            "text_id": ["a", "b", "c"],
            "source": ["reddit_eli5", "medicine", "wiki"],
            "question": ["Why cats?", "Why dogs?", "Why not?"],
        }
    ).to_parquet(tmp_path / "texts.parquet", index=False)
    pd.DataFrame(
        {"method": ["ig", "ig"], "device": ["cpu", "cpu"], "seconds": [1.0, 2.0]}
    ).to_parquet(tmp_path / "explanations.parquet", index=False)
    cfg = OmegaConf.create({"name": "prototype", "paths": {"interim": str(tmp_path)}})

    out_path = create_report(cfg)

    assert out_path == tmp_path / "report.md"
    report = out_path.read_text(encoding="utf-8")
    assert "| paraphrase | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |" in report
    assert "| synonyms | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |" in report
    assert "| Changed words (synonyms) | 2 / 6 |" in report
    assert "| Mean absolute change in p_chatgpt | 0.200 |" in report
    assert "| Original | 2 | 1 |" in report
    assert "| synonyms | ig | 2 | 1.000 | 0.400 | 0.400 | 0.750 | 0.750 |" in report
    assert "| synonyms | lime | 1 | 1.000 | 0.100 | 0.100 | 0.000 | 0.000 |" in report
    assert "| ig | cpu | 2 | 1.500 | 1.500 | 3.000 |" in report
    assert "| accepted | lime | 0.100 | 0.000 | 0.100 |" in report
    assert report.count("### accepted") == 1
    assert "**Source:** reddit_eli5" in report
    assert "**Question:** Why cats?" in report
