import pandas as pd
from omegaconf import OmegaConf

from xai_inv.report import create_report


def test_create_report_summarizes_filtering_and_metrics(tmp_path):
    pd.DataFrame(
        {
            "pair_id": ["accepted", "rejected", "invalid"],
            "text_id": ["a", "b", "c"],
            "transformation_status": ["ok", "ok", "rejected"],
            "filter_status": ["accepted", "rejected", "rejected"],
            "filter_reason": [
                "same_detector_decision",
                "detector_decision_changed",
                "transformation_rejected",
            ],
            "original_text": ["A small cat.", "A calm dog.", "Unchanged text."],
            "transformed_text": ["A little cat.", "A quiet dog.", None],
            "original_predicted_label": [0, 1, None],
            "transformed_predicted_label": [0, 0, None],
            "original_p_chatgpt": [0.2, 0.9, None],
            "transformed_p_chatgpt": [0.3, 0.4, None],
        }
    ).to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["accepted", "rejected"],
            "text_id": ["a", "b"],
            "transformation": ["synonyms", "synonyms"],
            "spearman": [0.2, 0.6],
            "top_k_jaccard": [0.5, 1.0],
        }
    ).to_parquet(tmp_path / "metrics.parquet", index=False)
    pd.DataFrame(
        {
            "text_id": ["a", "b", "c"],
            "source": ["reddit_eli5", "medicine", "wiki"],
            "question": ["Why cats?", "Why dogs?", "Why not?"],
        }
    ).to_parquet(tmp_path / "texts.parquet", index=False)
    cfg = OmegaConf.create({"name": "prototype", "paths": {"interim": str(tmp_path)}})

    out_path = create_report(cfg)

    assert out_path == tmp_path / "report.md"
    report = out_path.read_text(encoding="utf-8")
    assert "| Changed words | 2 / 6 |" in report
    assert "| Mean absolute change in p_chatgpt | 0.300 |" in report
    assert "| Original | 1 | 1 |" in report
    assert "| synonyms | 2 | 0.400 | 0.400 | 0.750 | 0.750 |" in report
    assert "| accepted | 0.200 | 0.500 | 0.100 |" in report
    assert "### accepted" in report
    assert "**Source:** reddit_eli5" in report
    assert "**Question:** Why cats?" in report
