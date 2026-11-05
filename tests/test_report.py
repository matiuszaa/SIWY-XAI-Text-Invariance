import pandas as pd
from omegaconf import OmegaConf

from xai_inv.report import create_report


def test_create_report_summarizes_filtering_and_metrics(tmp_path):
    pd.DataFrame(
        {
            "transformation_status": ["ok", "ok", "rejected"],
            "filter_status": ["accepted", "rejected", "rejected"],
            "filter_reason": [
                "same_detector_decision",
                "detector_decision_changed",
                "transformation_rejected",
            ],
        }
    ).to_parquet(tmp_path / "filtered_pairs.parquet", index=False)
    pd.DataFrame(
        {
            "transformation": ["synonyms", "synonyms"],
            "spearman": [0.2, 0.6],
            "top_k_jaccard": [0.5, 1.0],
        }
    ).to_parquet(tmp_path / "metrics.parquet", index=False)
    cfg = OmegaConf.create({"name": "prototype", "paths": {"interim": str(tmp_path)}})

    out_path = create_report(cfg)

    assert out_path == tmp_path / "report.md"
    assert out_path.read_text(encoding="utf-8") == (
        "# Experiment report: prototype\n"
        "\n"
        "## Pair filtering\n"
        "\n"
        "| Measure | Value |\n"
        "|---|---:|\n"
        "| Input texts | 3 |\n"
        "| Valid transformations | 2 |\n"
        "| Accepted detector-stable pairs | 1 |\n"
        "| Rejected after detector decision change | 1 |\n"
        "\n"
        "## Invariance metrics\n"
        "\n"
        "| Transformation | Pairs | Spearman mean | Spearman median | "
        "Top-k Jaccard mean | Top-k Jaccard median |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        "| synonyms | 2 | 0.400 | 0.400 | 0.750 | 0.750 |\n"
    )
