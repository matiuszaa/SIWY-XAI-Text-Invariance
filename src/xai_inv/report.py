from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def _number(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}"


def create_report(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    pairs_path = interim_dir / "filtered_pairs.parquet"
    metrics_path = interim_dir / "metrics.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Filtered pairs not found: {pairs_path}. Run 'filter' first.")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}. Run 'metrics' first.")

    pairs = pd.read_parquet(pairs_path)
    metrics = pd.read_parquet(metrics_path)
    required_pair_columns = {"transformation_status", "filter_status"}
    required_metric_columns = {"transformation", "spearman", "top_k_jaccard"}
    missing_pairs = required_pair_columns - set(pairs.columns)
    missing_metrics = required_metric_columns - set(metrics.columns)
    if missing_pairs:
        raise ValueError(
            f"Filtered pairs missing required columns: {', '.join(sorted(missing_pairs))}"
        )
    if missing_metrics:
        raise ValueError(f"Metrics missing required columns: {', '.join(sorted(missing_metrics))}")

    total_pairs = len(pairs)
    transformed_pairs = int(pairs.transformation_status.eq("ok").sum())
    accepted_pairs = int(pairs.filter_status.eq("accepted").sum())
    detector_rejections = int(pairs.filter_reason.eq("detector_decision_changed").sum())

    lines = [
        f"# Experiment report: {cfg.name}",
        "",
        "## Pair filtering",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Input texts | {total_pairs} |",
        f"| Valid transformations | {transformed_pairs} |",
        f"| Accepted detector-stable pairs | {accepted_pairs} |",
        f"| Rejected after detector decision change | {detector_rejections} |",
        "",
        "## Invariance metrics",
        "",
        "| Transformation | Pairs | Spearman mean | Spearman median | "
        "Top-k Jaccard mean | Top-k Jaccard median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if metrics.empty:
        lines.append("| — | 0 | — | — | — | — |")
    else:
        for transformation, group in metrics.groupby("transformation", sort=True):
            lines.append(
                "| "
                f"{transformation} | {len(group)} | {_number(group.spearman.mean())} | "
                f"{_number(group.spearman.median())} | {_number(group.top_k_jaccard.mean())} | "
                f"{_number(group.top_k_jaccard.median())} |"
            )

    out_path = interim_dir / "report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
