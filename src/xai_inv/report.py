from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def _number(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}"


def _changed_words(original_text: str, transformed_text: str) -> tuple[int, int]:
    original_words = original_text.split()
    transformed_words = transformed_text.split()
    changed = sum(a != b for a, b in zip(original_words, transformed_words, strict=False))
    changed += abs(len(original_words) - len(transformed_words))
    return changed, len(original_words)


def _markdown_text(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def create_report(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    pairs_path = interim_dir / "filtered_pairs.parquet"
    metrics_path = interim_dir / "metrics.parquet"
    texts_path = interim_dir / "texts.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Filtered pairs not found: {pairs_path}. Run 'filter' first.")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}. Run 'metrics' first.")
    if not texts_path.is_file():
        raise FileNotFoundError(f"Prepared texts not found: {texts_path}. Run 'prepare' first.")

    pairs = pd.read_parquet(pairs_path)
    metrics = pd.read_parquet(metrics_path)
    texts = pd.read_parquet(texts_path)
    required_pair_columns = {
        "pair_id",
        "text_id",
        "transformation_status",
        "filter_status",
        "filter_reason",
        "original_text",
        "transformed_text",
        "original_predicted_label",
        "transformed_predicted_label",
        "original_p_chatgpt",
        "transformed_p_chatgpt",
    }
    required_metric_columns = {"pair_id", "text_id", "transformation", "spearman", "top_k_jaccard"}
    required_text_columns = {"text_id", "source", "question"}
    missing_pairs = required_pair_columns - set(pairs.columns)
    missing_metrics = required_metric_columns - set(metrics.columns)
    missing_texts = required_text_columns - set(texts.columns)
    if missing_pairs:
        raise ValueError(
            f"Filtered pairs missing required columns: {', '.join(sorted(missing_pairs))}"
        )
    if missing_metrics:
        raise ValueError(f"Metrics missing required columns: {', '.join(sorted(missing_metrics))}")
    if missing_texts:
        raise ValueError(
            f"Prepared texts missing required columns: {', '.join(sorted(missing_texts))}"
        )

    total_pairs = len(pairs)
    transformed_pairs = int(pairs.transformation_status.eq("ok").sum())
    accepted_pairs = int(pairs.filter_status.eq("accepted").sum())
    detector_rejections = int(pairs.filter_reason.eq("detector_decision_changed").sum())
    valid_pairs = pairs[pairs.transformation_status.eq("ok")].copy()
    word_changes = valid_pairs.apply(
        lambda row: _changed_words(row.original_text, row.transformed_text), axis=1
    )
    changed_words = sum(change[0] for change in word_changes)
    total_words = sum(change[1] for change in word_changes)
    replacement_rates = [changed / total for changed, total in word_changes if total > 0]
    changed_word_rate = changed_words / total_words if total_words else float("nan")
    median_replacement_rate = pd.Series(replacement_rates).median()
    confidence_changes = (
        pd.to_numeric(valid_pairs.original_p_chatgpt, errors="coerce")
        - pd.to_numeric(valid_pairs.transformed_p_chatgpt, errors="coerce")
    ).abs()

    original_labels = pd.to_numeric(valid_pairs.original_predicted_label, errors="coerce")
    transformed_labels = pd.to_numeric(valid_pairs.transformed_predicted_label, errors="coerce")
    metrics_with_confidence = metrics.merge(
        valid_pairs[["pair_id", "original_p_chatgpt", "transformed_p_chatgpt"]],
        on="pair_id",
        how="left",
    ).merge(texts[["text_id", "source", "question"]], on="text_id", how="left")
    metrics_with_confidence["abs_p_chatgpt_change"] = (
        metrics_with_confidence.original_p_chatgpt - metrics_with_confidence.transformed_p_chatgpt
    ).abs()

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
        "## Transformation impact",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Changed words | {changed_words} / {total_words} |",
        f"| Changed-word rate | {_number(changed_word_rate)} |",
        f"| Median pair-level changed-word rate | {_number(median_replacement_rate)} |",
        f"| Mean absolute change in p_chatgpt | {_number(confidence_changes.mean())} |",
        f"| Median absolute change in p_chatgpt | {_number(confidence_changes.median())} |",
        f"| Maximum absolute change in p_chatgpt | {_number(confidence_changes.max())} |",
        "",
        "## Detector decisions",
        "",
        "| Text version | Human (0) | ChatGPT (1) |",
        "|---|---:|---:|",
        f"| Original | {int((original_labels == 0).sum())} | {int((original_labels == 1).sum())} |",
        f"| Transformed | {int((transformed_labels == 0).sum())} | "
        f"{int((transformed_labels == 1).sum())} |",
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

    lines.extend(
        [
            "",
            "## Least stable explanation pairs",
            "",
            "| Pair ID | Spearman | Top-k Jaccard | Absolute change in p_chatgpt |",
            "|---|---:|---:|---:|",
        ]
    )
    if metrics_with_confidence.empty:
        lines.append("| — | — | — | — |")
    else:
        least_stable = metrics_with_confidence.nsmallest(3, "spearman")
        for row in least_stable.itertuples(index=False):
            lines.append(
                f"| {row.pair_id} | {_number(row.spearman)} | {_number(row.top_k_jaccard)} | "
                f"{_number(row.abs_p_chatgpt_change)} |"
            )
        for row in least_stable.itertuples(index=False):
            lines.extend(
                [
                    "",
                    f"### {row.pair_id}",
                    "",
                    f"**Source:** {_markdown_text(row.source)}  ",
                    f"**Question:** {_markdown_text(row.question)}",
                ]
            )

    out_path = interim_dir / "report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
