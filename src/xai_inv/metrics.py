from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr

METRIC_COLUMNS = [
    "pair_id",
    "text_id",
    "transformation",
    "n_aligned_words",
    "spearman",
    "top_k",
    "top_k_jaccard",
]


def align_attributions(
    original_scores: list[float] | np.ndarray,
    transformed_scores: list[float] | np.ndarray,
    word_alignment: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(original_scores, dtype=float)
    transformed = np.asarray(transformed_scores, dtype=float)
    alignment = np.asarray(word_alignment, dtype=int)
    if original.ndim != transformed.ndim or original.ndim != 1:
        raise ValueError("Attribution scores must be one-dimensional")
    if len(original) != len(alignment):
        raise ValueError("Word alignment must contain one entry for every original word")
    if len(alignment) == 0:
        return original, transformed[:0]
    if (alignment < 0).any() or (alignment >= len(transformed)).any():
        raise ValueError("Word alignment contains an out-of-range transformed-word index")
    return original, transformed[alignment]


def spearman_similarity(original: np.ndarray, transformed: np.ndarray) -> float:
    if len(original) < 2:
        return float("nan")
    return float(spearmanr(original, transformed).statistic)


def top_k_jaccard(
    original: np.ndarray, transformed: np.ndarray, fraction: float
) -> tuple[int, float]:
    if not 0 < fraction <= 1:
        raise ValueError(f"metrics.top_k_fraction must be in (0, 1], got {fraction}")
    if len(original) != len(transformed):
        raise ValueError("Top-k Jaccard requires equally sized aligned attribution vectors")
    if len(original) == 0:
        return 0, float("nan")

    k = min(len(original), max(1, ceil(len(original) * fraction)))
    original_top = set(np.argsort(-np.abs(original), kind="stable")[:k])
    transformed_top = set(np.argsort(-np.abs(transformed), kind="stable")[:k])
    return k, len(original_top & transformed_top) / len(original_top | transformed_top)


def compute_metrics(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    pairs_path = interim_dir / "filtered_pairs.parquet"
    explanations_path = interim_dir / "explanations.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Filtered pairs not found: {pairs_path}. Run 'filter' first.")
    if not explanations_path.is_file():
        raise FileNotFoundError(
            f"Explanations not found: {explanations_path}. Run 'explain' first."
        )

    pairs = pd.read_parquet(pairs_path)
    explanations = pd.read_parquet(explanations_path)
    required_pair_columns = {
        "pair_id",
        "text_id",
        "transformation",
        "word_alignment",
        "filter_status",
    }
    required_explanation_columns = {"pair_id", "version", "ig_scores"}
    missing_pairs = required_pair_columns - set(pairs.columns)
    missing_explanations = required_explanation_columns - set(explanations.columns)
    if missing_pairs:
        raise ValueError(
            f"Filtered pairs missing required columns: {', '.join(sorted(missing_pairs))}"
        )
    if missing_explanations:
        raise ValueError(
            f"Explanations missing required columns: {', '.join(sorted(missing_explanations))}"
        )

    fraction = float(cfg.metrics.top_k_fraction)
    rows = []
    for pair in pairs[pairs.filter_status.eq("accepted")].itertuples(index=False):
        pair_explanations = explanations[explanations.pair_id.eq(pair.pair_id)]
        by_version = pair_explanations.set_index("version")
        if set(by_version.index) != {"original", "transformed"} or by_version.index.has_duplicates:
            raise ValueError(
                f"Expected one original and one transformed explanation for {pair.pair_id}"
            )

        original, transformed = align_attributions(
            by_version.loc["original", "ig_scores"],
            by_version.loc["transformed", "ig_scores"],
            pair.word_alignment,
        )
        k, jaccard = top_k_jaccard(original, transformed, fraction)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "text_id": pair.text_id,
                "transformation": pair.transformation,
                "n_aligned_words": len(original),
                "spearman": spearman_similarity(original, transformed),
                "top_k": k,
                "top_k_jaccard": jaccard,
            }
        )

    out_path = interim_dir / "metrics.parquet"
    pd.DataFrame(rows, columns=METRIC_COLUMNS).to_parquet(out_path, index=False)
    return out_path
