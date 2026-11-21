from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

REVIEW_COLUMNS = ["pair_id", "original_text", "transformed_text", "meaning_preserved", "notes"]
ANSWERS = {
    "yes": True,
    "y": True,
    "tak": True,
    "1": True,
    "no": False,
    "n": False,
    "nie": False,
    "0": False,
}


def sample_for_review(pairs: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    candidates = pairs[pairs.semantic_similarity.notna()]
    transformations = sorted(candidates.transformation.unique())
    if not transformations:
        raise ValueError("No transformed pairs to review")
    parts = []
    for i, name in enumerate(transformations):
        n = n_pairs // len(transformations) + (i < n_pairs % len(transformations))
        group = candidates[candidates.transformation.eq(name)]
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed)
    return sample.assign(meaning_preserved="", notes="")[REVIEW_COLUMNS]


def export_review_sample(cfg: DictConfig, force: bool = False) -> Path:
    interim_dir = Path(cfg.paths.interim)
    pairs_path = interim_dir / "filtered_pairs.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Filtered pairs not found: {pairs_path}. Run 'filter' first.")
    out_path = interim_dir / "review.csv"
    if out_path.is_file() and not force:
        raise FileExistsError(f"{out_path} already exists, use --force to overwrite it")

    pairs = pd.read_parquet(pairs_path)
    sample = sample_for_review(pairs, int(cfg.review.n_pairs), int(cfg.seed))
    sample.to_csv(out_path, index=False)
    return out_path


def parse_answer(value: str) -> bool | None:
    value = value.strip().lower()
    if not value:
        return None
    if value not in ANSWERS:
        raise ValueError(f"Unknown meaning_preserved answer: {value!r}")
    return ANSWERS[value]


def threshold_table(annotated: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    good = annotated.meaning_preserved.astype(bool)
    rows = []
    for threshold in thresholds:
        kept = annotated.semantic_similarity >= threshold
        good_kept = int((kept & good).sum())
        rows.append(
            {
                "threshold": threshold,
                "kept": int(kept.sum()),
                "good_kept": good_kept,
                "bad_kept": int((kept & ~good).sum()),
                "good_lost": int((~kept & good).sum()),
                "precision": good_kept / kept.sum() if kept.any() else float("nan"),
                "recall": good_kept / good.sum() if good.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _number(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}"


def summarize_review(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    review_path = interim_dir / "review.csv"
    pairs_path = interim_dir / "filtered_pairs.parquet"
    if not review_path.is_file():
        raise FileNotFoundError(f"Review file not found: {review_path}. Run 'review' first.")
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Filtered pairs not found: {pairs_path}. Run 'filter' first.")

    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    review["meaning_preserved"] = review.meaning_preserved.map(parse_answer)
    annotated = review[review.meaning_preserved.notna()]
    if annotated.empty:
        raise ValueError(f"No annotated pairs in {review_path}")
    pairs = pd.read_parquet(pairs_path)
    annotated = annotated.merge(
        pairs[["pair_id", "transformation", "semantic_similarity"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    unknown = annotated.pair_id[annotated.semantic_similarity.isna()].tolist()
    if unknown:
        raise ValueError(f"Annotated pairs without similarity: {', '.join(unknown)}")

    lines = [
        f"# Manual review: {cfg.name}",
        "",
        f"Annotated pairs: {len(annotated)} / {len(review)}",
        "",
        "| Transformation | Annotated | Meaning preserved |",
        "|---|---:|---:|",
    ]
    for name, group in annotated.groupby("transformation", sort=True):
        lines.append(f"| {name} | {len(group)} | {int(group.meaning_preserved.sum())} |")

    lines.extend(
        [
            "",
            "## SBERT threshold",
            "",
            "| Threshold | Kept | Good kept | Bad kept | Good lost | Precision | Recall |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_table(annotated, list(cfg.review.thresholds)).itertuples(index=False):
        lines.append(
            f"| {row.threshold:.2f} | {row.kept} | {row.good_kept} | {row.bad_kept} | "
            f"{row.good_lost} | {_number(row.precision)} | {_number(row.recall)} |"
        )

    lines.extend(
        [
            "",
            "## Pairs that changed meaning",
            "",
            "| Pair ID | Similarity | Notes |",
            "|---|---:|---|",
        ]
    )
    bad = annotated[~annotated.meaning_preserved.astype(bool)]
    for row in bad.sort_values("semantic_similarity", ascending=False).itertuples(index=False):
        notes = " ".join(row.notes.split()).replace("|", "\\|")
        lines.append(f"| {row.pair_id} | {row.semantic_similarity:.3f} | {notes} |")
    if bad.empty:
        lines.append("| — | — | — |")

    out_path = interim_dir / "review_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
