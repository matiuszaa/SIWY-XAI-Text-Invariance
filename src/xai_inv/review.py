from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

REVIEW_COLUMNS = ["pair_id", "original_text", "transformed_text", "meaning_preserved", "notes"]


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
