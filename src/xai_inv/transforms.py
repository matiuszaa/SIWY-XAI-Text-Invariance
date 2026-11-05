import random
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class Augmenter(Protocol):
    def augment(self, text: str) -> list[str]: ...


PAIR_COLUMNS = [
    "pair_id",
    "text_id",
    "label",
    "transformation",
    "original_text",
    "transformed_text",
    "original_n_words",
    "transformed_n_words",
    "word_alignment",
    "transformation_status",
    "transformation_reason",
]


def build_synonym_augmenter(fraction: float) -> Augmenter:
    if not 0 < fraction <= 1:
        raise ValueError(f"synonyms.fraction must be in (0, 1], got {fraction}")

    from textattack.augmentation import WordNetAugmenter

    return WordNetAugmenter(pct_words_to_swap=fraction, transformations_per_example=1)


def synonym_pair(text: str, augmenter: Augmenter) -> dict[str, object]:
    original_words = text.split()
    candidates = augmenter.augment(text)
    if not candidates:
        return {
            "transformed_text": None,
            "transformed_n_words": None,
            "word_alignment": None,
            "transformation_status": "rejected",
            "transformation_reason": "no_candidate",
        }

    transformed = candidates[0].strip()
    transformed_words = transformed.split()
    if transformed == text:
        reason = "unchanged"
    elif len(transformed_words) != len(original_words):
        reason = "word_count_changed"
    else:
        return {
            "transformed_text": transformed,
            "transformed_n_words": len(transformed_words),
            "word_alignment": list(range(len(original_words))),
            "transformation_status": "ok",
            "transformation_reason": None,
        }

    return {
        "transformed_text": transformed,
        "transformed_n_words": len(transformed_words),
        "word_alignment": None,
        "transformation_status": "rejected",
        "transformation_reason": reason,
    }


def transform_synonyms(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    texts_path = interim_dir / "texts.parquet"
    if not texts_path.is_file():
        raise FileNotFoundError(f"Prepared texts not found: {texts_path}. Run 'prepare' first.")

    texts = pd.read_parquet(texts_path)
    required_columns = {"text_id", "label", "text"}
    missing_columns = required_columns - set(texts.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prepared texts missing required columns: {missing}")

    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    augmenter = build_synonym_augmenter(float(cfg.transforms.synonyms.fraction))

    rows = []
    for row in texts.itertuples(index=False):
        pair = synonym_pair(row.text, augmenter)
        rows.append(
            {
                "pair_id": f"{row.text_id}-synonyms",
                "text_id": row.text_id,
                "label": int(row.label),
                "transformation": "synonyms",
                "original_text": row.text,
                "original_n_words": len(row.text.split()),
                **pair,
            }
        )

    out_path = interim_dir / "pairs.parquet"
    pd.DataFrame(rows, columns=PAIR_COLUMNS).to_parquet(out_path, index=False)
    return out_path
