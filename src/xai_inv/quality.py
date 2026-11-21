from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from xai_inv.detector import Detector

FILTER_COLUMNS = [
    "original_predicted_label",
    "transformed_predicted_label",
    "original_p_chatgpt",
    "transformed_p_chatgpt",
    "length_ratio",
    "semantic_similarity",
    "filter_status",
    "filter_reason",
]


def length_ratio(original: str, transformed: str) -> float:
    return len(transformed.split()) / len(original.split())


def semantic_similarity(
    originals: list[str], transformed: list[str], model_name: str, device: str
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    a = model.encode(originals, convert_to_tensor=True, normalize_embeddings=True)
    b = model.encode(transformed, convert_to_tensor=True, normalize_embeddings=True)
    return (a * b).sum(dim=-1).cpu().numpy()


def apply_filters(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    pairs_path = interim_dir / "pairs.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Transformed pairs not found: {pairs_path}. Run 'transform' first."
        )

    pairs = pd.read_parquet(pairs_path)
    required_columns = {"original_text", "transformed_text", "transformation_status"}
    missing_columns = required_columns - set(pairs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Transformed pairs missing required columns: {missing}")

    min_ratio = float(cfg.filter.min_length_ratio)
    max_ratio = float(cfg.filter.max_length_ratio)
    if not 0 < min_ratio <= 1 <= max_ratio:
        raise ValueError(f"Invalid length ratio range: [{min_ratio}, {max_ratio}]")
    min_similarity = float(cfg.filter.min_similarity)
    if not -1 <= min_similarity <= 1:
        raise ValueError(f"filter.min_similarity must be in [-1, 1], got {min_similarity}")

    result = pairs.copy()
    for column in FILTER_COLUMNS:
        result[column] = None

    valid = result.transformation_status.eq("ok") & result.transformed_text.notna()
    decision_changed = pd.Series(False, index=result.index)
    ratios = pd.Series(np.nan, index=result.index)
    similarities = pd.Series(np.nan, index=result.index)
    if valid.any():
        originals = result.loc[valid, "original_text"].tolist()
        transformed_texts = result.loc[valid, "transformed_text"].tolist()
        detector = Detector(cfg.detector.model, device=cfg.detector.device)
        original = detector.predict(originals)
        transformed = detector.predict(transformed_texts)

        result.loc[valid, "original_predicted_label"] = original.labels.astype(int)
        result.loc[valid, "transformed_predicted_label"] = transformed.labels.astype(int)
        result.loc[valid, "original_p_chatgpt"] = original.probs.astype(float)
        result.loc[valid, "transformed_p_chatgpt"] = transformed.probs.astype(float)
        decision_changed[valid] = original.labels != transformed.labels
        ratios[valid] = [
            length_ratio(o, t) for o, t in zip(originals, transformed_texts, strict=True)
        ]
        similarities[valid] = semantic_similarity(
            originals, transformed_texts, cfg.filter.sbert_model, cfg.filter.device
        )

    result["length_ratio"] = ratios
    result["semantic_similarity"] = similarities
    checks = [
        ("transformation_rejected", ~valid),
        ("length_ratio_out_of_range", valid & ~ratios.between(min_ratio, max_ratio)),
        ("low_semantic_similarity", valid & (similarities < min_similarity)),
        ("detector_decision_changed", valid & decision_changed),
    ]
    result["filter_status"] = "accepted"
    result["filter_reason"] = "passed"
    for reason, failed in reversed(checks):
        result.loc[failed, "filter_status"] = "rejected"
        result.loc[failed, "filter_reason"] = reason

    out_path = interim_dir / "filtered_pairs.parquet"
    result.to_parquet(out_path, index=False)
    return out_path
