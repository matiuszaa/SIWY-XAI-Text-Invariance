from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from xai_inv.detector import Detector

PREDICTION_COLUMNS = [
    "original_predicted_label",
    "transformed_predicted_label",
    "original_p_chatgpt",
    "transformed_p_chatgpt",
    "filter_status",
    "filter_reason",
]


def filter_by_detector(cfg: DictConfig) -> Path:
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

    result = pairs.copy()
    for column in PREDICTION_COLUMNS:
        result[column] = None

    valid = result.transformation_status.eq("ok") & result.transformed_text.notna()
    if valid.any():
        detector = Detector(cfg.detector.model, device=cfg.detector.device)
        original = detector.predict(result.loc[valid, "original_text"].tolist())
        transformed = detector.predict(result.loc[valid, "transformed_text"].tolist())

        result.loc[valid, "original_predicted_label"] = original.labels.astype(int)
        result.loc[valid, "transformed_predicted_label"] = transformed.labels.astype(int)
        result.loc[valid, "original_p_chatgpt"] = original.probs.astype(float)
        result.loc[valid, "transformed_p_chatgpt"] = transformed.probs.astype(float)

        same_decision = original.labels == transformed.labels
        valid_indices = result.index[valid]
        accepted_indices = valid_indices[same_decision]
        rejected_indices = valid_indices[~same_decision]
        result.loc[accepted_indices, "filter_status"] = "accepted"
        result.loc[accepted_indices, "filter_reason"] = "same_detector_decision"
        result.loc[rejected_indices, "filter_status"] = "rejected"
        result.loc[rejected_indices, "filter_reason"] = "detector_decision_changed"

    invalid = ~valid
    result.loc[invalid, "filter_status"] = "rejected"
    result.loc[invalid, "filter_reason"] = "transformation_rejected"

    out_path = interim_dir / "filtered_pairs.parquet"
    result.to_parquet(out_path, index=False)
    return out_path
