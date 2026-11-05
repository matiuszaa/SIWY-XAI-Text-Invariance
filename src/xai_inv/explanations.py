from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from xai_inv.detector import Detector
from xai_inv.explainers import explain_ig

EXPLANATION_COLUMNS = [
    "pair_id",
    "text_id",
    "transformation",
    "version",
    "text",
    "true_label",
    "predicted_label",
    "p_chatgpt",
    "words",
    "ig_scores",
    "nonword_score",
]


def explain_texts(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    pairs_path = interim_dir / "filtered_pairs.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Filtered pairs not found: {pairs_path}. Run 'filter' first.")

    pairs = pd.read_parquet(pairs_path)
    required_columns = {
        "pair_id",
        "text_id",
        "label",
        "transformation",
        "original_text",
        "transformed_text",
        "original_predicted_label",
        "transformed_predicted_label",
        "original_p_chatgpt",
        "transformed_p_chatgpt",
        "filter_status",
    }
    missing_columns = required_columns - set(pairs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Filtered pairs missing required columns: {missing}")

    accepted_pairs = pairs[pairs.filter_status.eq("accepted")]
    rows = []
    if not accepted_pairs.empty:
        detector = Detector(cfg.detector.model, device=cfg.detector.device)
        for pair in accepted_pairs.itertuples(index=False):
            for version in ("original", "transformed"):
                text = getattr(pair, f"{version}_text")
                predicted_label = int(getattr(pair, f"{version}_predicted_label"))
                attribution = explain_ig(
                    detector,
                    text,
                    target=predicted_label,
                    n_steps=cfg.explainers.ig.n_steps,
                )
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "text_id": pair.text_id,
                        "transformation": pair.transformation,
                        "version": version,
                        "text": text,
                        "true_label": int(pair.label),
                        "predicted_label": predicted_label,
                        "p_chatgpt": float(getattr(pair, f"{version}_p_chatgpt")),
                        "words": attribution.words,
                        "ig_scores": attribution.scores.tolist(),
                        "nonword_score": attribution.nonword_score,
                    }
                )

    out_path = interim_dir / "explanations.parquet"
    pd.DataFrame(rows, columns=EXPLANATION_COLUMNS).to_parquet(out_path, index=False)
    return out_path
