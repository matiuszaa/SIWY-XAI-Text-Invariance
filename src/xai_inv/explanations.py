from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from xai_inv.detector import Detector
from xai_inv.explainers import explain_ig


def explain_texts(cfg: DictConfig) -> Path:
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

    detector = Detector(cfg.detector.model, device=cfg.detector.device)
    predictions = detector.predict(texts.text.tolist())

    rows = []
    for row, predicted_label, p_chatgpt in zip(
        texts.itertuples(index=False), predictions.labels, predictions.probs, strict=True
    ):
        attribution = explain_ig(
            detector,
            row.text,
            target=int(predicted_label),
            n_steps=cfg.explainers.ig.n_steps,
        )
        rows.append(
            {
                "text_id": row.text_id,
                "true_label": row.label,
                "predicted_label": int(predicted_label),
                "p_chatgpt": float(p_chatgpt),
                "words": attribution.words,
                "ig_scores": attribution.scores.tolist(),
                "nonword_score": attribution.nonword_score,
            }
        )

    out_path = interim_dir / "explanations.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path
