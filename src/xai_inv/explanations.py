import hashlib
import json
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from xai_inv.data import text_key
from xai_inv.detector import Detector
from xai_inv.explainers import WordAttribution, explain_ig, explain_lime, explain_shap

SAVE_EVERY = 20
EXPLANATION_COLUMNS = [
    "text_key",
    "text",
    "method",
    "run",
    "seed",
    "target",
    "p_chatgpt",
    "words",
    "scores",
    "nonword_score",
    "params_hash",
]


def run_explainer(
    detector: Detector, method: str, params: DictConfig, text: str, target: int, seed: int | None
) -> WordAttribution:
    if method == "ig":
        return explain_ig(detector, text, target, n_steps=int(params.n_steps))
    if method == "lime":
        return explain_lime(detector, text, target, n_samples=int(params.n_samples), seed=seed)
    if method == "shap":
        return explain_shap(detector, text, target, max_evals=int(params.max_evals), seed=seed)
    raise ValueError(f"Unknown explainer: {method}")


def params_hash(model: str, method: str, params: DictConfig, target: int, seed: int | None) -> str:
    payload = {
        "model": model,
        "method": method,
        "params": OmegaConf.to_container(params),
        "target": target,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def collect_texts(pairs: pd.DataFrame) -> dict[str, dict]:
    texts = {}
    for pair in pairs[pairs.filter_status.eq("accepted")].itertuples(index=False):
        for version in ("original", "transformed"):
            text = getattr(pair, f"{version}_text")
            entry = texts.setdefault(
                text_key(text),
                {
                    "text": text,
                    "target": int(getattr(pair, f"{version}_predicted_label")),
                    "p_chatgpt": float(getattr(pair, f"{version}_p_chatgpt")),
                    "is_original": False,
                },
            )
            entry["is_original"] |= version == "original"
    return texts


def load_cached(path: Path) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    cached = pd.read_parquet(path)
    if "params_hash" not in cached.columns:
        return {}
    return {(row["text_key"], row["method"], row["run"]): row for row in cached.to_dict("records")}


def write_explanations(rows: list[dict], path: Path) -> None:
    tmp_path = path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=EXPLANATION_COLUMNS).to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def explain_texts(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    pairs_path = interim_dir / "filtered_pairs.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Filtered pairs not found: {pairs_path}. Run 'filter' first.")

    pairs = pd.read_parquet(pairs_path)
    required_columns = {
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

    out_path = interim_dir / "explanations.parquet"
    done = load_cached(out_path)
    texts = collect_texts(pairs)
    detector = None
    rows = []
    n_new = 0
    for method, params in cfg.explainers.items():
        for key, entry in texts.items():
            runs = [("main", params.get("seed"))]
            if entry["is_original"] and "rerun_seed" in params:
                runs.append(("rerun", params.rerun_seed))
            for run, seed in runs:
                seed = None if seed is None else int(seed)
                expected_hash = params_hash(
                    cfg.detector.model, method, params, entry["target"], seed
                )
                row = done.get((key, method, run))
                if row is None or row["params_hash"] != expected_hash:
                    if detector is None:
                        detector = Detector(cfg.detector.model, device=cfg.detector.device)
                    attribution = run_explainer(
                        detector, method, params, entry["text"], entry["target"], seed
                    )
                    row = {
                        "text_key": key,
                        "text": entry["text"],
                        "method": method,
                        "run": run,
                        "seed": seed,
                        "target": entry["target"],
                        "p_chatgpt": entry["p_chatgpt"],
                        "words": attribution.words,
                        "scores": attribution.scores.tolist(),
                        "nonword_score": attribution.nonword_score,
                        "params_hash": expected_hash,
                    }
                    done[(key, method, run)] = row
                    n_new += 1
                    if n_new % SAVE_EVERY == 0:
                        write_explanations(list(done.values()), out_path)
                rows.append(row)

    write_explanations(rows, out_path)
    return out_path
