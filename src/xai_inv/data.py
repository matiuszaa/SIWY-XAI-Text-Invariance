import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

HC3_FILE = "all.jsonl"
LABELS = {"human": 0, "chatgpt": 1}
COLUMNS = ["text_id", "label", "source", "question", "text", "n_words"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_texts(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for row in df[["text_id", "label", "text"]].itertuples(index=False):
        h.update(f"{row.text_id}\t{row.label}\t{row.text}\n".encode())
    return h.hexdigest()


def download_hc3(dataset: str, revision: str, raw_dir: Path) -> Path:
    path = hf_hub_download(
        dataset, HC3_FILE, repo_type="dataset", revision=revision, local_dir=raw_dir / "hc3"
    )
    return Path(path)


def load_answers(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for q_idx, line in enumerate(f):
            item = json.loads(line)
            for kind, label in LABELS.items():
                for a_idx, text in enumerate(item[f"{kind}_answers"]):
                    text = text.strip()
                    rows.append(
                        {
                            "text_id": f"hc3-{q_idx}-{kind}-{a_idx}",
                            "label": label,
                            "source": item["source"],
                            "question": item["question"],
                            "text": text,
                            "n_words": len(text.split()),
                        }
                    )
    return pd.DataFrame(rows, columns=COLUMNS)


def sample_texts(
    answers: pd.DataFrame, n_texts: int, min_words: int, max_words: int, seed: int
) -> pd.DataFrame:
    pool = answers[answers.n_words.between(min_words, max_words)]
    pool = pool.drop_duplicates("text")
    n_per_label = {0: n_texts // 2, 1: n_texts - n_texts // 2}
    parts = []
    for label, n in n_per_label.items():
        candidates = pool[pool.label == label]
        if len(candidates) < n:
            raise ValueError(f"Only {len(candidates)} texts with label {label}, need {n}")
        parts.append(candidates.sample(n=n, random_state=seed))
    return pd.concat(parts).sort_values("text_id").reset_index(drop=True)


def prepare(cfg: DictConfig) -> Path:
    out_dir = Path(cfg.paths.interim)
    out_path = out_dir / "texts.parquet"
    manifest_path = out_dir / "texts.json"
    if out_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256_texts"] == sha256_texts(pd.read_parquet(out_path)):
            return out_path

    raw_path = download_hc3(cfg.data.dataset, cfg.data.revision, Path(cfg.paths.raw))
    texts = sample_texts(
        load_answers(raw_path),
        n_texts=cfg.data.n_texts,
        min_words=cfg.data.min_words,
        max_words=cfg.data.max_words,
        seed=cfg.seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    texts.to_parquet(out_path, index=False)
    manifest = {
        "dataset": cfg.data.dataset,
        "revision": cfg.data.revision,
        "file": HC3_FILE,
        "sha256_raw": sha256_file(raw_path),
        "seed": cfg.seed,
        "n_texts": len(texts),
        "n_human": int((texts.label == LABELS["human"]).sum()),
        "n_chatgpt": int((texts.label == LABELS["chatgpt"]).sum()),
        "sha256_texts": sha256_texts(texts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out_path
