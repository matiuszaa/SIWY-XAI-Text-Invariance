from pathlib import Path

import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from xai_inv.explanations import timing_summary

METRIC_NAMES = ["alignment_coverage", "spearman", "top_k_jaccard"]


def tracking_enabled(cfg: DictConfig) -> bool:
    return "wandb" in cfg and cfg.wandb.mode != "disabled"


def summarize_results(
    pairs: pd.DataFrame, explanations: pd.DataFrame, metrics: pd.DataFrame
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    scalars = {}
    filtering = (
        pairs.groupby(["transformation", "filter_reason"]).size().rename("pairs").reset_index()
    )
    for name, group in pairs.groupby("transformation"):
        scalars[f"filter/{name}/accepted_rate"] = float(group.filter_status.eq("accepted").mean())

    timing = timing_summary(explanations)
    for row in timing.itertuples(index=False):
        scalars[f"time/{row.method}/mean_seconds"] = float(row.mean)

    invariance = metrics.groupby(["transformation", "method"])[METRIC_NAMES].mean().reset_index()
    for row in invariance.to_dict("records"):
        for metric in METRIC_NAMES:
            key = f"invariance/{row['transformation']}/{row['method']}/{metric}"
            scalars[key] = float(row[metric])

    return scalars, {"filtering": filtering, "timing": timing, "invariance": invariance}


def log_results(cfg: DictConfig) -> bool:
    if not tracking_enabled(cfg):
        return False
    interim_dir = Path(cfg.paths.interim)
    scalars, tables = summarize_results(
        pd.read_parquet(interim_dir / "filtered_pairs.parquet"),
        pd.read_parquet(interim_dir / "explanations.parquet"),
        pd.read_parquet(interim_dir / "metrics.parquet"),
    )
    run = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=cfg.name,
        job_type="report",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run.summary.update(scalars)
    run.log({name: wandb.Table(dataframe=table) for name, table in tables.items()})
    run.finish()
    return True
