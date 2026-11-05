from pathlib import Path
from typing import Annotated

import typer

from xai_inv import data
from xai_inv.config import load_config
from xai_inv.explanations import explain_texts
from xai_inv.metrics import compute_metrics
from xai_inv.quality import filter_by_detector
from xai_inv.report import create_report
from xai_inv.transforms import transform_synonyms

app = typer.Typer(help="XAI explanation invariance under meaning-preserving text transformations.")

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Path to the experiment YAML config.")
]
DEFAULT_CONFIG = Path("configs/small.yaml")

STAGES = ["prepare", "transform", "filter", "explain", "metrics", "report"]


def _not_implemented(stage: str, config: Path) -> None:
    cfg = load_config(config)
    typer.echo(f"[{cfg.name}] stage '{stage}' is not implemented yet.")


@app.command()
def prepare(config: ConfigOption = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    out_path = data.prepare(cfg)
    typer.echo(f"[{cfg.name}] texts written to {out_path}")


@app.command()
def transform(config: ConfigOption = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    out_path = transform_synonyms(cfg)
    typer.echo(f"[{cfg.name}] transformed pairs written to {out_path}")


@app.command(name="filter")
def filter_pairs(config: ConfigOption = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    out_path = filter_by_detector(cfg)
    typer.echo(f"[{cfg.name}] filtered pairs written to {out_path}")


@app.command()
def explain(config: ConfigOption = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    out_path = explain_texts(cfg)
    typer.echo(f"[{cfg.name}] explanations written to {out_path}")


@app.command()
def metrics(config: ConfigOption = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    out_path = compute_metrics(cfg)
    typer.echo(f"[{cfg.name}] metrics written to {out_path}")


@app.command()
def report(config: ConfigOption = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    out_path = create_report(cfg)
    typer.echo(f"[{cfg.name}] report written to {out_path}")


@app.command()
def run(config: ConfigOption = DEFAULT_CONFIG) -> None:
    prepare(config)
    transform(config)
    filter_pairs(config)
    explain(config)
    metrics(config)
    report(config)


if __name__ == "__main__":
    app()
