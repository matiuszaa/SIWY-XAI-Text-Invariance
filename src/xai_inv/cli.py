from pathlib import Path
from typing import Annotated

import typer

from xai_inv import data
from xai_inv.config import load_config
from xai_inv.explanations import explain_texts
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
    _not_implemented("filter", config)


@app.command()
def explain(config: ConfigOption = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    out_path = explain_texts(cfg)
    typer.echo(f"[{cfg.name}] explanations written to {out_path}")


@app.command()
def metrics(config: ConfigOption = DEFAULT_CONFIG) -> None:
    _not_implemented("metrics", config)


@app.command()
def report(config: ConfigOption = DEFAULT_CONFIG) -> None:
    _not_implemented("report", config)


@app.command()
def run(config: ConfigOption = DEFAULT_CONFIG) -> None:
    prepare(config)
    transform(config)
    explain(config)
    for stage in ["filter", "metrics", "report"]:
        _not_implemented(stage, config)


if __name__ == "__main__":
    app()
