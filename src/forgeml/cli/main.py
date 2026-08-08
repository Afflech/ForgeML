from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from forgeml.config.forge_config import ForgeConfig
from forgeml.config.run_config import RunConfig, SourceSpec, TrainingSpec
from forgeml.core.errors import ConfigError, LockError, PackagingError
from forgeml.core.logging import get_logger
from forgeml.project.packaging import create_bundle, get_git_commit, make_run_id

logger = get_logger(__name__)
console = Console()
app = typer.Typer(name="forge", help="ForgeML — automated local-to-Kaggle ML training pipeline")

FORGE_YAML = "forge.yaml"
LOCK_FILE = ".forge.lock"


def _load_config() -> ForgeConfig:
    cwd = Path.cwd()
    cfg_path = cwd / FORGE_YAML
    if not cfg_path.exists():
        console.print(f"[red]forge.yaml not found in {cwd}. Run 'forge init' first.[/red]")
        raise typer.Exit(1)
    return ForgeConfig.from_yaml(cfg_path)


@app.command()
def init() -> None:
    """Initialize a ForgeML project in the current directory."""
    cfg_path = Path.cwd() / FORGE_YAML
    if cfg_path.exists():
        console.print("[yellow]forge.yaml already exists — skipping.[/yellow]")
        return
    # Copy example config
    import shutil
    template = Path(__file__).resolve().parents[3] / "forge.yaml"
    if template.exists():
        shutil.copy(template, cfg_path)
    else:
        cfg_path.write_text("project:\n  name: myproject\n\nprovider:\n  name: kaggle\n\nkaggle:\n  kernel: myproject-training\n  dataset: myproject-source\n")
    console.print(f"[green]Created forge.yaml[/green] — edit kaggle.kernel and kaggle.dataset.")


@app.command()
def validate() -> None:
    """Validate forge.yaml and check project structure."""
    try:
        cfg = _load_config()
    except SystemExit:
        return

    console.print(f"[green]✓[/green] forge.yaml valid")
    console.print(f"  project : {cfg.project.name}")
    console.print(f"  provider: {cfg.provider.name}")
    console.print(f"  kernel  : {cfg.kaggle.kernel}")
    console.print(f"  dataset : {cfg.kaggle.dataset}")

    # Check IndustrialAD source structure
    cwd = Path.cwd()
    for item in ["src", "configs", "requirements.txt"]:
        path = cwd / item
        status = "[green]✓[/green]" if path.exists() else "[red]✗[/red]"
        console.print(f"  {status} {item}")


@app.command()
def status() -> None:
    """Show the current run state (if any)."""
    lock = Path.cwd() / LOCK_FILE
    if not lock.exists():
        console.print("No active run.")
        return
    data = json.loads(lock.read_text())
    table = Table(show_header=False)
    for k, v in data.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def run(
    model: str = typer.Option("patchcore", help="Model name (patchcore|padim|fastflow|efficientad)"),
    dataset: str = typer.Option("mvtec", help="Dataset name"),
    category: str = typer.Option("bottle", help="MVTec category"),
    seed: int = typer.Option(42, help="Random seed"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Package only, do not upload"),
) -> None:
    """Package source, upload to Kaggle, and submit training run."""
    from forgeml.workflow.runner import WorkflowRunner

    cfg_path = Path.cwd() / FORGE_YAML
    if not cfg_path.exists():
        console.print("[red]forge.yaml not found. Run 'forge init' first.[/red]")
        raise typer.Exit(1)

    forge_cfg = ForgeConfig.from_yaml(cfg_path)

    try:
        runner = WorkflowRunner(forge_cfg, cwd=Path.cwd())
        runner.execute(
            model=model,
            dataset=dataset,
            category=category,
            seed=seed,
            dry_run=dry_run,
        )
    except LockError as e:
        console.print(f"[red]Lock error:[/red] {e}")
        raise typer.Exit(1)
    except (ConfigError, PackagingError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
