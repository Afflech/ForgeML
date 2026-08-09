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

    # Check project structure and capabilities
    from forgeml.project.inspector import ProjectInspector
    inspector = ProjectInspector(Path.cwd())
    structure = inspector.check_structure()

    console.print("\n[bold]Project Structure:[/bold]")
    all_ok = True
    for item, exists in structure.items():
        if exists:
            console.print(f"  [green]✓[/green] {item}")
        else:
            console.print(f"  [red]✗[/red] {item}")
            all_ok = False

    console.print("\n[bold]Git Status:[/bold]")
    if not inspector.is_git_repo():
        console.print("  [red]✗ Not a git repository[/red]")
        all_ok = False
    elif inspector.is_git_dirty():
        console.print("  [yellow]! Working tree has uncommitted changes[/yellow]")
    else:
        console.print("  [green]✓ Clean working tree[/green]")

    if not all_ok:
        console.print("\n[yellow]Warning: Some validation checks failed.[/yellow]")


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
def history(
    limit: int = typer.Option(10, help="Number of recent runs to show"),
) -> None:
    """List past runs from the local SQLite experiment tracker."""
    from sqlmodel import Session, select
    from forgeml.db.engine import get_engine
    from forgeml.db.models import Run

    engine = get_engine(Path.cwd())
    with Session(engine) as session:
        statement = select(Run).order_by(Run.started_at.desc()).limit(limit)
        runs = session.exec(statement).all()

    if not runs:
        console.print("No runs found in local history.")
        return

    from rich import box
    table = Table(title="ForgeML Run History", expand=True, show_lines=True, box=box.HEAVY_EDGE)
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Config", style="magenta")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Info", style="blue")
    table.add_column("Metrics", style="green")

    for r in runs:
        # Dynamic config representation
        config_parts = []
        if r.model: config_parts.append(f"m={r.model}")
        if r.dataset: config_parts.append(f"d={r.dataset}")
        if r.category: config_parts.append(f"c={r.category}")
        config_text = "\n".join(config_parts)

        status_color = "green" if r.status == "completed" else "red" if "failed" in r.status.lower() else "yellow"
        status_text = f"[{status_color}]{r.status}[/{status_color}]"

        duration = ""
        if r.started_at and r.finished_at:
            dur_s = (r.finished_at - r.started_at).total_seconds()
            duration = f"{dur_s:.1f}s"

        info_text = ""
        metrics_text = ""
        if r.metrics_json:
            try:
                m = json.loads(r.metrics_json)
                info_parts = []
                metrics_parts = []
                for k, v in m.items():
                    if k in ["checkpoint", "checkpoint_sha256"]:
                        continue
                    if isinstance(v, float):
                        metrics_parts.append(f"{k}:{v:.3f}")
                    elif isinstance(v, (int, str)):
                        info_parts.append(f"{k}:{v}")
                info_text = "\n".join(info_parts)
                metrics_text = "\n".join(metrics_parts)
            except Exception:
                pass

        table.add_row(r.id, config_text, status_text, duration, info_text, metrics_text)

    console.print(table)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Natural language request (e.g. 'run padim on screw')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Package only, do not upload"),
) -> None:
    """Use AI to parse a natural language request and execute the run."""
    try:
        from forgeml.llm.planner import LLMPlanner, PlannerError
    except ImportError:
        console.print("[red]Missing dependencies.[/red] Ensure 'openai' is installed.")
        raise typer.Exit(1)

    console.print(f"[cyan]Analyzing request:[/cyan] '{prompt}'...")
    try:
        planner = LLMPlanner()
        plan = planner.plan(prompt)
    except PlannerError as e:
        console.print(f"[red]Planner error:[/red] {e}")
        raise typer.Exit(1)

    # Show preview
    console.print("\n[bold]Planned Configuration:[/bold]")
    table = Table(show_header=False)
    table.add_row("Model", f"[green]{plan.model}[/green]")
    table.add_row("Dataset", plan.dataset)
    table.add_row("Category", f"[green]{plan.category}[/green]")
    table.add_row("Seed", str(plan.seed))
    if plan.reasoning:
        table.add_row("Reasoning", f"[dim]{plan.reasoning}[/dim]")
    console.print(table)

    confirm = typer.confirm("\nExecute this run?")
    if not confirm:
        console.print("[yellow]Aborted by user.[/yellow]")
        return

    # Pass to the normal run execution logic
    from forgeml.workflow.runner import WorkflowRunner
    cfg_path = Path.cwd() / FORGE_YAML
    if not cfg_path.exists():
        console.print("[red]forge.yaml not found. Run 'forge init' first.[/red]")
        raise typer.Exit(1)

    forge_cfg = ForgeConfig.from_yaml(cfg_path)
    try:
        runner = WorkflowRunner(forge_cfg, cwd=Path.cwd())
        runner.execute(
            model=plan.model,
            dataset=plan.dataset,
            category=plan.category,
            seed=plan.seed,
            dry_run=dry_run,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def run(
    model: str = typer.Option("patchcore", help="Model name (patchcore|padim|fastflow|efficientad)"),
    dataset: str = typer.Option("mvtec", help="Dataset name"),
    category: str = typer.Option("bottle", help="MVTec category"),
    seed: int = typer.Option(42, help="Random seed"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Package only, do not upload"),
    run_id: Optional[str] = typer.Option(None, help="Resume an existing run_id (skips packaging)"),
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
            resume_run_id=run_id,
        )
    except LockError as e:
        console.print(f"[red]Lock error:[/red] {e}")
        raise typer.Exit(1)
    except (ConfigError, PackagingError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
