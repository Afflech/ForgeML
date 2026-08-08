from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from forgeml.config.forge_config import ForgeConfig
from forgeml.config.run_config import RunConfig, SourceSpec, TrainingSpec
from forgeml.core.errors import LockError, PackagingError
from forgeml.core.logging import get_logger
from forgeml.core.states import RunState, FailureState
from forgeml.project.packaging import create_bundle, get_git_commit, make_run_id

logger = get_logger(__name__)
console = Console()

LOCK_FILE = ".forge.lock"


class WorkflowRunner:
    def __init__(self, forge_cfg: ForgeConfig, cwd: Path) -> None:
        self.cfg = forge_cfg
        self.cwd = cwd
        self.lock_path = cwd / LOCK_FILE

    # ------------------------------------------------------------------
    # Lock
    # ------------------------------------------------------------------

    def _acquire_lock(self, run_id: str) -> None:
        if self.lock_path.exists():
            data = json.loads(self.lock_path.read_text())
            raise LockError(
                f"Another run is active: {data.get('run_id')} ({data.get('state')}). "
                "Wait for it to finish or delete .forge.lock if it is stale."
            )
        self.lock_path.write_text(json.dumps({"run_id": run_id, "state": RunState.CREATED}))

    def _update_lock(self, run_id: str, state: RunState | FailureState) -> None:
        data = json.loads(self.lock_path.read_text())
        data["state"] = state
        self.lock_path.write_text(json.dumps(data, indent=2))

    def _release_lock(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(
        self,
        model: str,
        dataset: str,
        category: str,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        run_id = make_run_id()
        console.print(f"\n[bold]ForgeML run[/bold] {run_id}")
        console.print(f"  model={model}  dataset={dataset}  category={category}  seed={seed}")

        self._acquire_lock(run_id)
        try:
            self._run(run_id, model, dataset, category, seed, dry_run)
        except Exception:
            self._update_lock(run_id, FailureState.FAILED_EXECUTION)
            raise
        finally:
            if not dry_run:
                self._release_lock()

    def _run(
        self,
        run_id: str,
        model: str,
        dataset: str,
        category: str,
        seed: int,
        dry_run: bool,
    ) -> None:
        # --- PACKAGING ---
        self._update_lock(run_id, RunState.PACKAGING)
        console.print(f"\n[cyan]▶ PACKAGING[/cyan]")

        git_commit = get_git_commit(self.cwd)
        console.print(f"  git commit: {git_commit[:12]}")

        staging_dir = self.cwd / "artifacts" / run_id / "staging"
        bundle_path, bundle_sha = create_bundle(self.cwd, staging_dir)
        console.print(f"  bundle: {bundle_path.name}  sha256: {bundle_sha[:16]}…")

        run_config = RunConfig(
            run_id=run_id,
            project=self.cfg.project.name,
            source=SourceSpec(git_commit=git_commit, bundle_sha256=bundle_sha),
            training=TrainingSpec(
                model=model,
                dataset=dataset,
                category=category,
                config_path=f"configs/{model}.yaml",
                seed=seed,
            ),
        )
        run_config.validate_capabilities()

        config_path = staging_dir / "run_config.json"
        config_path.write_text(run_config.model_dump_json(indent=2))
        console.print(f"  run_config.json written")

        if dry_run:
            console.print("\n[yellow]--dry-run: stopping after packaging.[/yellow]")
            console.print(f"  Staging dir: {staging_dir}")
            return

        # --- DATASET UPLOADING ---
        self._update_lock(run_id, RunState.DATASET_UPLOADING)
        console.print(f"\n[cyan]▶ DATASET_UPLOADING[/cyan]")

        from forgeml.providers.kaggle.dataset_manager import DatasetManager
        dm = DatasetManager(self.cfg)
        dm.upload(staging_dir, run_id)

        # --- KERNEL SUBMITTING ---
        self._update_lock(run_id, RunState.KERNEL_SUBMITTING)
        console.print(f"\n[cyan]▶ KERNEL_SUBMITTING[/cyan]")

        from forgeml.providers.kaggle.kernel_manager import KernelManager
        km = KernelManager(self.cfg)
        km.submit(run_id)

        # --- MONITOR ---
        self._update_lock(run_id, RunState.QUEUED)
        console.print(f"\n[cyan]▶ MONITORING[/cyan]")
        final_state = km.monitor(run_id, on_running=lambda: self._update_lock(run_id, RunState.RUNNING))

        if final_state != "complete":
            self._update_lock(run_id, FailureState.FAILED_EXECUTION)
            raise RuntimeError(f"Kernel ended with state: {final_state}")

        # --- COLLECTING ---
        self._update_lock(run_id, RunState.COLLECTING)
        console.print(f"\n[cyan]▶ COLLECTING[/cyan]")
        output_dir = self.cwd / "artifacts" / run_id / "output"
        km.download_output(run_id, output_dir)

        self._update_lock(run_id, RunState.COMPLETED)
        console.print(f"\n[green bold]✓ COMPLETED[/green bold]  artifacts → {output_dir}")
