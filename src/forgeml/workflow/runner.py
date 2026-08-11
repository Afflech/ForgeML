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

    def _check_dependencies(self) -> None:
        """Verify required local CLI tools before starting the run."""
        import subprocess
        from forgeml.core.errors import ConfigError

        # Check Git
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            # Check if inside git repo
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.cwd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise ConfigError("Not a valid git repository or no commits found. Git is required to track source code.") from e
        except FileNotFoundError as e:
            raise ConfigError("Git CLI not found. Please install git.") from e

        # Check Kaggle API authentication
        try:
            from kaggle import api
            api.authenticate()
        except Exception as e:
            raise ConfigError(f"Kaggle authentication failed: {e}\nEnsure ~/.kaggle/kaggle.json exists or KAGGLE_USERNAME/KAGGLE_KEY are set.") from e

    def execute(
        self,
        model: str,
        dataset: str,
        category: str,
        seed: int = 42,
        dry_run: bool = False,
        resume_run_id: Optional[str] = None,
    ) -> None:
        run_id = resume_run_id if resume_run_id else make_run_id()
        console.print(f"\n[bold]ForgeML run[/bold] {run_id}{' (resumed)' if resume_run_id else ''}")

        # If resuming, load config from the existing staging dir
        if resume_run_id:
            import json
            staging_dir = self.cwd / "artifacts" / run_id / "staging"
            config_path = staging_dir / "run_config.json"
            if not config_path.exists():
                raise RuntimeError(f"Cannot resume: {config_path} not found.")
            cfg_data = json.loads(config_path.read_text())
            t = cfg_data.get("training", {})
            model = t.get("model", model)
            dataset = t.get("dataset", dataset)
            category = t.get("category", category)
            seed = t.get("seed", seed)

        console.print(f"  model={model}  dataset={dataset}  category={category}  seed={seed}")

        # Dependency diagnostics
        self._check_dependencies()

        self._acquire_lock(run_id)

        # Init DB tracking
        from forgeml.db.engine import get_engine
        from forgeml.db.models import Run
        from sqlmodel import Session
        from datetime import datetime, timezone

        engine = get_engine(self.cwd)
        with Session(engine) as session:
            db_run = session.get(Run, run_id)
            if not db_run:
                db_run = Run(
                    id=run_id,
                    project=self.cfg.project.name,
                    provider=self.cfg.provider.name,
                    status=RunState.CREATED,
                    model=model,
                    dataset=dataset,
                    category=category,
                    git_commit="unknown",
                    bundle_sha256="unknown",
                    started_at=datetime.now(timezone.utc)
                )
                session.add(db_run)
                session.commit()

        try:
            self._run(run_id, model, dataset, category, seed, dry_run, engine, resume=bool(resume_run_id))
        except Exception as e:
            from forgeml.core.errors import ArtifactError, ConfigError, ProviderError, PackagingError, QuotaError, AuthError

            if isinstance(e, (ConfigError, PackagingError)):
                fail_state = FailureState.FAILED_CONFIG
            elif isinstance(e, ArtifactError):
                fail_state = FailureState.FAILED_ARTIFACT
            elif isinstance(e, QuotaError):
                fail_state = FailureState.BLOCKED_QUOTA
            elif isinstance(e, AuthError):
                fail_state = FailureState.BLOCKED_AUTH
            elif isinstance(e, ProviderError):
                fail_state = FailureState.FAILED_TRANSIENT
            else:
                fail_state = FailureState.FAILED_EXECUTION

            self._update_lock(run_id, fail_state)
            with Session(engine) as session:
                db_run = session.get(Run, run_id)
                if db_run:
                    db_run.status = fail_state
                    db_run.error_type = type(e).__name__
                    db_run.finished_at = datetime.now(timezone.utc)
                    session.add(db_run)
                    session.commit()
            raise
        finally:
            self._release_lock()

    def _run(
        self,
        run_id: str,
        model: str,
        dataset: str,
        category: str,
        seed: int,
        dry_run: bool,
        engine,
        resume: bool = False,
    ) -> None:
        staging_dir = self.cwd / "artifacts" / run_id / "staging"
        from sqlmodel import Session
        from forgeml.db.models import Run
        from datetime import datetime, timezone

        if not resume:
            # --- PACKAGING ---
            self._update_lock(run_id, RunState.PACKAGING)
            console.print(f"\n[cyan]▶ PACKAGING[/cyan]")

            git_commit = get_git_commit(self.cwd)
            console.print(f"  git commit: {git_commit[:12]}")

            bundle_path, bundle_sha = create_bundle(self.cwd, staging_dir)
            console.print(f"  bundle: {bundle_path.name}  sha256: {bundle_sha[:16]}…")

            with Session(engine) as session:
                db_run = session.get(Run, run_id)
                if db_run:
                    db_run.git_commit = git_commit
                    db_run.bundle_sha256 = bundle_sha
                    session.add(db_run)
                    session.commit()

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
        else:
            console.print(f"\n[yellow]Resuming run (skipping packaging)[/yellow]")

        # --- DATASET UPLOADING ---
        self._update_lock(run_id, RunState.DATASET_UPLOADING)
        console.print(f"\n[cyan]▶ DATASET_UPLOADING[/cyan]")

        from forgeml.providers.kaggle.audit import ProviderAuditor
        run_dir = self.cwd / "artifacts" / run_id
        auditor = ProviderAuditor(run_dir)

        from forgeml.providers.kaggle.dataset_manager import DatasetManager
        dm = DatasetManager(self.cfg, auditor)
        dm.upload(staging_dir, run_id)

        # --- KERNEL SUBMITTING ---
        self._update_lock(run_id, RunState.KERNEL_SUBMITTING)
        console.print(f"\n[cyan]▶ KERNEL_SUBMITTING[/cyan]")

        from forgeml.providers.kaggle.kernel_manager import KernelManager
        km = KernelManager(self.cfg, auditor)
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

        # Update final run record
        import json
        metrics_file = output_dir / "metrics.json"
        manifest_file = output_dir / "run_manifest.json"

        metrics_json = None
        artifact_path = None
        if metrics_file.exists():
            metrics_json = metrics_file.read_text()
        if manifest_file.exists():
            try:
                manifest_data = json.loads(manifest_file.read_text())
                outputs = manifest_data.get("outputs", {})
                artifact_path = outputs.get("checkpoint")
                expected_sha = outputs.get("checkpoint_sha256")

                # Integrity check
                if artifact_path and expected_sha:
                    from forgeml.project.packaging import sha256_file
                    from forgeml.core.errors import ArtifactError

                    local_checkpoint = output_dir / artifact_path
                    if not local_checkpoint.exists():
                        raise ArtifactError(f"Checkpoint not found at expected path: {local_checkpoint}")

                    actual_sha = sha256_file(local_checkpoint)
                    if actual_sha != expected_sha:
                        raise ArtifactError(
                            f"Artifact integrity check failed!\n"
                            f"Expected SHA256: {expected_sha}\n"
                            f"Actual SHA256:   {actual_sha}"
                        )
                    console.print(f"  [green]Integrity check passed[/green] ({expected_sha[:16]}...)")
            except Exception as e:
                # If it's our own error, re-raise it
                from forgeml.core.errors import ArtifactError
                if isinstance(e, ArtifactError):
                    raise
                # Otherwise pass
                pass

        with Session(engine) as session:
            db_run = session.get(Run, run_id)
            if db_run:
                db_run.status = RunState.COMPLETED
                db_run.metrics_json = metrics_json
                db_run.artifact_path = artifact_path
                db_run.finished_at = datetime.now(timezone.utc)
                session.add(db_run)
                session.commit()

        self._update_lock(run_id, RunState.COMPLETED)
        console.print(f"\n[green bold]✓ COMPLETED[/green bold]  artifacts → {output_dir}")
