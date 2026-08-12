from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from rich.console import Console

from forgeml.config.forge_config import ForgeConfig
from forgeml.config.run_config import RunConfig, SourceSpec, TrainingSpec
from forgeml.core.errors import LockError, PackagingError
from forgeml.core.logging import get_logger
from forgeml.core.states import (
    RunState,
    FailureState,
    RunStateMachine,
    RESUME_TARGETS,
)
from forgeml.project.packaging import create_bundle, get_git_commit, make_run_id

logger = get_logger(__name__)
console = Console()

LOCK_FILE = ".forge.lock"


# ---------------------------------------------------------------------------
# Provider protocol — real Kaggle implementation lives in providers/kaggle,
# stub implementations can be injected for testing.
# ---------------------------------------------------------------------------

@runtime_checkable
class RemoteProvider(Protocol):
    """Interface for remote execution providers (Kaggle, stubs, etc.)."""

    def upload_dataset(self, staging_dir: Path, run_id: str) -> None: ...

    def submit_kernel(self, run_id: str) -> str: ...

    def monitor_kernel(
        self,
        run_id: str,
        remote_id: Optional[str] = None,
        on_running: Optional[callable] = None,
    ) -> str: ...

    def download_output(self, run_id: str, output_dir: Path) -> None: ...


class KaggleProvider:
    """Wraps existing DatasetManager + KernelManager behind RemoteProvider."""

    def __init__(self, cfg: ForgeConfig, run_dir: Path) -> None:
        from forgeml.providers.kaggle.audit import ProviderAuditor
        from forgeml.providers.kaggle.auth import get_kaggle_api
        from forgeml.providers.kaggle.dataset_manager import DatasetManager
        from forgeml.providers.kaggle.kernel_manager import KernelManager

        self._cfg = cfg
        self._auditor = ProviderAuditor(run_dir)
        self.api = get_kaggle_api()
        self._dm = DatasetManager(self._cfg, api=self.api, auditor=self._auditor)
        self._km = KernelManager(self._cfg, api=self.api, auditor=self._auditor)

    def upload_dataset(self, staging_dir: Path, run_id: str) -> None:
        self._dm.upload(staging_dir, run_id)

    def submit_kernel(self, run_id: str) -> str:
        return self._km.submit(run_id)

    def monitor_kernel(
        self,
        run_id: str,
        remote_id: Optional[str] = None,
        on_running: Optional[callable] = None,
    ) -> str:
        return self._km.monitor(run_id, remote_id=remote_id, on_running=on_running)

    def download_output(self, run_id: str, output_dir: Path) -> None:
        self._km.download_output(run_id, output_dir)


class WorkflowRunner:
    def __init__(
        self,
        forge_cfg: ForgeConfig,
        cwd: Path,
        provider: Optional[RemoteProvider] = None,
    ) -> None:
        self.cfg = forge_cfg
        self.cwd = cwd
        self.lock_path = cwd / LOCK_FILE
        self._provider = provider  # None → will create KaggleProvider lazily

    # ------------------------------------------------------------------
    # Lock & state persistence
    # ------------------------------------------------------------------

    def _acquire_lock(self, run_id: str) -> None:
        import os
        import time
        if self.lock_path.exists():
            data = json.loads(self.lock_path.read_text())
            pid = data.get("pid")
            timestamp = data.get("timestamp", 0)
            
            stale = False
            if pid:
                try:
                    os.kill(pid, 0)
                except OSError:
                    stale = True
            
            # Also consider stale if untouched for 12 hours
            if not stale and time.time() - timestamp > 12 * 3600:
                stale = True

            if not stale:
                raise LockError(
                    f"Another run is active: {data.get('run_id')} ({data.get('state')}). "
                    "Wait for it to finish or delete .forge.lock if it is stale."
                )
            else:
                logger.warning("Found stale lock file, overwriting.")

        self.lock_path.write_text(json.dumps({
            "run_id": run_id, 
            "state": RunState.CREATED.value,
            "pid": os.getpid(),
            "timestamp": time.time()
        }, indent=2))

    def _persist_state(self, run_id: str, state: RunState | FailureState) -> None:
        """Write current state to lock file. Tracks last_stage for RunStates."""
        import os
        import time
        if self.lock_path.exists():
            data = json.loads(self.lock_path.read_text())
        else:
            data = {"run_id": run_id}
        data["state"] = state.value if hasattr(state, "value") else str(state)
        data["pid"] = os.getpid()
        data["timestamp"] = time.time()
        # Track the last RunState separately so resume can find it after a failure
        if isinstance(state, RunState):
            data["last_stage"] = state.value
        self.lock_path.write_text(json.dumps(data, indent=2))

    def _release_lock(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()

    def _read_persisted_state(self, run_id: str) -> tuple[Optional[str], Optional[str]]:
        """Read (status, last_stage) for a run from DB, falling back to lock file.

        last_stage is the last RunState reached before any failure — this is
        what resume logic uses to decide where to restart.
        """
        try:
            from forgeml.db.engine import get_engine
            from forgeml.db.models import Run
            from sqlmodel import Session

            engine = get_engine(self.cwd)
            with Session(engine) as session:
                db_run = session.get(Run, run_id)
                if db_run:
                    return db_run.status, db_run.last_stage
        except Exception:
            pass

        # Fallback: read from lock file
        lock_file = self.cwd / LOCK_FILE
        if lock_file.exists():
            data = json.loads(lock_file.read_text())
            if data.get("run_id") == run_id:
                return data.get("state"), data.get("last_stage")
        return None, None

    # ------------------------------------------------------------------
    # Dependency check (preflight)
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

    # ------------------------------------------------------------------
    # Execute (public entry point)
    # ------------------------------------------------------------------

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

        # Determine resume state
        resume_from: Optional[RunState] = None
        if resume_run_id:
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

            # Determine where to resume from based on persisted state
            persisted_status, persisted_last_stage = self._read_persisted_state(run_id)
            if persisted_last_stage:
                # Use last_stage — this is the last RunState before any failure
                try:
                    last_state = RunState(persisted_last_stage)
                    resume_from = RESUME_TARGETS.get(last_state, RunState.PACKAGING)
                except ValueError:
                    resume_from = RunState.PACKAGING
            elif persisted_status:
                # Fallback: try to parse status directly (for runs without last_stage)
                try:
                    last_state = RunState(persisted_status)
                    resume_from = RESUME_TARGETS.get(last_state, RunState.PACKAGING)
                except ValueError:
                    resume_from = RunState.PACKAGING

        console.print(f"  model={model}  dataset={dataset}  category={category}  seed={seed}")
        if resume_from:
            console.print(f"  [yellow]Resuming from: {resume_from.value}[/yellow]")

        # Dependency diagnostics (skip during resume if provider is injected)
        if self._provider is None:
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
            self._run(
                run_id, model, dataset, category, seed,
                dry_run, engine, resume_from=resume_from,
            )
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

            self._persist_state(run_id, fail_state)
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

    # ------------------------------------------------------------------
    # Internal workflow stages
    # ------------------------------------------------------------------

    def _run(
        self,
        run_id: str,
        model: str,
        dataset: str,
        category: str,
        seed: int,
        dry_run: bool,
        engine,
        resume_from: Optional[RunState] = None,
    ) -> None:
        staging_dir = self.cwd / "artifacts" / run_id / "staging"
        from sqlmodel import Session
        from forgeml.db.models import Run
        from datetime import datetime, timezone

        # Initialize the state machine
        sm = RunStateMachine()

        # Determine which stages to execute
        all_stages = [
            RunState.PACKAGING,
            RunState.DATASET_UPLOADING,
            RunState.DATASET_READY,
            RunState.KERNEL_SUBMITTING,
            RunState.QUEUED,
            RunState.RUNNING,
            RunState.COLLECTING,
            RunState.COMPLETED,
        ]

        if resume_from:
            # Fast-forward the state machine to the resume point
            try:
                resume_idx = all_stages.index(resume_from)
            except ValueError:
                resume_idx = 0

            # Walk the state machine forward to just before resume point
            for state in all_stages[:resume_idx]:
                sm.transition(state)
            console.print(f"\n[yellow]Resuming run from {resume_from.value} (skipping completed stages)[/yellow]")
        else:
            resume_idx = 0  # start from beginning

        def _advance(target: RunState) -> None:
            """Transition state machine and persist state + last_stage."""
            sm.transition(target)
            self._persist_state(run_id, target)
            with Session(engine) as session:
                db_run = session.get(Run, run_id)
                if db_run:
                    db_run.status = target
                    db_run.last_stage = target.value
                    session.add(db_run)
                    session.commit()

        # Get or create provider
        provider = self._provider
        if provider is None:
            run_dir = self.cwd / "artifacts" / run_id
            provider = KaggleProvider(self.cfg, run_dir)

        # --- PACKAGING ---
        if resume_idx <= all_stages.index(RunState.PACKAGING):
            _advance(RunState.PACKAGING)
            console.print(f"\n[cyan]▶ PACKAGING[/cyan]")

            git_commit = get_git_commit(self.cwd)
            console.print(f"  git commit: {git_commit[:12]}")

            bundle_path, bundle_sha = create_bundle(
                self.cwd, staging_dir, includes=self.cfg.project.bundle_includes
            )
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

        # --- DATASET UPLOADING ---
        if resume_idx <= all_stages.index(RunState.DATASET_UPLOADING):
            _advance(RunState.DATASET_UPLOADING)
            console.print(f"\n[cyan]▶ DATASET_UPLOADING[/cyan]")
            provider.upload_dataset(staging_dir, run_id)

        # --- DATASET READY ---
        if resume_idx <= all_stages.index(RunState.DATASET_READY):
            _advance(RunState.DATASET_READY)
            console.print(f"  [green]Dataset ready[/green]")

        # --- KERNEL SUBMITTING ---
        if resume_idx <= all_stages.index(RunState.KERNEL_SUBMITTING):
            _advance(RunState.KERNEL_SUBMITTING)
            console.print(f"\n[cyan]▶ KERNEL_SUBMITTING[/cyan]")
            remote_id = provider.submit_kernel(run_id)
            
            with Session(engine) as session:
                db_run = session.get(Run, run_id)
                if db_run:
                    db_run.kaggle_run_id = remote_id
                    session.add(db_run)
                    session.commit()

        # Fetch remote_id from DB for resume cases
        remote_id = None
        with Session(engine) as session:
            db_run = session.get(Run, run_id)
            if db_run:
                remote_id = db_run.kaggle_run_id

        # --- MONITORING (QUEUED → RUNNING) ---
        if resume_idx <= all_stages.index(RunState.QUEUED):
            _advance(RunState.QUEUED)
            console.print(f"\n[cyan]▶ MONITORING[/cyan]")

            def _on_running():
                _advance(RunState.RUNNING)

            final_state = provider.monitor_kernel(run_id, remote_id=remote_id, on_running=_on_running)

            if final_state != "complete":
                sm.fail(FailureState.FAILED_EXECUTION)
                self._persist_state(run_id, FailureState.FAILED_EXECUTION)
                raise RuntimeError(f"Kernel ended with state: {final_state}")
        elif resume_idx == all_stages.index(RunState.RUNNING):
            # Resuming from RUNNING — re-enter monitoring
            _advance(RunState.RUNNING)
            console.print(f"\n[cyan]▶ MONITORING (resumed)[/cyan]")

            final_state = provider.monitor_kernel(run_id)

            if final_state != "complete":
                sm.fail(FailureState.FAILED_EXECUTION)
                self._persist_state(run_id, FailureState.FAILED_EXECUTION)
                raise RuntimeError(f"Kernel ended with state: {final_state}")

        # --- COLLECTING ---
        if resume_idx <= all_stages.index(RunState.COLLECTING):
            _advance(RunState.COLLECTING)
            console.print(f"\n[cyan]▶ COLLECTING[/cyan]")
            output_dir = self.cwd / "artifacts" / run_id / "output"
            provider.download_output(run_id, output_dir)

            # Verify artifacts
            metrics_json = None
            artifact_path = None
            metrics_file = output_dir / "metrics.json"
            manifest_file = output_dir / "run_manifest.json"

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
                    db_run.metrics_json = metrics_json
                    db_run.artifact_path = artifact_path
                    session.add(db_run)
                    session.commit()

        # --- COMPLETED ---
        _advance(RunState.COMPLETED)
        with Session(engine) as session:
            db_run = session.get(Run, run_id)
            if db_run:
                db_run.status = RunState.COMPLETED
                db_run.finished_at = datetime.now(timezone.utc)
                session.add(db_run)
                session.commit()
        console.print(f"\n[green bold]✓ COMPLETED[/green bold]  artifacts → {self.cwd / 'artifacts' / run_id / 'output'}")
