"""Test resume from various interrupted states.

Each test follows the crash-restart pattern:
1. Create a run with a failing provider → run crashes at a specific stage
2. The first WorkflowRunner and StubProvider are discarded (simulating process death)
3. A NEW WorkflowRunner instance is created (simulating a new process)
4. The new runner reads persisted state from the DB via _read_persisted_state()
5. Resume continues from the correct stage based on DB state

This verifies that resume works across process boundaries, not just within
a single runner instance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.states import RunState, FailureState, RESUME_TARGETS
from forgeml.workflow.runner import WorkflowRunner
from tests.workflow.conftest import StubProvider


def _crash_run_at(
    forge_project: Path,
    forge_cfg: ForgeConfig,
    interrupt_at: str,
) -> str:
    """Run a workflow that crashes at the specified provider stage.

    Returns the run_id. After this function returns, the first runner
    and provider are gone — only the DB and staging artifacts remain.
    """
    from forgeml.core.errors import ProviderError

    failing_provider = StubProvider(
        fail_at=interrupt_at,
        fail_error=ProviderError(f"Simulated crash at {interrupt_at}"),
    )
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=failing_provider)

    run_id = None
    try:
        runner.execute(model="patchcore", dataset="mvtec", category="bottle")
    except Exception:
        artifacts_dir = forge_project / "artifacts"
        run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        run_id = run_dirs[0].name

    assert run_id is not None

    # Explicitly delete runner and provider to simulate process death.
    # After this, only the DB file and staging directory persist.
    del runner
    del failing_provider

    return run_id


def _verify_db_state(forge_project: Path, run_id: str) -> tuple[str, str]:
    """Read status and last_stage from the DB — a fresh engine, fresh session.

    Returns (status, last_stage).
    """
    from forgeml.db.engine import get_engine
    from forgeml.db.models import Run
    from sqlmodel import Session

    engine = get_engine(forge_project)
    with Session(engine) as session:
        db_run = session.get(Run, run_id)
        assert db_run is not None, f"Run {run_id} not found in DB"
        return db_run.status, db_run.last_stage


def test_resume_after_upload_failure(forge_project: Path, forge_cfg: ForgeConfig):
    """Crash during upload → resume skips packaging, re-uploads."""
    run_id = _crash_run_at(forge_project, forge_cfg, "upload_dataset")

    # Verify DB state before resume (cross-process verification)
    status, last_stage = _verify_db_state(forge_project, run_id)
    assert status == FailureState.FAILED_TRANSIENT.value
    assert last_stage == RunState.DATASET_UPLOADING.value  # last RunState reached

    # --- Simulate new process: new runner, new provider, fresh from disk ---
    resume_provider = StubProvider()
    new_runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=resume_provider)
    new_runner.execute(
        model="patchcore", dataset="mvtec", category="bottle",
        resume_run_id=run_id,
    )

    # Upload was incomplete → resume re-uploads, but does NOT re-package
    assert "upload_dataset" in resume_provider.calls
    assert "submit_kernel" in resume_provider.calls
    assert "monitor_kernel" in resume_provider.calls
    assert "download_output" in resume_provider.calls

    # Verify final state via a fresh DB read
    status, last_stage = _verify_db_state(forge_project, run_id)
    assert status == RunState.COMPLETED.value


def test_resume_after_submit_failure(forge_project: Path, forge_cfg: ForgeConfig):
    """Crash during kernel submit → resume skips packaging + upload, re-submits."""
    run_id = _crash_run_at(forge_project, forge_cfg, "submit_kernel")

    # DB should show failure, but last_stage = KERNEL_SUBMITTING
    status, last_stage = _verify_db_state(forge_project, run_id)
    assert status == FailureState.FAILED_TRANSIENT.value
    assert last_stage == RunState.KERNEL_SUBMITTING.value

    # --- New process ---
    resume_provider = StubProvider()
    new_runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=resume_provider)
    new_runner.execute(
        model="patchcore", dataset="mvtec", category="bottle",
        resume_run_id=run_id,
    )

    # Should skip packaging AND upload, go directly to submit
    assert "upload_dataset" not in resume_provider.calls
    assert "submit_kernel" in resume_provider.calls
    assert "monitor_kernel" in resume_provider.calls

    status, _ = _verify_db_state(forge_project, run_id)
    assert status == RunState.COMPLETED.value


def test_resume_after_monitor_failure(forge_project: Path, forge_cfg: ForgeConfig):
    """Crash during monitoring → resume re-enters monitoring."""
    run_id = _crash_run_at(forge_project, forge_cfg, "monitor_kernel")

    status, last_stage = _verify_db_state(forge_project, run_id)
    assert status == FailureState.FAILED_TRANSIENT.value
    assert last_stage == RunState.QUEUED.value  # monitor is called after QUEUED

    # --- New process ---
    resume_provider = StubProvider()
    new_runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=resume_provider)
    new_runner.execute(
        model="patchcore", dataset="mvtec", category="bottle",
        resume_run_id=run_id,
    )

    # Should skip everything up to monitoring
    assert "upload_dataset" not in resume_provider.calls
    assert "submit_kernel" not in resume_provider.calls
    assert "monitor_kernel" in resume_provider.calls
    assert "download_output" in resume_provider.calls

    status, _ = _verify_db_state(forge_project, run_id)
    assert status == RunState.COMPLETED.value


def test_resume_after_download_failure(forge_project: Path, forge_cfg: ForgeConfig):
    """Crash during download → resume re-downloads."""
    run_id = _crash_run_at(forge_project, forge_cfg, "download_output")

    status, last_stage = _verify_db_state(forge_project, run_id)
    assert status == FailureState.FAILED_TRANSIENT.value
    assert last_stage == RunState.COLLECTING.value

    # --- New process ---
    resume_provider = StubProvider()
    new_runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=resume_provider)
    new_runner.execute(
        model="patchcore", dataset="mvtec", category="bottle",
        resume_run_id=run_id,
    )

    assert "upload_dataset" not in resume_provider.calls
    assert "submit_kernel" not in resume_provider.calls
    assert "monitor_kernel" not in resume_provider.calls
    assert "download_output" in resume_provider.calls

    status, _ = _verify_db_state(forge_project, run_id)
    assert status == RunState.COMPLETED.value


def test_resume_preserves_original_config(forge_project: Path, forge_cfg: ForgeConfig):
    """Resumed run must use config from original run, not CLI args."""
    run_id = _crash_run_at(forge_project, forge_cfg, "upload_dataset")

    # --- New process with DIFFERENT CLI args ---
    resume_provider = StubProvider()
    new_runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=resume_provider)
    new_runner.execute(
        model="padim",       # different from original "patchcore"
        dataset="mvtec",
        category="cable",    # different from original "bottle"
        resume_run_id=run_id,
    )

    # Config should still be from the original run (read from staging/run_config.json)
    staging_dir = forge_project / "artifacts" / run_id / "staging"
    cfg = json.loads((staging_dir / "run_config.json").read_text())
    assert cfg["training"]["model"] == "patchcore"  # original, not "padim"
    assert cfg["training"]["category"] == "bottle"   # original, not "cable"


def test_resume_nonexistent_run_id_fails(forge_project: Path, forge_cfg: ForgeConfig):
    """Resuming with a run_id that doesn't exist should fail clearly."""
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=StubProvider())
    with pytest.raises(RuntimeError, match="Cannot resume"):
        runner.execute(
            model="patchcore", dataset="mvtec", category="bottle",
            resume_run_id="nonexistent-run-id",
        )


def test_resume_reads_state_from_db_not_lock(forge_project: Path, forge_cfg: ForgeConfig):
    """Verify resume derives state from the DB, not from in-memory state.

    After a crash, the lock file is cleaned up (finally block), so the DB
    is the only surviving source of truth.
    """
    run_id = _crash_run_at(forge_project, forge_cfg, "submit_kernel")

    # Lock file should NOT exist (released in finally block)
    assert not (forge_project / ".forge.lock").exists()

    # But DB should have the state
    status, last_stage = _verify_db_state(forge_project, run_id)
    assert last_stage is not None  # DB has last_stage

    # Create a completely new runner and verify it reads from DB
    new_runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=StubProvider())
    persisted_status, persisted_last_stage = new_runner._read_persisted_state(run_id)
    assert persisted_status == FailureState.FAILED_TRANSIENT.value
    assert persisted_last_stage == RunState.KERNEL_SUBMITTING.value
