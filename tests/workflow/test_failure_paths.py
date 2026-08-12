"""Test failure paths and error classification."""
from __future__ import annotations

from pathlib import Path

import pytest

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import ProviderError, ArtifactError, PackagingError
from forgeml.core.states import FailureState
from forgeml.workflow.runner import WorkflowRunner
from tests.workflow.conftest import StubProvider


def test_provider_error_classified_as_transient(
    forge_project: Path, forge_cfg: ForgeConfig,
):
    """ProviderError during upload → FAILED_TRANSIENT."""
    provider = StubProvider(
        fail_at="upload_dataset",
        fail_error=ProviderError("network timeout"),
    )
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=provider)

    with pytest.raises(ProviderError):
        runner.execute(model="patchcore", dataset="mvtec", category="bottle")

    from forgeml.db.engine import get_engine
    from forgeml.db.models import Run
    from sqlmodel import Session, select

    engine = get_engine(forge_project)
    with Session(engine) as session:
        run = session.exec(select(Run)).first()
        assert run.status == FailureState.FAILED_TRANSIENT.value
        assert run.error_type == "ProviderError"


def test_generic_error_classified_as_execution_failure(
    forge_project: Path, forge_cfg: ForgeConfig,
):
    """Generic RuntimeError during monitoring → FAILED_EXECUTION."""
    provider = StubProvider(
        fail_at="monitor_kernel",
        fail_error=RuntimeError("unexpected crash"),
    )
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=provider)

    with pytest.raises(RuntimeError):
        runner.execute(model="patchcore", dataset="mvtec", category="bottle")

    from forgeml.db.engine import get_engine
    from forgeml.db.models import Run
    from sqlmodel import Session, select

    engine = get_engine(forge_project)
    with Session(engine) as session:
        run = session.exec(select(Run)).first()
        assert run.status == FailureState.FAILED_EXECUTION.value


def test_lock_prevents_concurrent_runs(
    forge_project: Path, forge_cfg: ForgeConfig,
):
    """A second run should fail with LockError if the lock is still held."""
    from forgeml.core.errors import LockError
    import json
    import time
    import os

    # Manually create a lock file as if a run is active
    lock_path = forge_project / ".forge.lock"
    lock_path.write_text(json.dumps({
        "run_id": "existing-run",
        "state": "RUNNING",
        "pid": os.getpid(),
        "timestamp": time.time()
    }))

    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=StubProvider())
    with pytest.raises(LockError, match="Another run is active"):
        runner.execute(model="patchcore", dataset="mvtec", category="bottle")


def test_lock_released_on_failure(
    forge_project: Path, forge_cfg: ForgeConfig,
):
    """Lock file should be removed even when the run fails."""
    provider = StubProvider(
        fail_at="upload_dataset",
        fail_error=ProviderError("boom"),
    )
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=provider)

    with pytest.raises(ProviderError):
        runner.execute(model="patchcore", dataset="mvtec", category="bottle")

    assert not (forge_project / ".forge.lock").exists()


def test_invalid_model_fails_at_packaging(
    forge_project: Path, forge_cfg: ForgeConfig,
):
    """Invalid model name should fail during packaging (capability validation)."""
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=StubProvider())

    with pytest.raises(ValueError, match="Unsupported model"):
        runner.execute(model="nonexistent", dataset="mvtec", category="bottle")


def test_invalid_category_fails_at_packaging(
    forge_project: Path, forge_cfg: ForgeConfig,
):
    """Invalid category should fail during packaging (capability validation)."""
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=StubProvider())

    with pytest.raises(ValueError, match="Unsupported category"):
        runner.execute(model="patchcore", dataset="mvtec", category="nonexistent")
