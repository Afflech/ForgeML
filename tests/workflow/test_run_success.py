"""Test full successful workflow run against a mocked provider."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.states import RunState
from forgeml.workflow.runner import WorkflowRunner
from tests.workflow.conftest import StubProvider


def test_successful_run(forge_project: Path, forge_cfg: ForgeConfig, stub_provider: StubProvider):
    """Full happy-path: CREATED → PACKAGING → … → COMPLETED."""
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=stub_provider)

    runner.execute(model="patchcore", dataset="mvtec", category="bottle")

    # All provider methods should have been called in order
    assert stub_provider.calls == [
        "upload_dataset",
        "submit_kernel",
        "monitor_kernel",
        "download_output",
    ]

    # Lock file should be cleaned up after success
    assert not (forge_project / ".forge.lock").exists()

    # DB record should be COMPLETED
    from forgeml.db.engine import get_engine
    from forgeml.db.models import Run
    from sqlmodel import Session, select

    engine = get_engine(forge_project)
    with Session(engine) as session:
        runs = session.exec(select(Run)).all()
        assert len(runs) == 1
        assert runs[0].status == RunState.COMPLETED
        assert runs[0].model == "patchcore"
        assert runs[0].dataset == "mvtec"
        assert runs[0].category == "bottle"
        assert runs[0].finished_at is not None


def test_successful_run_records_git_and_bundle(
    forge_project: Path, forge_cfg: ForgeConfig, stub_provider: StubProvider
):
    """Verify that git commit and bundle hash are recorded after packaging."""
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=stub_provider)
    runner.execute(model="patchcore", dataset="mvtec", category="bottle")

    from forgeml.db.engine import get_engine
    from forgeml.db.models import Run
    from sqlmodel import Session, select

    engine = get_engine(forge_project)
    with Session(engine) as session:
        run = session.exec(select(Run)).first()
        assert run is not None
        assert run.git_commit != "unknown"
        assert run.bundle_sha256 != "unknown"
        assert len(run.git_commit) == 40  # full SHA


def test_staging_artifacts_created(
    forge_project: Path, forge_cfg: ForgeConfig, stub_provider: StubProvider
):
    """Verify packaging creates staging dir with bundle and run_config.json."""
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=stub_provider)
    runner.execute(model="patchcore", dataset="mvtec", category="bottle")

    # Find the run directory (there should be exactly one)
    artifacts_dir = forge_project / "artifacts"
    run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1

    staging = run_dirs[0] / "staging"
    assert staging.exists()
    assert (staging / "bundle.tar.gz").exists()
    assert (staging / "run_config.json").exists()

    # Verify run_config.json content
    cfg = json.loads((staging / "run_config.json").read_text())
    assert cfg["training"]["model"] == "patchcore"
    assert cfg["training"]["category"] == "bottle"


def test_dry_run_stops_after_packaging(
    forge_project: Path, forge_cfg: ForgeConfig, stub_provider: StubProvider
):
    """dry_run=True should package but not call any provider methods."""
    runner = WorkflowRunner(forge_cfg, cwd=forge_project, provider=stub_provider)
    runner.execute(model="patchcore", dataset="mvtec", category="bottle", dry_run=True)

    # No provider calls
    assert stub_provider.calls == []

    # But staging dir should exist with run_config.json
    artifacts_dir = forge_project / "artifacts"
    run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "staging" / "run_config.json").exists()
