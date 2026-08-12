"""Shared fixtures and mock provider for workflow tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from forgeml.config.forge_config import ForgeConfig
from forgeml.workflow.runner import RemoteProvider


class StubProvider:
    """A mock remote provider that records calls and succeeds by default.

    Set ``fail_at`` to the method name to simulate a failure at that stage.
    """

    def __init__(self, *, fail_at: Optional[str] = None, fail_error: Optional[Exception] = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at
        self.fail_error = fail_error or RuntimeError("simulated failure")

    def _maybe_fail(self, method: str) -> None:
        if self.fail_at == method:
            raise self.fail_error

    def upload_dataset(self, staging_dir: Path, run_id: str) -> None:
        self.calls.append("upload_dataset")
        self._maybe_fail("upload_dataset")

    def submit_kernel(self, run_id: str) -> str:
        self.calls.append("submit_kernel")
        self._maybe_fail("submit_kernel")
        return "v1"

    def monitor_kernel(
        self,
        run_id: str,
        remote_id: Optional[str] = None,
        on_running: Optional[callable] = None,
    ) -> str:
        self.calls.append("monitor_kernel")
        self._maybe_fail("monitor_kernel")
        if on_running:
            on_running()
        return "complete"

    def download_output(self, run_id: str, output_dir: Path) -> None:
        self.calls.append("download_output")
        self._maybe_fail("download_output")
        output_dir.mkdir(parents=True, exist_ok=True)


# --- Fixtures ---

MINIMAL_FORGE_YAML = """\
project:
  name: test-project
kaggle:
  kernel: test-kernel
  dataset: test-dataset
"""


@pytest.fixture
def forge_project(tmp_path: Path) -> Path:
    """Create a minimal forge project directory with required structure."""
    # forge.yaml
    (tmp_path / "forge.yaml").write_text(MINIMAL_FORGE_YAML)

    # Git repo (needed for packaging)
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True)

    # Required source structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "patchcore.yaml").write_text("model: patchcore")
    (tmp_path / "requirements.txt").write_text("torch>=2.0")

    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)

    return tmp_path


@pytest.fixture
def forge_cfg(forge_project: Path) -> ForgeConfig:
    """Load ForgeConfig from the test project."""
    return ForgeConfig.from_yaml(forge_project / "forge.yaml")


@pytest.fixture
def stub_provider() -> StubProvider:
    """Create a fresh stub provider."""
    return StubProvider()
