from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import AuthError, ProviderError, QuotaError
from forgeml.core.logging import get_logger
from forgeml.core.utils import retry_transient
from forgeml.providers.kaggle.audit import ProviderAuditor

from forgeml.providers.kaggle.auth import get_kaggle_api

logger = get_logger(__name__)

POLL_INTERVAL_S = 30
TIMEOUT_S = 3 * 3600  # 3 hours max

# Kaggle kernel terminal states
TERMINAL_STATES = {"complete", "error", "failed", "cancelled"}

# kernel-metadata.json filename expected by kaggle.kernels_push()
KERNEL_METADATA_FILE = "kernel-metadata.json"
KERNEL_ENTRYPOINT_FILE = "kernel_entrypoint.py"


VALID_ACCELERATORS = {"None", "NvidiaTeslaT4", "NvidiaTeslaP100", "TPUv3"}

class KernelManager:
    def __init__(
        self,
        cfg: ForgeConfig,
        api=None,
        auditor: Optional[ProviderAuditor] = None,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.api = api or get_kaggle_api()
        self.auditor = auditor
        self._templates_dir = Path(__file__).resolve().parents[4] / "templates"
        self._run_dir = run_dir

    def _kernel_id(self) -> str:
        username = self.api.get_config_value("username")
        return f"{username}/{self.cfg.kaggle.kernel}"

    def _prepare_kernel_files(self, run_id: str, dataset_version: int) -> Path:
        """Create the per-run immutable Kaggle push directory."""
        if not dataset_version:
            raise ProviderError("Cannot submit kernel: missing strictly pinned dataset_version.")

        username = self.api.get_config_value("username")
        source_dataset_name = self.cfg.kaggle.source_dataset
        data_mount_slug = self.cfg.kaggle.dataset_slug

        pinned_dataset_id = f"{username}/{source_dataset_name}/{dataset_version}"

        acc = self.cfg.kaggle.accelerator
        enable_gpu = acc not in ("None", "TPUv3")
        enable_tpu = acc == "TPUv3"

        meta = {
            "id": self._kernel_id(),
            "title": self.cfg.kaggle.kernel,
            "code_file": KERNEL_ENTRYPOINT_FILE,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": enable_gpu,
            "enable_tpu": enable_tpu,
            "enable_internet": self.cfg.kaggle.internet,
            "dataset_sources": [src for src in (pinned_dataset_id, data_mount_slug) if src],
            "competition_sources": [],
            "kernel_sources": [],
        }

        run_dir = self._run_dir or Path("artifacts") / run_id
        remote_dir = run_dir / "remote"
        remote_dir.mkdir(parents=True, exist_ok=True)

        source_entrypoint = self._templates_dir / KERNEL_ENTRYPOINT_FILE
        if not source_entrypoint.exists():
            raise ProviderError(f"Kernel template not found: {source_entrypoint}")

        shutil.copy2(source_entrypoint, remote_dir / KERNEL_ENTRYPOINT_FILE)
        meta_path = remote_dir / KERNEL_METADATA_FILE
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Wrote %s", meta_path)
        return remote_dir

    @retry_transient(max_attempts=3, initial_wait_s=5)
    def submit(self, run_id: str, dataset_version: int) -> str:
        """Push the fixed Kernel to Kaggle to trigger a new run. Returns version number."""
        acc = self.cfg.kaggle.accelerator
        if acc not in VALID_ACCELERATORS:
            from forgeml.core.errors import ConfigError
            raise ConfigError(f"Unsupported Kaggle accelerator '{acc}'. Valid options: {', '.join(VALID_ACCELERATORS)}")

        kernel_dir = self._prepare_kernel_files(run_id, dataset_version)
        try:
            req_data = {"folder": str(kernel_dir)}
            kwargs = {}
            if acc != "None":
                kwargs["acc"] = acc
            result = self.api.kernels_push(str(kernel_dir), **kwargs)
            if self.auditor:
                # `result` is often a wrapper object, so we convert it to str
                self.auditor.record("kernels_push", req_data, str(result))
        except Exception as e:
            err = str(e).lower()
            if self.auditor:
                self.auditor.record("kernels_push", {"folder": str(kernel_dir)}, f"ERROR: {e}")
            if "quota" in err or "limit" in err:
                raise QuotaError(f"Kaggle GPU quota exceeded: {e}") from e
            raise ProviderError(f"Kernel submit failed: {e}") from e

        if result is None:
            raise ProviderError("Kernel push returned no result — check Kaggle dashboard.")

        if getattr(result, "error", None):
            raise ProviderError(f"Kernel push error: {result.error}")

        invalid_sources = getattr(result, "invalidDatasetSources", None)
        if invalid_sources:
            raise ProviderError(
                f"Kernel rejected dataset sources: {invalid_sources}\n"
                "The dataset may not be fully indexed yet. Wait a minute and retry."
            )

        version = str(getattr(result, "version_number", getattr(result, "versionNumber", "")))
        logger.info("Kernel submitted: %s (version %s)", self._kernel_id(), version)
        return version

    def monitor(
        self,
        run_id: str,
        remote_id: Optional[str] = None,
        on_running: Optional[Callable] = None,
        poll_interval: int = POLL_INTERVAL_S,
        timeout: int = TIMEOUT_S,
    ) -> str:
        """Poll Kernel status until terminal state. Returns final status string."""
        kernel_id = self._kernel_id()
        if remote_id:
            logger.info("Monitoring kernel %s (targeting version %s)", kernel_id, remote_id)
        
        deadline = time.time() + timeout
        last_state = ""
        seen_running = False

        while time.time() < deadline:
            try:
                # Bypass self.api.kernels_status which drops the version label
                from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
                owner_slug, kernel_slug, _ = self.api.parse_kernel_string(kernel_id)
                with self.api.build_kaggle_client() as client:
                    req = ApiGetKernelSessionStatusRequest()
                    req.user_name = owner_slug
                    req.kernel_slug = kernel_slug
                    response = client.kernels.kernels_api_client.get_kernel_session_status(req)

                raw = str(getattr(response, "status", "unknown")).lower()
                # Kaggle 2.x returns "kernelworkerstatus.running" etc — strip prefix
                state = raw.split(".")[-1]
            except Exception as e:
                logger.warning("Status poll error: %s — retrying in %ds", e, poll_interval)
                time.sleep(poll_interval)
                continue

            if state != last_state:
                logger.info("Kernel %s → %s", kernel_id, state)
                last_state = state

            if state == "running" and not seen_running:
                seen_running = True
                if on_running:
                    on_running()

            if state in TERMINAL_STATES:
                return state

            time.sleep(poll_interval)

        raise ProviderError(f"Kernel {kernel_id} timed out after {timeout}s")

    @retry_transient(max_attempts=3, initial_wait_s=5)
    def download_output(self, run_id: str, output_dir: Path) -> None:
        """Download Kernel output files to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        kernel_id = self._kernel_id()
        try:
            self.api.kernels_output_cli(
                kernel=kernel_id,
                path=str(output_dir),
                force=True,
                quiet=True,
            )
            logger.info("Output downloaded to %s", output_dir)
        except Exception as e:
            raise ProviderError(f"Failed to download Kernel output: {e}") from e
