from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import AuthError, ProviderError, QuotaError
from forgeml.core.logging import get_logger
from forgeml.core.utils import retry_transient
from forgeml.providers.kaggle.audit import ProviderAuditor

logger = get_logger(__name__)

POLL_INTERVAL_S = 30
TIMEOUT_S = 3 * 3600  # 3 hours max

# Kaggle kernel terminal states
TERMINAL_STATES = {"complete", "error", "failed", "cancelled"}

# kernel-metadata.json filename expected by kaggle.kernels_push()
KERNEL_METADATA_FILE = "kernel-metadata.json"


def _api():
    """Return the authenticated Kaggle API singleton (kaggle 2.2.4+)."""
    try:
        from kaggle import api
        api.authenticate()
        return api
    except Exception as e:
        raise AuthError(
            f"Kaggle authentication failed: {e}\n"
            "Run 'kaggle auth login' to authenticate via the Kaggle CLI."
        ) from e


class KernelManager:
    def __init__(self, cfg: ForgeConfig, auditor: Optional[ProviderAuditor] = None) -> None:
        self.cfg = cfg
        self.api = _api()
        self.auditor = auditor
        self._templates_dir = Path(__file__).resolve().parents[4] / "templates"

    def _kernel_id(self) -> str:
        username = self.api.get_config_value("username")
        return f"{username}/{self.cfg.kaggle.kernel}"

    def _write_kernel_metadata(self) -> None:
        """Write kernel-metadata.json into the templates folder before pushing."""
        username = self.api.get_config_value("username")
        dataset_slug = self.cfg.kaggle.dataset
        mvtec_slug = self.cfg.kaggle.mvtec_dataset

        meta = {
            "id": self._kernel_id(),
            "title": self.cfg.kaggle.kernel,
            "code_file": "kernel_entrypoint.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": self.cfg.kaggle.accelerator != "None",
            "enable_internet": self.cfg.kaggle.internet,
            "dataset_sources": [
                f"{username}/{dataset_slug}",
                mvtec_slug,
            ],
            "competition_sources": [],
            "kernel_sources": [],
        }
        meta_path = self._templates_dir / KERNEL_METADATA_FILE
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Wrote %s", meta_path)

    @retry_transient(max_attempts=3, initial_wait_s=5)
    def submit(self, run_id: str) -> None:
        """Push the fixed Kernel to Kaggle to trigger a new run."""
        self._write_kernel_metadata()
        try:
            req_data = {"folder": str(self._templates_dir)}
            result = self.api.kernels_push(str(self._templates_dir))
            if self.auditor:
                # `result` is often a wrapper object, so we convert it to str
                self.auditor.record("kernels_push", req_data, str(result))
        except Exception as e:
            err = str(e).lower()
            if self.auditor:
                self.auditor.record("kernels_push", {"folder": str(self._templates_dir)}, f"ERROR: {e}")
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

        logger.info("Kernel submitted: %s", self._kernel_id())

    def monitor(
        self,
        run_id: str,
        on_running: Optional[Callable] = None,
        poll_interval: int = POLL_INTERVAL_S,
        timeout: int = TIMEOUT_S,
    ) -> str:
        """Poll Kernel status until terminal state. Returns final status string."""
        kernel_id = self._kernel_id()
        deadline = time.time() + timeout
        last_state = ""
        seen_running = False

        while time.time() < deadline:
            try:
                response = self.api.kernels_status(kernel_id)
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
                quiet=False,
            )
            logger.info("Output downloaded to %s", output_dir)
        except Exception as e:
            raise ProviderError(f"Failed to download Kernel output: {e}") from e
