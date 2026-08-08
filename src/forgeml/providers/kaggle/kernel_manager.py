from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import AuthError, ProviderError, QuotaError
from forgeml.core.logging import get_logger

logger = get_logger(__name__)

POLL_INTERVAL_S = 30
TIMEOUT_S = 3 * 3600  # 3 hours max


def _kaggle_api():
    """Return authenticated KaggleApi instance.

    Authentication is handled by the Kaggle CLI (v2.2.4+).
    Run 'kaggle auth login' to store credentials in ~/.kaggle/access_token.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        return api
    except Exception as e:
        raise AuthError(
            f"Kaggle authentication failed: {e}\n"
            "Run 'kaggle auth login' to authenticate via the Kaggle CLI."
        ) from e


class KernelManager:
    def __init__(self, cfg: ForgeConfig) -> None:
        self.cfg = cfg
        self.api = _kaggle_api()

    def submit(self, run_id: str) -> None:
        """Push a new version of the fixed Kernel to trigger training."""
        username = self.api.get_config_value("username")
        kernel_slug = self.cfg.kaggle.kernel
        dataset_slug = self.cfg.kaggle.dataset
        mvtec_slug = self.cfg.kaggle.mvtec_dataset

        kernel_meta = {
            "id": f"{username}/{kernel_slug}",
            "title": kernel_slug,
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

        try:
            self.api.kernels_push_cli(
                folder=str(Path(__file__).resolve().parents[5] / "templates"),
                metadata_path=None,
                metadata=kernel_meta,
                quiet=False,
            )
            logger.info("Kernel submitted: %s/%s", username, kernel_slug)
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "limit" in err:
                raise QuotaError(f"Kaggle GPU quota exceeded: {e}") from e
            raise ProviderError(f"Kernel submit failed: {e}") from e

    def monitor(
        self,
        run_id: str,
        on_running: Optional[Callable] = None,
        poll_interval: int = POLL_INTERVAL_S,
        timeout: int = TIMEOUT_S,
    ) -> str:
        """Poll Kernel status until terminal state. Returns final status string."""
        username = self.api.get_config_value("username")
        kernel_slug = self.cfg.kaggle.kernel
        kernel_id = f"{username}/{kernel_slug}"

        deadline = time.time() + timeout
        last_state = ""
        seen_running = False

        while time.time() < deadline:
            try:
                status = self.api.kernel_status(username, kernel_slug)
                state = status.get("status", "unknown")
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

            if state in ("complete", "error", "failed", "cancelled"):
                return state

            time.sleep(poll_interval)

        raise ProviderError(f"Kernel {kernel_id} timed out after {timeout}s")

    def download_output(self, run_id: str, output_dir: Path) -> None:
        """Download Kernel output files to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        username = self.api.get_config_value("username")
        kernel_slug = self.cfg.kaggle.kernel

        try:
            self.api.kernels_output_cli(
                kernel=f"{username}/{kernel_slug}",
                path=str(output_dir),
                force=True,
                quiet=False,
            )
            logger.info("Output downloaded to %s", output_dir)
        except Exception as e:
            raise ProviderError(f"Failed to download Kernel output: {e}") from e
