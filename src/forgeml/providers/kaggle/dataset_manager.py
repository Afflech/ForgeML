from __future__ import annotations

import json
import time
from pathlib import Path

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import AuthError, ProviderError
from forgeml.core.logging import get_logger

logger = get_logger(__name__)

DATASET_READY_POLL_S = 10
DATASET_READY_TIMEOUT_S = 300  # 5 minutes


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


class DatasetManager:
    def __init__(self, cfg: ForgeConfig) -> None:
        self.cfg = cfg
        self.api = _api()

    def _dataset_id(self) -> str:
        username = self.api.get_config_value("username")
        return f"{username}/{self.cfg.kaggle.dataset}"

    def upload(self, staging_dir: Path, run_id: str) -> None:
        """
        Upload staging_dir contents as a new version of the private source Dataset.
        Creates the Dataset on first use, then waits until Kaggle marks it ready.
        """
        dataset_id = self._dataset_id()

        meta = {
            "title": self.cfg.kaggle.dataset,
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        }
        meta_path = staging_dir / "dataset-metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        try:
            self.api.dataset_create_version(
                str(staging_dir),
                version_notes=f"run_id={run_id}",
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
            )
            logger.info("Dataset version created: %s", dataset_id)
        except Exception as e:
            err = str(e).lower()
            # Kaggle 2.x returns 403 when dataset does not exist yet
            if "not found" in err or "404" in err or "does not exist" in err or "403" in err:
                logger.info("Dataset not found — creating new: %s", dataset_id)
                try:
                    self.api.dataset_create_new(
                        str(staging_dir),
                        public=False,
                        quiet=False,
                        convert_to_csv=False,
                    )
                    logger.info("Dataset created: %s", dataset_id)
                except Exception as create_err:
                    raise ProviderError(f"Failed to create dataset: {create_err}") from create_err
            else:
                raise ProviderError(f"Failed to upload dataset: {e}") from e

        self._wait_until_ready(dataset_id)

    def _wait_until_ready(self, dataset_id: str) -> None:
        """Poll until dataset files are accessible via the API (i.e. Kaggle has indexed it)."""
        logger.info("Waiting for dataset %s to be indexed…", dataset_id)
        deadline = time.time() + DATASET_READY_TIMEOUT_S
        while time.time() < deadline:
            try:
                result = self.api.dataset_list_files(dataset_id)
                files = getattr(result, "files", None) or getattr(result, "datasetFiles", None)
                if files is not None:
                    logger.info("Dataset indexed — %d file(s) visible.", len(files))
                    return
                logger.info("Dataset not yet indexed — retrying in %ds…", DATASET_READY_POLL_S)
            except Exception as e:
                logger.info("Dataset not yet indexed (%s) — retrying in %ds…", e, DATASET_READY_POLL_S)
            time.sleep(DATASET_READY_POLL_S)

        logger.warning("Dataset indexing timed out after %ds — proceeding anyway", DATASET_READY_TIMEOUT_S)
