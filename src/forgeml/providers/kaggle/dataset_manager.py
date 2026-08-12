from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import AuthError, ProviderError
from forgeml.core.logging import get_logger
from forgeml.core.utils import retry_transient
from forgeml.providers.kaggle.audit import ProviderAuditor

from forgeml.providers.kaggle.auth import get_kaggle_api

logger = get_logger(__name__)

DATASET_READY_POLL_S = 10
DATASET_READY_TIMEOUT_S = 300  # 5 minutes


class DatasetManager:
    def __init__(self, cfg: ForgeConfig, api=None, auditor: Optional[ProviderAuditor] = None) -> None:
        self.cfg = cfg
        self.api = api or get_kaggle_api()
        self.auditor = auditor

    def _dataset_id(self) -> str:
        username = self.api.get_config_value("username")
        return f"{username}/{self.cfg.kaggle.dataset}"

    @retry_transient(max_attempts=3, initial_wait_s=5)
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
        
        meta_dir = staging_dir / "kaggle_meta"
        meta_dir.mkdir(exist_ok=True)
        meta_path = meta_dir / "dataset-metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # Hardlink user artifacts into an isolated source/ folder
        source_dir = meta_dir / "source"
        source_dir.mkdir(exist_ok=True)
        import os
        for f in staging_dir.iterdir():
            if f.name == "kaggle_meta":
                continue
            dst = source_dir / f.name
            if not dst.exists():
                os.link(f, dst)

        try:
            req_data = {"folder": str(meta_dir), "version_notes": f"run_id={run_id}"}
            self.api.dataset_create_version(
                str(meta_dir),
                version_notes=f"run_id={run_id}",
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="tar"
            )
            if self.auditor:
                self.auditor.record("dataset_create_version", req_data, "SUCCESS")
            logger.info("Dataset version created: %s", dataset_id)
        except Exception as e:
            err = str(e).lower()
            # Kaggle 2.x returns 403 when dataset does not exist yet
            if "not found" in err or "404" in err or "does not exist" in err or "403" in err:
                if self.auditor:
                    self.auditor.record("dataset_create_version", req_data, f"ERROR: {e} (Falling back to create_new)")
                logger.info("Dataset not found — creating new: %s", dataset_id)
                try:
                    req_new = {"folder": str(meta_dir), "public": False}
                    self.api.dataset_create_new(
                        str(meta_dir),
                        public=False,
                        quiet=False,
                        convert_to_csv=False,
                    )
                    if self.auditor:
                        self.auditor.record("dataset_create_new", req_new, "SUCCESS")
                    logger.info("Dataset created: %s", dataset_id)
                except Exception as create_err:
                    if self.auditor:
                        self.auditor.record("dataset_create_new", req_new, f"ERROR: {create_err}")
                    raise ProviderError(f"Failed to create dataset: {create_err}") from create_err
            else:
                if self.auditor:
                    self.auditor.record("dataset_create_version", req_data, f"ERROR: {e}")
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
