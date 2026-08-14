from __future__ import annotations

import json
import logging
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
        return f"{username}/{self.cfg.kaggle.source_dataset}"

    @retry_transient(max_attempts=3, initial_wait_s=5)
    def upload(self, staging_dir: Path, run_id: str) -> int:
        """
        Upload staging_dir contents as a new version of the private source Dataset.
        Creates the Dataset on first use, then waits until Kaggle marks it ready.
        Returns the confirmed version number.
        """
        dataset_id = self._dataset_id()

        meta = {
            "title": self.cfg.kaggle.source_dataset,
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        }
        meta_dir = staging_dir / "kaggle_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "dataset-metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        source_dir = meta_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        import os
        for f in staging_dir.iterdir():
            if f.name == "kaggle_meta":
                continue
            dst = source_dir / f.name
            if not dst.exists():
                os.link(f, dst)

        old_version = 0
        username, slug = dataset_id.split("/", 1)
        try:
            datasets = self.api.dataset_list(user=username, search=slug)
            for ds in datasets:
                ref = getattr(ds, "ref", getattr(ds, "id", str(ds)))
                if ref == dataset_id:
                    old_version = getattr(ds, "currentVersionNumber", getattr(ds, "current_version_number", 0))
                    break
        except Exception as e:
            logger.info("Could not list dataset %s before creating version: %s", dataset_id, e)

        target_version = None
        try:
            req_data = {"folder": str(meta_dir), "version_notes": f"run_id={run_id}"}
            response = self.api.dataset_create_version(
                str(meta_dir),
                version_notes=f"run_id={run_id}",
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="tar"
            )
            url = getattr(response, "url", "")
            if url and "/versions/" in url:
                try:
                    target_version = int(url.split("/versions/")[-1])
                except ValueError:
                    pass
            
            if self.auditor:
                self.auditor.record("dataset_create_version", req_data, "SUCCESS")
            
            if target_version is not None:
                logger.info("Dataset version created: %s (target version parsed: %d)", dataset_id, target_version)
            else:
                logger.info("Dataset version created: %s (version unknown, waiting for version > %d)", dataset_id, old_version)
        except ProviderError:
            raise
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
                    target_version = 1
                    if self.auditor:
                        self.auditor.record("dataset_create_new", req_new, "SUCCESS")
                    logger.info("Dataset created: %s (target version: 1)", dataset_id)
                except Exception as create_err:
                    if self.auditor:
                        self.auditor.record("dataset_create_new", req_new, f"ERROR: {create_err}")
                    raise ProviderError(f"Failed to create dataset: {create_err}") from create_err
            else:
                if self.auditor:
                    self.auditor.record("dataset_create_version", req_data, f"ERROR: {e}")
                raise ProviderError(f"Failed to upload dataset: {e}") from e

        return self._wait_until_ready(dataset_id, target_version=target_version, old_version=old_version)

    def _wait_until_ready(self, dataset_id: str, target_version: Optional[int] = None, old_version: int = 0) -> int:
        """Poll until dataset files are accessible via the API (i.e. Kaggle has indexed it). Returns the confirmed version."""
        logger.info("Waiting for dataset %s to be indexed…", dataset_id)
        deadline = time.time() + DATASET_READY_TIMEOUT_S
        username, slug = dataset_id.split("/", 1)
        
        while time.time() < deadline:
            try:
                datasets = self.api.dataset_list(user=username, search=slug)
                found = False
                for ds in datasets:
                    ref = getattr(ds, "ref", getattr(ds, "id", str(ds)))
                    if ref == dataset_id:
                        found = True
                        cur_version = getattr(ds, "currentVersionNumber", getattr(ds, "current_version_number", 0))
                        
                        if target_version is not None:
                            if cur_version >= target_version:
                                logger.info("Dataset version %d indexed.", cur_version)
                                time.sleep(2)  # brief grace period for API propagation
                                return cur_version
                            else:
                                logger.info("Dataset at version %d (waiting for %d)…", cur_version, target_version)
                        else:
                            if cur_version > old_version:
                                logger.info("Dataset new version %d indexed (old was %d).", cur_version, old_version)
                                time.sleep(2)
                                return cur_version
                            else:
                                logger.info("Dataset at version %d (waiting for > %d)…", cur_version, old_version)
                if not found:
                    logger.info("Dataset not returned in list yet…")
            except Exception as e:
                logger.info("Dataset not yet indexed (%s) — retrying in %ds…", e, DATASET_READY_POLL_S)
            time.sleep(DATASET_READY_POLL_S)

        raise ProviderError(f"Dataset indexing timed out after {DATASET_READY_TIMEOUT_S}s")
