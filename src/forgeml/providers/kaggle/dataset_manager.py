from __future__ import annotations

import json
import os
from pathlib import Path

from forgeml.config.forge_config import ForgeConfig
from forgeml.core.errors import AuthError, ProviderError
from forgeml.core.logging import get_logger

logger = get_logger(__name__)


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


class DatasetManager:
    def __init__(self, cfg: ForgeConfig) -> None:
        self.cfg = cfg
        self.api = _kaggle_api()

    def upload(self, staging_dir: Path, run_id: str) -> None:
        """
        Upload staging_dir contents as a new version of the private source Dataset.
        Creates the Dataset on first use.
        """
        dataset_slug = self.cfg.kaggle.dataset
        username = self.api.get_config_value("username")
        dataset_id = f"{username}/{dataset_slug}"

        # Write dataset-metadata.json required by Kaggle API
        meta = {
            "title": dataset_slug,
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        }
        meta_path = staging_dir / "dataset-metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # Try to create; if exists, add new version
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
            if "not found" in err or "404" in err:
                logger.info("Dataset not found — creating: %s", dataset_id)
                self.api.dataset_create_new(
                    str(staging_dir),
                    public=False,
                    quiet=False,
                    convert_to_csv=False,
                )
            else:
                raise ProviderError(f"Failed to upload dataset: {e}") from e
