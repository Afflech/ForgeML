from __future__ import annotations

import hashlib
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from forgeml.core.errors import PackagingError
from forgeml.core.logging import get_logger

logger = get_logger(__name__)

# Directories and files to include in the source bundle
BUNDLE_INCLUDES = ["src", "configs", "requirements.txt"]


def get_git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise PackagingError(f"Failed to get git commit: {e.stderr}") from e


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def create_bundle(project_root: Path, output_dir: Path) -> tuple[Path, str]:
    """
    Pack src/, configs/, requirements.txt from project_root into bundle.tar.gz.
    Returns (bundle_path, sha256).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "bundle.tar.gz"

    missing = [p for p in BUNDLE_INCLUDES if not (project_root / p).exists()]
    if missing:
        raise PackagingError(f"Required paths not found in {project_root}: {missing}")

    logger.info("Creating bundle %s", bundle_path)
    with tarfile.open(bundle_path, "w:gz") as tar:
        for item in BUNDLE_INCLUDES:
            src = project_root / item
            tar.add(src, arcname=item)
            logger.info("  + %s", item)

    digest = sha256_file(bundle_path)
    logger.info("Bundle ready — sha256=%s…", digest[:16])
    return bundle_path, digest


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import secrets
    suffix = secrets.token_hex(2)
    return f"{ts}-{suffix}"
