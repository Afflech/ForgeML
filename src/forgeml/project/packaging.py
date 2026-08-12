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


def _deterministic_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Reset metadata to ensure reproducible sha256 across identical source files."""
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = "forge"
    tarinfo.gname = "forge"
    tarinfo.mtime = 0
    return tarinfo


def create_bundle(project_root: Path, output_dir: Path, includes: list[str] = None) -> tuple[Path, str]:
    """
    Pack selected files/dirs from project_root into bundle.tar.gz deterministically.
    Returns (bundle_path, sha256).
    """
    if includes is None:
        includes = ["src", "configs", "requirements.txt"]

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "bundle.tar.gz"

    missing = [p for p in includes if not (project_root / p).exists()]
    if missing:
        raise PackagingError(f"Required paths not found in {project_root}: {missing}")

    logger.info("Creating bundle %s", bundle_path)
    import gzip
    with gzip.GzipFile(bundle_path, "wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for item in sorted(includes):
                src = project_root / item
                
                # Walk and add files in sorted order to ensure deterministic archive
                if src.is_dir():
                    # Add directory itself
                    tar.add(src, arcname=item, filter=_deterministic_filter, recursive=False)
                    # Recursively sort and add all contents
                    paths = sorted(src.rglob("*"))
                    for p in paths:
                        tar.add(p, arcname=p.relative_to(project_root).as_posix(), filter=_deterministic_filter, recursive=False)
                else:
                    tar.add(src, arcname=item, filter=_deterministic_filter, recursive=False)
                
                logger.info("  + %s", item)

    digest = sha256_file(bundle_path)
    logger.info("Bundle ready — sha256=%s…", digest[:16])
    return bundle_path, digest


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import secrets
    suffix = secrets.token_hex(2)
    return f"{ts}-{suffix}"
