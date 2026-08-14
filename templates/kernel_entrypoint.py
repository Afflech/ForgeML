"""
ForgeML — Generic Kernel Entrypoint (v2.0 Adapter Phase)
Runs on Kaggle. Reads run_config.json from the attached source Dataset,
extracts bundle, installs dependencies, and runs the user-defined entrypoint.
Outputs must be saved by the user script into /kaggle/working/outputs/.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUTS_DIR = KAGGLE_WORKING / "outputs"


def find_source_dataset() -> Path:
    print(f"[init] Full /kaggle/input tree:")
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        depth = root.replace(str(KAGGLE_INPUT), "").count(os.sep)
        if depth > 3:
            continue
        indent = "  " * depth
        print(f"[init] {indent}{os.path.basename(root)}/")
        for f in files:
            print(f"[init] {indent}  {f}")

    print(f"[init] Searching for run_config.json…")
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if "run_config.json" in files:
            found = Path(root)
            print(f"[init] Found run_config.json at {found}")
            return found

    raise FileNotFoundError("No run_config.json found anywhere under /kaggle/input")


def load_config(source_dir: Path) -> dict:
    config_path = source_dir / "run_config.json"
    with open(config_path) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack_bundle(source_dir: Path, cfg: dict) -> Path:
    bundle_gz = source_dir / "bundle.tar.gz"
    bundle_dir = source_dir / "bundle"

    if bundle_gz.exists():
        actual_sha = sha256_file(bundle_gz)
        expected_sha = cfg["source"]["bundle_sha256"]
        if actual_sha != expected_sha:
            raise ValueError(f"Bundle SHA256 mismatch: {expected_sha} vs {actual_sha}")
        
        unpack_dir = KAGGLE_WORKING / "source"
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_gz, "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(unpack_dir, filter="data")
            else:
                tar.extractall(unpack_dir)
        print(f"[bundle] Extracted bundle.tar.gz → {unpack_dir}")
        return unpack_dir

    if bundle_dir.exists():
        print(f"[bundle] Kaggle auto-extracted bundle found at {bundle_dir}")
        return bundle_dir

    raise FileNotFoundError("Neither bundle.tar.gz nor bundle/ found")


def install_dependencies(source_dir: Path) -> dict:
    req_path = source_dir / "requirements.txt"
    if not req_path.exists():
        return {}

    print(f"[deps] Installing from {req_path}")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path), "-q"], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("pip install failed")
        
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    versions = {}
    for line in freeze.stdout.splitlines():
        if "==" in line:
            pkg, ver = line.split("==", 1)
            versions[pkg.lower()] = ver
    return versions


def _cuda_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"available": True, "device": torch.cuda.get_device_name(0)}
        return {"available": False}
    except:
        return {"available": False}


def write_manifest(cfg: dict, pkg_versions: dict) -> None:
    archive_path = KAGGLE_WORKING / "outputs.tar.gz"
    if OUTPUTS_DIR.exists():
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(OUTPUTS_DIR, arcname="outputs")
    
    archive_sha256 = sha256_file(archive_path) if archive_path.exists() else None

    manifest = {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_config": cfg,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_available": _cuda_info(),
        },
        "packages": pkg_versions,
        "outputs": {
            "archive": "outputs.tar.gz",
            "archive_sha256": archive_sha256
        }
    }
    
    # We write manifest to the parent, because outputs/ is for the user
    # ForgeML kernel manager will download BOTH outputs/ and run_manifest.json
    manifest_path = KAGGLE_WORKING / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[output] manifest → {manifest_path}")


def main() -> None:
    print("=" * 60)
    print("  ForgeML Generic Kernel Entrypoint (v2.0)")
    print("=" * 60)

    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        source_dataset_dir = find_source_dataset()
        cfg = load_config(source_dataset_dir)
        source_dir = unpack_bundle(source_dataset_dir, cfg)
        pkg_versions = install_dependencies(source_dir)

        # Run user entrypoint
        entrypoint_rel = cfg.get("training", {}).get("entrypoint", "scripts/kaggle_adapter.py")
        entrypoint_path = source_dir / entrypoint_rel
        
        if not entrypoint_path.exists():
            raise FileNotFoundError(f"Entrypoint script not found: {entrypoint_path}")
            
        print(f"[exec] Launching entrypoint: {entrypoint_path}")
        
        env = os.environ.copy()
        # Set pythonpath so imports work
        env["PYTHONPATH"] = str(source_dir)
        
        cmd = [sys.executable, str(entrypoint_path)]
        args_str = cfg.get("training", {}).get("args")
        if args_str:
            import shlex
            cmd.extend(shlex.split(args_str))
            
        # We run it from the source_dir so local relative paths might work better
        result = subprocess.run(
            cmd,
            cwd=str(source_dir),
            env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Entrypoint script failed with exit code {result.returncode}")

        write_manifest(cfg, pkg_versions)
        print("\n[done] Run completed successfully.")

    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[FAILED_CONFIG] {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\n[FAILED_EXECUTION] {exc}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
