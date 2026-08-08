"""
ForgeML — Fixed Kernel Entrypoint
Runs on Kaggle. Reads run_config.json from the attached source Dataset,
validates it, trains the requested model, and writes metrics + manifest.

Input layout (Kaggle Dataset mounted at /kaggle/input/<dataset-slug>/):
    run_config.json
    bundle.tar.gz        — source archive (src/, configs/, requirements.txt)

Output (written to /kaggle/working/):
    metrics.json
    artifacts/           — model checkpoint
    run_manifest.json    — lineage record
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
MVTEC_DATASET_SLUG = "ipythonx/mvtec-ad"


def find_source_dataset() -> Path:
    """Return the first dataset directory that contains run_config.json."""
    for d in sorted(KAGGLE_INPUT.iterdir()):
        if (d / "run_config.json").exists():
            return d
    raise FileNotFoundError(
        f"No run_config.json found in any subdirectory of {KAGGLE_INPUT}.\n"
        f"Available: {[str(p) for p in KAGGLE_INPUT.iterdir()]}"
    )


def find_mvtec_root() -> Path:
    """Return the MVTec AD root directory from the attached public Dataset."""
    for d in sorted(KAGGLE_INPUT.iterdir()):
        # MVTec AD dataset has a 'bottle' subdirectory at root
        if (d / "bottle").exists() and (d / "bottle" / "train").exists():
            return d
    raise FileNotFoundError(
        "Could not locate MVTec AD dataset. "
        "Ensure the dataset 'ipythonx/mvtec-ad' is attached to this Kernel."
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {"patchcore", "padim", "fastflow", "efficientad"}
SUPPORTED_DATASETS = {"mvtec"}
MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
}


def load_and_validate_config(source_dir: Path) -> dict:
    config_path = source_dir / "run_config.json"
    print(f"[config] Loading {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)

    # Schema version check
    if cfg.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version: {cfg.get('schema_version')}. Expected 1.")

    required_top = {"run_id", "action", "project", "source", "training", "output"}
    missing = required_top - cfg.keys()
    if missing:
        raise ValueError(f"Missing required fields in run_config.json: {missing}")

    t = cfg["training"]
    if t["model"] not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{t['model']}'. Supported: {sorted(SUPPORTED_MODELS)}")
    if t["dataset"] not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{t['dataset']}'. Supported: {sorted(SUPPORTED_DATASETS)}")
    if t["category"] not in MVTEC_CATEGORIES:
        raise ValueError(f"Unsupported category '{t['category']}'. Supported: {sorted(MVTEC_CATEGORIES)}")

    print(f"[config] OK — run_id={cfg['run_id']} model={t['model']} category={t['category']}")
    return cfg


# ---------------------------------------------------------------------------
# Source bundle
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack_bundle(source_dir: Path, cfg: dict) -> Path:
    """Unpack bundle.tar.gz and verify its sha256 against run_config."""
    bundle_path = source_dir / "bundle.tar.gz"
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle.tar.gz not found in {source_dir}")

    actual_sha = sha256_file(bundle_path)
    expected_sha = cfg["source"]["bundle_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(
            f"Bundle SHA256 mismatch.\n  expected: {expected_sha}\n  actual:   {actual_sha}"
        )
    print(f"[bundle] SHA256 verified: {actual_sha[:16]}…")

    unpack_dir = KAGGLE_WORKING / "source"
    unpack_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(unpack_dir)
    print(f"[bundle] Unpacked to {unpack_dir}")
    return unpack_dir


# ---------------------------------------------------------------------------
# Dependency install
# ---------------------------------------------------------------------------

def install_dependencies(source_dir: Path) -> dict:
    req_path = source_dir / "requirements.txt"
    if not req_path.exists():
        print("[deps] No requirements.txt found — skipping install")
        return {}

    print(f"[deps] Installing from {req_path}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path), "-q"],
        capture_output=True, text=True
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"pip install failed (exit {result.returncode})")
    print(f"[deps] Done in {elapsed:.1f}s")

    # Capture installed versions for manifest
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )
    versions = {}
    for line in freeze.stdout.splitlines():
        if "==" in line:
            pkg, ver = line.split("==", 1)
            versions[pkg.lower()] = ver
    return versions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(cfg: dict, source_dir: Path, mvtec_root: Path) -> dict:
    """Import IndustrialAD from the unpacked bundle and run training."""
    t = cfg["training"]
    model_name = t["model"]
    category = t["category"]
    seed = t.get("seed", 42)

    # Add source to path so IndustrialAD src/ is importable
    sys.path.insert(0, str(source_dir))

    import torch
    torch.manual_seed(seed)

    from torch.utils.data import DataLoader

    # Set env vars so IndustrialAD paths.py resolves correctly
    artifact_dir = KAGGLE_WORKING / cfg["output"]["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MVTEC_DATASET_PATH"] = str(mvtec_root)
    os.environ["INDUSTRIALAD_MODELS_DIR"] = str(artifact_dir)

    from src.datasets.dataset import MVTecDataset
    from src.datasets.transforms import get_default_transform
    from src.models.factory import MODEL_REGISTRY

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in MODEL_REGISTRY: {list(MODEL_REGISTRY.keys())}")

    # Load YAML config if present
    config_path = source_dir / t.get("config_path", f"configs/{model_name}.yaml")
    model_cfg = {}
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            full_cfg = yaml.safe_load(f)
        model_cfg = full_cfg.get("model", {}).copy()
        model_cfg.pop("name", None)
    model_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={model_cfg['device']} model={model_name} category={category} seed={seed}")

    transform = get_default_transform()
    train_dataset = MVTecDataset(str(mvtec_root), category, "train", transform)
    test_dataset = MVTecDataset(str(mvtec_root), category, "test", transform)
    print(f"[train] n_train={len(train_dataset)}  n_test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_cls, _ = MODEL_REGISTRY[model_name]
    model = model_cls(config=model_cfg)

    t0 = time.time()
    model.fit(train_loader)
    train_time = time.time() - t0
    print(f"[train] fit done in {train_time:.1f}s")

    t0 = time.time()
    results = model.predict(test_loader)
    infer_time = time.time() - t0

    # Compute metrics
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import torch as th

    image_scores = results["image_scores"].numpy()
    anomaly_maps = results["anomaly_maps"].numpy()

    gt_labels, gt_masks = [], []
    for _, lbl, msk in DataLoader(test_dataset, batch_size=64, shuffle=False):
        gt_labels.append(lbl)
        gt_masks.append(msk)
    gt_labels = th.cat(gt_labels).numpy()
    gt_masks = (th.cat(gt_masks).numpy() > 0.5).astype(np.uint8)

    image_auroc = float(roc_auc_score(gt_labels, image_scores))

    pixel_auroc = float("nan")
    anomaly_idx = gt_labels == 1
    if anomaly_idx.any() and gt_masks[anomaly_idx].sum() > 0:
        pixel_auroc = float(roc_auc_score(
            gt_masks[anomaly_idx].flatten(),
            anomaly_maps[anomaly_idx].flatten(),
        ))

    print(f"[eval] image_auroc={image_auroc:.4f}  pixel_auroc={pixel_auroc:.4f}")

    # Save checkpoint
    ckpt_path = artifact_dir / f"{model_name}" / f"{category}.pkl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_path))
    ckpt_sha = sha256_file(ckpt_path)
    print(f"[artifact] Saved checkpoint → {ckpt_path}")

    return {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "train_time_s": round(train_time, 1),
        "infer_time_s": round(infer_time, 1),
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "checkpoint": str(ckpt_path.relative_to(KAGGLE_WORKING)),
        "checkpoint_sha256": ckpt_sha,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_outputs(cfg: dict, metrics: dict, pkg_versions: dict) -> None:
    run_id = cfg["run_id"]

    metrics_path = KAGGLE_WORKING / cfg["output"]["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[output] metrics → {metrics_path}")

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_config": cfg,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_available": _cuda_info(),
        },
        "packages": pkg_versions,
        "outputs": metrics,
    }
    manifest_path = KAGGLE_WORKING / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[output] manifest → {manifest_path}")


def _cuda_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"available": True, "device": torch.cuda.get_device_name(0)}
        return {"available": False}
    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  ForgeML Kernel Entrypoint")
    print("=" * 60)

    try:
        source_dataset_dir = find_source_dataset()
        print(f"[init] Source dataset: {source_dataset_dir}")

        cfg = load_and_validate_config(source_dataset_dir)
        source_dir = unpack_bundle(source_dataset_dir, cfg)
        pkg_versions = install_dependencies(source_dir)

        mvtec_root = find_mvtec_root()
        print(f"[init] MVTec root: {mvtec_root}")

        metrics = run_training(cfg, source_dir, mvtec_root)
        write_outputs(cfg, metrics, pkg_versions)

        print("\n[done] Run completed successfully.")

    except (ValueError, FileNotFoundError) as exc:
        # Config/dependency errors — do not retry
        print(f"\n[FAILED_CONFIG] {exc}", file=sys.stderr)
        sys.exit(2)

    except Exception as exc:
        # Unexpected runtime errors
        print(f"\n[FAILED_EXECUTION] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
