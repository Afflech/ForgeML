from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

# TODO: sync manually with IndustrialAD MODEL_REGISTRY until Phase D
SUPPORTED_MODELS = {"patchcore", "padim", "fastflow", "efficientad"}
MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
}

class SourceSpec(BaseModel):
    git_commit: str
    bundle_sha256: str

class TrainingSpec(BaseModel):
    entrypoint: str = "scripts/kaggle_adapter.py"
    # Legacy fields (optional cho generic workloads)
    model: Optional[str] = None
    dataset: Optional[str] = None
    category: Optional[str] = None
    config_path: Optional[str] = None
    seed: int = 42
    # Generic field (để truyền tham số cho các script tuỳ ý)
    args: Optional[str] = None

class OutputSpec(BaseModel):
    metrics_file: str = "metrics.json"
    artifact_dir: str = "artifacts"

class RunConfig(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    action: Literal["train"] = "train"
    project: str
    source: SourceSpec
    training: TrainingSpec
    output: OutputSpec = Field(default_factory=OutputSpec)

    def validate_capabilities(self) -> None:
        """Raise ValueError if any field is outside supported capabilities."""
        t = self.training
        
        # Chỉ validate gắt gao nếu đang dùng adapter mặc định (IndustrialAD pipeline)
        if t.entrypoint == "scripts/kaggle_adapter.py":
            missing = []
            if not t.model: missing.append("model")
            if not t.dataset: missing.append("dataset")
            if not t.category: missing.append("category")
            if not t.config_path: missing.append("config_path")
            
            if missing:
                raise ValueError(f"Legacy adapter requires the following fields: {', '.join(missing)}")
                
            if t.model not in SUPPORTED_MODELS:
                raise ValueError(f"Unsupported model '{t.model}'. Supported: {sorted(SUPPORTED_MODELS)}")
            if t.category not in MVTEC_CATEGORIES:
                raise ValueError(f"Unsupported category '{t.category}'. Supported: {sorted(MVTEC_CATEGORIES)}")
