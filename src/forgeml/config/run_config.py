from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


SCHEMA_VERSION = 1

SUPPORTED_MODELS = {"patchcore", "padim", "fastflow", "efficientad"}
SUPPORTED_DATASETS = {"mvtec"}
MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
}


class SourceSpec(BaseModel):
    git_commit: str
    bundle_sha256: str


class TrainingSpec(BaseModel):
    model: str
    dataset: str
    category: str
    config_path: str
    seed: int = 42


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
        if t.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{t.model}'. Supported: {sorted(SUPPORTED_MODELS)}")
        if t.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset '{t.dataset}'. Supported: {sorted(SUPPORTED_DATASETS)}")
        if t.category not in MVTEC_CATEGORIES:
            raise ValueError(f"Unsupported category '{t.category}'. Supported: {sorted(MVTEC_CATEGORIES)}")
