from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


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

def fetch_capabilities_catalog(project_root: str, capabilities_script: str) -> dict:
    import subprocess, json, sys
    from forgeml.core.errors import ProviderError
    result = subprocess.run(
        [sys.executable, capabilities_script],
        cwd=project_root, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise ProviderError(f"capabilities_script failed: {result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ProviderError(f"capabilities_script did not return valid JSON: {e}")

class RunConfig(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    action: Literal["train"] = "train"
    project: str
    source: SourceSpec
    training: TrainingSpec
    output: OutputSpec = Field(default_factory=OutputSpec)

    def validate_capabilities(self, project_root: Optional[str] = None, capabilities_script: Optional[str] = None) -> None:
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
                from forgeml.core.errors import ConfigError
                raise ConfigError(f"Legacy adapter requires the following fields: {', '.join(missing)}")

            if capabilities_script and project_root:
                catalog = fetch_capabilities_catalog(project_root, capabilities_script)

                if t.model not in catalog.get("models", []):
                    raise ValueError(f"Unsupported model '{t.model}'. Supported: {sorted(catalog.get('models', []))}")
                if t.category not in catalog.get("categories", []):
                    raise ValueError(f"Unsupported category '{t.category}'. Supported: {sorted(catalog.get('categories', []))}")
