from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class KaggleConfig(BaseModel):
    kernel: str
    dataset: str
    mvtec_dataset: str = "ipythonx/mvtec-ad"
    accelerator: str = "NvidiaTeslaT4"
    internet: bool = True


class TrainingDefaults(BaseModel):
    default_model: str = "patchcore"
    default_dataset: str = "mvtec"
    default_category: str = "bottle"


class ArtifactsConfig(BaseModel):
    directory: str = "artifacts"


class AIConfig(BaseModel):
    planner: str = "python-sdk"
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o"


class NotificationsConfig(BaseModel):
    discord: bool = False


class ProviderConfig(BaseModel):
    name: str = "kaggle"


class ProjectConfig(BaseModel):
    name: str
    bundle_includes: list[str] = Field(default_factory=lambda: ["src", "configs", "requirements.txt"])


class ForgeConfig(BaseModel):
    project: ProjectConfig
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    kaggle: KaggleConfig
    training: TrainingDefaults = Field(default_factory=TrainingDefaults)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ForgeConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
