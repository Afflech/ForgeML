import pytest
from pathlib import Path
from forgeml.config.run_config import RunConfig, SourceSpec, TrainingSpec, OutputSpec
from forgeml.config.forge_config import ForgeConfig


def test_run_config_valid():
    config = RunConfig(
        run_id="test-run",
        project="test-proj",
        source=SourceSpec(git_commit="abc", bundle_sha256="def"),
        training=TrainingSpec(
            model="patchcore",
            dataset="mvtec",
            category="bottle",
            config_path="cfg.yaml"
        )
    )
    assert config.run_id == "test-run"
    config.validate_capabilities()


def test_run_config_invalid_capabilities():
    config = RunConfig(
        run_id="test-run",
        project="test-proj",
        source=SourceSpec(git_commit="abc", bundle_sha256="def"),
        training=TrainingSpec(
            model="invalid-model",
            dataset="mvtec",
            category="bottle",
            config_path="cfg.yaml"
        )
    )
    with pytest.raises(ValueError, match="Unsupported model"):
        config.validate_capabilities()
        
    config.training.model = "patchcore"
    config.training.category = "invalid-category"
    with pytest.raises(ValueError, match="Unsupported category"):
        config.validate_capabilities()


def test_forge_config_from_yaml(tmp_path: Path):
    yaml_content = """
project:
  name: test-forge
kaggle:
  kernel: test-kernel
  source_dataset: test-dataset
"""
    yaml_file = tmp_path / "forge.yaml"
    yaml_file.write_text(yaml_content)
    
    config = ForgeConfig.from_yaml(yaml_file)
    assert config.project.name == "test-forge"
    assert config.kaggle.kernel == "test-kernel"
    assert config.kaggle.source_dataset == "test-dataset"
    assert config.training.default_model == "patchcore"


def test_forge_config_missing_required():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        # Missing kaggle
        ForgeConfig.model_validate({
            "project": {"name": "test-forge"}
        })
