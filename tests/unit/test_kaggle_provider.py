import pytest
from pathlib import Path
import os
from forgeml.core.errors import AuthError, ConfigError
from forgeml.config.forge_config import ForgeConfig
from forgeml.providers.kaggle.kernel_manager import KernelManager
from forgeml.providers.kaggle.auth import get_kaggle_api

def test_auth_fallback_missing_credentials(monkeypatch):
    """Test that auth raises AuthError if no env vars and no config file exist."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    
    # Mock Path.exists to return False for config files
    original_exists = Path.exists
    def mock_exists(self):
        if "kaggle.json" in str(self):
            return False
        return original_exists(self)
    
    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    
    with pytest.raises(AuthError, match="Kaggle authentication credentials not found"):
        get_kaggle_api()

from forgeml.config.forge_config import ForgeConfig, ProjectConfig, KaggleConfig

def test_kernel_manager_validates_accelerator(monkeypatch, tmp_path):
    cfg = ForgeConfig(
        project=ProjectConfig(name="test"),
        kaggle=KaggleConfig(kernel="test", source_dataset="test", accelerator="InvalidAcc")
    )
    
    # Bypass _write_kernel_metadata and kernels_push since we only care about validation
    km = KernelManager(cfg, api="dummy", run_dir=tmp_path)
    
    with pytest.raises(ConfigError, match="Unsupported Kaggle accelerator 'InvalidAcc'"):
        km.submit("run123", 1)

def test_kernel_manager_valid_accelerator_proceeds(monkeypatch, tmp_path):
    cfg = ForgeConfig(
        project=ProjectConfig(name="test"),
        kaggle=KaggleConfig(kernel="test", source_dataset="test", accelerator="TPUv3")
    )
    
    class DummyApi:
        pushed_folder = None

        def get_config_value(self, key):
            return "dummy_user"
        def kernels_push(self, *args, **kwargs):
            self.pushed_folder = Path(args[0])
            class DummyResult:
                version_number = "5"
            return DummyResult()

    api = DummyApi()
    km = KernelManager(cfg, api=api, run_dir=tmp_path)

    version = km.submit("run123", 1)
    assert version == "5"
    assert api.pushed_folder == tmp_path / "remote"
    assert (tmp_path / "remote" / "kernel_entrypoint.py").exists()
    assert (tmp_path / "remote" / "kernel-metadata.json").exists()
    assert not (km._templates_dir / "kernel-metadata.json").exists()
