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

def test_kernel_manager_validates_accelerator(monkeypatch):
    cfg = ForgeConfig(
        project=ProjectConfig(name="test"),
        kaggle=KaggleConfig(kernel="test", source_dataset="test", accelerator="InvalidAcc")
    )
    
    # Bypass _write_kernel_metadata and kernels_push since we only care about validation
    km = KernelManager(cfg, api="dummy")
    
    with pytest.raises(ConfigError, match="Unsupported Kaggle accelerator 'InvalidAcc'"):
        km.submit("run123", 1)

def test_kernel_manager_valid_accelerator_proceeds(monkeypatch):
    cfg = ForgeConfig(
        project=ProjectConfig(name="test"),
        kaggle=KaggleConfig(kernel="test", source_dataset="test", accelerator="TPUv3")
    )
    
    class DummyApi:
        def get_config_value(self, key):
            return "dummy_user"
        def kernels_push(self, *args, **kwargs):
            class DummyResult:
                version_number = "5"
            return DummyResult()
            
    km = KernelManager(cfg, api=DummyApi())
    
    # Mock write_kernel_metadata so it doesn't write to the real templates folder
    monkeypatch.setattr(km, "_write_kernel_metadata", lambda dataset_version: None)
    
    version = km.submit("run123", 1)
    assert version == "5"
