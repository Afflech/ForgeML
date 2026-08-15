from pathlib import Path
from forgeml.workflow.runner import WorkflowRunner
from forgeml.config.forge_config import ForgeConfig, ProjectConfig, ProviderConfig, KaggleConfig, TrainingDefaults
from tests.workflow.conftest import StubProvider
from sqlmodel import SQLModel
from forgeml.db.engine import get_engine
import json
import tarfile

project_dir = Path("/tmp/dummy_project")
cfg = ForgeConfig(
    project=ProjectConfig(name="test"),
    provider=ProviderConfig(name="kaggle"),
    kaggle=KaggleConfig(kernel="test-kernel", source_dataset="test-ds"),
    training=TrainingDefaults(default_entrypoint="scripts/dummy.py")
)

provider = StubProvider()
def fake_download(*args, **kwargs):
    from forgeml.core.errors import ProviderError
    raise ProviderError("Network error while downloading")

provider.download_output = fake_download
runner = WorkflowRunner(cfg, cwd=project_dir, provider=provider)
runner.execute(model="test_model", category="test_cat")
