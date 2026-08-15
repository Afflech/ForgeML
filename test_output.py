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
def fake_download(run_id, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    archive_path = output_dir / "outputs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        pass
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {"outputs": {"archive_sha256": sha}}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (output_dir / "kernel.log").write_text("")
    (output_dir / "outputs" / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs" / "metrics.json").write_text('{"accuracy": 0.99, "loss": 0.01}')
    for i in range(5):
        (output_dir / f"extra_file_{i}.txt").write_text("")

provider.download_output = fake_download
runner = WorkflowRunner(cfg, cwd=project_dir, provider=provider)
runner.execute(model="test_model", category="test_cat")
