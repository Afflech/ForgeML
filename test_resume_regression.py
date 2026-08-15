import sys
from pathlib import Path
from forgeml.workflow.runner import WorkflowRunner
from forgeml.config.forge_config import ForgeConfig, ProjectConfig, ProviderConfig, KaggleConfig, TrainingDefaults
from tests.workflow.conftest import StubProvider
import json

project_dir = Path("/tmp/dummy_project")

# Initialize DB
from sqlmodel import SQLModel
from forgeml.db.engine import get_engine
SQLModel.metadata.create_all(get_engine(project_dir))

cfg = ForgeConfig(
    project=ProjectConfig(name="test"),
    provider=ProviderConfig(name="kaggle"),
    kaggle=KaggleConfig(kernel="test-kernel", source_dataset="test-ds"),
    training=TrainingDefaults(default_entrypoint="scripts/kaggle_adapter.py", capabilities_script=None)
)

# First run: fail at UPLOADING
provider1 = StubProvider()
def fail_upload(*args, **kwargs):
    from forgeml.core.errors import ProviderError
    raise ProviderError("Failed at upload")
provider1.upload_dataset = fail_upload

runner1 = WorkflowRunner(cfg, cwd=project_dir, provider=provider1)
try:
    runner1.execute(model="patchcore", category="bottle")
except Exception as e:
    pass

import os
run_id = sorted(os.listdir("/tmp/dummy_project/artifacts"))[-1]
print(f"Run ID: {run_id}")

# Second run: Resume
provider2 = StubProvider()
def fake_download(r_id, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    import hashlib, tarfile
    archive_path = output_dir / "outputs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar: pass
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (output_dir / "run_manifest.json").write_text(json.dumps({"outputs": {"archive_sha256": sha}}))
provider2.download_output = fake_download

runner2 = WorkflowRunner(cfg, cwd=project_dir, provider=provider2)
try:
    runner2.execute(resume_run_id=run_id)
    print("SUCCESS!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
    sys.exit(1)
