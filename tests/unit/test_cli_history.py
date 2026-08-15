from pathlib import Path
from typer.testing import CliRunner
from sqlmodel import Session
from forgeml.cli.main import app
from forgeml.db.engine import get_engine
from forgeml.db.models import Run
from forgeml.core.states import RunState
from datetime import datetime, timezone

runner = CliRunner()

def test_history_shows_dash_for_missing_metrics(tmp_path: Path):
    forge_project = tmp_path
    engine = get_engine(forge_project)
    
    # Create a generic run without model/dataset/category, and metrics_json = None
    with Session(engine) as session:
        test_run = Run(
            id="test-history-1",
            project="test-proj",
            provider="kaggle",
            status=RunState.COMPLETED.value,
            git_commit="abcdef",
            bundle_sha256="123456",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            metrics_json=None
        )
        session.add(test_run)
        session.commit()

    # We need to run the CLI within the forge_project dir so it uses its db
    import os
    os.chdir(forge_project)
    
    # Create dummy forge.yaml to test fallback entrypoint
    (forge_project / "forge.yaml").write_text('''
project:
  name: test
provider:
  name: kaggle
kaggle: {}
training:
  default_entrypoint: scripts/train.py
''')

    result = runner.invoke(app, ["history"])
    
    assert result.exit_code == 0
    # Expect train.py fallback
    assert "train.py" in result.stdout
    # Expect EM DASH
    assert "—" in result.stdout
