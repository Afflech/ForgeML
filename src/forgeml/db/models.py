from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel

class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    project: str
    provider: str
    status: str

    model: str
    dataset: str
    category: str

    git_commit: str
    bundle_sha256: str

    # Last successfully reached RunState (before any failure).
    # Used by resume logic to determine where to restart from.
    last_stage: Optional[str] = None

    kaggle_dataset_version: Optional[str] = None
    kaggle_run_id: Optional[str] = None

    started_at: datetime
    finished_at: Optional[datetime] = None

    metrics_json: Optional[str] = None
    artifact_path: Optional[str] = None
    error_type: Optional[str] = None

