from pathlib import Path
from sqlmodel import create_engine, SQLModel

def get_engine(cwd: Path):
    sqlite_file = cwd / "forge.sqlite"
    sqlite_url = f"sqlite:///{sqlite_file}"
    engine = create_engine(sqlite_url)
    SQLModel.metadata.create_all(engine)
    return engine
