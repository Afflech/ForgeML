import os
import tarfile
from pathlib import Path
import pytest
from forgeml.project.packaging import create_bundle, PackagingError

def test_create_bundle_deterministic(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('hello')")
    
    # Create bundle twice, modifying mtime and access times of the source in between
    out1 = tmp_path / "out1"
    bundle1, sha1 = create_bundle(tmp_path, out1, includes=["src"])
    
    import time
    time.sleep(0.1) # ensure a different timestamp if used
    (src_dir / "main.py").touch()
    
    out2 = tmp_path / "out2"
    bundle2, sha2 = create_bundle(tmp_path, out2, includes=["src"])
    
    assert sha1 == sha2

def test_create_bundle_missing_includes(tmp_path: Path):
    with pytest.raises(PackagingError, match="Required paths not found"):
        create_bundle(tmp_path, tmp_path / "out", includes=["missing_dir"])

def test_create_bundle_includes_correct_files(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "cfg.yaml").write_text("")
    
    bundle_path, _ = create_bundle(tmp_path, tmp_path / "out", includes=["src"])
    
    with tarfile.open(bundle_path, "r:gz") as tar:
        names = tar.getnames()
        assert "src" in names
        assert "src/a.py" in names
        assert "configs" not in names
        assert "configs/cfg.yaml" not in names
