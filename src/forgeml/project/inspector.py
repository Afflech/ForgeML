from __future__ import annotations

import subprocess
from pathlib import Path

class ProjectInspector:
    """Inspects local project structure and capabilities."""
    
    def __init__(self, cwd: Path):
        self.cwd = cwd

    def check_structure(self) -> dict[str, bool]:
        """Check for required source files/directories."""
        required = ["src", "configs", "requirements.txt"]
        return {item: (self.cwd / item).exists() for item in required}
    
    def is_git_repo(self) -> bool:
        """Check if cwd is inside a git repository."""
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.cwd,
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def is_git_dirty(self) -> bool:
        """Check if git working tree has uncommitted changes."""
        if not self.is_git_repo():
            return False
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.cwd,
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip()) > 0
