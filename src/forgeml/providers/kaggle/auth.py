from __future__ import annotations

import os
from pathlib import Path

from forgeml.core.errors import AuthError


def get_kaggle_api():
    """Authenticate and return the Kaggle API singleton.
    
    Checks in order:
    1. OAuth session (via kaggle library's internal logic)
    2. KAGGLE_USERNAME / KAGGLE_KEY env vars
    3. ~/.kaggle/kaggle.json or ~/.config/kaggle/kaggle.json
    
    Raises AuthError with actionable instructions if none found.
    """
    import kaggle

    # Check 1: Env vars
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    
    # Check 2 & 3: Config files (classic legacy json or new oauth creds)
    # The kaggle package uses XDG_CONFIG_HOME or ~/.config/kaggle or ~/.kaggle
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir:
        paths = [
            Path(config_dir) / "kaggle.json",
            Path(config_dir) / "credentials.json",
        ]
    else:
        paths = [
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / ".kaggle" / "credentials.json",
            Path.home() / ".config" / "kaggle" / "kaggle.json",
            Path.home() / ".config" / "kaggle" / "credentials.json",
        ]
        
    has_file = any(p.exists() and p.is_file() for p in paths)
    
    if not (has_env or has_file):
        raise AuthError(
            "Kaggle authentication credentials not found.\n"
            "Please authenticate using one of the following methods:\n"
            "1. Run 'kaggle auth login' to authenticate via OAuth (Kaggle CLI v2.2.0+).\n"
            "2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
            "3. Place a kaggle.json file in ~/.kaggle/ or ~/.config/kaggle/."
        )
        
    try:
        kaggle.api.authenticate()
        return kaggle.api
    except SystemExit:
        raise AuthError(
            "Kaggle API authentication failed (SystemExit raised by kaggle library).\n"
            "Please verify your credentials in ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY."
        )
    except Exception as e:
        raise AuthError(f"Kaggle authentication failed: {e}")
