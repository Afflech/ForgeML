import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class ProviderAuditor:
    def __init__(self, run_dir: Path):
        self.log_file = run_dir / "provider.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def record(self, action: str, request: dict[str, Any], response: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "request": request,
            "response": str(response)
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
