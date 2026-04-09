import json
import os
import time
import uuid
from typing import Any
from utils.path_tool import get_abs_path

SESSION_LOG_ROOT = get_abs_path("logs/agent_sessions")
os.makedirs(SESSION_LOG_ROOT, exist_ok=True)

class AgentSession:
    def __init__(self) -> None:
        self.session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.file_path = os.path.join(SESSION_LOG_ROOT, f"{self.session_id}.jsonl")
        self._write_line({
            "timestamp": self._now(),
            "event": "session_start",
            "session_id": self.session_id,
        })

    def _now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _write_line(self, payload: dict[str, Any]) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log(self, event: str, payload: dict[str, Any]) -> None:
        self._write_line({
            "timestamp": self._now(),
            "event": event,
            "payload": payload,
        })
