from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AgentResult:
    raw: str
    exit_code: int
    parsed: Optional[dict] = None
    duration_seconds: Optional[float] = None

    @classmethod
    def from_raw(cls, raw_str: str, exit_code: int, duration: float = 0.0) -> "AgentResult":
        parsed = None
        try:
            parsed = json.loads(raw_str)
        except (json.JSONDecodeError, TypeError):
            pass
        return cls(raw=raw_str, exit_code=exit_code, parsed=parsed, duration_seconds=duration)
