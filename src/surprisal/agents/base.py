from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AgentResult:
    raw: str
    exit_code: int
    parsed: Optional[dict] = None
    duration_seconds: Optional[float] = None
    session_id: Optional[str] = None

    @classmethod
    def from_raw(
        cls,
        raw_str: str,
        exit_code: int,
        duration: float = 0.0,
        session_id: Optional[str] = None,
        parsed: Optional[dict] = None,
    ) -> "AgentResult":
        if parsed is None:
            try:
                parsed = json.loads(raw_str)
            except (json.JSONDecodeError, TypeError):
                parsed = None

        if session_id is None and isinstance(parsed, dict):
            parsed_session_id = parsed.get("session_id")
            if isinstance(parsed_session_id, str) and parsed_session_id.strip():
                session_id = parsed_session_id

        return cls(
            raw=raw_str,
            exit_code=exit_code,
            parsed=parsed,
            duration_seconds=duration,
            session_id=session_id,
        )
