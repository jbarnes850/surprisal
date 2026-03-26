import fcntl
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ExplorationMeta:
    id: str
    domain: str
    budget: int = 100
    status: str = "initialized"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


def _sessions_path(exp_dir: Path) -> Path:
    return exp_dir / "sessions.json"


def load_branch_sessions(exp_dir: Path, branch_id: str) -> dict[str, str]:
    """Load persisted session IDs for a branch."""
    path = _sessions_path(exp_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    branch_data = data.get(branch_id, {})
    if not isinstance(branch_data, dict):
        return {}
    normalized: dict[str, str] = {}

    research_session = branch_data.get("research_claude_session_id") or branch_data.get("claude_session_id")
    if isinstance(research_session, str) and research_session.strip():
        normalized["research_claude_session_id"] = research_session.strip()
        normalized["claude_session_id"] = research_session.strip()

    code_session = branch_data.get("code_session_id")
    code_provider = branch_data.get("code_provider")
    if not isinstance(code_session, str) or not code_session.strip():
        legacy_codex = branch_data.get("codex_session_id")
        if isinstance(legacy_codex, str) and legacy_codex.strip():
            code_session = legacy_codex.strip()
            code_provider = "codex"
    if isinstance(code_session, str) and code_session.strip():
        normalized["code_session_id"] = code_session.strip()
        if isinstance(code_provider, str) and code_provider.strip():
            normalized["code_provider"] = code_provider.strip()
        elif "codex_session_id" in branch_data:
            normalized["code_provider"] = "codex"
        if normalized.get("code_provider") == "codex":
            normalized["codex_session_id"] = code_session.strip()

    runner_session = branch_data.get("runner_claude_session_id") or branch_data.get("runner_session_id")
    if isinstance(runner_session, str) and runner_session.strip():
        normalized["runner_claude_session_id"] = runner_session.strip()

    return normalized


def save_branch_sessions(
    exp_dir: Path,
    branch_id: str,
    *,
    claude_session_id: Optional[str] = None,
    codex_session_id: Optional[str] = None,
    research_claude_session_id: Optional[str] = None,
    code_session_id: Optional[str] = None,
    code_provider: Optional[str] = None,
    runner_claude_session_id: Optional[str] = None,
) -> dict[str, str]:
    """Persist the latest known branch sessions for Claude/Codex-backed roles."""
    path = _sessions_path(exp_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    updates: dict[str, str] = {}
    research_session = research_claude_session_id or claude_session_id
    if isinstance(research_session, str) and research_session.strip():
        updates["research_claude_session_id"] = research_session.strip()
    effective_code_session = code_session_id
    effective_code_provider = code_provider
    if effective_code_session is None and isinstance(codex_session_id, str) and codex_session_id.strip():
        effective_code_session = codex_session_id.strip()
        effective_code_provider = effective_code_provider or "codex"
    if isinstance(effective_code_session, str) and effective_code_session.strip():
        updates["code_session_id"] = effective_code_session.strip()
        if isinstance(effective_code_provider, str) and effective_code_provider.strip():
            updates["code_provider"] = effective_code_provider.strip()
    if isinstance(runner_claude_session_id, str) and runner_claude_session_id.strip():
        updates["runner_claude_session_id"] = runner_claude_session_id.strip()

    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            raw = handle.read().strip()
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                data = {}
            branch_data = data.get(branch_id, {})
            if not isinstance(branch_data, dict):
                branch_data = {}
            branch_data.update(updates)
            data[branch_id] = branch_data
            handle.seek(0)
            handle.truncate()
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return load_branch_sessions(exp_dir, branch_id)


def create_exploration(base_dir: Path, domain: str, budget: int = 100) -> ExplorationMeta:
    """Create a new exploration directory structure. Idempotent by domain."""
    # Check for existing exploration with same domain
    if base_dir.exists():
        for child in base_dir.iterdir():
            if child.is_dir():
                meta_path = child / "meta.json"
                if meta_path.exists():
                    existing = load_exploration(child)
                    if existing.domain == domain:
                        sessions_path = _sessions_path(child)
                        if not sessions_path.exists():
                            sessions_path.write_text("{}\n")
                        return existing

    exp_id = uuid.uuid4().hex[:12]
    exp_dir = base_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "workspaces").mkdir(exist_ok=True)
    (exp_dir / "prompts").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "dedup").mkdir(exist_ok=True)
    _sessions_path(exp_dir).write_text("{}\n")

    # Write meta.json
    meta = ExplorationMeta(id=exp_id, domain=domain, budget=budget)
    (exp_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))

    return meta


def load_exploration(exp_dir: Path) -> ExplorationMeta:
    """Load exploration metadata from its directory."""
    meta_path = exp_dir / "meta.json"
    data = json.loads(meta_path.read_text())
    return ExplorationMeta(**data)


def find_latest_exploration(base_dir: Path) -> Optional[Path]:
    """Find the most recently created exploration directory."""
    if not base_dir.exists():
        return None
    candidates = []
    for child in base_dir.iterdir():
        if child.is_dir() and (child / "meta.json").exists():
            meta = load_exploration(child)
            candidates.append((meta.created_at, child))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]
