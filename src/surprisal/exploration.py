import json
import uuid
from dataclasses import dataclass, asdict
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
                        return existing

    exp_id = uuid.uuid4().hex[:12]
    exp_dir = base_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "workspaces").mkdir(exist_ok=True)
    (exp_dir / "prompts").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "dedup").mkdir(exist_ok=True)

    # Write meta.json
    meta = ExplorationMeta(id=exp_id, domain=domain, budget=budget)
    (exp_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))

    # Initialize sessions.json
    (exp_dir / "sessions.json").write_text("{}")

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


def update_session(exp_dir: Path, branch_id: str,
                   claude_session_id: Optional[str] = None,
                   codex_session_id: Optional[str] = None) -> None:
    """Update sessions.json with session IDs for a branch."""
    sessions_path = exp_dir / "sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else {}
    sessions[branch_id] = {
        "claude_session_id": claude_session_id,
        "codex_session_id": codex_session_id,
    }
    sessions_path.write_text(json.dumps(sessions, indent=2))


def get_session_ids(exp_dir: Path, branch_id: str) -> Optional[dict]:
    """Get session IDs for a branch. Returns None if branch not found."""
    sessions_path = exp_dir / "sessions.json"
    if not sessions_path.exists():
        return None
    sessions = json.loads(sessions_path.read_text())
    return sessions.get(branch_id)
