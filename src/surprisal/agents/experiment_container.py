"""Local GPU container backend — full implementation in Task 4."""
from pathlib import Path
from surprisal.agents.base import AgentResult
from surprisal.config import SandboxConfig, CredentialsConfig


class ExperimentContainer:
    def __init__(self, config: SandboxConfig, credentials: CredentialsConfig):
        self.config = config
        self.credentials = credentials

    async def execute(self, experiment_prompt: str, workspace: Path, config: SandboxConfig) -> AgentResult:
        raise NotImplementedError("Full implementation in Task 4")
