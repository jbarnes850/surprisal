"""Sandbox backend protocol, auto-detection, and factory."""
import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable

from surprisal.agents.base import AgentResult
from surprisal.config import SandboxConfig, CredentialsConfig


@runtime_checkable
class SandboxBackend(Protocol):
    async def execute(
        self,
        experiment_prompt: str,
        workspace: Path,
        config: SandboxConfig,
        system_prompt_file: str | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        ...


async def detect_gpu() -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0 and len(stdout) > 0
    except (FileNotFoundError, asyncio.TimeoutError):
        return False


def create_backend(
    config: SandboxConfig,
    credentials: CredentialsConfig,
    gpu_available: bool | None = None,
) -> SandboxBackend:
    from surprisal.agents.experiment_container import ExperimentContainer
    from surprisal.agents.hf_jobs import HFJobsSandbox

    if config.backend == "hf_jobs":
        return HFJobsSandbox(config=config, credentials=credentials)

    if config.backend == "local" or config.backend == "auto":
        if config.backend == "auto" and gpu_available is not None:
            config.gpu = gpu_available
        return ExperimentContainer(config=config, credentials=credentials)

    return ExperimentContainer(config=config, credentials=credentials)
