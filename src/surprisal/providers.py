"""Auto-detect available agent providers and route roles accordingly."""
import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger("surprisal")


@dataclass
class ProviderStatus:
    claude_available: bool
    codex_available: bool

    @property
    def any_available(self) -> bool:
        return self.claude_available or self.codex_available

    @property
    def both_available(self) -> bool:
        return self.claude_available and self.codex_available


async def check_claude() -> bool:
    """Check if Claude CLI is authenticated."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "auth", "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0 and b"loggedIn" in stdout
    except (FileNotFoundError, asyncio.TimeoutError):
        return False


async def check_codex() -> bool:
    """Check if Codex CLI is available and authenticated."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "codex", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0
    except (FileNotFoundError, asyncio.TimeoutError):
        return False


async def detect_providers() -> ProviderStatus:
    """Detect which agent providers are available."""
    claude, codex = await asyncio.gather(check_claude(), check_codex())
    status = ProviderStatus(claude_available=claude, codex_available=codex)

    if status.both_available:
        logger.info("Both Claude and Codex available -- using heterogeneous agent mapping")
    elif status.claude_available:
        logger.info("Only Claude available -- routing all roles through Claude")
    elif status.codex_available:
        logger.info("Only Codex available -- routing all roles through Codex")
    else:
        logger.error("No agent providers available -- run 'claude auth login' or install Codex")

    return status
