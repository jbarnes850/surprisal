"""Auto-detect available agent providers and route roles accordingly."""
import asyncio
import logging
import os
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


@dataclass
class LiteratureStatus:
    provider: str  # "alphaxiv", "huggingface"

    @property
    def has_semantic_search(self) -> bool:
        return self.provider == "alphaxiv"


async def detect_literature_provider() -> LiteratureStatus:
    """Detect which literature search provider is available."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "mcp", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if b"alphaxiv" in stdout:
            logger.info("Literature: alphaxiv MCP connected (semantic search + paper Q&A)")
            return LiteratureStatus(provider="alphaxiv")
    except (FileNotFoundError, asyncio.TimeoutError):
        pass
    logger.info("Literature: using HuggingFace Papers API (public, no semantic search)")
    return LiteratureStatus(provider="huggingface")


async def check_claude_auth_method() -> str | None:
    """Return the Claude auth method ('claude.ai' or 'apiKey') or None."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "auth", "status", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            return None
        import json
        data = json.loads(stdout)
        return data.get("authMethod")
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        return None


def ensure_runner_auth(config_path=None, save_config_fn=None, cfg=None) -> bool:
    """Ensure CLAUDE_CODE_OAUTH_TOKEN is set for Docker runner auth.

    If the user is subscription-backed and the token is missing from the
    environment, check the config for a cached token. If not cached, prompt
    the user once and cache it.

    Returns True if auth is available, False if not.
    """
    # Already in environment — nothing to do
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return True
    # API-key auth doesn't need the token
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True

    # Check config for cached token
    if cfg and cfg.credentials.claude_oauth_token:
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = cfg.credentials.claude_oauth_token
        return True

    # Detect if subscription-backed
    auth_method = asyncio.run(check_claude_auth_method())
    if auth_method != "claude.ai":
        return True  # API-key auth or unknown — let it proceed

    # Subscription auth without token — prompt the user
    import sys
    print(
        "\n"
        "Claude subscription auth detected, but the Docker runner needs an OAuth token.\n"
        "Run this once in your terminal:\n"
        "\n"
        "  claude setup-token\n"
        "\n"
        "Then paste the token here (starts with sk-ant-oat01-).",
        file=sys.stderr,
    )
    try:
        token = input("Token: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False

    if not token or not token.startswith("sk-ant-"):
        print("Invalid token. Run 'claude setup-token' to generate one.", file=sys.stderr)
        return False

    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = token

    # Cache in config so the user never has to do this again
    if cfg and config_path and save_config_fn:
        cfg.credentials.claude_oauth_token = token
        save_config_fn(cfg, config_path)
        print(f"Token cached in {config_path} — you won't be asked again.", file=sys.stderr)

    return True


async def detect_providers() -> ProviderStatus:
    """Detect which agent providers are available."""
    claude, codex = await asyncio.gather(check_claude(), check_codex())
    status = ProviderStatus(claude_available=claude, codex_available=codex)

    if status.both_available:
        logger.info("Claude + Codex detected -- routing research/belief to Claude and analysis/review to Codex")
    elif status.claude_available:
        logger.info("Claude available -- routing all roles through Claude")
    elif status.codex_available:
        logger.warning("Only Codex detected -- Claude is still required for generator and belief roles")
    else:
        logger.error("No agent providers available -- run 'claude auth login'")

    return status
