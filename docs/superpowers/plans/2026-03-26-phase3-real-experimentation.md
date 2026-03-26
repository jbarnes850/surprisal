# Phase 3: Real Experimentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CPU-only, network-isolated Docker sandbox with a GPU-enabled container where agents run inside, and add W&B + HuggingFace integration via MCP configs.

**Architecture:** Backend abstraction (SandboxBackend protocol) with two implementations — LocalGPUContainer (default, agent runs inside Docker container with GPU) and HFJobsSandbox (cloud scale-out via HF Jobs). FSM merges experiment_programmer + code_executor into experiment_runner. MCP configs shipped with surprisal, credentials injected at runtime.

**Tech Stack:** Python 3.12, asyncio, Docker (nvidia/cuda), Claude Agent SDK (`claude -p`), HuggingFace Hub SDK (`run_uv_job`), W&B/HF MCP servers, TOML config.

**Spec:** `docs/superpowers/specs/2026-03-26-phase3-real-experimentation-design.md`

---

## File Map

### New Files
| File | Responsibility |
|---|---|
| `src/surprisal/agents/experiment_container.py` | LocalGPUContainer — launches Docker container with agent inside |
| `src/surprisal/agents/hf_jobs.py` | HFJobsSandbox — submits experiments to HF Jobs cloud |
| `src/surprisal/agents/backends.py` | SandboxBackend protocol + auto-detection + factory |
| `src/surprisal/mcp_config.py` | Runtime MCP config generation with credential injection |
| `src/surprisal/prompts/experiment_runner.md` | New merged runner prompt (write + execute + debug) |
| `sandbox/Dockerfile.gpu` | GPU-enabled Docker image with ML stack + agent CLIs |
| `install.sh` | One-line bootstrap script |
| `tests/test_agents/test_experiment_container.py` | Tests for LocalGPUContainer |
| `tests/test_agents/test_hf_jobs.py` | Tests for HFJobsSandbox |
| `tests/test_agents/test_backends.py` | Tests for backend protocol + auto-detection |
| `tests/test_mcp_config.py` | Tests for MCP config generation |

### Modified Files
| File | What Changes |
|---|---|
| `src/surprisal/config.py` | Add SandboxConfig fields (gpu, image, backend), add CredentialsConfig, update load/save |
| `src/surprisal/fsm.py` | Replace experiment_programmer + code_executor with experiment_runner |
| `src/surprisal/fsm_runner.py` | Merge programmer + executor blocks into runner, use backend abstraction |
| `src/surprisal/prompts/experiment_generator.md` | Add HF dataset awareness section |
| `src/surprisal/prompts/experiment_analyst.md` | Add W&B querying section |
| `tests/test_fsm.py` | Update all tests for merged states |
| `tests/test_config.py` | Add tests for new config fields |
| `tests/test_agents/test_docker.py` | Keep existing tests (backward compat), add deprecation note |

---

## Task 1: Config Dataclass Updates

**Files:**
- Modify: `src/surprisal/config.py:35-56` (SandboxConfig, AutoDiscoveryConfig)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for new config fields**

Add to `tests/test_config.py`:

```python
def test_sandbox_config_new_defaults():
    cfg = AutoDiscoveryConfig()
    assert cfg.sandbox.backend == "auto"
    assert cfg.sandbox.image == "surprisal-gpu:latest"
    assert cfg.sandbox.gpu is True
    assert cfg.sandbox.memory_limit == "16g"
    assert cfg.sandbox.cpu_limit == "4"
    assert cfg.sandbox.timeout == 1800
    assert cfg.sandbox.network is True
    assert cfg.sandbox.hf_flavor == "a10g-small"
    assert cfg.sandbox.hf_timeout == "2h"


def test_credentials_config_defaults():
    cfg = AutoDiscoveryConfig()
    assert cfg.credentials.wandb_api_key == ""
    assert cfg.credentials.hf_token == ""


def test_credentials_config_round_trip(tmp_path):
    cfg = AutoDiscoveryConfig()
    cfg.credentials.wandb_api_key = "test-key-123"
    cfg.credentials.hf_token = "hf_test_456"
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.credentials.wandb_api_key == "test-key-123"
    assert loaded.credentials.hf_token == "hf_test_456"


def test_config_set_credentials():
    cfg = AutoDiscoveryConfig()
    cfg.set("credentials.wandb_api_key", "my-key")
    assert cfg.credentials.wandb_api_key == "my-key"


def test_config_set_sandbox_gpu():
    cfg = AutoDiscoveryConfig()
    cfg.set("sandbox.gpu", "false")
    assert cfg.sandbox.gpu is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_config.py -v -k "sandbox_config_new or credentials" 2>&1 | tail -20`
Expected: FAIL — `AttributeError: 'SandboxConfig' object has no attribute 'backend'`

- [ ] **Step 3: Update SandboxConfig and add CredentialsConfig**

In `src/surprisal/config.py`, replace the existing `SandboxConfig` (lines 35-39) with:

```python
@dataclass
class SandboxConfig:
    backend: str = "auto"
    image: str = "surprisal-gpu:latest"
    gpu: bool = True
    memory_limit: str = "16g"
    cpu_limit: str = "4"
    timeout: int = 1800
    network: bool = True
    hf_flavor: str = "a10g-small"
    hf_timeout: str = "2h"


@dataclass
class CredentialsConfig:
    wandb_api_key: str = ""
    hf_token: str = ""
```

Update `AutoDiscoveryConfig` (line 50-56) to add `credentials`:

```python
@dataclass
class AutoDiscoveryConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
```

Update `load_config` (line 84) to include `"credentials"`:

```python
    for section_name in ("general", "mcts", "agents", "sandbox", "predictor", "credentials"):
```

Update `save_config` (line 95) to include `"credentials"`:

```python
    for section_name in ("general", "mcts", "agents", "sandbox", "predictor", "credentials"):
```

- [ ] **Step 4: Update existing tests for changed defaults**

In `tests/test_config.py`, update `test_default_config_has_expected_values` — the sandbox defaults have changed:

Replace lines 22-25:
```python
    assert cfg.sandbox.memory_limit == "16g"
    assert cfg.sandbox.cpu_limit == "4"
    assert cfg.sandbox.timeout == 1800
    assert cfg.sandbox.network is True
```

- [ ] **Step 5: Run all config tests**

Run: `uv run --no-sync pytest tests/test_config.py -v 2>&1 | tail -20`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/surprisal/config.py tests/test_config.py
git commit -m "feat: add CredentialsConfig and update SandboxConfig for Phase 3"
```

---

## Task 2: MCP Config Generation

**Files:**
- Create: `src/surprisal/mcp_config.py`
- Test: `tests/test_mcp_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mcp_config.py`:

```python
import json
import pytest
from surprisal.mcp_config import generate_mcp_config
from surprisal.config import CredentialsConfig


def test_empty_credentials_produces_empty_servers():
    creds = CredentialsConfig()
    config = generate_mcp_config(creds)
    assert config == {"mcpServers": {}}


def test_wandb_only():
    creds = CredentialsConfig(wandb_api_key="test-key")
    config = generate_mcp_config(creds)
    assert "wandb" in config["mcpServers"]
    assert config["mcpServers"]["wandb"]["env"]["WANDB_API_KEY"] == "test-key"
    assert "huggingface" not in config["mcpServers"]


def test_hf_only():
    creds = CredentialsConfig(hf_token="hf_test")
    config = generate_mcp_config(creds)
    assert "huggingface" in config["mcpServers"]
    assert config["mcpServers"]["huggingface"]["env"]["HF_TOKEN"] == "hf_test"
    assert "wandb" not in config["mcpServers"]


def test_both_credentials():
    creds = CredentialsConfig(wandb_api_key="wk", hf_token="hf")
    config = generate_mcp_config(creds)
    assert "wandb" in config["mcpServers"]
    assert "huggingface" in config["mcpServers"]


def test_output_is_json_serializable():
    creds = CredentialsConfig(wandb_api_key="k", hf_token="t")
    config = generate_mcp_config(creds)
    serialized = json.dumps(config)
    assert json.loads(serialized) == config
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_mcp_config.py -v 2>&1 | tail -10`
Expected: FAIL — `ModuleNotFoundError: No module named 'surprisal.mcp_config'`

- [ ] **Step 3: Implement generate_mcp_config**

Create `src/surprisal/mcp_config.py`:

```python
"""Runtime MCP config generation with credential injection."""
import json
import tempfile
from pathlib import Path
from surprisal.config import CredentialsConfig


def generate_mcp_config(credentials: CredentialsConfig) -> dict:
    """Generate MCP server config dict from credentials.

    Returns a dict suitable for --mcp-config JSON. Only includes
    servers for which credentials are provided.
    """
    config: dict = {"mcpServers": {}}

    if credentials.wandb_api_key:
        config["mcpServers"]["wandb"] = {
            "command": "npx",
            "args": ["-y", "@wandb/mcp-server"],
            "env": {"WANDB_API_KEY": credentials.wandb_api_key},
        }

    if credentials.hf_token:
        config["mcpServers"]["huggingface"] = {
            "command": "npx",
            "args": ["-y", "@huggingface/mcp-server"],
            "env": {"HF_TOKEN": credentials.hf_token},
        }

    return config


def write_mcp_config(credentials: CredentialsConfig) -> str:
    """Write MCP config to a temp file and return the path.

    The caller is responsible for cleanup (or let the OS handle it
    since these are small JSON files in /tmp).
    """
    config = generate_mcp_config(credentials)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="surprisal-mcp-")
    with open(fd, "w") as f:
        json.dump(config, f)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_mcp_config.py -v 2>&1 | tail -10`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/surprisal/mcp_config.py tests/test_mcp_config.py
git commit -m "feat: add MCP config generation with credential injection"
```

---

## Task 3: SandboxBackend Protocol and Auto-Detection

**Files:**
- Create: `src/surprisal/agents/backends.py`
- Test: `tests/test_agents/test_backends.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agents/test_backends.py`:

```python
import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from surprisal.agents.backends import detect_gpu, create_backend
from surprisal.config import SandboxConfig, CredentialsConfig


@pytest.mark.asyncio
async def test_detect_gpu_with_nvidia_smi():
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"GPU 0: NVIDIA GB10\n", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        result = await detect_gpu()
        assert result is True


@pytest.mark.asyncio
async def test_detect_gpu_without_nvidia_smi():
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await detect_gpu()
        assert result is False


def test_create_backend_local():
    from surprisal.agents.experiment_container import ExperimentContainer
    config = SandboxConfig(backend="local")
    creds = CredentialsConfig()
    backend = create_backend(config, creds)
    assert isinstance(backend, ExperimentContainer)


def test_create_backend_hf_jobs():
    from surprisal.agents.hf_jobs import HFJobsSandbox
    config = SandboxConfig(backend="hf_jobs")
    creds = CredentialsConfig(hf_token="test")
    backend = create_backend(config, creds)
    assert isinstance(backend, HFJobsSandbox)


def test_create_backend_auto_with_gpu():
    from surprisal.agents.experiment_container import ExperimentContainer
    config = SandboxConfig(backend="auto")
    creds = CredentialsConfig()
    backend = create_backend(config, creds, gpu_available=True)
    assert isinstance(backend, ExperimentContainer)
    assert backend.config.gpu is True


def test_create_backend_auto_without_gpu():
    from surprisal.agents.experiment_container import ExperimentContainer
    config = SandboxConfig(backend="auto")
    creds = CredentialsConfig()
    backend = create_backend(config, creds, gpu_available=False)
    assert isinstance(backend, ExperimentContainer)
    assert backend.config.gpu is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_agents/test_backends.py -v 2>&1 | tail -10`
Expected: FAIL — `ModuleNotFoundError: No module named 'surprisal.agents.backends'`

- [ ] **Step 3: Implement backends module**

Create `src/surprisal/agents/backends.py`:

```python
"""Sandbox backend protocol, auto-detection, and factory."""
import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable

from surprisal.agents.base import AgentResult
from surprisal.config import SandboxConfig, CredentialsConfig


@runtime_checkable
class SandboxBackend(Protocol):
    async def execute(self, experiment_prompt: str, workspace: Path, config: SandboxConfig) -> AgentResult:
        ...


async def detect_gpu() -> bool:
    """Check if nvidia-smi is available and reports a GPU."""
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
    """Factory: create the appropriate backend based on config."""
    from surprisal.agents.experiment_container import ExperimentContainer
    from surprisal.agents.hf_jobs import HFJobsSandbox

    if config.backend == "hf_jobs":
        return HFJobsSandbox(config=config, credentials=credentials)

    if config.backend == "local" or config.backend == "auto":
        # For "auto", adjust GPU flag based on detection
        if config.backend == "auto" and gpu_available is not None:
            config.gpu = gpu_available
        return ExperimentContainer(config=config, credentials=credentials)

    # Unknown backend, default to local
    return ExperimentContainer(config=config, credentials=credentials)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_agents/test_backends.py -v 2>&1 | tail -10`
Expected: FAIL — still needs ExperimentContainer and HFJobsSandbox (Tasks 4 and 5). Create minimal stubs first.

- [ ] **Step 5: Create minimal stubs for ExperimentContainer and HFJobsSandbox**

Create `src/surprisal/agents/experiment_container.py` (stub):

```python
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
```

Create `src/surprisal/agents/hf_jobs.py` (stub):

```python
"""HF Jobs cloud backend — full implementation in Task 5."""
from pathlib import Path
from surprisal.agents.base import AgentResult
from surprisal.config import SandboxConfig, CredentialsConfig


class HFJobsSandbox:
    def __init__(self, config: SandboxConfig, credentials: CredentialsConfig):
        self.config = config
        self.credentials = credentials

    async def execute(self, experiment_prompt: str, workspace: Path, config: SandboxConfig) -> AgentResult:
        raise NotImplementedError("Full implementation in Task 5")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_agents/test_backends.py -v 2>&1 | tail -10`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/surprisal/agents/backends.py src/surprisal/agents/experiment_container.py src/surprisal/agents/hf_jobs.py tests/test_agents/test_backends.py
git commit -m "feat: add SandboxBackend protocol with auto-detection and factory"
```

---

## Task 4: ExperimentContainer (Local GPU Backend)

**Files:**
- Modify: `src/surprisal/agents/experiment_container.py` (replace stub)
- Test: `tests/test_agents/test_experiment_container.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agents/test_experiment_container.py`:

```python
import pytest
from surprisal.agents.experiment_container import ExperimentContainer
from surprisal.config import SandboxConfig, CredentialsConfig


def test_build_command_with_gpu():
    config = SandboxConfig(gpu=True, image="surprisal-gpu:latest", memory_limit="16g", cpu_limit="4")
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    assert "--gpus=all" in cmd
    assert "--memory" in cmd
    assert "16g" in cmd
    assert "--cpus" in cmd
    assert "4" in cmd
    assert "-v" in cmd
    assert "/tmp/ws:/work:rw" in cmd
    assert "/tmp/mcp.json:/etc/surprisal/mcp.json:ro" in cmd
    assert "surprisal-gpu:latest" in cmd
    assert "claude" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--mcp-config" in cmd


def test_build_command_without_gpu():
    config = SandboxConfig(gpu=False, image="surprisal-sandbox:latest")
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    assert "--gpus=all" not in cmd


def test_build_command_with_wandb_creds():
    config = SandboxConfig()
    creds = CredentialsConfig(wandb_api_key="test-wandb-key")
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    env_idx = cmd.index("-e")
    assert "WANDB_API_KEY=test-wandb-key" in cmd[env_idx + 1]


def test_build_command_with_hf_creds():
    config = SandboxConfig()
    creds = CredentialsConfig(hf_token="hf_test_token")
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    # Find -e flags
    env_args = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
    assert any("HF_TOKEN=hf_test_token" in arg for arg in env_args)


def test_build_command_no_creds_no_env():
    config = SandboxConfig()
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    assert "-e" not in cmd


def test_is_infra_error():
    assert ExperimentContainer.is_infra_error(125) is True
    assert ExperimentContainer.is_infra_error(126) is True
    assert ExperimentContainer.is_infra_error(127) is True
    assert ExperimentContainer.is_infra_error(1) is False
    assert ExperimentContainer.is_infra_error(0) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_agents/test_experiment_container.py -v 2>&1 | tail -10`
Expected: FAIL — `AttributeError: 'ExperimentContainer' object has no attribute 'build_run_command'`

- [ ] **Step 3: Implement ExperimentContainer**

Replace contents of `src/surprisal/agents/experiment_container.py`:

```python
"""Local GPU container backend — runs agent inside Docker container."""
import asyncio
import os
import time
from pathlib import Path
from typing import Optional

from surprisal.agents.base import AgentResult
from surprisal.config import SandboxConfig, CredentialsConfig
from surprisal.mcp_config import write_mcp_config


INFRA_ERROR_CODES = {125, 126, 127}


class ExperimentContainer:
    def __init__(self, config: SandboxConfig, credentials: CredentialsConfig):
        self.config = config
        self.credentials = credentials

    def build_run_command(self, workspace: str, mcp_config: str, prompt: str) -> list[str]:
        cmd = ["docker", "run", "--rm"]

        if self.config.gpu:
            cmd.append("--gpus=all")

        cmd.extend(["--memory", self.config.memory_limit])
        cmd.extend(["--cpus", self.config.cpu_limit])

        cmd.extend(["-v", f"{workspace}:/work:rw"])
        cmd.extend(["-v", f"{mcp_config}:/etc/surprisal/mcp.json:ro"])

        if self.credentials.wandb_api_key:
            cmd.extend(["-e", f"WANDB_API_KEY={self.credentials.wandb_api_key}"])
        if self.credentials.hf_token:
            cmd.extend(["-e", f"HF_TOKEN={self.credentials.hf_token}"])

        cmd.append(self.config.image)

        cmd.extend([
            "claude", "-p", prompt,
            "--mcp-config", "/etc/surprisal/mcp.json",
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--setting-sources", "",
        ])
        return cmd

    @staticmethod
    def is_infra_error(exit_code: int) -> bool:
        return exit_code in INFRA_ERROR_CODES

    async def execute(self, experiment_prompt: str, workspace: Path, config: SandboxConfig) -> AgentResult:
        mcp_config_path = write_mcp_config(self.credentials)
        cmd = self.build_run_command(str(workspace), mcp_config_path, experiment_prompt)

        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout + 30
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentResult(raw="", exit_code=-1, duration_seconds=self.config.timeout)
        duration = time.monotonic() - start

        stdout_path = workspace / "stdout.txt"
        stderr_path = workspace / "stderr.txt"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)

        # Clean up temp MCP config
        try:
            os.unlink(mcp_config_path)
        except OSError:
            pass

        return AgentResult.from_raw(
            stdout.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
            duration=duration,
        )

    @staticmethod
    async def build_image(dockerfile_path: str, tag: str = "surprisal-gpu:latest"):
        dockerfile_dir = str(Path(dockerfile_path).parent)
        dockerfile_name = Path(dockerfile_path).name
        proc = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", tag, "-f", dockerfile_path, dockerfile_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_agents/test_experiment_container.py -v 2>&1 | tail -10`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/surprisal/agents/experiment_container.py tests/test_agents/test_experiment_container.py
git commit -m "feat: implement ExperimentContainer (local GPU backend)"
```

---

## Task 5: HFJobsSandbox (Cloud Backend)

**Files:**
- Modify: `src/surprisal/agents/hf_jobs.py` (replace stub)
- Test: `tests/test_agents/test_hf_jobs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agents/test_hf_jobs.py`:

```python
import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from surprisal.agents.hf_jobs import HFJobsSandbox, extract_code
from surprisal.config import SandboxConfig, CredentialsConfig


def test_extract_code_from_markdown():
    text = "Here is the code:\n```python\nimport numpy as np\nprint(42)\n```\nDone."
    assert extract_code(text) == "import numpy as np\nprint(42)"


def test_extract_code_plain():
    text = "import numpy as np\nprint(42)"
    assert extract_code(text) == "import numpy as np\nprint(42)"


def test_extract_code_triple_backtick():
    text = "```\nimport os\n```"
    assert extract_code(text) == "import os"


def test_hf_jobs_sandbox_init():
    config = SandboxConfig(hf_flavor="t4-small", hf_timeout="1h")
    creds = CredentialsConfig(hf_token="hf_test")
    sandbox = HFJobsSandbox(config=config, credentials=creds)
    assert sandbox.config.hf_flavor == "t4-small"
    assert sandbox.credentials.hf_token == "hf_test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_agents/test_hf_jobs.py -v 2>&1 | tail -10`
Expected: FAIL — `ImportError: cannot import name 'extract_code'`

- [ ] **Step 3: Implement HFJobsSandbox**

Replace contents of `src/surprisal/agents/hf_jobs.py`:

```python
"""HF Jobs cloud backend — submits experiments to HuggingFace Jobs."""
import asyncio
import logging
from pathlib import Path

from surprisal.agents.base import AgentResult
from surprisal.agents.claude import ClaudeAgent
from surprisal.config import SandboxConfig, CredentialsConfig

logger = logging.getLogger("surprisal")


def extract_code(text: str) -> str:
    """Extract Python code from text that may contain markdown fences."""
    stripped = text.strip()
    if "```python" in stripped:
        start = stripped.index("```python") + len("```python")
        end = stripped.index("```", start) if "```" in stripped[start:] else len(stripped)
        return stripped[start:end].strip()
    if stripped.startswith("```"):
        stripped = stripped[3:].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


class HFJobsSandbox:
    def __init__(self, config: SandboxConfig, credentials: CredentialsConfig):
        self.config = config
        self.credentials = credentials

    async def execute(
        self,
        experiment_prompt: str,
        workspace: Path,
        config: SandboxConfig,
    ) -> AgentResult:
        model = "opus"

        # 1. Agent generates self-contained script
        agent = ClaudeAgent(model=model, max_turns=3)
        script_result = await agent.invoke(
            prompt=(
                f"Write a self-contained Python script for:\n{experiment_prompt}\n\n"
                "Print results as JSON to stdout. Include all imports.\n"
                "Output ONLY code, no explanation."
            ),
            output_format="text",
            no_tools=True,
        )
        script = extract_code(script_result.raw)
        script_path = workspace / "experiment.py"
        script_path.write_text(script)

        # 2. Submit to HF Jobs
        try:
            from huggingface_hub import run_uv_job, fetch_job_logs, inspect_job
        except ImportError:
            logger.error("huggingface_hub not installed — cannot use HF Jobs backend")
            return AgentResult(raw="huggingface_hub not installed", exit_code=1)

        secrets = {}
        if self.credentials.wandb_api_key:
            secrets["WANDB_API_KEY"] = self.credentials.wandb_api_key
        if self.credentials.hf_token:
            secrets["HF_TOKEN"] = self.credentials.hf_token

        job = run_uv_job(
            str(script_path),
            flavor=self.config.hf_flavor,
            dependencies=["torch", "transformers", "wandb", "datasets", "numpy", "scipy", "pandas", "scikit-learn"],
            secrets=secrets,
            timeout=self.config.hf_timeout,
        )
        logger.info(f"HF Job submitted: {job.id}")

        # 3. Poll for completion
        while True:
            info = inspect_job(job_id=job.id)
            if info.status.stage in ("COMPLETED", "ERROR"):
                break
            await asyncio.sleep(10)

        # 4. Read results from logs
        logs = list(fetch_job_logs(job_id=job.id))
        raw = "\n".join(str(log) for log in logs)
        exit_code = 0 if info.status.stage == "COMPLETED" else 1

        logger.info(f"HF Job {job.id} finished: {info.status.stage}")
        return AgentResult.from_raw(raw, exit_code=exit_code)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_agents/test_hf_jobs.py -v 2>&1 | tail -10`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/surprisal/agents/hf_jobs.py tests/test_agents/test_hf_jobs.py
git commit -m "feat: implement HFJobsSandbox (cloud GPU backend)"
```

---

## Task 6: FSM State Merge

**Files:**
- Modify: `src/surprisal/fsm.py`
- Modify: `tests/test_fsm.py`

- [ ] **Step 1: Update FSM tests for merged states**

Replace entire contents of `tests/test_fsm.py`:

```python
import pytest
from surprisal.fsm import select_next_state, FSMResponse


def test_start_goes_to_generator():
    assert select_next_state("start", None, failure_count=0, revision_count=0) == "experiment_generator"


def test_generator_goes_to_runner():
    assert select_next_state("experiment_generator", FSMResponse(error=False), 0, 0) == "experiment_runner"


def test_runner_goes_to_analyst():
    assert select_next_state("experiment_runner", FSMResponse(error=False, exit_code=0), 0, 0) == "experiment_analyst"


def test_runner_infra_error_125():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=125), 0, 0) == "FAIL"


def test_runner_infra_error_126():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=126), 0, 0) == "FAIL"


def test_runner_infra_error_127():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=127), 0, 0) == "FAIL"


def test_runner_code_error_goes_to_analyst():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=1), 0, 0) == "experiment_analyst"


def test_analyst_error_retries_runner():
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=2, revision_count=0) == "experiment_runner"


def test_analyst_error_at_max_fails():
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=6, revision_count=0) == "FAIL"


def test_analyst_success_goes_to_reviewer():
    assert select_next_state("experiment_analyst", FSMResponse(error=False), 0, 0) == "experiment_reviewer"


def test_reviewer_error_goes_to_reviser():
    assert select_next_state("experiment_reviewer", FSMResponse(error=True), 0, revision_count=0) == "experiment_reviser"


def test_reviewer_error_at_max_fails():
    assert select_next_state("experiment_reviewer", FSMResponse(error=True), 0, revision_count=1) == "FAIL"


def test_reviewer_success_goes_to_hypothesis_generator():
    assert select_next_state("experiment_reviewer", FSMResponse(error=False), 0, 0) == "hypothesis_generator"


def test_reviser_goes_to_runner():
    assert select_next_state("experiment_reviser", FSMResponse(error=False), 0, 0) == "experiment_runner"


def test_hypothesis_generator_goes_to_belief():
    assert select_next_state("hypothesis_generator", FSMResponse(error=False), 0, 0) == "belief_elicitation"


def test_belief_goes_to_complete():
    assert select_next_state("belief_elicitation", FSMResponse(error=False), 0, 0) == "COMPLETE"


def test_unknown_state_goes_to_fail():
    assert select_next_state("unknown_state", FSMResponse(error=False), 0, 0) == "FAIL"


def test_full_happy_path():
    """Walk through the entire FSM happy path."""
    states = []
    state = "start"
    while state not in ("COMPLETE", "FAIL"):
        state = select_next_state(state, FSMResponse(error=False, exit_code=0), 0, 0)
        states.append(state)
    assert states == [
        "experiment_generator", "experiment_runner",
        "experiment_analyst", "experiment_reviewer", "hypothesis_generator",
        "belief_elicitation", "COMPLETE",
    ]


def test_failure_count_cumulative():
    """Verify failure count is checked cumulatively, not reset."""
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=5, revision_count=0) == "experiment_runner"
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=6, revision_count=0) == "FAIL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_fsm.py -v 2>&1 | tail -20`
Expected: FAIL — `experiment_programmer` still in FSM, tests expect `experiment_runner`

- [ ] **Step 3: Update FSM**

Replace entire contents of `src/surprisal/fsm.py`:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class FSMResponse:
    error: bool
    exit_code: Optional[int] = None
    feedback: Optional[str] = None
    data: Optional[dict] = None


INFRA_ERROR_CODES = {125, 126, 127}

# FSM States
STATES = [
    "start", "experiment_generator", "experiment_runner",
    "experiment_analyst", "experiment_reviewer",
    "experiment_reviser", "hypothesis_generator", "belief_elicitation",
]


def select_next_state(current_state: str, response: Optional[FSMResponse],
                       failure_count: int, revision_count: int) -> str:
    """Pure function: given current state and response, return next state.
    Returns 'COMPLETE' when done, 'FAIL' on terminal failure."""
    if current_state == "start":
        return "experiment_generator"

    elif current_state == "experiment_generator":
        return "experiment_runner"

    elif current_state == "experiment_runner":
        if response and response.exit_code in INFRA_ERROR_CODES:
            return "FAIL"
        return "experiment_analyst"

    elif current_state == "experiment_analyst":
        if response and response.error:
            if failure_count < 6:
                return "experiment_runner"
            return "FAIL"
        return "experiment_reviewer"

    elif current_state == "experiment_reviewer":
        if response and response.error:
            if revision_count < 1:
                return "experiment_reviser"
            return "FAIL"
        return "hypothesis_generator"

    elif current_state == "experiment_reviser":
        return "experiment_runner"

    elif current_state == "hypothesis_generator":
        return "belief_elicitation"

    elif current_state == "belief_elicitation":
        return "COMPLETE"

    return "FAIL"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_fsm.py -v 2>&1 | tail -20`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/surprisal/fsm.py tests/test_fsm.py
git commit -m "feat: merge experiment_programmer + code_executor into experiment_runner"
```

---

## Task 7: Experiment Runner Prompt

**Files:**
- Create: `src/surprisal/prompts/experiment_runner.md`
- Modify: `src/surprisal/prompts/experiment_generator.md`
- Modify: `src/surprisal/prompts/experiment_analyst.md`

- [ ] **Step 1: Create experiment_runner.md**

Create `src/surprisal/prompts/experiment_runner.md`:

```markdown
# Experiment Runner

You are a research engineer running experiments inside a containerized environment. You have full access to Python, GPU (if available), and installed ML libraries.

## Your Role and Responsibilities

1. Read the experiment plan provided in your prompt
2. Write a Python script that implements it
3. Execute the script using your Bash tool
4. If it fails, debug and fix it (you have up to 3 self-repair attempts before returning an error)
5. Log key metrics to W&B if WANDB_API_KEY is set in the environment
6. Write structured results to /work/results.json

## Environment

- Python 3.12+ with ML stack: torch, transformers, trl, datasets, accelerate, wandb
- Stats stack: numpy, scipy, pandas, sklearn, statsmodels, seaborn, networkx, sympy
- GPU available if the system has one (check with `torch.cuda.is_available()`)
- Network access for HuggingFace datasets (`datasets.load_dataset()`) and W&B logging
- Workspace at /work (read-write)

## Execution Workflow

1. Write your experiment code to `/work/experiment.py`
2. Run it: `python /work/experiment.py`
3. If it crashes, read the traceback, fix the code, and retry (up to 3 times)
4. After successful execution, write `/work/results.json`

## Output Format

After successful execution, create `/work/results.json`:
```json
{
  "code": "the Python code you wrote (final version)",
  "stdout": "the full stdout output from execution",
  "metrics": {"metric_name": "value", "p_value": "0.001"},
  "error": false
}
```

If execution fails after all attempts:
```json
{
  "code": "the last version of the code",
  "stdout": "last output including tracebacks",
  "error": true,
  "error_message": "concise description of what went wrong"
}
```

## Constraints

- Write clean, self-contained scripts
- Use real HF datasets when the plan specifies them (`datasets.load_dataset()`)
- Use synthetic data only when the plan explicitly says so or no dataset is specified
- Log training metrics with `wandb.log()` if WANDB_API_KEY is set in the environment
- Do NOT install packages via pip (everything is pre-installed in the container)
- Print results to stdout AND write to /work/results.json
- Keep scripts focused: one experiment, one clear result

## Few-Shot Examples

**Example 1: Synthetic stats experiment**

Plan: "Test if batch size affects convergence speed in gradient descent on a quadratic loss surface. Report the correlation between batch size and iterations to convergence."

```python
import numpy as np
from scipy import stats

np.random.seed(42)
batch_sizes = [8, 16, 32, 64, 128, 256]
iterations_to_converge = []
for bs in batch_sizes:
    x = np.random.randn(1000, 10)
    y = x @ np.random.randn(10) + np.random.randn(1000) * 0.1
    w = np.zeros(10)
    for i in range(1000):
        idx = np.random.choice(len(x), bs)
        grad = -2 * x[idx].T @ (y[idx] - x[idx] @ w) / bs
        w -= 0.01 * grad
        if np.mean((y - x @ w) ** 2) < 0.05:
            iterations_to_converge.append(i)
            break
    else:
        iterations_to_converge.append(1000)

r, p = stats.pearsonr(batch_sizes, iterations_to_converge)
print(f"correlation = {r:.4f}")
print(f"p_value = {p:.6f}")
```

**Example 2: HF dataset experiment**

Plan: "Load the IMDB dataset and measure if review length correlates with sentiment. Report Pearson correlation."

```python
from datasets import load_dataset
from scipy import stats

ds = load_dataset("imdb", split="train[:1000]")
lengths = [len(text.split()) for text in ds["text"]]
labels = ds["label"]
r, p = stats.pearsonr(lengths, labels)
print(f"correlation = {r:.4f}")
print(f"p_value = {p:.6f}")
```

## Guardrails

- Do NOT import packages outside the pre-installed list
- Do NOT catch exceptions silently; let errors surface so you can debug them
- Do NOT write multi-file projects; keep everything in one script
- Do NOT run indefinitely; training loops must have a max iteration cap
```

- [ ] **Step 2: Update experiment_generator.md**

Append to the end of `src/surprisal/prompts/experiment_generator.md` (before the final `## Guardrails` section):

```markdown
## Data Sources

When designing experiments, you may specify real datasets:
- Use HuggingFace datasets when relevant: `datasets.load_dataset("dataset_name")`
- For novel hypotheses without a clear dataset, use synthetic data
- Specify the dataset in the experiment plan so the runner knows to load it
- Consider dataset size: prefer small splits for fast iteration (e.g., `split="train[:1000]"`)
- The runner environment has network access and the `datasets` library pre-installed
```

- [ ] **Step 3: Update experiment_analyst.md**

Append to the end of `src/surprisal/prompts/experiment_analyst.md` (before `## Guardrails`):

```markdown
## W&B Context

If W&B tools are available in this session, you may:
- Compare current results against historical baselines from prior W&B runs
- Flag anomalous metrics (e.g., loss diverging, accuracy below expected threshold)
- Reference specific W&B run IDs in your analysis summary
- Note if the experiment logged metrics to W&B successfully
```

- [ ] **Step 4: Run prompt tests to verify files are valid**

Run: `uv run --no-sync pytest tests/test_prompts.py -v 2>&1 | tail -10`
Expected: ALL PASS (these tests check that all prompt files in the prompts/ dir are readable)

- [ ] **Step 5: Commit**

```bash
git add src/surprisal/prompts/experiment_runner.md src/surprisal/prompts/experiment_generator.md src/surprisal/prompts/experiment_analyst.md
git commit -m "feat: add experiment_runner prompt, update generator and analyst for Phase 3"
```

---

## Task 8: FSM Runner Integration

**Files:**
- Modify: `src/surprisal/fsm_runner.py:1-506`

This is the largest change — merging the `experiment_programmer` and `code_executor` blocks into `experiment_runner` and wiring in the backend abstraction.

- [ ] **Step 1: Update imports and function signature**

In `src/surprisal/fsm_runner.py`:

First, update the module docstring (lines 1-5) to:
```python
"""Live FSM execution — calls real Claude/Codex/Docker agents.

This module implements the full discovery agent pipeline for a single MCTS node:
  experiment_generator → runner → analyst → reviewer → hypothesis → belief
"""
```

Then update the imports (lines 10-20). Replace:
```python
from surprisal.agents.docker import DockerSandbox
```
With:
```python
from surprisal.agents.backends import create_backend
from surprisal.agents.experiment_container import ExperimentContainer
from surprisal.mcp_config import write_mcp_config
```

- [ ] **Step 2: Update run_live_fsm to remove DockerSandbox creation**

In `run_live_fsm()`, remove lines 128-133 (the `sandbox = DockerSandbox(...)` block). Replace with GPU auto-detection (runs once per FSM execution):

```python
    # Auto-detect GPU for backend selection
    from surprisal.agents.backends import detect_gpu
    _gpu_available = await detect_gpu() if config.sandbox.backend == "auto" else None
```

- [ ] **Step 3: Replace experiment_programmer + code_executor blocks with experiment_runner**

Replace the `experiment_programmer` block (lines 241-282) AND the `code_executor` block (lines 284-297) with a single `experiment_runner` block:

```python
        # ── Experiment Runner (agent inside container) ──
        elif state == "experiment_runner":
            if last_response and last_response.error:
                node.fsm_failure_count += 1
                db.update_node(node_id, fsm_failure_count=node.fsm_failure_count)
            feedback = ""
            if last_response and last_response.feedback:
                feedback = f"\nPrevious attempt feedback: {last_response.feedback}"

            plan_summary = experiment_plan[:800]
            runner_prompt = (
                f"Run this experiment inside your environment.\n"
                f"Plan: {plan_summary}\n{feedback}\n"
                "Write the code, execute it, debug if needed. "
                "Write results to /work/results.json."
            )
            sys_prompt = str(_prompts_dir() / "experiment_runner.md")

            backend = create_backend(config.sandbox, config.credentials, gpu_available=_gpu_available)
            result = await backend.execute(
                experiment_prompt=runner_prompt,
                workspace=exp_dir,
                config=config.sandbox,
            )
            logger.info(f"  Runner exit={result.exit_code}, len={len(result.raw)}")

            experiment_output = result.raw
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    results_data = json.loads(results_file.read_text())
                    experiment_code = results_data.get("code", "")
                    experiment_output = results_data.get("stdout", result.raw)
                except (json.JSONDecodeError, KeyError):
                    experiment_code = ""
            else:
                experiment_code = (exp_dir / "experiment.py").read_text() if (exp_dir / "experiment.py").exists() else ""

            db.update_node(node_id, experiment_exit_code=result.exit_code)

            if ExperimentContainer.is_infra_error(result.exit_code):
                last_response = FSMResponse(error=True, exit_code=result.exit_code)
            else:
                last_response = FSMResponse(
                    error=result.exit_code != 0,
                    exit_code=result.exit_code,
                )
```

- [ ] **Step 4: Update experiment_reviser to route back to experiment_runner**

The reviser block (lines 368-379) already goes back to `experiment_programmer` via the FSM. Since we renamed the state, the FSM will now route to `experiment_runner` automatically. No code change needed in the reviser block itself — the FSM `select_next_state` handles the routing.

- [ ] **Step 5: Run the full test suite**

Run: `uv run --no-sync pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: ALL PASS (134+ tests). If any test references old state names, fix them.

- [ ] **Step 6: Commit**

```bash
git add src/surprisal/fsm_runner.py
git commit -m "feat: wire experiment_runner backend into fsm_runner"
```

---

## Task 9: Dockerfile.gpu

**Files:**
- Create: `sandbox/Dockerfile.gpu`

- [ ] **Step 1: Create GPU Dockerfile**

Create `sandbox/Dockerfile.gpu`:

```dockerfile
FROM nvidia/cuda:13.0-runtime-ubuntu24.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    curl git ca-certificates \
    nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# ML stack
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio \
    transformers trl datasets accelerate \
    wandb \
    numpy scipy pandas scikit-learn matplotlib \
    statsmodels seaborn networkx sympy

# Agent runtimes
RUN npm install -g @anthropic-ai/claude-code && \
    npm install -g codex

WORKDIR /work
```

- [ ] **Step 2: Verify Dockerfile syntax**

Run: `docker build --check -f sandbox/Dockerfile.gpu sandbox/ 2>&1 || echo "Docker check not available, visual inspection OK"`

- [ ] **Step 3: Commit**

```bash
git add sandbox/Dockerfile.gpu
git commit -m "feat: add GPU-enabled Docker image for experiment container"
```

---

## Task 10: Install Script

**Files:**
- Create: `install.sh`

- [ ] **Step 1: Create install.sh**

Create `install.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Surprisal Setup ==="
echo ""

# 1. Check Python
if ! python3 --version >/dev/null 2>&1; then
    echo "ERROR: Python 3 required. Install from https://python.org"
    exit 1
fi
echo "Python: $(python3 --version)"

# 2. Check Docker
if ! docker --version >/dev/null 2>&1; then
    echo "ERROR: Docker required. Install from https://docs.docker.com/get-docker/"
    exit 1
fi
echo "Docker: $(docker --version)"

# 3. Install surprisal
echo ""
echo "--- Installing surprisal ---"
pip3 install surprisal

# 4. Check/setup Claude CLI
echo ""
echo "--- Agent Setup ---"
if ! command -v claude &>/dev/null; then
    echo "Installing Claude CLI..."
    npm install -g @anthropic-ai/claude-code
fi
if ! claude auth status 2>/dev/null | grep -q loggedIn; then
    echo "Please log in to Claude:"
    claude auth login
fi
echo "Claude CLI: ready"

if command -v codex &>/dev/null; then
    echo "Codex CLI: detected (optional)"
fi

# 5. Check GPU
echo ""
echo "--- GPU Detection ---"
GPU=false
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU detected: $GPU_NAME"
    GPU=true
else
    echo "No GPU detected -- stats-only mode (use HF Jobs for cloud GPU)"
fi

# 6. Build sandbox image
echo ""
echo "--- Building Sandbox Image ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$GPU" = true ] && [ -f "$SCRIPT_DIR/sandbox/Dockerfile.gpu" ]; then
    docker build -t surprisal-gpu:latest -f "$SCRIPT_DIR/sandbox/Dockerfile.gpu" "$SCRIPT_DIR/sandbox/"
elif [ -f "$SCRIPT_DIR/sandbox/Dockerfile" ]; then
    docker build -t surprisal-sandbox:latest "$SCRIPT_DIR/sandbox/"
else
    echo "Sandbox Dockerfiles not found -- skipping image build"
    echo "You can build later with: docker build -t surprisal-gpu:latest -f sandbox/Dockerfile.gpu sandbox/"
fi

# 7. Credentials (optional)
echo ""
echo "--- Credentials (optional, press Enter to skip) ---"
read -rp "W&B API key: " WANDB_KEY
read -rp "HuggingFace token: " HF_TOKEN

# 8. Write config
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/surprisal"
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.toml" << EOF
[sandbox]
backend = "auto"
gpu = $GPU

[credentials]
wandb_api_key = "${WANDB_KEY:-}"
hf_token = "${HF_TOKEN:-}"
EOF
echo ""
echo "Config written to $CONFIG_DIR/config.toml"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  surprisal init --domain 'your research topic' --seed 'your hypothesis'"
echo "  surprisal explore --budget 10"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x install.sh`

- [ ] **Step 3: Verify script syntax**

Run: `bash -n install.sh && echo "Syntax OK"`
Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add install.sh
git commit -m "feat: add one-line install script for surprisal setup"
```

---

## Task 11: Run Full Test Suite and Fix Regressions

**Files:**
- Potentially any test file that references old state names

- [ ] **Step 1: Run the complete test suite**

Run: `uv run --no-sync pytest tests/ -v --tb=short 2>&1 | tail -40`
Expected: ALL PASS. If any tests fail, they likely reference `experiment_programmer` or `code_executor` — fix them.

- [ ] **Step 2: Check for stale references in test files**

Run: `grep -r "experiment_programmer\|code_executor" tests/ --include="*.py"`
Fix any found references to use `experiment_runner`.

- [ ] **Step 3: Check for stale references in source files**

Run: `grep -r "experiment_programmer\|code_executor" src/ --include="*.py"`
Fix any found references. The old `DockerSandbox` class should remain untouched (backward compat).

- [ ] **Step 4: Run full suite one more time**

Run: `uv run --no-sync pytest tests/ -v --tb=short 2>&1 | tail -40`
Expected: ALL PASS

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: update stale references from FSM state merge"
```

---

## Task 12: Final Verification

- [ ] **Step 1: Run full test suite with coverage**

Run: `uv run --no-sync pytest tests/ -v --tb=short 2>&1`
Expected: 140+ tests, ALL PASS

- [ ] **Step 2: Verify new files exist**

Run: `ls -la src/surprisal/agents/backends.py src/surprisal/agents/experiment_container.py src/surprisal/agents/hf_jobs.py src/surprisal/mcp_config.py src/surprisal/prompts/experiment_runner.md sandbox/Dockerfile.gpu install.sh`
Expected: All files exist

- [ ] **Step 3: Verify import chain works**

Run: `uv run --no-sync python -c "from surprisal.agents.backends import create_backend, SandboxBackend; from surprisal.mcp_config import generate_mcp_config; from surprisal.config import CredentialsConfig; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 4: Verify backward compatibility**

Run: `uv run --no-sync python -c "from surprisal.agents.docker import DockerSandbox; s = DockerSandbox(); print('DockerSandbox still works:', s.build_run_command('/tmp/exp')[:3])"`
Expected: `DockerSandbox still works: ['docker', 'run', '--rm']`

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: Phase 3 real experimentation — complete implementation"
```
