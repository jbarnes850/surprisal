# Phase 3: Real Experimentation

**Date:** 2026-03-26
**Status:** Draft
**Author:** Jarrod Barnes + Claude

## 1. Problem Statement

Surprisal currently runs synthetic-only experiments: 10-30 line Python scripts using simulated data inside a CPU-only, network-isolated Docker container. This limits the system to statistical hypothesis testing and prevents researchers from running real ML experiments with actual datasets, model training, and experiment tracking.

Phase 3 connects surprisal to real research infrastructure so researchers can dogfood it on their own work.

## 2. Design Goals

1. **W&B integration** -- agents log experiment metrics and query prior runs
2. **HuggingFace integration** -- agents load real datasets and access models
3. **GPU experiments** -- agents write and execute training scripts on local GPU hardware
4. **Cloud scale-out** -- HF Jobs as escape hatch for GPU beyond local capacity
5. **One-line install** -- new users go from zero to running in one command
6. **Backend abstraction** -- single interface for experiment execution regardless of compute backend

## 3. Architecture

### 3.1 Overview

The core change: the experiment agent runs inside a GPU-enabled Docker container with full tool access, replacing the current model where an agent writes code externally and a dumb container executes it.

```
Before (Phase 2):
  Claude subprocess -> writes code text -> saved to file -> Docker runs file -> stdout captured

After (Phase 3):
  surprisal orchestrator -> launches GPU Docker container ->
    Claude/Codex agent runs INSIDE container ->
      writes code, executes, debugs, logs to W&B, loads HF data ->
    structured results returned to orchestrator
```

### 3.2 Backend Abstraction

```
SandboxBackend (Protocol)
  |
  |- LocalGPUContainer  -- default, full agent inside docker run --gpus
  |- HFJobsSandbox       -- cloud GPU via run_uv_job(), training scale-out only
  +- (future: Modal, RunPod, etc.)
```

Both backends implement the same interface:

```python
class SandboxBackend(Protocol):
    async def execute(self, experiment_prompt: str, workspace: Path, config: SandboxConfig) -> AgentResult:
        """Send experiment prompt, get structured results back."""
        ...
```

**Auto-detection** (`backend = "auto"` in config):
- `nvidia-smi` succeeds -> LocalGPUContainer
- No local GPU -> LocalGPUContainer without `--gpus` (stats-only mode)
- User explicitly sets `backend = "hf_jobs"` -> HFJobsSandbox for cloud GPU

### 3.3 What Changes

| Component | Before | After |
|---|---|---|
| `DockerSandbox` | CPU-only, no network, runs Python file | `LocalGPUContainer`: GPU, network, agent runs inside |
| FSM states | `experiment_programmer` + `code_executor` (separate) | `experiment_runner` (merged, agent writes + executes) |
| `SandboxConfig` | memory, cpu, timeout, network | + gpu, image, backend, mcp_config_path |
| Agent prompts | Programmer outputs code text only | Runner prompt: write, execute, debug, log metrics |
| MCP configs | alphaxiv only (forwarded to generator) | + W&B, HF (shipped with surprisal, injected into container) |
| Config file | No credentials section | `[credentials]` section for API keys |
| Install | Manual setup | `install.sh` one-line bootstrap |

### 3.4 What Doesn't Change

- FSM state machine logic (pure function, just different state names)
- MCTS engine, Bayesian surprise, belief elicitation
- Database schema (no new columns needed)
- Export, CLI commands (init/explore/status/export/prune/config)
- Literature search (alphaxiv MCP + HF fallback)
- Agent prompt architecture (system prompt markdown files)

## 4. Local GPU Container (Primary Backend)

### 4.1 Docker Image

New `sandbox/Dockerfile.gpu`:

```dockerfile
FROM nvidia/cuda:13.0-runtime-ubuntu24.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl git nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# ML stack
RUN pip3 install --no-cache-dir \
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

The existing `sandbox/Dockerfile` (python:3.12-slim, stats-only) is preserved as the lightweight fallback for users without GPU.

### 4.2 Container Launch

`ExperimentContainer` replaces `DockerSandbox`:

```python
class ExperimentContainer:
    def __init__(self, config: SandboxConfig, credentials: CredentialsConfig):
        self.config = config
        self.credentials = credentials

    def build_run_command(self, workspace: str, mcp_config: str, prompt: str) -> list[str]:
        cmd = ["docker", "run", "--rm"]

        # GPU passthrough
        if self.config.gpu:
            cmd.append("--gpus=all")

        # Resources
        cmd.extend(["--memory", self.config.memory_limit])
        cmd.extend(["--cpus", self.config.cpu_limit])

        # Workspace + MCP config mount
        cmd.extend(["-v", f"{workspace}:/work:rw"])
        cmd.extend(["-v", f"{mcp_config}:/etc/surprisal/mcp.json:ro"])

        # Credentials as env vars (not baked into image)
        if self.credentials.wandb_api_key:
            cmd.extend(["-e", f"WANDB_API_KEY={self.credentials.wandb_api_key}"])
        if self.credentials.hf_token:
            cmd.extend(["-e", f"HF_TOKEN={self.credentials.hf_token}"])

        cmd.append(self.config.image)

        # Agent runs inside container
        cmd.extend([
            "claude", "-p", prompt,
            "--mcp-config", "/etc/surprisal/mcp.json",
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--setting-sources", "",
        ])
        return cmd
```

### 4.3 Agent Inside Container

The agent inside the container has:
- **Bash tool** -- write and execute Python scripts, install packages
- **Read/Write/Edit tools** -- modify files in /work
- **MCP tools** -- W&B logging, HF dataset loading (via injected mcp.json)
- **Network access** -- pip install, API calls to W&B/HF

The agent receives an experiment prompt and autonomously:
1. Reads the experiment plan
2. Writes Python code
3. Executes it
4. If it fails, reads the error and fixes it (self-debugging)
5. Logs metrics to W&B if configured
6. Returns structured JSON: `{code, stdout, metrics, error}`

This self-debugging capability means most failures are resolved without the FSM retry loop, reducing the current 6-attempt cycle.

## 5. HF Jobs Backend (Cloud Scale-Out)

### 5.1 When to Use

HF Jobs is opt-in for training workloads that exceed local GPU capacity. The user sets `backend = "hf_jobs"` in config or the experiment generator flags a job as GPU-intensive.

### 5.2 Execution Flow

```python
class HFJobsSandbox:
    async def execute(self, experiment_prompt: str, workspace: Path, config: SandboxConfig) -> AgentResult:
        # 1. Agent generates self-contained script (runs outside HF Jobs)
        agent = ClaudeAgent(model=agents_config.claude_model)
        script_result = await agent.invoke(
            prompt=f"Write a self-contained Python script for: {experiment_prompt}\n"
                   "Print results as JSON to stdout. Include all imports and deps.",
            output_format="text",
            no_tools=True,
        )
        script = extract_code(script_result.raw)
        script_path = workspace / "experiment.py"
        script_path.write_text(script)

        # 2. Submit to HF Jobs
        from huggingface_hub import run_uv_job, fetch_job_logs, inspect_job
        job = run_uv_job(
            str(script_path),
            flavor=config.hf_flavor,  # e.g. "a10g-small"
            dependencies=["torch", "transformers", "wandb", "datasets"],
            secrets={"WANDB_API_KEY": self.credentials.wandb_api_key},
            timeout=config.timeout,
        )

        # 3. Poll for completion
        while True:
            info = inspect_job(job_id=job.id)
            if info.status.stage in ("COMPLETED", "ERROR"):
                break
            await asyncio.sleep(10)

        # 4. Read results from logs
        logs = list(fetch_job_logs(job_id=job.id))
        return AgentResult.from_raw(
            "\n".join(logs),
            exit_code=0 if info.status.stage == "COMPLETED" else 1,
        )
```

### 5.3 Difference from Local

| | Local GPU Container | HF Jobs |
|---|---|---|
| Agent location | Inside container | Outside (generates script) |
| Self-debugging | Yes (agent has Bash tool) | No (agent reads error logs, resubmits) |
| GPU access | Local hardware | Cloud (T4 through 8xH200) |
| Network | Full | Full |
| Latency | Seconds (container start) | Minutes (job queue + cold start) |
| Cost | Free (user's hardware) | Pay-per-minute |

## 6. MCP Server Configs

### 6.1 Shipped Configs

surprisal ships MCP server configuration files that are injected into containers at runtime:

```
mcp/
  wandb.json        -- W&B MCP server config
  huggingface.json  -- HF MCP server config
  alphaxiv.json     -- alphaxiv MCP server config (existing)
  combined.json     -- merged config for container injection
```

### 6.2 Runtime Generation

`surprisal` generates the combined MCP config at runtime, injecting credentials from `surprisal.toml`:

```python
def generate_mcp_config(credentials: CredentialsConfig) -> dict:
    config = {"mcpServers": {}}

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
```

The generated config is written to a temp file and mounted into the container at `/etc/surprisal/mcp.json`.

### 6.3 Agent Tool Access

Inside the container, the Claude/Codex agent gains MCP tools:
- `mcp__wandb__log_run` -- log metrics to a W&B run
- `mcp__wandb__query_runs` -- query prior runs for comparison
- `mcp__huggingface__load_dataset` -- load HF datasets
- `mcp__huggingface__push_to_hub` -- push results/models

The exact tool names depend on the MCP server implementations. The agent prompts reference these capabilities without hardcoding tool names.

## 7. Configuration

### 7.1 surprisal.toml Changes

```toml
[sandbox]
backend = "auto"          # "auto" | "local" | "hf_jobs"
image = "surprisal-gpu:latest"  # Docker image for local backend
gpu = true                # enable --gpus all
memory_limit = "16g"      # container memory limit
cpu_limit = "4"           # container CPU limit
timeout = 1800            # 30 min default for GPU experiments
network = true            # enable network (W&B, HF Hub)

[sandbox.hf_jobs]
flavor = "a10g-small"     # HF Jobs GPU flavor
timeout = "2h"            # HF Jobs timeout

[credentials]
wandb_api_key = ""        # W&B API key (optional)
hf_token = ""             # HuggingFace token (optional)
```

### 7.2 Config Dataclass Changes

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

@dataclass
class AutoDiscoveryConfig:
    general: GeneralConfig
    mcts: MCTSConfig
    agents: AgentsConfig
    sandbox: SandboxConfig
    predictor: PredictorConfig
    credentials: CredentialsConfig  # NEW
```

## 8. FSM Changes

### 8.1 State Rename

```python
STATES = [
    "start",
    "experiment_generator",
    "experiment_runner",       # was: experiment_programmer + code_executor
    "experiment_analyst",
    "experiment_reviewer",
    "experiment_reviser",
    "hypothesis_generator",
    "belief_elicitation",
]
```

### 8.2 Transition Changes

```python
def select_next_state(current_state, response, failure_count, revision_count):
    ...
    elif current_state == "experiment_generator":
        return "experiment_runner"       # was: experiment_programmer

    elif current_state == "experiment_runner":  # was: code_executor
        if response and response.exit_code in INFRA_ERROR_CODES:
            return "FAIL"
        return "experiment_analyst"

    elif current_state == "experiment_analyst":
        if response and response.error:
            if failure_count < 6:
                return "experiment_runner"   # retry goes back to runner, not programmer
            return "FAIL"
        return "experiment_reviewer"
    ...
```

The FSM loses the `experiment_programmer -> code_executor` transition. `experiment_runner` handles both writing and executing code. On analyst rejection, it goes back to `experiment_runner` which re-prompts the agent with the error feedback.

### 8.3 fsm_runner.py Changes

The `experiment_programmer` and `code_executor` blocks in `run_live_fsm()` merge into a single `experiment_runner` block:

```python
elif state == "experiment_runner":
    backend = create_backend(config, credentials)
    prompt = build_runner_prompt(experiment_plan, feedback, config)
    result = await backend.execute(prompt, workspace=exp_dir, config=config.sandbox)

    experiment_output = result.raw
    experiment_code = (exp_dir / "experiment.py").read_text() if (exp_dir / "experiment.py").exists() else ""
    db.update_node(node_id, experiment_exit_code=result.exit_code)

    if result.exit_code in INFRA_ERROR_CODES:
        last_response = FSMResponse(error=True, exit_code=result.exit_code)
    else:
        last_response = FSMResponse(error=result.exit_code != 0, exit_code=result.exit_code)
```

## 9. Agent Prompts

### 9.1 New: experiment_runner.md

Replaces `experiment_programmer.md`. The runner prompt tells the agent to both write AND execute code:

```markdown
# Experiment Runner

You are a research engineer running experiments inside a containerized environment.
You have full access to Python, GPU (if available), and installed ML libraries.

## Your Role

1. Read the experiment plan
2. Write a Python script that implements it
3. Execute the script using your Bash tool
4. If it fails, debug and fix it (you have up to 3 self-repair attempts)
5. Log key metrics to W&B if WANDB_API_KEY is set
6. Return structured results

## Environment

- Python 3.12+ with: torch, transformers, trl, datasets, accelerate, wandb
- Stats: numpy, scipy, pandas, sklearn, statsmodels, seaborn
- GPU available if the system has one (check with torch.cuda.is_available())
- Network access for HuggingFace datasets and W&B logging
- Workspace at /work (read-write)

## Output Format

After successful execution, create /work/results.json:
{
  "code": "the Python code you wrote",
  "stdout": "execution output",
  "metrics": {"metric_name": value, ...},
  "error": false
}

If execution fails after all attempts:
{
  "code": "the last version of the code",
  "stdout": "last output",
  "error": true,
  "error_message": "what went wrong"
}

## Constraints

- Write clean, self-contained scripts
- Use real HF datasets when the plan specifies them (datasets.load_dataset())
- Use synthetic data only when the plan explicitly says so
- Log training metrics with wandb.log() if WANDB_API_KEY is available
- Do NOT install packages via pip inside the container (everything is pre-installed)
- Print results to stdout AND write to /work/results.json
```

### 9.2 Updated: experiment_analyst.md

Add W&B querying capability:

```markdown
## Additional Context (Phase 3)

If W&B tools are available, you may query prior experiment runs for comparison:
- Compare current results against historical baselines
- Flag anomalous metrics (e.g., loss diverging, accuracy below threshold)
- Reference specific W&B run IDs in your analysis
```

### 9.3 Updated: experiment_generator.md

Add HF dataset awareness:

```markdown
## Data Sources (Phase 3)

When designing experiments, you may specify real datasets:
- Use HuggingFace datasets when relevant: datasets.load_dataset("dataset_name")
- For novel hypotheses without a clear dataset, use synthetic data
- Specify the dataset in the experiment plan so the runner knows to load it
- Consider dataset size: prefer small/medium datasets for fast iteration
```

## 10. One-Line Install

### 10.1 install.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Surprisal Setup ==="

# 1. Check Python
python3 --version >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }

# 2. Check Docker
docker --version >/dev/null 2>&1 || { echo "Docker required: https://docs.docker.com/get-docker/"; exit 1; }

# 3. Install surprisal
pip3 install surprisal

# 4. Check/setup agent CLIs
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

if command -v codex &>/dev/null; then
    echo "Codex CLI detected (optional, Claude is primary)"
fi

# 5. Check GPU
echo ""
echo "--- GPU Detection ---"
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU detected: $GPU_NAME"
    GPU=true
else
    echo "No GPU detected -- stats-only mode"
    GPU=false
fi

# 6. Build sandbox image
echo ""
echo "--- Building Sandbox Image ---"
if [ "$GPU" = true ]; then
    docker build -t surprisal-gpu:latest -f sandbox/Dockerfile.gpu sandbox/
else
    docker build -t surprisal-sandbox:latest sandbox/
fi

# 7. Credentials (optional)
echo ""
echo "--- Credentials (optional, press Enter to skip) ---"
read -p "W&B API key: " WANDB_KEY
read -p "HuggingFace token: " HF_TOKEN

# 8. Write config
mkdir -p ~/.config/surprisal
cat > ~/.config/surprisal/config.toml << EOF
[sandbox]
backend = "auto"
gpu = $GPU

[credentials]
wandb_api_key = "${WANDB_KEY:-}"
hf_token = "${HF_TOKEN:-}"
EOF

echo ""
echo "=== Setup Complete ==="
echo "Run: surprisal init --domain 'your research topic'"
echo "Then: surprisal explore --budget 10"
```

### 10.2 Usage

```bash
curl -fsSL https://raw.githubusercontent.com/jbarnes850/surprisal/main/install.sh | bash
```

## 11. Backward Compatibility

- The existing `surprisal-sandbox:latest` image (stats-only, CPU) continues to work
- Users without GPU or Docker get degraded but functional behavior (stats-only experiments)
- The `experiment_programmer.md` prompt is replaced by `experiment_runner.md` but the old prompt stays in the repo for reference
- Database schema is unchanged -- no migration needed
- All 134 existing tests continue to pass (they mock the sandbox layer)
- CLI commands are unchanged

## 12. Testing Strategy

### 12.1 Unit Tests

- `test_experiment_container.py` -- ExperimentContainer builds correct docker commands
- `test_hf_jobs_sandbox.py` -- HFJobsSandbox submit/poll/parse flow (mocked)
- `test_sandbox_backend.py` -- backend auto-detection logic
- `test_mcp_config_generation.py` -- MCP config generation with various credential combos
- `test_fsm_merged_states.py` -- FSM transitions with merged experiment_runner state

### 12.2 Integration Tests

- Build GPU image and run a simple experiment end-to-end (requires GPU)
- Run stats-only experiment with the new runner flow (no GPU required)
- Verify W&B MCP config injection and agent tool access
- Verify HF dataset loading inside container

### 12.3 What Stays Mocked

- Actual Claude/Codex API calls (as today)
- Actual W&B/HF API calls (MCP servers mocked)
- HF Jobs submission (mocked with recorded responses)

## 13. Implementation Priority

1. **ExperimentContainer class** -- replaces DockerSandbox, adds GPU + network + agent-inside
2. **FSM merge** -- collapse programmer + executor into runner
3. **experiment_runner.md prompt** -- new agent prompt for merged execution
4. **MCP config generation** -- runtime MCP config with credential injection
5. **SandboxConfig + CredentialsConfig** -- config dataclass updates
6. **Dockerfile.gpu** -- GPU-enabled Docker image
7. **HFJobsSandbox class** -- cloud backend (can ship after local is working)
8. **install.sh** -- one-line bootstrap
9. **Updated prompts** -- generator (HF datasets), analyst (W&B querying)
10. **Tests** -- unit + integration for all new components

## 14. Open Questions

None -- all design decisions resolved during brainstorming:
- Backend: local GPU container (primary) + HF Jobs (scale-out)
- Sandbox model: agent runs inside container with full tool access
- FSM: merged programmer + executor into experiment_runner
- MCP: shipped configs, runtime credential injection
- Install: one-line curl script
