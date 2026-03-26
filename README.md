<p align="center">
  <img src="public/banner.jpg" alt="Surprisal: open-ended scientific discovery via Bayesian surprise" width="100%">
</p>

<p align="center">
  <a href="https://github.com/jbarnes850/surprisal/actions"><img src="https://img.shields.io/github/actions/workflow/status/jbarnes850/surprisal/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="https://github.com/jbarnes850/surprisal/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python 3.12+"></a>
  <a href="https://arxiv.org/abs/2602.07670"><img src="https://img.shields.io/badge/arXiv-2602.07670-b31b1b?style=flat-square" alt="arXiv"></a>
</p>

Describe what you're curious about. Get back ranked scientific discoveries.

Surprisal takes a research question in plain English, autonomously generates hypotheses grounded in recent literature, writes and executes experiments on real data and models, and ranks discoveries by how much they shift the model's own beliefs.

```bash
curl -fsSL https://raw.githubusercontent.com/jbarnes850/surprisal/main/install.sh | bash
surprisal init --domain "neural scaling laws" --seed "your hypothesis here"
surprisal explore --budget 20
surprisal export --top 5
```

## Quick start

**One-line install** (handles dependencies, GPU detection, Docker image, and credentials):

```bash
curl -fsSL https://raw.githubusercontent.com/jbarnes850/surprisal/main/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/jbarnes850/surprisal && cd surprisal
uv sync

# Build the GPU-enabled sandbox image
docker build -t surprisal-gpu:latest -f sandbox/Dockerfile.gpu sandbox/

# Initialize an exploration
uv run surprisal init \
  --domain "AI for scientific discovery" \
  --seed "LLM self-evaluation accuracy correlates inversely with task compositional depth"

# Run MCTS exploration (budget = number of hypothesis nodes to expand)
uv run surprisal explore --budget 10 --concurrency 1

# Check the hypothesis tree
uv run surprisal status --tree

# Export ranked discoveries
uv run surprisal export --top 5 --format md
```

## Example

```bash
uv run surprisal init \
  --domain "neural scaling laws across modalities" \
  --seed "Vision transformer scaling exponents differ from language model scaling exponents on equivalent compute budgets"

uv run surprisal explore --budget 20 --concurrency 2

uv run surprisal status --tree
# Exploration: a3f7e2 (neural scaling laws across modalities)
# Nodes: 21 total, 18 verified, 0 expanding, 3 failed
# Surprisals: 4 found (22.2% rate)
# Tree depth: max 5
#
#  [0] (verified) Vision transformer scaling exponents differ from language...
#    [1] (verified) Log-linear fits to simulated ViT loss curves show steeper... BS=2.31 SHIFTED!
#      [2] (verified) The scaling exponent gap widens when attention head count...
#      [2] (verified) Cross-modal transfer learning efficiency follows a power... BS=1.87 SHIFTED!
#    [1] (verified) Compute-optimal model size ratios (Chinchilla) transfer...

uv run surprisal export --top 3 --format json | jq '.hypotheses[].hypothesis'
# "Log-linear fits to simulated ViT loss curves show steeper scaling exponents..."
# "Cross-modal transfer learning efficiency follows a power law..."
# "Attention head scaling exhibits phase transitions at critical compute thresholds..."
```

The system generated 20 hypotheses, ran experiments in GPU-accelerated containers, and identified 4 where the experimental evidence genuinely surprised the model. These are the discoveries worth investigating further.

## How it works

Each MCTS iteration:

1. **Selection.** UCT picks the most promising branch to explore.
2. **Literature search.** The generator searches recent papers (via alphaxiv or HuggingFace Papers API) to identify gaps and ground new hypotheses in the literature.
3. **Experiment design.** Claude proposes a hypothesis and experiment plan, citing the papers that motivated it.
4. **Execution.** An agent runs inside a GPU-enabled Docker container, writing and executing code against real HuggingFace datasets or synthetic data. The agent self-debugs failures and logs metrics to W&B if configured.
5. **Analysis and review.** A separate agent analyzes the output, comparing against prior W&B runs where available. A reviewer validates the results.
6. **Belief elicitation.** The model is asked "is this hypothesis true?" before and after seeing evidence (n=30 samples each).
7. **Bayesian surprise.** KL divergence between prior and posterior Beta distributions measures belief shift.
8. **Backpropagation.** Surprisal scores flow up the tree, guiding future exploration toward high-information regions.

The system optimizes for **variance**: it seeks hypotheses where the model genuinely does not know what to expect, following Shi & Evans (2023) on the relationship between surprising research combinations and scientific impact.

## Commands

| Command | Description |
|---------|-------------|
| `surprisal init` | Create a new exploration |
| `surprisal explore` | Run MCTS exploration |
| `surprisal status` | Show tree state and hypothesis tree |
| `surprisal export` | Export ranked hypotheses (JSON, CSV, markdown, training data) |
| `surprisal prune` | Remove low-value branches (`--dry-run` supported) |
| `surprisal config` | Manage settings |
| `surprisal resume` | Resume a branch or exploration |

All commands accept `--json` for machine-readable output. All commands are idempotent.

## Architecture

```
MCTS Engine (deterministic Python, never calls LLMs directly)
    |
    +-- Persistence (SQLite WAL + filesystem workspaces)
    |
    +-- Backend Abstraction (SandboxBackend protocol)
    |    +-- LocalGPUContainer (default: Claude agent runs inside Docker with GPU)
    |    +-- HF Jobs (cloud scale-out for training beyond local GPU)
    |
    +-- Agent Dispatcher
    |    +-- Claude Opus: hypothesis generation, literature search, belief elicitation
    |    +-- Codex or Claude Sonnet: experiment analysis, review, revision
    |    +-- Claude (inside container): experiment execution with self-debugging
    |
    +-- MCP Tools (injected into containers at runtime)
         +-- W&B (experiment tracking and run comparison)
         +-- HuggingFace (dataset access, model hub)
         +-- alphaxiv (semantic paper search)
```

**Discovery agent FSM (per hypothesis node):**
```
generator (Opus) -> runner (agent inside container) -> analyst (Codex/Sonnet) -> reviewer (Codex/Sonnet)
                                                                                    |
                                                                         [reject] reviser (Codex/Sonnet) -> runner (retry, up to 6x)
                                                                         [approve] hypothesis_generator (Opus) -> belief_elicitation (Opus, x60)
```

## Key concepts

- **Bayesian surprise.** D_KL(posterior || prior), measuring how much experimental evidence shifts the model's beliefs about a hypothesis.
- **Belief shift.** Whether the expected belief crosses the 0.5 decision boundary. The model changed its mind from "likely true" to "likely false" or vice versa.
- **Progressive widening.** Tree breadth grows as sqrt(visits), preventing premature width while allowing exploration to broaden over time.
- **Virtual loss.** Parallel workers avoid selecting the same branch simultaneously via temporary visit count inflation.
- **Literature grounding.** Every hypothesis cites the recent papers that motivated it. The generator searches alphaxiv (semantic search) or HuggingFace Papers API (public fallback) for 2024-2026 work.
- **Agent-in-container execution.** The experiment runner operates inside a GPU-enabled Docker container with full tool access: Bash, file I/O, MCP tools for W&B and HuggingFace. It writes code, executes it, and self-debugs failures before returning results.

## Requirements

- Python 3.12+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) authenticated (`claude auth login`)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU experiments)
- [`uv`](https://docs.astral.sh/uv/) package manager

Optional:

- [Codex CLI](https://github.com/openai/codex) for heterogeneous agent mode (Codex handles analysis/review, Claude handles research/belief)
- [W&B](https://wandb.ai/) account for experiment tracking
- [HuggingFace](https://huggingface.co/) token for dataset access and HF Jobs cloud GPU
- [alphaxiv MCP](https://www.alphaxiv.org/docs/mcp) for semantic paper search

## Configuration

```bash
uv run surprisal config --show
uv run surprisal config --set mcts.c_explore 2.0         # more exploration
uv run surprisal config --set mcts.belief_samples 30      # production (60 calls/node)
uv run surprisal config --set agents.claude_model opus    # use Opus for research
uv run surprisal config --set sandbox.gpu true            # enable GPU passthrough
uv run surprisal config --set credentials.wandb_api_key YOUR_KEY
uv run surprisal config --set credentials.hf_token YOUR_TOKEN
```

| Setting | Default | Description |
|---------|---------|-------------|
| `mcts.c_explore` | 1.414 | UCT exploration constant (higher = more exploration) |
| `mcts.belief_samples` | 30 | Samples per belief phase (total = 2x this per node) |
| `mcts.max_depth` | 30 | Maximum hypothesis tree depth |
| `mcts.dedup_interval` | 50 | Run deduplication every N nodes |
| `agents.claude_model` | opus | Model for Claude roles |
| `sandbox.backend` | auto | Experiment backend: `auto`, `local`, or `hf_jobs` |
| `sandbox.gpu` | true | Enable GPU passthrough for local container |
| `sandbox.timeout` | 1800 | Container execution timeout (seconds) |
| `sandbox.hf_flavor` | a10g-small | HF Jobs GPU flavor (when using cloud backend) |
| `credentials.wandb_api_key` | | W&B API key for experiment tracking |
| `credentials.hf_token` | | HuggingFace token for datasets and HF Jobs |

## References

- Agarwal et al., [AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise](https://openreview.net/forum?id=kJqTkj2HhF) (NeurIPS 2025)
- Shi & Evans, [Surprising combinations of research contents and contexts are related to impact](https://www.nature.com/articles/s41467-023-36741-4) (Nature Communications 2023)
- [Surprisal-Guided Selection](https://arxiv.org/abs/2602.07670) (2026)

## License

MIT
