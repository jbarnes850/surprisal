# autodiscovery

Open-ended scientific discovery via Bayesian surprise. Uses MCTS to explore a hypothesis tree, dispatching Claude and Codex agents to generate experiments, execute code in Docker sandboxes, and measure belief shifts.

Based on [AutoDiscovery (NeurIPS 2025)](https://github.com/allenai/autodiscovery) by Agarwal et al.

## Quick start

```bash
# Install
cd ~/autodiscovery && uv sync

# Build the sandbox image
docker build -t autodiscovery-sandbox:latest sandbox/

# Initialize an exploration
uv run autodiscovery init \
  --domain "AI for scientific discovery" \
  --seed "LLM self-evaluation accuracy correlates inversely with task compositional depth"

# Run MCTS exploration (budget = number of hypothesis nodes to expand)
uv run autodiscovery explore --budget 10 --concurrency 1

# Check status
uv run autodiscovery status --tree

# Export top hypotheses
uv run autodiscovery export --top 5 --format md
```

## How it works

Each MCTS iteration:

1. **Selection** -- UCT picks the most promising branch to explore
2. **Expansion** -- Claude generates a hypothesis and experiment plan
3. **Execution** -- Code is written, run in a Docker sandbox, analyzed, and reviewed
4. **Belief elicitation** -- Claude is asked "is this hypothesis true?" before and after seeing evidence
5. **Bayesian surprise** -- KL divergence between prior and posterior beliefs measures how much the evidence shifted the model's mind
6. **Backpropagation** -- Surprisal scores flow up the tree, guiding future exploration

The system optimizes for **variance** -- it seeks hypotheses where the model genuinely doesn't know what to expect.

## Commands

| Command | Description |
|---------|-------------|
| `autodiscovery init` | Create a new exploration |
| `autodiscovery explore` | Run MCTS exploration |
| `autodiscovery status` | Show tree state and hypothesis tree |
| `autodiscovery export` | Export ranked hypotheses (JSON, CSV, markdown) |
| `autodiscovery prune` | Remove low-value branches (`--dry-run` supported) |
| `autodiscovery config` | Manage settings |
| `autodiscovery resume` | Resume a branch or exploration |

All commands accept `--json` for machine-readable output.

## Architecture

```
MCTS Engine (deterministic Python, never calls LLMs)
    |
    +-- Persistence (SQLite WAL + filesystem workspaces)
    |
    +-- Agent Dispatcher
         +-- Claude (hypothesis generation, belief elicitation) via claude -p
         +-- Codex (code writing, analysis, review) via codex exec
         +-- Docker (sandboxed experiment execution, --network=none)
```

**Agent FSM per node:**
```
generator -> programmer -> Docker executor -> analyst -> reviewer
                                                          |
                                               [reject] reviser -> programmer (retry)
                                               [approve] hypothesis_generator -> belief (x60)
```

## Key concepts

- **Bayesian surprise**: D_KL(posterior || prior) -- how much experimental evidence shifts the model's beliefs
- **Belief shift**: Binary indicator -- did the expected belief cross the 0.5 decision boundary?
- **Progressive widening**: Tree breadth grows as sqrt(visits), balancing depth vs breadth
- **Virtual loss**: Parallel workers avoid selecting the same branch simultaneously
- **Surprisal predictor** (Phase 2): Learn to predict surprise before running experiments

## Requirements

- Python 3.12+
- Claude CLI authenticated (`claude auth login`) with Max subscription
- Codex CLI authenticated (Pro plan)
- Docker (for sandbox execution)
- `uv` package manager

## Config

```bash
uv run autodiscovery config --show          # view all settings
uv run autodiscovery config --set mcts.c_explore 2.0      # more exploration
uv run autodiscovery config --set mcts.belief_samples 30   # production belief elicitation
uv run autodiscovery config --set agents.claude_model opus  # use Opus for research
```

Key settings:
- `mcts.c_explore` -- UCT exploration constant (higher = more exploration, default sqrt(2))
- `mcts.belief_samples` -- samples per belief phase (30 = 60 total Claude calls per node)
- `mcts.max_depth` -- maximum tree depth (default 30)
- `agents.claude_model` -- model for Claude roles (sonnet or opus)

## Spec

Full design specification with algorithm details, data model, and FSM transitions:
`/home/jarrodbarnes/docs/superpowers/specs/2026-03-25-autodiscovery-design.md`
