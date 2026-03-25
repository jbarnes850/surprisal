# surprisal

Open-ended scientific discovery via Bayesian surprise. Uses MCTS to explore a hypothesis tree, dispatching Claude and Codex agents to generate experiments, execute code in Docker sandboxes, and measure belief shifts.

Based on [AutoDiscovery (NeurIPS 2025)](https://github.com/allenai/autodiscovery) by Agarwal et al., with extensions for parallel heterogeneous agents and learned surprisal prediction.

## Quick start

```bash
# Install
git clone https://github.com/jbarnes850/autodiscovery && cd surprisal
uv sync

# Build the sandbox image (network-isolated Python environment)
docker build -t surprisal-sandbox:latest sandbox/

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

The system generated 20 hypotheses, ran experiments in Docker sandboxes, and identified 4 where the experimental evidence genuinely surprised the model -- these are the discoveries worth investigating further.

## How it works

Each MCTS iteration:

1. **Selection** -- UCT picks the most promising branch to explore
2. **Expansion** -- Claude generates a hypothesis and experiment plan
3. **Execution** -- Code is written, run in a Docker sandbox, analyzed, and reviewed by a multi-agent FSM
4. **Belief elicitation** -- The model is asked "is this hypothesis true?" before and after seeing evidence (n=30 samples each)
5. **Bayesian surprise** -- KL divergence between prior and posterior Beta distributions measures belief shift
6. **Backpropagation** -- Surprisal scores flow up the tree, guiding future exploration toward high-information regions

The system optimizes for **variance** -- it seeks hypotheses where the model genuinely doesn't know what to expect, following Shi & Evans (2023) finding that surprising research combinations correlate with scientific impact.

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
    +-- Agent Dispatcher
         +-- Claude (hypothesis generation, belief elicitation) via claude -p
         +-- Codex (code writing, analysis, review) via codex exec
         +-- Docker (sandboxed experiment execution, --network=none)
```

**Discovery agent FSM (per hypothesis node):**
```
generator -> programmer -> Docker executor -> analyst -> reviewer
                                                          |
                                               [reject] reviser -> programmer (retry, up to 6x)
                                               [approve] hypothesis_generator -> belief_elicitation (x60)
```

## Key concepts

- **Bayesian surprise**: D_KL(posterior || prior) -- how much experimental evidence shifts the model's beliefs about a hypothesis
- **Belief shift**: Did the expected belief cross the 0.5 decision boundary? (model changed its mind from "likely true" to "likely false" or vice versa)
- **Progressive widening**: Tree breadth grows as sqrt(visits), preventing premature width while allowing exploration to broaden over time
- **Virtual loss**: Parallel workers avoid selecting the same branch simultaneously via temporary visit count inflation
- **Heterogeneous agents**: Claude handles research/reasoning, Codex handles code/analysis -- cross-model feedback prevents echo chambers

## Requirements

- Python 3.12+
- [Claude CLI](https://claude.ai/install.sh) authenticated (`claude auth login`)
- [Codex CLI](https://github.com/openai/codex) authenticated
- Docker
- [`uv`](https://docs.astral.sh/uv/) package manager

## Configuration

```bash
uv run surprisal config --show
uv run surprisal config --set mcts.c_explore 2.0       # more exploration
uv run surprisal config --set mcts.belief_samples 30    # production (60 calls/node)
uv run surprisal config --set agents.claude_model opus  # use Opus for research
```

| Setting | Default | Description |
|---------|---------|-------------|
| `mcts.c_explore` | 1.414 | UCT exploration constant (higher = more exploration) |
| `mcts.belief_samples` | 30 | Samples per belief phase (total = 2x this per node) |
| `mcts.max_depth` | 30 | Maximum hypothesis tree depth |
| `mcts.dedup_interval` | 50 | Run deduplication every N nodes |
| `agents.claude_model` | opus | Model for Claude roles |
| `sandbox.timeout` | 600 | Docker execution timeout (seconds) |

## References

- Agarwal et al., [AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise](https://openreview.net/forum?id=kJqTkj2HhF) (NeurIPS 2025)
- Shi & Evans, [Surprising combinations of research contents and contexts are related to impact](https://www.nature.com/articles/s41467-023-36741-4) (Nature Communications 2023)
- [Surprisal-Guided Selection](https://arxiv.org/abs/2602.07670) (2026)

## License

MIT
