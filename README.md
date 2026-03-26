# Surprisal

MCTS with Bayesian surprise for open-ended scientific discovery.

Surprisal is inspired by AllenAI's [AutoDiscovery](https://github.com/allenai/autodiscovery) and the Surprisal-Guided Selection paper cited below. It explores a research domain by generating literature-grounded hypotheses, running bounded experiments in a sandbox with real tools and network access, and ranking branches by how much the evidence changes the model's beliefs.

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/jbarnes850/surprisal/main/install.sh | bash

uv run surprisal init \
  --domain "AI for scientific discovery" \
  --seed "LLM self-evaluation accuracy drops as task compositional depth increases"

uv run surprisal explore --budget 10 --concurrency 1
uv run surprisal status --tree
uv run surprisal export --top 5 --format md
```

## What it does

Each expansion runs a per-node FSM:

1. `experiment_generator`: Claude searches recent literature and proposes one hypothesis plus one executable plan.
2. `experiment_runner`: a sandbox backend executes the plan with Python, Bash, local files, public network access, HuggingFace resources, and optional W&B logging.
3. `experiment_analyst`: Codex or Claude reviews the execution for fidelity and validity.
4. `experiment_reviewer`: Codex or Claude decides whether the evidence is usable.
5. `experiment_reviser`: if needed, the plan is revised and retried within configured bounds.
6. `hypothesis_generator`: Claude formalizes the post-experiment hypothesis record.
7. `belief_elicitation`: Claude samples prior and posterior binary judgments and Surprisal computes Bayesian surprise.

The deterministic MCTS layer never calls LLMs directly. It only consumes node state and reward signals.

## Runtime model

- Claude is required for research-facing roles: generator, hypothesis formalization, and belief elicitation.
- If Codex is available, it handles analysis, review, and revision roles.
- If Codex is not available, Claude handles all roles.
- Agent sessions persist per branch in `sessions.json`: Claude research sessions, code-analysis sessions, and runner sessions are tracked separately and resumed automatically across nodes on the same branch.
- Belief elicitation forks from the persisted research session instead of mutating it, so prior and posterior samples stay independent while still inheriting branch context.
- Experiment execution uses the configured sandbox backend:
  - `local`: Docker-based local sandbox with the full runner contract
  - `hf_jobs`: one-shot Hugging Face Jobs execution path for batch runs
  - `auto`: local backend with GPU autodetection

## Commands

| Command | Purpose | Machine-readable output |
| --- | --- | --- |
| `surprisal init` | Create or reuse an exploration for a domain | `--json` |
| `surprisal explore` | Run exploration on the latest or a specific exploration | `--json` |
| `surprisal status` | Show exploration summary and optional tree | `--json` |
| `surprisal export` | Export results as markdown, CSV, JSON, or JSONL training data | `--format json` or `--json` |
| `surprisal resume` | Alias for `explore` against the latest or a specific exploration | `--json` |
| `surprisal prune` | Mark low-value branches as pruned | `--json` |
| `surprisal config` | Show, set, or reset config | `--json` |

`resume` resumes an exploration, not a per-agent conversational session.

## Architecture

Three layers:

1. `src/surprisal/mcts.py`
   Deterministic tree policy, UCT scoring, progressive widening, and backpropagation.
2. `src/surprisal/db.py`, `src/surprisal/exploration.py`, `src/surprisal/workspace.py`
   SQLite WAL persistence plus per-branch workspaces.
3. `src/surprisal/orchestrator.py`, `src/surprisal/fsm_runner.py`
   Async worker orchestration and the multi-agent experiment FSM.

Key files:

- `src/surprisal/fsm_runner.py`: per-node live FSM
- `src/surprisal/orchestrator.py`: worker pool, selection, branching, and dedup scheduling
- `src/surprisal/bayesian.py`: Beta posterior updates and belief-shift scoring
- `src/surprisal/prompts/`: prompt contracts for generator, runner, analyst, reviewer, reviser, and belief stages

## Configuration

Exploration state defaults to `~/.surprisal`.

Config is loaded from:

- `${SURPRISAL_HOME}/config.toml` when `SURPRISAL_HOME` is set
- `~/.surprisal/config.toml` when that file exists
- otherwise `${XDG_CONFIG_HOME:-~/.config}/surprisal/config.toml`

Show the active config:

```bash
uv run surprisal config --show
```

Live config knobs:

| Setting | Default | Description |
| --- | --- | --- |
| `general.default_budget` | `100` | Default exploration budget |
| `general.default_concurrency` | `2` | Default worker count |
| `mcts.c_explore` | `1.414` | UCT exploration constant |
| `mcts.k_progressive` | `1.0` | Progressive widening coefficient |
| `mcts.alpha_progressive` | `0.5` | Progressive widening exponent |
| `mcts.max_depth` | `30` | Maximum tree depth |
| `mcts.belief_samples` | `30` | Samples per prior and posterior belief phase |
| `mcts.virtual_loss` | `2` | Virtual loss applied during parallel selection |
| `mcts.dedup_interval` | `50` | Run deduplication every N completed expansions |
| `agents.claude_model` | `opus` | Claude model for research roles |
| `agents.codex_model` | `gpt-5.4` | Codex model for analysis, review, and revision roles |
| `agents.max_turns` | `20` | Max Claude turns per invocation |
| `agents.code_attempts` | `6` | Total runner attempts before failure |
| `agents.revision_attempts` | `1` | Total plan revisions after rejection |
| `agents.generator_timeout` | `180` | Generator timeout in seconds |
| `sandbox.backend` | `auto` | `auto`, `local`, or `hf_jobs` (`local` is the canonical research-grade path) |
| `sandbox.image` | `surprisal-gpu:latest` | Local sandbox image |
| `sandbox.gpu` | `true` | Enable GPU passthrough for the local sandbox |
| `sandbox.memory_limit` | `16g` | Local sandbox memory limit |
| `sandbox.cpu_limit` | `4` | Local sandbox CPU limit |
| `sandbox.timeout` | `1800` | Sandbox timeout in seconds |
| `sandbox.network` | `true` | Allow public network access in the sandbox |
| `sandbox.hf_flavor` | `a10g-small` | HF Jobs hardware flavor |
| `sandbox.hf_timeout` | `2h` | HF Jobs timeout |
| `credentials.wandb_api_key` | `""` | Optional W&B API key |
| `credentials.hf_token` | `""` | Optional HuggingFace token |

## Literature grounding

The generator prefers alphaxiv MCP when available and falls back to the HuggingFace Papers API otherwise.

One-time alphaxiv setup:

```bash
claude mcp add --transport http alphaxiv https://api.alphaxiv.org/mcp/v1
```

Each hypothesis stores the papers that motivated it.

## Validation

Run the test suite:

```bash
uv run pytest tests/ -q --tb=short
```

## References

- Agarwal et al., [AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise](https://openreview.net/forum?id=kJqTkj2HhF)
- Shi and Evans, [Surprising combinations of research contents and contexts are related to impact](https://www.nature.com/articles/s41467-023-36741-4)
- Barnes et al., [Surprisal-Guided Selection](https://arxiv.org/abs/2602.07670)

## License

MIT
