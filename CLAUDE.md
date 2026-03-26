# Surprisal

MCTS with Bayesian surprise for open-ended scientific discovery.

## Architecture

Three layers:

1. `src/surprisal/mcts.py`
   Deterministic tree search: UCT, progressive widening, and backpropagation.
2. `src/surprisal/db.py`, `src/surprisal/exploration.py`, `src/surprisal/workspace.py`
   SQLite WAL persistence plus exploration and branch workspaces.
3. `src/surprisal/orchestrator.py`, `src/surprisal/fsm_runner.py`
   Async orchestration and the per-node experiment FSM.

## Agent Roles

| Role | Purpose |
| --- | --- |
| Experiment Generator | Search literature, identify gaps, propose a hypothesis and one executable plan |
| Experiment Runner | Implement and execute the plan inside the sandbox |
| Experiment Analyst | Check execution fidelity, metrics, and failure modes |
| Experiment Reviewer | Decide whether the evidence is valid enough to use |
| Experiment Reviser | Repair rejected plans without changing the hypothesis |
| Hypothesis Generator | Formalize the post-experiment hypothesis record |
| Belief Agent | Sample binary prior and posterior judgments for Bayesian surprise |

Claude is required for generator, hypothesis, and belief roles. If Codex is available it handles analyst, reviewer, and reviser roles; otherwise Claude handles those too.

Claude research sessions, code-analysis sessions, and runner sessions are persisted separately per branch in `sessions.json` and resumed automatically. Belief sampling forks from the persisted research session so repeated samples do not contaminate one another.

## Sandbox

The experiment runner operates in a bounded sandbox with:

- Python and Bash
- local filesystem access
- public network access
- HuggingFace datasets and models
- optional W&B logging if configured
- GPU access when available and enabled

The runtime is not restricted to synthetic-only micro-experiments. Plans may use real public datasets or models when they are the right instrument for the hypothesis.

The local Docker backend is the canonical path for the full runner contract. The HF Jobs backend is available for one-shot remote execution, but it is intentionally narrower.

## Commands

```bash
uv run surprisal init --domain "your research topic" --seed "your hypothesis"
uv run surprisal explore --budget 10 --concurrency 1
uv run surprisal status --tree
uv run surprisal export --top 5 --format md
```

Machine-readable output:

- `init`, `explore`, `status`, `resume`, `prune`, and `config` support `--json`
- `export` supports `--format json` and `--json`

`resume` resumes an exploration, not a per-agent conversational session.

## Configuration

Exploration state defaults to `~/.surprisal`.

Config is loaded from:

- `${SURPRISAL_HOME}/config.toml` when `SURPRISAL_HOME` is set
- `~/.surprisal/config.toml` when that file exists
- otherwise `${XDG_CONFIG_HOME:-~/.config}/surprisal/config.toml`

Show config:

```bash
uv run surprisal config --show
```

Live knobs are defined in `src/surprisal/config.py`. Removed or unsupported knobs should not be reintroduced without wiring them into runtime behavior and tests.

## Literature Search

The generator uses alphaxiv MCP when available and falls back to the HuggingFace Papers API otherwise.

Setup:

```bash
claude mcp add --transport http alphaxiv https://api.alphaxiv.org/mcp/v1
```

## Testing

```bash
uv run pytest tests/ -q --tb=short
```
