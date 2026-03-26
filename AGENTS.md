# Surprisal

MCTS with Bayesian surprise for open-ended scientific discovery.

## Architecture

Three layers:

1. `src/surprisal/mcts.py`
   Deterministic tree policy, UCT scoring, progressive widening, and backpropagation.
2. `src/surprisal/db.py`, `src/surprisal/exploration.py`, `src/surprisal/workspace.py`
   SQLite WAL persistence plus per-branch workspaces and session state.
3. `src/surprisal/orchestrator.py`, `src/surprisal/fsm_runner.py`
   Async worker orchestration and the live multi-agent FSM.

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

## Runtime Notes

- The local Docker backend is the canonical path for the full runner contract.
- Runner plans may use real public datasets, models, public network access, and optional W&B logging when that is the right instrument for the hypothesis.
- Agent sessions persist per branch in `sessions.json`:
  research Claude sessions, code-analysis sessions, and runner sessions are tracked separately.
- Belief sampling forks from the persisted research session so repeated samples remain independent.

## Commands

```bash
uv run surprisal init --domain "your research topic" --seed "your hypothesis"
uv run surprisal explore --budget 10 --concurrency 1
uv run surprisal status --tree
uv run surprisal export --top 5 --format md
```

`resume` resumes an exploration, not a per-agent conversational session.

## Literature Search

The generator uses alphaxiv MCP when available and falls back to the HuggingFace Papers API otherwise.

```bash
claude mcp add --transport http alphaxiv https://api.alphaxiv.org/mcp/v1
```

## Testing

```bash
uv run pytest tests/ -q --tb=short
```
