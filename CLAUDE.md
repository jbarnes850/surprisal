# Surprisal

MCTS with Bayesian surprise for open-ended scientific discovery.

## Architecture

Three layers:
1. **MCTS Engine** (`mcts.py`) — deterministic search, never calls LLMs directly
2. **Persistence** (`db.py`, `workspace.py`, `exploration.py`) — SQLite WAL + filesystem
3. **Agent Dispatcher** (`fsm_runner.py`, `orchestrator.py`) — spawns Claude and Codex agents for hypothesis generation, code writing, experiment execution, and belief elicitation

## Agent Roles

| Role | Purpose |
|------|---------|
| Experiment Generator | Searches literature, identifies gaps, proposes hypotheses |
| Experiment Programmer | Writes Python code to test hypotheses |
| Code Executor | Runs code in a network-isolated Docker sandbox |
| Experiment Analyst | Analyzes execution output, provides feedback on failures |
| Experiment Reviewer | Approves or rejects experiments based on result validity |
| Experiment Reviser | Revises rejected experiment plans |
| Hypothesis Generator | Formalizes experimental results into structured hypotheses |
| Belief Agent | Evaluates hypothesis truth before and after evidence (Bayesian surprise) |

## Key Files

- `src/surprisal/fsm_runner.py` — multi-agent FSM pipeline per hypothesis node
- `src/surprisal/orchestrator.py` — async worker pool with parallel MCTS
- `src/surprisal/mcts.py` — UCT, progressive widening, backpropagation
- `src/surprisal/bayesian.py` — Beta distribution estimation, KL divergence, belief shift detection
- `src/surprisal/prompts/` — system prompts for all 8 agent roles

## Testing

```bash
uv run pytest tests/ -q --tb=short
```

## Running

```bash
uv run surprisal init --domain "your research topic" --seed "your hypothesis"
uv run surprisal explore --budget 10 --concurrency 1
uv run surprisal status --tree
uv run surprisal export --top 5 --format md
```

## Literature Search

The experiment generator searches for recent papers via [alphaxiv MCP](https://www.alphaxiv.org/docs/mcp) (semantic search) or the [HuggingFace Papers API](https://huggingface.co/docs/hub/api#papers) (public fallback). Each hypothesis tracks which papers informed it.

Setup (one-time):
```bash
claude mcp add --transport http alphaxiv https://api.alphaxiv.org/mcp/v1
```
