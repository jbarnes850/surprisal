# Surprisal

Package name: `surprisal` (renamed from `autodiscovery` to avoid confusion with AI2's AutoDiscovery)

## Architecture

MCTS with Bayesian surprise for open-ended scientific discovery. Three layers:
1. **MCTS Engine** (`mcts.py`) — deterministic, never calls LLMs
2. **Persistence** (`db.py`, `workspace.py`, `exploration.py`) — SQLite WAL + filesystem
3. **Agent Dispatcher** (`fsm_runner.py`, `orchestrator.py`) — spawns Claude/Codex as subprocesses

## CLI subprocess gotchas

When spawning `claude -p` as subprocess:
- `--bare` breaks auth on this machine (Vercel plugin hooks are part of auth flow)
- Use `--setting-sources ""` to skip hooks/plugins while keeping auth
- Use `--disallowedTools` for text-only roles (prevents tool-use loops)
- Use `--mcp-config ~/.claude.json` to forward MCP servers to generator step
- Long prompts (>1000 chars) via `-p` argument can hang — use system prompt files
- `max_turns=1` causes "max turns exceeded" — use at least 3

## Agent role mapping

| Role | Provider | Notes |
|------|----------|-------|
| Experiment Generator | Claude | Has MCP access (alphaxiv), generates literature-grounded hypotheses |
| Experiment Programmer | Claude (sonnet) | `no_tools=True`, outputs code as text |
| Code Executor | Docker | `--network=none`, image: `surprisal-sandbox:latest` |
| Experiment Analyst | Claude (sonnet) | Reviews execution output |
| Experiment Reviewer | Claude (sonnet) | Approves/rejects experiments |
| Experiment Reviser | Claude (sonnet) | Revises rejected experiment plans |
| Hypothesis Generator | Claude | Formalizes experimental results |
| Belief Agent | Claude | 30 prior + 30 posterior samples for Bayesian surprise |

## Key files

- `src/surprisal/fsm_runner.py` — the live FSM that calls agents
- `src/surprisal/orchestrator.py` — async worker pool with selection lock
- `src/surprisal/mcts.py` — UCT, progressive widening, backpropagation
- `src/surprisal/bayesian.py` — Beta estimation, KL divergence, belief shift
- `src/surprisal/prompts/` — 8 system prompt files (864 lines total)

## Testing

```bash
uv run pytest tests/ -q --tb=short --ignore=tests/test_integration.py --ignore=tests/test_cli.py
```

Integration/CLI tests call real agents and are slow. Unit tests run in <2s.

## Running

```bash
uv run surprisal init --domain "your topic" --seed "your hypothesis"
uv run surprisal explore --budget 10 --concurrency 1
uv run surprisal status --tree
uv run surprisal export --top 5 --format md
```
