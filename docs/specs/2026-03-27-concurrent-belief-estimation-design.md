# Concurrent Belief Estimation

Date: 2026-03-27
Status: Approved
Scope: Belief elicitation phase only — all other FSM stages untouched

## Problem

The belief elicitation phase runs 60 sequential Claude CLI subprocess calls per node (30 prior + 30 posterior, each via `claude -p --fork-session`). Each subprocess has ~5-15 seconds of overhead (CLI startup, auth check, model load). Total belief phase wall time: 5-15 minutes per node.

For a `--budget 5` run, belief alone consumes 25-75 minutes. This blocks launch.

## Context

AutoDiscovery (AllenAI) uses `--n_belief_samples=30` as a default, designed for fast OpenAI API calls (~1-2s each). Surprisal inherited the `n=30` default but pays ~10x per-sample latency through CLI subprocesses.

GPT-5 models do not support logprobs (confirmed deprecated as of Aug 2025). Claude has never exposed logprobs. Multi-sample binary estimation is the only available approach for empirical belief distribution estimation on frontier models.

## Design

### 1. Reduce default belief_samples from 30 to 10

Standard error of a proportion at `n=10` is ~16% (worst case). Sufficient for detecting meaningful belief shifts — interesting discoveries produce large shifts that `n=10` captures trivially. Researchers can increase via config for publication-grade runs.

### 2. Run belief samples concurrently within each phase

Replace the sequential `for i in range(n_samples)` loops with `asyncio.gather()`. Prior phase fires all `n` concurrently, waits, counts `k_prior`. Then posterior phase fires all `n` concurrently, waits, counts `k_post`. Same `compute_surprisal(k_prior, k_post, n)` call.

Each sample still uses `--fork-session` from the research session. Independence preserved — every sample forks from the same base, no sample sees another's response, research session is not mutated.

If any sample fails (exception or missing `believes_hypothesis`), fail the node. Same fail-fast behavior as the current sequential code — if Claude is broken, it's broken for all samples.

### 3. No changes to downstream math

`compute_surprisal(k_prior, k_post, n)` in `bayesian.py` is parameterized by `n`. No code changes needed.

## Files Changed

| File | Change |
|------|--------|
| `src/surprisal/config.py` | `belief_samples` default: 30 → 10 |
| `src/surprisal/models.py` | `Node.n_belief_samples` default: 30 → 10 |
| `src/surprisal/db.py` | Schema DEFAULT 30 → 10 |
| `src/surprisal/fsm_runner.py` | Replace sequential belief loops with concurrent batch; write `n_belief_samples` to node record |
| `tests/test_fsm_runner.py` | Update belief tests for concurrent path |
| `tests/test_config.py` | Assert `belief_samples == 10` (line 21) |

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| CLI calls per node (belief) | 60 | 20 |
| Belief phase wall time per node | 5-15 min | 30-40 sec |
| `--budget 5` (concurrency 2) | 20-50 min | 8-25 min |

## Validation

1. `uv run pytest tests/ -q --tb=short` — all tests pass
2. `surprisal explore --budget 1 --concurrency 1` completes with concurrent belief path
