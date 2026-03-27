# Concurrent Belief Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 60 sequential Claude CLI calls in belief elicitation with 20 concurrent calls (10 prior + 10 posterior), reducing belief phase from 5-15 minutes to 30-40 seconds per node.

**Architecture:** Change the default `belief_samples` from 30 to 10 across config, models, and DB schema. Replace the two sequential `for i in range(n_samples)` loops in `fsm_runner.py` with `asyncio.gather()` batches. All other FSM stages, prompts, and downstream math are untouched.

**Tech Stack:** Python 3.12+, asyncio, pytest, SQLite WAL

**Spec:** `docs/specs/2026-03-27-concurrent-belief-estimation-design.md`

---

### Task 1: Update defaults from 30 to 10

**Files:**
- Modify: `src/surprisal/config.py:23` — `belief_samples` default
- Modify: `src/surprisal/models.py:29` — `Node.n_belief_samples` default
- Modify: `src/surprisal/db.py:43` — SQL schema DEFAULT
- Modify: `tests/test_config.py:21` — assertion on default value

- [ ] **Step 1: Update config.py**

In `src/surprisal/config.py`, change line 23:

```python
# Before:
    belief_samples: int = 30
# After:
    belief_samples: int = 10
```

- [ ] **Step 2: Update models.py**

In `src/surprisal/models.py`, change line 29:

```python
# Before:
    n_belief_samples: int = 30
# After:
    n_belief_samples: int = 10
```

- [ ] **Step 3: Update db.py**

In `src/surprisal/db.py`, change line 43:

```python
# Before:
                n_belief_samples INTEGER DEFAULT 30,
# After:
                n_belief_samples INTEGER DEFAULT 10,
```

- [ ] **Step 4: Update test_config.py**

In `tests/test_config.py`, change line 21:

```python
# Before:
    assert cfg.mcts.belief_samples == 30
# After:
    assert cfg.mcts.belief_samples == 10
```

- [ ] **Step 5: Run tests to verify defaults**

Run: `uv run pytest tests/test_config.py tests/test_db.py -q --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/surprisal/config.py src/surprisal/models.py src/surprisal/db.py tests/test_config.py
git commit -m "chore: reduce default belief_samples from 30 to 10"
```

---

### Task 2: Replace sequential belief loops with concurrent batch

**Files:**
- Modify: `src/surprisal/fsm_runner.py:722-844` — belief_elicitation block

- [ ] **Step 1: Add the `_run_belief_batch` helper**

Add this function before `run_live_fsm` in `src/surprisal/fsm_runner.py` (after the existing helper functions, around line 120):

```python
async def _run_belief_batch(
    agent,
    prompt: str,
    n: int,
    research_session_id: str | None,
    db: Database,
    node_id: str,
    phase: str,
    workspace: Path,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Run n belief samples concurrently. Returns count of True beliefs."""
    emit_progress(progress_callback, f"Node {node_id}: sampling {n} {phase} beliefs concurrently.")

    async def _sample(i: int) -> tuple[int, AgentResult]:
        result = await agent.invoke(
            prompt=prompt,
            output_format="text",
            cwd=str(workspace),
            **_forked_belief_session_args(research_session_id),
        )
        return i, result

    results = await asyncio.gather(*[_sample(i) for i in range(n)])

    k = 0
    for i, result in results:
        _record_invocation(db, node_id, f"belief_elicitation_{phase}", "claude", prompt, result)
        data, stage_error = _extract_stage_json(result, f"belief_elicitation_{phase}")
        if stage_error:
            return _fail_node(db, node_id, "belief_elicitation", stage_error)
        if "believes_hypothesis" not in data or not isinstance(data["believes_hypothesis"], bool):
            return _fail_node(
                db, node_id, "belief_elicitation",
                f"belief_elicitation_{phase} failed: missing boolean believes_hypothesis",
            )
        believes = data["believes_hypothesis"]
        if believes:
            k += 1
        db.insert_belief_sample(BeliefSample(
            node_id=node_id, phase=phase,
            sample_index=i, believes_hypothesis=bool(believes),
            raw_response=result.raw[:500],
        ))

    emit_progress(progress_callback, f"Node {node_id}: {phase} belief complete — {k}/{n} positive.")
    return k
```

- [ ] **Step 2: Replace the belief_elicitation block**

Replace the entire `elif state == "belief_elicitation":` block (lines 722-844) in `run_live_fsm` with:

```python
        # ── Belief Elicitation (concurrent) ──
        elif state == "belief_elicitation":
            node = db.get_node(node_id)  # refresh
            n_samples = config.mcts.belief_samples
            hypothesis_text = node.hypothesis

            # Prior elicitation (no evidence)
            prior_prompt = _belief_prompt(hypothesis_text)
            k_prior_result = await _run_belief_batch(
                research_agent, prior_prompt, n_samples,
                research_session_id, db, node_id, "prior", workspace, progress_callback,
            )
            if isinstance(k_prior_result, bool):
                return k_prior_result  # _fail_node was called
            k_prior = k_prior_result

            # Posterior elicitation (with evidence)
            posterior_prompt = _belief_prompt(
                hypothesis_text,
                evidence=(
                    f"Execution Output:\n{experiment_output[:2000]}\n\n"
                    f"Analysis:\n{analysis_summary}"
                ),
            )
            k_post_result = await _run_belief_batch(
                research_agent, posterior_prompt, n_samples,
                research_session_id, db, node_id, "posterior", workspace, progress_callback,
            )
            if isinstance(k_post_result, bool):
                return k_post_result  # _fail_node was called
            k_post = k_post_result

            # Compute surprisal
            surprisal_result = compute_surprisal(k_prior, k_post, n_samples)
            logger.info(
                f"  Surprisal: k_prior={k_prior}, k_post={k_post}, "
                f"BS={surprisal_result.bayesian_surprise:.3f}, "
                f"shifted={surprisal_result.belief_shifted}"
            )

            db.update_node(node_id,
                prior_alpha=surprisal_result.prior_alpha,
                prior_beta=surprisal_result.prior_beta,
                posterior_alpha=surprisal_result.posterior_alpha,
                posterior_beta=surprisal_result.posterior_beta,
                k_prior=k_prior,
                k_post=k_post,
                n_belief_samples=n_samples,
                bayesian_surprise=surprisal_result.bayesian_surprise,
                belief_shifted=surprisal_result.belief_shifted,
            )

            # Save belief summary
            (exp_dir / "belief.json").write_text(json.dumps({
                "k_prior": k_prior, "k_post": k_post, "n": n_samples,
                "prior_alpha": surprisal_result.prior_alpha,
                "prior_beta": surprisal_result.prior_beta,
                "posterior_alpha": surprisal_result.posterior_alpha,
                "posterior_beta": surprisal_result.posterior_beta,
                "bayesian_surprise": surprisal_result.bayesian_surprise,
                "belief_shifted": surprisal_result.belief_shifted,
                "surprisal": surprisal_result.surprisal,
            }, indent=2))

            last_response = FSMResponse(error=False)
            state = "belief_elicitation"  # will transition to COMPLETE on next iteration
```

- [ ] **Step 3: Add `import asyncio` if not already present**

Check the imports at the top of `fsm_runner.py`. It does not currently import `asyncio`. Add it:

```python
import asyncio
```

- [ ] **Step 4: Run existing tests**

Run: `uv run pytest tests/test_fsm_runner.py -q --tb=short`
Expected: All pass. The existing tests use `belief_samples = 1`, which means `asyncio.gather()` fires 1 concurrent task — functionally identical to the old sequential loop.

- [ ] **Step 5: Commit**

```bash
git add src/surprisal/fsm_runner.py
git commit -m "feat: concurrent belief estimation via asyncio.gather"
```

---

### Task 3: Update session-reuse test for concurrent belief calls

**Files:**
- Modify: `tests/test_fsm_runner.py:519-532` — fixed-index call assertions

- [ ] **Step 1: Update the session-reuse test assertions**

The test `test_run_live_fsm_reuses_branch_sessions_and_updates_latest_ids` at line 519 uses fixed indices to check belief calls. With `belief_samples=1`, the concurrent batch fires 1 sample per phase, so calls[2] is prior and calls[3] is posterior. But the belief calls now go through `_run_belief_batch`, which still calls `agent.invoke`. With `belief_samples=1`, the call ordering is the same.

Verify that the test still passes without changes:

Run: `uv run pytest tests/test_fsm_runner.py::test_run_live_fsm_reuses_branch_sessions_and_updates_latest_ids -v`
Expected: PASS (because `belief_samples=1` means gather fires exactly 1 coroutine, same as sequential)

- [ ] **Step 2: Commit if no changes needed**

If the test passes without changes, no commit needed for this task.

---

### Task 4: Run full test suite

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -q --tb=short`
Expected: All pass (85+ tests)

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/surprisal/fsm_runner.py src/surprisal/config.py src/surprisal/models.py`
Expected: No errors

---

### Task 5: Live end-to-end verification

**Files:** None (verification only)

This is the critical validation. Run a real exploration with the concurrent belief path and verify the full UX.

- [ ] **Step 1: Verify Claude CLI is authenticated**

Run: `claude auth status`
Expected: Shows `loggedIn`

- [ ] **Step 2: Initialize a fresh exploration**

Run: `uv run surprisal init --domain "AI for scientific discovery" --seed "LLM self-evaluation accuracy drops as task compositional depth increases"`
Expected: Prints exploration ID and root node ID

- [ ] **Step 3: Run a budget-1 exploration**

Run: `uv run surprisal explore --budget 1 --concurrency 1`

Watch for:
- Progress messages for each FSM stage (generator, runner, analyst, reviewer, hypothesis, belief)
- Belief phase should show "sampling 10 prior beliefs concurrently" and "sampling 10 posterior beliefs concurrently"
- Belief phase should complete in under 60 seconds (not 5-15 minutes)
- Final result should show `"status": "completed"` with `"iterations": 1`

- [ ] **Step 4: Check exploration status**

Run: `uv run surprisal status --tree`
Expected: Shows the exploration tree with at least 1 verified node beyond the root

- [ ] **Step 5: Export results**

Run: `uv run surprisal export --top 5 --format md`
Expected: Prints a markdown table with ranked hypotheses, including Bayesian surprise scores

- [ ] **Step 6: Verify belief data integrity**

Check that the belief phase produced valid data:

```bash
uv run python3 -c "
from surprisal.cli import get_home
from surprisal.exploration import find_latest_exploration
from surprisal.db import Database

home = get_home()
exp_dir = find_latest_exploration(home)
db = Database(exp_dir / 'tree.db')
db.initialize()

# Get verified nodes (not root)
nodes = db.execute('SELECT id, hypothesis, k_prior, k_post, n_belief_samples, bayesian_surprise, belief_shifted FROM nodes WHERE depth > 0 AND status = \"verified\"').fetchall()
for nid, hyp, kp, kpo, n, bs, shifted in nodes:
    print(f'Node {nid}:')
    print(f'  hypothesis: {hyp[:80]}')
    print(f'  belief: k_prior={kp}, k_post={kpo}, n={n}')
    print(f'  surprise: BS={bs}, shifted={shifted}')
    prior_samples = db.execute('SELECT COUNT(*) FROM belief_samples WHERE node_id=? AND phase=\"prior\"', (nid,)).fetchone()[0]
    post_samples = db.execute('SELECT COUNT(*) FROM belief_samples WHERE node_id=? AND phase=\"posterior\"', (nid,)).fetchone()[0]
    print(f'  audit: {prior_samples} prior samples, {post_samples} posterior samples')

db.close()
"
```

Expected:
- `n_belief_samples` should be 10
- `k_prior` and `k_post` should each be between 0 and 10
- Prior and posterior sample counts should each be 10
- `bayesian_surprise` should be a float (may be 0.0 if no belief shift)
