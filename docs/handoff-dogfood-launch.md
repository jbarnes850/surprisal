## Handoff: Dogfood Experiment and Launch Blog

Date: 2026-03-26
Repo: `/Users/jarrodbarnes/surprisal`
Validated commit: `3e67348`

### Completed

- Closed the implementation/runtime drift against the README across experiment scope, scaffolding, config/runtime alignment, session persistence, and docs.
- Added persisted Claude/Codex/runner session tracking and resume/fork handling across the FSM and sandbox paths.
- Refreshed `README.md`, `CLAUDE.md`, and repo-level `AGENTS.md` to match the live runtime.
- Revalidated the repo after the changes:
  - `uv run ruff check src tests`
  - `uv run pytest tests/ -q --tb=short`

### Current Product State

- The runtime is in a research-grade state for bounded autonomous discovery runs.
- The remaining gap is not implementation fidelity; it is product usage framing and dogfood validation against a realistic end-user workflow.
- The next scope is exploratory/product-facing rather than core runtime repair.

### Next Session Scope

Primary goal:
- Scope a first Surprisal run that mirrors a real data-research workflow for a research scientist working on pre/post-training pipelines and data infrastructure.

Secondary goal:
- Scope a personal launch blog for Surprisal using the personal website repo once that codebase is in view.

### Suggested First Experiment

Use a literature-to-RQ-to-bounded-experiment workflow, not a frontier-scale reproduction.

Candidate framing:
- Research question: can an agentic, iterative data-curation pipeline produce a better domain corpus than static keyword filtering under equal collection and review budget?

Suggested artifacts:
- literature review
- 3 candidate research questions
- selected RQ with rationale
- dataset construction plan
- bounded experiment plan
- audit rubric and sample review process
- final recommendation with limitations

### User-Provided Sources

- `https://arxiv.org/abs/2602.11089`
- `https://arxiv.org/html/2603.17074v2`
- `https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_11a.pdf`
- `https://arxiv.org/abs/2601.21343`
- `https://arxiv.org/pdf/2603.14420v1`
- `https://github.com/karpathy/autoresearch`
- Core inspiration repo to review at the start of the next session: `https://github.com/allenai/autodiscovery`

### Proposed Next-Session Deliverables

1. Review AutoDiscovery and extract the workflow/product patterns worth inheriting.
2. Turn the supplied papers into an explicit Surprisal dogfood prompt and run plan.
3. Define a small-model/local-compute experiment configuration for the demo.
4. Scope the launch blog narrative:
   - why data-centric auto-research matters
   - Surprisal method
   - related work
   - usage
   - launch framing
5. If the website repo is available, convert the blog scope into file-level implementation work.

### Inputs Needed Next Session

- The personal website repo path or workspace if implementation work is expected there.
- A decision on whether the next step is:
  - pure scoping and prompt design
  - actually running the first Surprisal dogfood experiment
  - drafting the launch blog content
