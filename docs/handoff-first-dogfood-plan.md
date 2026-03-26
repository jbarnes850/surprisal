## Handoff: First Surprisal Dogfood Plan

Date: 2026-03-26
Repo: `/Users/jarrodbarnes/surprisal`
Website repo: `/Users/jarrodbarnes/jbarnes850.github.io`

### Direct Conclusion

The right first Surprisal dogfood is not a frontier-scale training run. It is a local, literature-grounded corpus design workflow for a researcher who starts with no hypothesis and needs to go from literature review to research question to dataset recipe to bounded verification. The core question should be:

> Under a fixed collection and review budget, can an iterative, literature-grounded corpus design loop produce a better science-reasoning adaptation set than static keyword filtering?

This matches the product better than a generic benchmark because it forces Surprisal to do what a real researcher needs:

- read recent papers,
- propose candidate research questions,
- choose one based on feasibility,
- build a dataset plan from public sources,
- run bounded evidence-gathering experiments,
- and return a recommendation with limitations.

It also stays MacBook-first, but it should assume real execution through the local Docker runner. No GPU is required for the first pass; GPU is an escalation path if the local proxy result is strong enough to justify a small training smoke.

### Source-Grounded Rationale

- AutoDiscovery is the closest product ancestor: it uses Bayesian surprise with MCTS, one bounded experiment per node, BYO dataset metadata, a reviewer/reviser loop, and resume-from-log exploration rather than a single monolithic run ([repo](https://github.com/allenai/autodiscovery), [paper](https://openreview.net/pdf?id=kJqTkj2HhF)).
- DataChef frames the right external problem: end-to-end data recipe generation for LLM adaptation, optimized with a proxy reward instead of full downstream retraining on every candidate ([arXiv:2602.11089](https://arxiv.org/abs/2602.11089)).
- Self-Improving Pretraining argues that post-trained models should be used upstream, during data selection and improvement, instead of only at the end of the pipeline ([arXiv:2601.21343](https://arxiv.org/abs/2601.21343)).
- DataEvolve shows a closed loop that identifies issues, generates candidate strategies, evaluates them on sampled data, and refines them across generations; that is exactly the right pattern for a bounded Surprisal demo ([arXiv:2603.14420](https://arxiv.org/abs/2603.14420)).
- PRISM gives the most important design constraint: data composition matters more at mid-training than RL mix, and science data in mid-training materially changes downstream reasoning performance ([arXiv:2603.17074v2](https://arxiv.org/html/2603.17074v2)).
- Berkeley’s synthetic-data lecture makes the practical budget argument: active curriculum design is replacing passive collection, small generators can be enough, and synthetic/rephrased data is currently a multiplier on real data, not a replacement ([slides](https://scalable-ai.eecs.berkeley.edu/assets/lecture_slides/lecture_11a.pdf)).
- `autoresearch` is useful as a contrast, not the main template: its key idea is bounded autonomous iteration, but it assumes a single-file training loop on a single NVIDIA GPU and is therefore the wrong first Surprisal demo for a hypothesis-free researcher on a MacBook ([repo](https://github.com/karpathy/autoresearch)).

### What To Inherit From AutoDiscovery

Keep these patterns:

- One hypothesis and one bounded experiment per expansion.
- A hard reviewer/reviser gate before any result counts as evidence.
- Resume/forkable exploration state rather than single-shot notebooks.
- Warm-start exploration before search: load data, inspect structure, then branch.
- Bring-your-own dataset metadata rather than hardcoded benchmark assumptions.
- Real execution in a bounded sandbox, not paper-only planning.

Do not inherit these as the first Surprisal demo:

- Dataset-first discovery without literature review. The first dogfood should begin from papers, not from an already-clean tabular dataset.
- GPU-first assumptions.
- Large branching factors. A small `budget` with `concurrency=1` is the right initial product test.

### Scoping Principle

We should not fully pre-scope the first dogfood experiment by hand.

Per the current architecture:

- the `experiment_generator` is responsible for searching recent literature, identifying a gap, and proposing one hypothesis plus one executable plan;
- the runner is responsible for executing and debugging inside the sandbox;
- the reviewer/reviser loop is responsible for narrowing or repairing scope only when execution or validity requires it.

That means the human-provided `domain` and `seed` should define:

- the user workflow to optimize for,
- the constraints of the demo,
- the rough problem family,
- and the success condition.

They should not hardcode the final research question, dataset, or exact evaluation design unless the user explicitly wants that.

So the correct stance is:

- **we scope the outer task**, meaning the product scenario and operating constraints;
- **the system scopes the inner experiment**, meaning the concrete RQ, resources, and bounded plan selected from literature.

### Recommended First Experiment Prompt

Use the following as the starting Surprisal framing.

Recommended `--domain`:

```text
Literature-grounded discovery of data recipes for science and reasoning adaptation corpora under fixed collection and review budgets.
```

Recommended `--seed`:

```text
Start from the supplied literature on data recipes, mid-training, and synthetic data. Your first job is to scope the problem, not assume a fixed hypothesis. Produce three candidate research questions for improving science/reasoning adaptation data under realistic researcher constraints, then select the most feasible one for a bounded local run. Prefer a question that compares an iterative literature-grounded corpus design loop against a simple static baseline under equal collection and review budget, but choose the exact question only after reviewing the literature. Use public papers and datasets, avoid frontier-scale training in the first pass, and prefer experiments that can run locally through the Docker sandbox on a MacBook. Success means returning a literature review, selected RQ with rationale, executable experiment design, measured artifacts, and a final recommendation with limitations.
```

This prompt is deliberately procedural. The user starts without a hypothesis; the system should derive the concrete RQ from the literature and defend why it is the right first RQ.

### Concrete Run Plan

#### 1. MacBook-first runtime configuration

```bash
uv run surprisal config --set sandbox.backend local
uv run surprisal config --set sandbox.gpu false
uv run surprisal config --set sandbox.cpu_limit 4
uv run surprisal config --set sandbox.memory_limit 8g
uv run surprisal config --set sandbox.timeout 1200
uv run surprisal config --set general.default_concurrency 1
```

Notes:

- This keeps the demo CPU-only and bounded while still using the real local Docker execution path.
- Treat the local Docker backend as the default and canonical first run.
- Do not move to DGX Spark unless the local proxy experiment clearly justifies a second-stage model-training smoke test.

#### 2. Initialize the exploration

```bash
uv run surprisal init \
  --domain "Literature-grounded discovery of data recipes for science and reasoning adaptation corpora under fixed collection and review budgets." \
  --seed "Start from the supplied literature on data recipes, mid-training, and synthetic data. First produce three candidate research questions for improving science/reasoning adaptation data. Then select the most feasible question for a local run. The default target question is: under a fixed collection and review budget, can an iterative literature-grounded corpus design loop produce a better science-reasoning adaptation set than static keyword filtering? Use public papers and datasets, avoid frontier-scale training, and prefer bounded experiments that can run on a MacBook. Success means returning a literature review, selected RQ, dataset recipe, baseline-vs-agentic comparison, audit results, and a final recommendation with limitations."
```

#### 3. Run a small first exploration

```bash
uv run surprisal explore --budget 6 --concurrency 1
uv run surprisal status --tree
uv run surprisal export --top 5 --format md
```

`budget=6` is enough for the first real dogfood because the goal is not search scale. It is whether one complete end-user workflow closes cleanly.

#### 3.5 Execution expectation

This should be a real executed run, not only a prompt-design exercise.

- The agents should actually retrieve literature, assemble the source pool, build the baseline and agentic corpora, and run the local proxy evaluations inside the Docker-backed sandbox.
- The first pass should avoid expensive model training, but it should still produce executed artifacts and measured results.
- If the local run shows a meaningful win for the agentic recipe, then escalate to a small GPU-backed smoke test rather than jumping directly to a larger training job.

#### 4. Expected expansion structure

The run should converge on this sequence:

1. Literature map.
   Output: 5-10 relevant papers, a short matrix of design axes, and three candidate RQs.
2. RQ selection.
   Output: one selected RQ with rationale, why the others were rejected, and an explicit evidence plan.
3. Raw source pool construction.
   Output: a lightweight paper/document pool from public sources.
4. Baseline recipe.
   Output: static keyword-filtered corpus from the same raw pool.
5. Agentic recipe.
   Output: iterative, literature-derived corpus construction process from the same raw pool.
6. Bounded evaluation and recommendation.
   Output: audit metrics, failure modes, and a decision on whether the agentic recipe actually beat the baseline.

### Dataset and Experiment Design

This section describes the most likely good outcome, not a hardcoded plan that the system must obey regardless of evidence. If the generator finds a better-scoped question within the same product scenario, prefer the system’s scoped experiment over this prior.

#### Target domain

Science/reasoning adaptation data for researchers working on pretraining, mid-training, synthetic data, and data infrastructure.

#### Raw source pool

Use a small public source pool, not a training-scale corpus:

- 1,000-2,000 paper abstracts and metadata from arXiv or Semantic Scholar style public sources.
- Time window: 2024-2026.
- Initial seed themes:
  - mid-training
  - data curation
  - synthetic data
  - science reasoning
  - reasoning traces
  - curriculum design
  - pretraining data quality

This keeps the run local and still realistic.

#### Baseline corpus

A simple keyword pipeline over the same raw pool:

- fixed keyword query set,
- light deduplication,
- fixed top-k selection,
- no iterative refinement.

#### Agentic corpus

An iterative recipe built from literature-derived criteria:

- query expansion from cited papers,
- document scoring against a relevance rubric,
- theme balancing across science, code, data quality, and reasoning,
- deduplication and redundancy control,
- optional lightweight normalization of abstracts or metadata if needed.

Important: do not make synthetic generation the main variable in the first demo. The first comparison should isolate recipe quality, not generator quality.

#### Bounded evaluations

Use cheap, local proxy evaluations. The first dogfood should not depend on full model retraining, but it should depend on actual execution.

1. Audit precision.
   Sample 50 documents from each corpus and score relevance against a rubric derived from the seed papers.
2. Coverage.
   Measure whether each corpus covers the key themes surfaced by the literature review.
3. Redundancy.
   Measure near-duplicate or semantically redundant document rate.
4. Retrieval utility.
   Build a held-out set of 15-20 factual claims or questions from the seed papers and measure whether each corpus retrieves evidence-bearing documents in top-k.

Optional DGX/GPU escalation, only if the local pass is strong:

- Small LoRA or continued-pretraining smoke on a 0.5B-1.5B open model.
- Very small token budget.
- One-pod GPU smoke only.
- Prefer the local machine first if an appropriate local GPU is available; otherwise use DGX Spark.

### Success Criteria

The run is successful if all of the following are true:

1. Surprisal produces three candidate RQs with cited-paper grounding.
2. It selects one RQ and explains the choice in feasibility terms, not only novelty terms.
3. It constructs both a baseline and an agentic corpus from the same raw pool.
4. The agentic corpus beats the baseline on at least three of these four checks:
   - audit precision by at least 15 percentage points,
   - retrieval recall@5 by at least 10 percentage points,
   - equal or better theme coverage,
   - lower redundancy rate.
5. The run ends with a clear recommendation:
   - proceed to a small training smoke,
   - revise the recipe design,
   - or reject the hypothesis.

If those conditions do not hold, the dogfood should be considered incomplete even if the system technically ran.

### Required Artifacts

These are the artifacts the run should explicitly produce inside the exploration workspace:

- `literature_review.md`
- `candidate_rqs.json`
- `selected_rq.md`
- `raw_source_pool.jsonl`
- `baseline_recipe.md`
- `baseline_corpus.jsonl`
- `agentic_recipe.md`
- `agentic_corpus.jsonl`
- `audit_rubric.md`
- `audit_sample.csv`
- `heldout_claims.jsonl`
- `evaluation_summary.md`
- `final_recommendation.md`

The important product check is whether a researcher could consume these outputs without opening the underlying tree database.

### Launch Blog Outline

Working title:

`Launching Surprisal: From Literature Review to Bounded Discovery`

Recommended structure:

1. Hook.
   Start from the real problem: researchers do not need another benchmark runner; they need a system that can start from a vague area, read the literature, choose a question, and run a bounded experiment.
2. Why now.
   Use the data-centric framing from DataChef, PRISM, DataEvolve, and the Berkeley lecture: data recipe design and curriculum construction are becoming first-class research problems.
3. What Surprisal is.
   Explain the MCTS + Bayesian surprise loop and the reviewer/reviser structure.
4. Why AutoDiscovery matters.
   Position AutoDiscovery as the closest research precursor, then explain the product shift: Surprisal is being dogfooded around a real research workflow rather than only structured dataset discovery.
5. The first dogfood experiment.
   Walk through the literature -> candidate RQs -> selected RQ -> corpus recipe comparison -> bounded evaluation flow.
6. Why local-first matters.
   Emphasize that the first useful run should work on a MacBook and only escalate to GPU when the evidence justifies it.
7. What success looks like.
   List the artifacts and decision criteria.
8. What comes next.
   Small-model training smoke on DGX Spark, broader domains, stronger artifact export, and better onboarding.

### Website Repo Scope

The website repo is available and does not need structural changes for a first launch post.

Recommended implementation scope in `/Users/jarrodbarnes/jbarnes850.github.io`:

- New post file:
  - `/Users/jarrodbarnes/jbarnes850.github.io/_posts/2026-03-27-launching-surprisal.md`
- Optional images if created later:
  - `/Users/jarrodbarnes/jbarnes850.github.io/assets/images/surprisal-dogfood-loop.png`
  - `/Users/jarrodbarnes/jbarnes850.github.io/assets/images/surprisal-dogfood-artifacts.png`

Do not start with:

- layout rewrites,
- navigation changes,
- CSS redesign,
- homepage edits.

The current Jekyll structure already supports a straightforward research post via `layout: post`.

### Recommended Next Session Split

The next step should be one of these, not both in the same session:

1. Run the dogfood experiment through the local Docker runner and evaluate the artifacts.
2. If the proxy result is strong, run one small GPU smoke test.
3. Draft the launch post in the website repo using this outline.

If the dogfood run succeeds and the artifacts are real, the blog draft will be much stronger.
