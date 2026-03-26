"""Live FSM execution — calls real Claude/Codex/Docker agents.

This module implements the full discovery agent pipeline for a single MCTS node:
  experiment_generator → runner → analyst → reviewer → hypothesis → belief
"""
import json
import logging
from pathlib import Path

from surprisal.agents.base import AgentResult
from surprisal.agents.claude import ClaudeAgent
from surprisal.agents.codex import CodexAgent
from surprisal.agents.backends import create_backend, detect_gpu
from surprisal.agents.experiment_container import ExperimentContainer
from surprisal.config import AutoDiscoveryConfig
from surprisal.db import Database
from surprisal.fsm import select_next_state, FSMResponse
from surprisal.models import Node, BeliefSample
from surprisal.bayesian import compute_surprisal
from surprisal.providers import LiteratureStatus, ProviderStatus
from surprisal.workspace import get_experiment_dir

logger = logging.getLogger("surprisal")


def _prompts_dir() -> Path:
    return Path(__file__).parent / "prompts"


def _load_prompt(name: str, **kwargs) -> str:
    """Load a prompt template and format with kwargs."""
    text = (_prompts_dir() / name).read_text()
    for k, v in kwargs.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text


def _extract_json(result: AgentResult) -> dict:
    """Try to extract JSON from an agent result."""
    if result.parsed:
        # If the output-format json wrapper has a 'result' key, parse that
        if isinstance(result.parsed, dict) and "result" in result.parsed:
            inner = result.parsed["result"]
            if isinstance(inner, str):
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    # Try to find JSON in the text
                    return _find_json_in_text(inner)
            return inner if isinstance(inner, dict) else {}
        return result.parsed
    return _find_json_in_text(result.raw)


def _find_json_in_text(text: str) -> dict:
    """Find and parse JSON from text that may contain markdown/prose."""
    # Try the whole text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find JSON block in markdown
    for start_marker in ["```json", "```"]:
        if start_marker in text:
            start = text.index(start_marker) + len(start_marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    # Try to find any { ... } block
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except json.JSONDecodeError:
                        break
    return {}


async def run_live_fsm(
    node_id: str,
    db: Database,
    config: AutoDiscoveryConfig,
    workspace: Path,
    domain: str,
    branch_path: list[Node],
    providers: ProviderStatus | None = None,
    literature_provider: LiteratureStatus | None = None,
) -> bool:
    """Run the full FSM for a single node with real agent calls.

    Returns True if the node was successfully verified (reached COMPLETE),
    False if it failed (reached FAIL).
    """
    # If no provider status passed, default to Claude-only
    if providers is None:
        providers = ProviderStatus(claude_available=True, codex_available=False)

    # Create agents based on available providers
    if providers.claude_available:
        research_agent = ClaudeAgent(model=config.agents.claude_model, max_turns=config.agents.max_turns)
    else:
        research_agent = None

    if providers.codex_available and providers.claude_available:
        # Heterogeneous: Codex for code/analysis roles
        code_agent = ClaudeAgent(model="sonnet", max_turns=config.agents.max_turns)
        # TODO: swap to real CodexAgent when codex exec subprocess is stable
    elif providers.claude_available:
        # Claude-only: use Claude for everything
        code_agent = ClaudeAgent(model="sonnet", max_turns=config.agents.max_turns)
    elif providers.codex_available:
        # Codex-only: not yet supported -- Codex subprocess needs work
        logger.error("Codex-only mode not yet supported")
        db.update_node(node_id, status="failed", virtual_loss=0)
        return False
    else:
        logger.error("No agent providers available")
        db.update_node(node_id, status="failed", virtual_loss=0)
        return False
    # Auto-detect GPU for backend selection (runs once per FSM execution)
    _gpu_available = await detect_gpu() if config.sandbox.backend == "auto" else None

    node = db.get_node(node_id)
    exp_dir = get_experiment_dir(workspace, node_id)
    state = "start"
    last_response = None
    experiment_plan = ""
    experiment_code = ""
    experiment_output = ""
    analysis_summary = ""

    # Build branch context for prompts
    branch_context = "\n".join(
        f"- Depth {n.depth}: {n.hypothesis}" for n in reversed(branch_path)
    )

    while state not in ("COMPLETE", "FAIL"):
        next_state = select_next_state(state, last_response,
                                        node.fsm_failure_count,
                                        node.fsm_revision_count)
        logger.info(f"  FSM {node_id}: {state} → {next_state}")

        if next_state == "FAIL":
            db.update_node(node_id, status="failed", fsm_state="FAIL", virtual_loss=0)
            return False

        if next_state == "COMPLETE":
            break

        state = next_state
        db.update_node(node_id, fsm_state=state)

        # ── Experiment Generator (Claude) ──
        if state == "experiment_generator":
            # IMPORTANT: The generator must NOT use no_tools=True because it needs
            # WebFetch for the HuggingFace fallback path when alphaxiv is unavailable.
            # See spec Section 6.3.

            # Forward user's MCP config for paper search
            gen_extra_args = []
            if literature_provider and literature_provider.provider == "alphaxiv":
                claude_json = Path.home() / ".claude.json"
                if claude_json.exists():
                    gen_extra_args = ["--mcp-config", str(claude_json)]

            # Build literature search instructions (conditional on provider)
            from datetime import date
            today = date.today().isoformat()
            lit_instructions = (
                "STEP 1: Search for 2-3 RECENT papers (2024-2026) related to this branch's topic.\n"
                "  Prioritize papers from the last 6 months. Older foundational works are OK as secondary references.\n"
                "STEP 2: Read their limitations or open problems.\n"
                "STEP 3: Identify a specific gap to test.\n"
                "STEP 4: Propose a hypothesis targeting that gap.\n"
            )
            if not (literature_provider and literature_provider.has_semantic_search):
                lit_instructions += (
                    f"\nTo find papers, use WebFetch:\n"
                    f"- Recent papers: fetch https://huggingface.co/api/daily_papers?date={today}\n"
                    f"- Search by topic: fetch https://huggingface.co/api/daily_papers to browse recent work\n"
                    "- Read a paper: fetch https://huggingface.co/papers/{{arxiv_id}}.md for full content\n"
                )

            prompt = (
                f"Domain: {domain}\n\n"
                f"Branch history:\n{branch_context}\n\n"
                f"Literature search:\n{lit_instructions}\n\n"
                "Propose a new hypothesis and a SIMPLE experiment plan.\n"
                "CRITICAL CONSTRAINTS:\n"
                "- The experiment MUST be 10-30 lines of Python. NOT more.\n"
                "- Use ONLY basic stats: t-test, correlation, regression, chi-square, or simple Monte Carlo.\n"
                "- Use ONLY synthetic/simulated data (no downloads, no APIs, no internet, no files)\n"
                "- Use only: numpy, scipy.stats, pandas, sklearn\n"
                "- Produce ONE clear numerical result (p-value, correlation, effect size)\n"
                "- The experiment_plan MUST be 2-3 sentences max. Do NOT describe multi-step procedures.\n"
                "- Think 'one scipy.stats function call on generated data' — not a simulation framework.\n\n"
                "Respond with JSON: {{\"hypothesis\": \"...\", \"context\": \"...\", "
                "\"variables\": [...], \"relationships\": [...], "
                "\"experiment_plan\": \"...\", "
                "\"cited_papers\": [{{\"arxiv_id\": \"...\", \"title\": \"...\", \"gap\": \"...\"}}]}}"
            )
            sys_prompt = str(_prompts_dir() / "experiment_generator.md")
            result = await research_agent.invoke(
                prompt=prompt,
                system_prompt_file=sys_prompt,
                output_format="text",
                cwd=str(workspace),
                timeout=config.agents.generator_timeout,
                extra_args=gen_extra_args,
            )
            logger.info(f"  Generator exit={result.exit_code}, len={len(result.raw)}")
            data = _extract_json(result)
            hypothesis = data.get("hypothesis", f"Auto-generated hypothesis at depth {node.depth}")
            experiment_plan = data.get("experiment_plan", "Analyze statistical patterns in the data")
            cited_papers = json.dumps(data.get("cited_papers", []))

            db.update_node(node_id,
                hypothesis=hypothesis,
                initial_hypothesis=hypothesis,
                context=data.get("context", ""),
                variables=json.dumps(data.get("variables", [])),
                relationships=json.dumps(data.get("relationships", [])),
                cited_papers=cited_papers,
            )
            (exp_dir / "plan.md").write_text(experiment_plan)
            last_response = FSMResponse(error=False, data=data)

        # ── Experiment Runner (agent inside container) ──
        elif state == "experiment_runner":
            if last_response and last_response.error:
                node.fsm_failure_count += 1
                db.update_node(node_id, fsm_failure_count=node.fsm_failure_count)
            feedback = ""
            if last_response and last_response.feedback:
                feedback = f"\nPrevious attempt feedback: {last_response.feedback}"

            plan_summary = experiment_plan[:800]
            runner_prompt = (
                f"Run this experiment inside your environment.\n"
                f"Plan: {plan_summary}\n{feedback}\n"
                "Write the code, execute it, debug if needed. "
                "Write results to /work/results.json."
            )

            backend = create_backend(config.sandbox, config.credentials, gpu_available=_gpu_available)
            result = await backend.execute(
                experiment_prompt=runner_prompt,
                workspace=exp_dir,
                config=config.sandbox,
            )
            logger.info(f"  Runner exit={result.exit_code}, len={len(result.raw)}")

            experiment_output = result.raw
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    results_data = json.loads(results_file.read_text())
                    experiment_code = results_data.get("code", "")
                    experiment_output = results_data.get("stdout", result.raw)
                except (json.JSONDecodeError, KeyError):
                    experiment_code = ""
            else:
                experiment_code = (exp_dir / "experiment.py").read_text() if (exp_dir / "experiment.py").exists() else ""

            db.update_node(node_id, experiment_exit_code=result.exit_code)

            if ExperimentContainer.is_infra_error(result.exit_code):
                last_response = FSMResponse(error=True, exit_code=result.exit_code)
            else:
                last_response = FSMResponse(
                    error=result.exit_code != 0,
                    exit_code=result.exit_code,
                )

        # ── Experiment Analyst (Codex) ──
        elif state == "experiment_analyst":
            stderr_text = ""
            stderr_path = exp_dir / "stderr.txt"
            if stderr_path.exists():
                stderr_text = stderr_path.read_text()[:2000]

            prompt = (
                f"Analyze this experiment output:\n\n"
                f"Exit code: {node.experiment_exit_code}\n"
                f"Stdout:\n{experiment_output[:3000]}\n"
                f"Stderr:\n{stderr_text}\n\n"
                f"Code:\n{experiment_code[:2000]}\n\n"
                "If error, return JSON: {{\"error\": true, \"feedback\": \"...\"}}\n"
                "If success, return JSON: {{\"error\": false, \"summary\": \"...\", \"key_results\": {{}}}}"
            )
            result = await code_agent.invoke(prompt=prompt, output_format="text", cwd=str(exp_dir), timeout=120)
            logger.info(f"  Analyst exit={result.exit_code}, len={len(result.raw)}")
            data = _extract_json(result)

            # Default: if experiment ran (exit=0) and we can't parse error status, assume success
            is_error = data.get("error", True) if data else (node.experiment_exit_code != 0)
            if is_error:
                # NOTE: failure_count is incremented when we actually re-enter programmer,
                # not here. select_next_state checks the current count for the threshold.
                last_response = FSMResponse(error=True, feedback=data.get("feedback", "Unknown error"))
            else:
                analysis_summary = data.get("summary", result.raw[:500])
                (exp_dir / "analysis.md").write_text(analysis_summary)
                last_response = FSMResponse(error=False, data=data)

        # ── Experiment Reviewer (Codex) ──
        elif state == "experiment_reviewer":
            sys_prompt_reviewer = str(_prompts_dir() / "experiment_reviewer.md")
            prompt = (
                f"Review this experiment for BASIC VALIDITY only.\n\n"
                f"Experiment plan: {experiment_plan}\n\n"
                f"Code output: {experiment_output[:2000]}\n\n"
                f"Analysis: {analysis_summary}\n\n"
                "APPROVE if the output contains numerical results that can be interpreted.\n"
                "REJECT ONLY if the code crashed or produced no scientific results.\n"
                "Bias toward approval. Do NOT reject for missing optional analyses or interpretation.\n\n"
                "Return JSON: {{\"error\": false, \"assessment\": \"...\"}} or "
                "{{\"error\": true, \"feedback\": \"...\"}}"
            )
            result = await code_agent.invoke(
                prompt=prompt,
                system_prompt_file=sys_prompt_reviewer,
                output_format="text",
                cwd=str(exp_dir),
                timeout=120,
            )
            logger.info(f"  Reviewer exit={result.exit_code}, len={len(result.raw)}")
            data = _extract_json(result)

            # Default to approved if we can't parse the response
            is_error = data.get("error", False) if data else False
            if is_error:
                # NOTE: do NOT increment revision_count here — select_next_state
                # checks the current value to decide if revision is allowed.
                # The count is incremented ONLY when we actually enter experiment_reviser.
                last_response = FSMResponse(error=True, feedback=data.get("feedback", ""))
                (exp_dir / "review.md").write_text(f"REJECTED: {data.get('feedback', '')}")
            else:
                assessment = data.get("assessment", result.raw[:500]) if data else result.raw[:500]
                (exp_dir / "review.md").write_text(f"APPROVED: {assessment}")
                last_response = FSMResponse(error=False, data=data)

        # ── Experiment Reviser (Codex) ──
        elif state == "experiment_reviser":
            node.fsm_revision_count += 1
            db.update_node(node_id, fsm_revision_count=node.fsm_revision_count)
            prompt = (
                f"The experiment was rejected. Feedback: {last_response.feedback if last_response else ''}\n\n"
                f"Original plan: {experiment_plan}\n\n"
                "Revise the experiment plan. Respond in natural language, no code."
            )
            result = await code_agent.invoke(prompt=prompt, output_format="text", cwd=str(exp_dir), timeout=120)
            experiment_plan = result.raw.strip()
            (exp_dir / "plan.md").write_text(f"REVISED:\n{experiment_plan}")
            last_response = FSMResponse(error=False)

        # ── Hypothesis Generator (Claude) ──
        elif state == "hypothesis_generator":
            prompt = (
                f"Based on this experiment result, propose a formal hypothesis.\n\n"
                f"Experiment: {experiment_plan}\n"
                f"Results: {experiment_output[:2000]}\n"
                f"Analysis: {analysis_summary}\n\n"
                "Respond with JSON: {{\"hypothesis\": \"...\", \"context\": \"...\", "
                "\"variables\": [...], \"relationships\": [...]}}"
            )
            sys_prompt = str(_prompts_dir() / "hypothesis_generator.md")
            result = await research_agent.invoke(
                prompt=prompt,
                system_prompt_file=sys_prompt,
                output_format="text",
                cwd=str(workspace),
            )
            data = _extract_json(result)
            if data.get("hypothesis"):
                db.update_node(node_id,
                    hypothesis=data["hypothesis"],
                    context=data.get("context", ""),
                    variables=json.dumps(data.get("variables", [])),
                    relationships=json.dumps(data.get("relationships", [])),
                )
            last_response = FSMResponse(error=False, data=data)

        # ── Belief Elicitation (Claude x60) ──
        elif state == "belief_elicitation":
            node = db.get_node(node_id)  # refresh
            n_samples = config.mcts.belief_samples
            hypothesis_text = node.hypothesis

            # Prior elicitation (no evidence)
            prior_prompt = (
                f"Assess whether this hypothesis is true or false. "
                f"Respond ONLY with JSON: {{\"believes_hypothesis\": true}} or {{\"believes_hypothesis\": false}}\n\n"
                f"Hypothesis: {hypothesis_text}\n\n"
                f"Consider from multiple angles. Express genuine uncertainty."
            )
            k_prior = 0
            for i in range(n_samples):
                result = await research_agent.invoke(
                    prompt=prior_prompt,
                    output_format="text",
                    cwd=str(workspace),
                )
                data = _extract_json(result)
                believes = data.get("believes_hypothesis", False)
                if believes:
                    k_prior += 1
                db.insert_belief_sample(BeliefSample(
                    node_id=node_id, phase="prior",
                    sample_index=i, believes_hypothesis=bool(believes),
                    raw_response=result.raw[:500],
                ))
                if i % 10 == 0:
                    logger.info(f"  Belief prior: {i+1}/{n_samples}, k_prior={k_prior}")

            # Posterior elicitation (with evidence)
            posterior_prompt = (
                f"Assess whether this hypothesis is true or false. "
                f"Respond ONLY with JSON: {{\"believes_hypothesis\": true}} or {{\"believes_hypothesis\": false}}\n\n"
                f"Hypothesis: {hypothesis_text}\n\n"
                f"Experimental Evidence:\n{experiment_output[:2000]}\n\n"
                f"Analysis: {analysis_summary}\n\n"
                f"Consider from multiple angles. Express genuine uncertainty."
            )
            k_post = 0
            for i in range(n_samples):
                result = await research_agent.invoke(
                    prompt=posterior_prompt,
                    output_format="text",
                    cwd=str(workspace),
                )
                data = _extract_json(result)
                believes = data.get("believes_hypothesis", False)
                if believes:
                    k_post += 1
                db.insert_belief_sample(BeliefSample(
                    node_id=node_id, phase="posterior",
                    sample_index=i, believes_hypothesis=bool(believes),
                    raw_response=result.raw[:500],
                ))
                if i % 10 == 0:
                    logger.info(f"  Belief posterior: {i+1}/{n_samples}, k_post={k_post}")

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

    # Mark as verified
    db.update_node(node_id, status="verified", virtual_loss=0)
    node = db.get_node(node_id)
    return True
