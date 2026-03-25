"""Live FSM execution — calls real Claude/Codex/Docker agents.

This module implements the full discovery agent pipeline for a single MCTS node:
  experiment_generator → programmer → executor → analyst → reviewer → hypothesis → belief
"""
import json
import logging
from pathlib import Path

from autodiscovery.agents.base import AgentResult
from autodiscovery.agents.claude import ClaudeAgent
from autodiscovery.agents.codex import CodexAgent
from autodiscovery.agents.docker import DockerSandbox
from autodiscovery.config import AutoDiscoveryConfig
from autodiscovery.db import Database
from autodiscovery.fsm import select_next_state, FSMResponse
from autodiscovery.models import Node, BeliefSample
from autodiscovery.surprisal import compute_surprisal
from autodiscovery.workspace import get_experiment_dir

logger = logging.getLogger("autodiscovery")


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
) -> bool:
    """Run the full FSM for a single node with real agent calls.

    Returns True if the node was successfully verified (reached COMPLETE),
    False if it failed (reached FAIL).
    """
    claude = ClaudeAgent(model=config.agents.claude_model, max_turns=config.agents.max_turns)
    # Use Claude for all roles initially — Codex subprocess invocation needs
    # further debugging for non-interactive mode. The heterogeneous agent
    # mapping (Claude=research, Codex=code) is preserved in the FSM state names
    # and can be swapped when Codex subprocess is verified.
    codex_as_claude = ClaudeAgent(model="sonnet", max_turns=config.agents.max_turns)
    sandbox = DockerSandbox(
        memory=config.sandbox.memory_limit,
        cpus=config.sandbox.cpu_limit,
        timeout=config.sandbox.timeout,
        network=config.sandbox.network,
    )

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
            prompt = (
                f"Domain: {domain}\n\n"
                f"Branch history:\n{branch_context}\n\n"
                "Propose a new hypothesis and a SIMPLE experiment plan.\n"
                "CRITICAL CONSTRAINTS:\n"
                "- The experiment must use ONLY synthetic/simulated data (no downloads, no APIs, no internet)\n"
                "- Keep the experiment under 50 lines of Python\n"
                "- Use only: numpy, scipy, pandas, sklearn, matplotlib\n"
                "- Print numerical results to stdout\n\n"
                "Respond with JSON: {{\"hypothesis\": \"...\", \"context\": \"...\", "
                "\"variables\": [...], \"relationships\": [...], "
                "\"experiment_plan\": \"...\"}}"
            )
            sys_prompt = str(_prompts_dir() / "experiment_generator.md")
            result = await claude.invoke(
                prompt=prompt,
                system_prompt_file=sys_prompt,
                output_format="text",
                cwd=str(workspace),
                timeout=config.sandbox.timeout,
            )
            logger.info(f"  Generator exit={result.exit_code}, len={len(result.raw)}")
            data = _extract_json(result)
            hypothesis = data.get("hypothesis", f"Auto-generated hypothesis at depth {node.depth}")
            experiment_plan = data.get("experiment_plan", "Analyze statistical patterns in the data")

            db.update_node(node_id,
                hypothesis=hypothesis,
                initial_hypothesis=hypothesis,
                context=data.get("context", ""),
                variables=json.dumps(data.get("variables", [])),
                relationships=json.dumps(data.get("relationships", [])),
            )
            (exp_dir / "plan.md").write_text(experiment_plan)
            last_response = FSMResponse(error=False, data=data)

        # ── Experiment Programmer (Codex/Claude) ──
        elif state == "experiment_programmer":
            feedback = ""
            if last_response and last_response.feedback:
                feedback = f"\n\nPrevious attempt feedback: {last_response.feedback}"

            # Truncate plan to avoid CLI argument length issues
            plan_summary = experiment_plan[:800]
            prompt = (
                f"Write a self-contained Python script. Output ONLY code.\n"
                f"Task: {plan_summary}\n{feedback}\n"
                "Libraries: numpy scipy pandas sklearn matplotlib statsmodels. "
                "No network. Print results to stdout."
            )
            # Programmer outputs code as text — no tools needed
            # The FSM saves the code to experiment.py
            prog_agent = ClaudeAgent(model=config.agents.claude_model, max_turns=3)
            result = await prog_agent.invoke(
                prompt=prompt,
                output_format="text",
                cwd=str(exp_dir),
                timeout=60,
                no_tools=True,
            )
            logger.info(f"  Programmer exit={result.exit_code}, len={len(result.raw)}")

            # Extract code from result
            code = result.raw.strip()
            # Strip markdown code fences if present
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            experiment_code = code
            (exp_dir / "experiment.py").write_text(code)
            last_response = FSMResponse(error=False)

        # ── Code Executor (Docker) ──
        elif state == "code_executor":
            result = await sandbox.execute(str(exp_dir))
            logger.info(f"  Executor exit={result.exit_code}, len={len(result.raw)}")
            experiment_output = result.raw
            db.update_node(node_id, experiment_exit_code=result.exit_code)

            if DockerSandbox.is_infra_error(result.exit_code):
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
            result = await codex_as_claude.invoke(prompt=prompt, output_format="text", cwd=str(exp_dir), timeout=120)
            logger.info(f"  Analyst exit={result.exit_code}, len={len(result.raw)}")
            data = _extract_json(result)

            # Default: if experiment ran (exit=0) and we can't parse error status, assume success
            is_error = data.get("error", True) if data else (node.experiment_exit_code != 0)
            if is_error:
                node.fsm_failure_count += 1
                db.update_node(node_id, fsm_failure_count=node.fsm_failure_count)
                last_response = FSMResponse(error=True, feedback=data.get("feedback", "Unknown error"))
            else:
                analysis_summary = data.get("summary", result.raw[:500])
                (exp_dir / "analysis.md").write_text(analysis_summary)
                last_response = FSMResponse(error=False, data=data)

        # ── Experiment Reviewer (Codex) ──
        elif state == "experiment_reviewer":
            prompt = (
                f"Review this experiment:\n\n"
                f"Experiment plan: {experiment_plan}\n\n"
                f"Code output: {experiment_output[:2000]}\n\n"
                f"Analysis: {analysis_summary}\n\n"
                "Was the experiment faithfully implemented? Are results interpretable?\n"
                "Return JSON: {{\"error\": false, \"assessment\": \"...\"}} or "
                "{{\"error\": true, \"feedback\": \"...\"}}"
            )
            result = await codex_as_claude.invoke(prompt=prompt, output_format="text", cwd=str(exp_dir), timeout=120)
            logger.info(f"  Reviewer exit={result.exit_code}, len={len(result.raw)}")
            data = _extract_json(result)

            # Default to approved if we can't parse the response
            is_error = data.get("error", False) if data else False
            if is_error:
                node.fsm_revision_count += 1
                db.update_node(node_id, fsm_revision_count=node.fsm_revision_count)
                last_response = FSMResponse(error=True, feedback=data.get("feedback", ""))
                (exp_dir / "review.md").write_text(f"REJECTED: {data.get('feedback', '')}")
            else:
                assessment = data.get("assessment", result.raw[:500]) if data else result.raw[:500]
                (exp_dir / "review.md").write_text(f"APPROVED: {assessment}")
                last_response = FSMResponse(error=False, data=data)

        # ── Experiment Reviser (Codex) ──
        elif state == "experiment_reviser":
            prompt = (
                f"The experiment was rejected. Feedback: {last_response.feedback if last_response else ''}\n\n"
                f"Original plan: {experiment_plan}\n\n"
                "Revise the experiment plan. Respond in natural language, no code."
            )
            result = await codex_as_claude.invoke(prompt=prompt, output_format="text", cwd=str(exp_dir), timeout=120)
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
            result = await claude.invoke(
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
                result = await claude.invoke(
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
                result = await claude.invoke(
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
