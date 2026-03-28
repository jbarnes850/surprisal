"""Live FSM execution — calls real Claude/Codex/Docker agents.

This module implements the full discovery agent pipeline for a single MCTS node:
  experiment_generator → runner → analyst → reviewer → hypothesis → belief
"""
import asyncio
import hashlib
import json
import logging
from pathlib import Path

from surprisal.agents.base import AgentResult
from surprisal.agents.claude import ClaudeAgent
from surprisal.agents.codex import CodexAgent
from surprisal.agents.backends import create_backend, detect_gpu
from surprisal.config import AutoDiscoveryConfig
from surprisal.db import Database
from surprisal.exploration import load_branch_sessions, save_branch_sessions
from surprisal.fsm import select_next_state, FSMResponse
from surprisal.models import AgentInvocation, Node, BeliefSample
from surprisal.bayesian import compute_surprisal
from surprisal.progress import ProgressCallback, emit_progress
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


def _belief_prompt(hypothesis: str, evidence: str = "") -> str:
    evidence_section = ""
    if evidence.strip():
        evidence_section = f"Experimental Evidence:\n{evidence.strip()}"
    return _load_prompt("belief.md", hypothesis=hypothesis, evidence_section=evidence_section)


def _results_contract_error(message: str, detail: str = "") -> str:
    if detail.strip():
        return f"runner results contract violated: {message}. {detail.strip()}"
    return f"runner results contract violated: {message}"


def _hash_text(text: str) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fail_node(db: Database, node_id: str, state: str, reason: str) -> bool:
    logger.error(f"  FSM {node_id}: {state} failed: {reason}")
    db.update_node(node_id, status="failed", fsm_state=state, virtual_loss=0)
    return False


def _agent_error_message(result: AgentResult) -> str:
    if isinstance(result.parsed, dict) and result.parsed.get("is_error") is True:
        inner = result.parsed.get("result")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
        if isinstance(inner, dict):
            return json.dumps(inner)[:500]
        return "provider reported is_error=true"
    if result.exit_code != 0:
        raw = result.raw.strip()
        if raw:
            return f"agent exited with code {result.exit_code}: {raw[:500]}"
        return f"agent exited with code {result.exit_code}"
    raw = result.raw.strip()
    return raw[:500] if raw else "agent returned no usable output"


def _record_invocation(
    db: Database,
    node_id: str,
    role: str,
    provider: str,
    prompt: str,
    result: AgentResult,
) -> None:
    db.insert_agent_invocation(AgentInvocation(
        node_id=node_id,
        role=role,
        provider=provider,
        prompt_hash=_hash_text(prompt),
        response_hash=_hash_text(result.raw),
        duration_seconds=result.duration_seconds,
        exit_code=result.exit_code,
    ))


def _resume_session_args(session_id: str | None) -> dict[str, object]:
    return {
        "session_id": session_id,
        "resume_session": bool(session_id),
    }


def _forked_belief_session_args(session_id: str | None) -> dict[str, object]:
    return {
        "session_id": session_id,
        "resume_session": bool(session_id),
        "fork_session": bool(session_id),
    }


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


def _unwrap_json_payload(parsed: object) -> object:
    """Unwrap provider envelopes while preserving direct JSON payloads."""
    if not isinstance(parsed, dict):
        return parsed

    # Claude JSON mode wraps successful responses in a transport envelope like:
    # {type, subtype, is_error, result, session_id, ...}. For these, the true
    # stage payload lives inside `result`.
    if "result" in parsed and (
        parsed.get("type") is not None
        or parsed.get("subtype") is not None
        or "session_id" in parsed
        or "is_error" in parsed
    ):
        return parsed["result"]

    # Preserve existing behavior for older wrappers that only expose `result`.
    if set(parsed.keys()) == {"result"}:
        return parsed["result"]

    return parsed


def _extract_json(result: AgentResult) -> dict:
    """Try to extract JSON from an agent result."""
    if result.parsed:
        payload = _unwrap_json_payload(result.parsed)
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return _find_json_in_text(payload)
        if isinstance(payload, dict):
            return payload
        return {}
    return _find_json_in_text(result.raw)


def _extract_stage_json(result: AgentResult, stage_name: str) -> tuple[dict | None, str | None]:
    """Extract a JSON payload for a stage, failing closed on provider errors."""
    if result.exit_code != 0:
        return None, f"{stage_name} failed: {_agent_error_message(result)}"
    if isinstance(result.parsed, dict) and result.parsed.get("is_error") is True:
        return None, f"{stage_name} failed: {_agent_error_message(result)}"
    data = _extract_json(result)
    if not data:
        return None, f"{stage_name} failed: agent returned no valid JSON payload"
    if not isinstance(data, dict):
        return None, f"{stage_name} failed: agent returned a non-dict JSON payload"
    return data, None


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
    progress_callback: ProgressCallback | None = None,
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
        # Heterogeneous: Codex for code/analysis, Claude for research/belief
        code_agent = CodexAgent(model=config.agents.codex_model)
        code_provider = "codex"
    elif providers.claude_available:
        # Claude-only: use Sonnet for code/analysis roles
        code_agent = ClaudeAgent(model="sonnet", max_turns=config.agents.max_turns)
        code_provider = "claude"
    elif providers.codex_available:
        # Codex-only: Codex handles code/analysis, but no research agent for
        # hypothesis generation and belief elicitation (these require Claude)
        logger.error("Codex-only mode requires Claude for hypothesis generation and belief elicitation")
        db.update_node(node_id, status="failed", virtual_loss=0)
        return False
    else:
        logger.error("No agent providers available")
        db.update_node(node_id, status="failed", virtual_loss=0)
        return False
    # Auto-detect GPU for Docker backend image selection
    _gpu_available = await detect_gpu() if config.sandbox.backend in ("docker", "local") else None

    node = db.get_node(node_id)
    branch_id = node.branch_id or workspace.name
    exploration_dir = workspace.parent.parent
    branch_sessions = load_branch_sessions(exploration_dir, branch_id)
    research_session_id = (
        node.claude_session_id
        or branch_sessions.get("research_claude_session_id")
        or branch_sessions.get("claude_session_id")
    )
    code_session_id = branch_sessions.get("code_session_id")
    stored_code_provider = branch_sessions.get("code_provider")
    if stored_code_provider != code_provider:
        code_session_id = None
    if code_provider == "codex":
        code_session_id = node.codex_session_id or code_session_id or branch_sessions.get("codex_session_id")
    runner_session_id = branch_sessions.get("runner_claude_session_id")
    initial_session_updates = {}
    if research_session_id and node.claude_session_id != research_session_id:
        initial_session_updates["claude_session_id"] = research_session_id
    if code_provider == "codex" and code_session_id and node.codex_session_id != code_session_id:
        initial_session_updates["codex_session_id"] = code_session_id
    if initial_session_updates:
        db.update_node(node_id, **initial_session_updates)
        node = db.get_node(node_id)
    exp_dir = get_experiment_dir(workspace, node_id)
    state = "start"
    last_response = None
    experiment_plan = ""
    experiment_code = ""
    experiment_output = ""
    analysis_summary = ""
    runner_feedback = ""
    runner_contract_valid = True

    # Build branch context for prompts
    branch_context = "\n".join(
        f"- Depth {n.depth}: {n.hypothesis}" for n in reversed(branch_path)
    )
    max_failures = max(config.agents.code_attempts - 1, 0)
    max_revisions = max(config.agents.revision_attempts, 0)

    def _remember_research_session(result: AgentResult) -> None:
        nonlocal research_session_id, node
        if not result.session_id:
            return
        if result.session_id == research_session_id:
            return
        research_session_id = result.session_id
        save_branch_sessions(
            exploration_dir,
            branch_id,
            research_claude_session_id=research_session_id,
        )
        db.update_node(node_id, claude_session_id=research_session_id)
        node = db.get_node(node_id)

    def _remember_code_session(result: AgentResult) -> None:
        nonlocal code_session_id, node
        if not result.session_id:
            return
        if result.session_id == code_session_id:
            return
        code_session_id = result.session_id
        save_branch_sessions(
            exploration_dir,
            branch_id,
            code_session_id=code_session_id,
            code_provider=code_provider,
        )
        if code_provider == "codex":
            db.update_node(node_id, codex_session_id=code_session_id)
            node = db.get_node(node_id)

    def _remember_runner_session(result: AgentResult) -> None:
        nonlocal runner_session_id
        if not result.session_id:
            return
        if result.session_id == runner_session_id:
            return
        runner_session_id = result.session_id
        save_branch_sessions(
            exploration_dir,
            branch_id,
            runner_claude_session_id=runner_session_id,
        )

    while state not in ("COMPLETE", "FAIL"):
        next_state = select_next_state(state, last_response,
                                        node.fsm_failure_count,
                                        node.fsm_revision_count,
                                        max_failures=max_failures,
                                        max_revisions=max_revisions)
        logger.info(f"  FSM {node_id}: {state} → {next_state}")

        if next_state == "FAIL":
            db.update_node(node_id, status="failed", fsm_state="FAIL", virtual_loss=0)
            return False

        if next_state == "COMPLETE":
            break

        state = next_state
        db.update_node(node_id, fsm_state=state)
        emit_progress(progress_callback, f"Node {node_id}: starting stage `{state}`.")

        # ── Experiment Generator (Claude) ──
        if state == "experiment_generator":
            # IMPORTANT: The generator must NOT use no_tools=True because it needs
            # WebFetch for the HuggingFace fallback path when alphaxiv is unavailable.
            # See spec Section 6.3.

            # The generator inherits the host's MCP servers (including alphaxiv)
            # automatically. No --mcp-config override needed.
            gen_extra_args = []

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
                "Propose one new hypothesis and one executable experiment plan.\n"
                "The runtime can access a full sandboxed research environment with Python, Bash, "
                "network access, HuggingFace datasets/models, and W&B logging when configured.\n"
                "Prefer real datasets, public models, or other sandbox-accessible resources when they materially "
                "strengthen the experiment. Use synthetic or simulated data only when no real dataset is a natural fit "
                "or when the hypothesis is fundamentally algorithmic/simulation-based.\n"
                "Keep the plan self-contained, executable in one workspace, and scoped for a single bounded run. "
                "Name the dataset/model/resource explicitly when using one, and state the main metric or evidence the "
                "runner should produce.\n\n"
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
                **_resume_session_args(research_session_id),
            )
            _remember_research_session(result)
            _record_invocation(db, node_id, "experiment_generator", "claude", prompt, result)
            logger.info(f"  Generator exit={result.exit_code}, len={len(result.raw)}")
            data, stage_error = _extract_stage_json(result, "experiment_generator")
            if stage_error:
                return _fail_node(db, node_id, state, stage_error)

            hypothesis = str(data.get("hypothesis", "")).strip()
            experiment_plan = str(data.get("experiment_plan", "")).strip()
            if not hypothesis:
                return _fail_node(db, node_id, state, "experiment_generator failed: missing hypothesis")
            if not experiment_plan:
                return _fail_node(db, node_id, state, "experiment_generator failed: missing experiment_plan")
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
                "You have full sandbox access, including Bash, local files, public network, HuggingFace datasets/models, "
                "and optional W&B logging. Prefer the most faithful implementation of the plan that is feasible inside one "
                "bounded run. Write the code, execute it, debug if needed, and write structured results to results.json in the current directory."
            )

            backend = create_backend(config.sandbox, config.credentials, gpu_available=_gpu_available)
            result = await backend.execute(
                experiment_prompt=runner_prompt,
                workspace=exp_dir,
                config=config.sandbox,
                system_prompt_file=str(_prompts_dir() / "experiment_runner.md"),
                session_id=runner_session_id,
                progress_callback=progress_callback,
                node_id=node_id,
            )
            _remember_runner_session(result)
            _record_invocation(
                db,
                node_id,
                "experiment_runner",
                type(backend).__name__,
                runner_prompt,
                result,
            )
            logger.info(f"  Runner exit={result.exit_code}, len={len(result.raw)}")

            experiment_output = result.raw
            results_summary = ""
            experiment_metrics = {}
            runner_feedback = ""
            runner_contract_valid = True
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    results_data = json.loads(results_file.read_text())
                except json.JSONDecodeError as exc:
                    results_data = None
                    runner_contract_valid = False
                    runner_feedback = _results_contract_error("results.json is not valid JSON", str(exc))
                if isinstance(results_data, dict):
                    code_value = results_data.get("code")
                    stdout_value = results_data.get("stdout")
                    error_value = results_data.get("error")

                    if not isinstance(code_value, str) or not code_value.strip():
                        runner_contract_valid = False
                        runner_feedback = _results_contract_error("results.json is missing non-empty string field 'code'")
                    elif not isinstance(stdout_value, str):
                        runner_contract_valid = False
                        runner_feedback = _results_contract_error("results.json is missing string field 'stdout'")
                    elif not isinstance(error_value, bool):
                        runner_contract_valid = False
                        runner_feedback = _results_contract_error("results.json is missing boolean field 'error'")
                    else:
                        experiment_code = code_value
                        experiment_output = stdout_value
                        runner_failed = error_value
                        # Extract structured fields (optional — older runners may not produce these)
                        summary_value = results_data.get("results_summary", "")
                        if isinstance(summary_value, str) and summary_value.strip():
                            results_summary = summary_value.strip()
                        metrics_value = results_data.get("metrics")
                        if isinstance(metrics_value, dict):
                            experiment_metrics = metrics_value
                        if runner_failed:
                            error_message = results_data.get("error_message", "")
                            if isinstance(error_message, str) and error_message.strip():
                                runner_feedback = error_message.strip()
                elif results_data is not None:
                    runner_contract_valid = False
                    runner_feedback = _results_contract_error("results.json must contain a JSON object")
            else:
                experiment_code = (exp_dir / "experiment.py").read_text() if (exp_dir / "experiment.py").exists() else ""
                runner_contract_valid = False
                runner_feedback = _results_contract_error("results.json was not written")

            if not runner_contract_valid:
                runner_failed = True
                if not experiment_output.strip():
                    experiment_output = result.raw
                if runner_feedback:
                    experiment_output = (
                        f"{experiment_output}\n\nRUNNER CONTRACT STATUS: {runner_feedback}".strip()
                    )
            elif runner_failed and runner_feedback:
                experiment_output = (
                    f"{experiment_output}\n\nRUNNER ERROR: {runner_feedback}".strip()
                )

            db.update_node(node_id, experiment_exit_code=result.exit_code)
            node.experiment_exit_code = result.exit_code

            # Docker-specific infra errors (exit 125-127) — only check for Docker backend
            is_infra = (
                hasattr(backend, "is_infra_error")
                and backend.is_infra_error(result.exit_code)
            )
            if is_infra:
                last_response = FSMResponse(error=True, exit_code=result.exit_code)
            else:
                last_response = FSMResponse(
                    error=runner_failed,
                    exit_code=result.exit_code,
                )

        # ── Experiment Analyst (Codex) ──
        elif state == "experiment_analyst":
            if not runner_contract_valid:
                last_response = FSMResponse(error=True, feedback=runner_feedback)
                continue

            stderr_text = ""
            stderr_path = exp_dir / "stderr.txt"
            if stderr_path.exists():
                stderr_text = stderr_path.read_text()[:2000]

            # Prefer structured results_summary + metrics over raw stdout.
            # Falls back to stdout tail when the runner didn't produce a summary.
            if results_summary:
                evidence_block = (
                    f"Results summary:\n{results_summary}\n\n"
                    f"Metrics:\n{json.dumps(experiment_metrics, indent=2) if experiment_metrics else 'none'}\n\n"
                    f"Stdout (tail):\n{experiment_output[-2000:]}"
                )
            else:
                evidence_block = f"Stdout:\n{experiment_output[-3000:]}"
            prompt = (
                f"Experiment plan:\n{experiment_plan}\n\n"
                f"Runner feedback:\n{runner_feedback or 'none'}\n\n"
                f"Exit code: {node.experiment_exit_code}\n\n"
                f"{evidence_block}\n\n"
                f"Stderr:\n{stderr_text}\n\n"
                f"Code:\n{experiment_code[:2000]}"
            )
            result = await code_agent.invoke(
                prompt=prompt,
                system_prompt_file=str(_prompts_dir() / "experiment_analyst.md"),
                output_format="text",
                cwd=str(exp_dir),
                timeout=120,
                **_resume_session_args(code_session_id),
            )
            _remember_code_session(result)
            _record_invocation(
                db,
                node_id,
                "experiment_analyst",
                type(code_agent).__name__,
                prompt,
                result,
            )
            logger.info(f"  Analyst exit={result.exit_code}, len={len(result.raw)}")
            data, stage_error = _extract_stage_json(result, "experiment_analyst")
            if stage_error:
                return _fail_node(db, node_id, state, stage_error)
            elif "error" not in data or not isinstance(data["error"], bool):
                return _fail_node(db, node_id, state, "experiment_analyst failed: missing boolean 'error' field")
            elif data["error"]:
                last_response = FSMResponse(
                    error=True,
                    feedback=str(data.get("feedback", "experiment_analyst reported an unspecified error")).strip(),
                )
            else:
                analysis_summary = str(data.get("summary", "")).strip()
                if not analysis_summary:
                    return _fail_node(
                        db, node_id, state,
                        "experiment_analyst failed: missing summary for successful analysis",
                    )
                (exp_dir / "analysis.md").write_text(analysis_summary)
                last_response = FSMResponse(error=False, data=data)

        # ── Experiment Reviewer (Codex) ──
        elif state == "experiment_reviewer":
            sys_prompt_reviewer = str(_prompts_dir() / "experiment_reviewer.md")
            if results_summary:
                reviewer_evidence = (
                    f"Results summary:\n{results_summary}\n\n"
                    f"Metrics:\n{json.dumps(experiment_metrics, indent=2) if experiment_metrics else 'none'}\n\n"
                    f"Execution output (tail):\n{experiment_output[-2000:]}"
                )
            else:
                reviewer_evidence = f"Execution output:\n{experiment_output[-2500:]}"
            prompt = (
                f"Hypothesis: {db.get_node(node_id).hypothesis}\n\n"
                f"Experiment plan:\n{experiment_plan}\n\n"
                f"{reviewer_evidence}\n\n"
                f"Analysis summary:\n{analysis_summary}\n\n"
                f"Code:\n{experiment_code[:2000]}"
            )
            # Fresh session: reviewer gets full context in the prompt and
            # Codex exec resume doesn't support -o for clean output capture.
            result = await code_agent.invoke(
                prompt=prompt,
                system_prompt_file=sys_prompt_reviewer,
                output_format="text",
                cwd=str(exp_dir),
                timeout=120,
            )
            _remember_code_session(result)
            _record_invocation(
                db,
                node_id,
                "experiment_reviewer",
                type(code_agent).__name__,
                prompt,
                result,
            )
            logger.info(f"  Reviewer exit={result.exit_code}, len={len(result.raw)}")
            data, stage_error = _extract_stage_json(result, "experiment_reviewer")
            if stage_error:
                (exp_dir / "review.md").write_text(f"FAILED: {stage_error}")
                return _fail_node(db, node_id, state, stage_error)
            elif "error" not in data or not isinstance(data["error"], bool):
                feedback = "experiment_reviewer failed: missing boolean 'error' field"
                (exp_dir / "review.md").write_text(f"FAILED: {feedback}")
                return _fail_node(db, node_id, state, feedback)
            elif data["error"]:
                # NOTE: do NOT increment revision_count here — select_next_state
                # checks the current value to decide if revision is allowed.
                # The count is incremented ONLY when we actually enter experiment_reviser.
                feedback = str(data.get("feedback", "experiment_reviewer rejected the experiment")).strip()
                last_response = FSMResponse(error=True, feedback=feedback)
                (exp_dir / "review.md").write_text(f"REJECTED: {feedback}")
            else:
                assessment = str(data.get("assessment", "")).strip()
                if not assessment:
                    feedback = "experiment_reviewer failed: missing assessment for approved review"
                    (exp_dir / "review.md").write_text(f"FAILED: {feedback}")
                    return _fail_node(db, node_id, state, feedback)
                (exp_dir / "review.md").write_text(f"APPROVED: {assessment}")
                last_response = FSMResponse(error=False, data=data)

        # ── Experiment Reviser (Codex) ──
        elif state == "experiment_reviser":
            node.fsm_revision_count += 1
            db.update_node(node_id, fsm_revision_count=node.fsm_revision_count)
            prompt = (
                f"The experiment was rejected. Feedback: {last_response.feedback if last_response else ''}\n\n"
                f"Hypothesis: {db.get_node(node_id).hypothesis}\n\n"
                f"Original plan: {experiment_plan}\n\n"
                "Revise the experiment plan. Respond in natural language, no code."
            )
            # Fresh session: reviser gets full context in the prompt.
            result = await code_agent.invoke(
                prompt=prompt,
                system_prompt_file=str(_prompts_dir() / "experiment_reviser.md"),
                output_format="text",
                cwd=str(exp_dir),
                timeout=120,
            )
            _record_invocation(
                db,
                node_id,
                "experiment_reviser",
                type(code_agent).__name__,
                prompt,
                result,
            )
            if result.exit_code != 0 or (isinstance(result.parsed, dict) and result.parsed.get("is_error") is True):
                return _fail_node(db, node_id, state, f"experiment_reviser failed: {_agent_error_message(result)}")
            experiment_plan = result.raw.strip()
            if not experiment_plan:
                return _fail_node(db, node_id, state, "experiment_reviser failed: empty revised plan")
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
                **_resume_session_args(research_session_id),
            )
            _remember_research_session(result)
            _record_invocation(db, node_id, "hypothesis_generator", "claude", prompt, result)
            data, stage_error = _extract_stage_json(result, "hypothesis_generator")
            if stage_error:
                return _fail_node(db, node_id, state, stage_error)

            hypothesis = str(data.get("hypothesis", "")).strip()
            if not hypothesis:
                return _fail_node(db, node_id, state, "hypothesis_generator failed: missing hypothesis")

            db.update_node(node_id,
                hypothesis=hypothesis,
                context=data.get("context", ""),
                variables=json.dumps(data.get("variables", [])),
                relationships=json.dumps(data.get("relationships", [])),
            )
            last_response = FSMResponse(error=False, data=data)

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

    # Mark as verified
    db.update_node(node_id, status="verified", virtual_loss=0)
    emit_progress(progress_callback, f"Node {node_id}: verified.")
    node = db.get_node(node_id)
    return True
