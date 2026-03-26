import asyncio
import logging
from pathlib import Path

import signal

from surprisal.config import AutoDiscoveryConfig
from surprisal.db import Database
from surprisal.dedup import deduplicate
from surprisal.mcts import select_node, backpropagate
from surprisal.models import Node
from surprisal.providers import LiteratureStatus, ProviderStatus, detect_literature_provider, detect_providers
from surprisal.workspace import (
    assign_branch_id,
    copy_parent_memory,
    create_workspace,
    write_branch_context,
    write_claude_md,
)

logger = logging.getLogger("surprisal")


class AtomicCounter:
    """Thread-safe counter for budget tracking."""
    def __init__(self, initial: int):
        self._value = initial
        self._lock = asyncio.Lock()

    async def decrement(self) -> bool:
        async with self._lock:
            if self._value <= 0:
                return False
            self._value -= 1
            return True

    @property
    def value(self) -> int:
        return self._value


class DedupCheckpoint:
    """Track the last completed-expansion count that triggered deduplication."""

    def __init__(self):
        self.last_completed = 0


def _format_worker_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def run_exploration(
    db: Database,
    exploration_dir: Path,
    budget: int,
    concurrency: int,
    c_explore: float,
    config: AutoDiscoveryConfig,
    root_id: str,
    domain: str,
    providers: ProviderStatus | None = None,
    literature_provider: LiteratureStatus | None = None,
) -> dict:
    """Main exploration loop -- spawns worker pool, handles shutdown."""
    # Detect available providers if not passed in
    if providers is None:
        providers = await detect_providers()
    if not providers.any_available:
        return {"status": "error", "message": "No agent providers available"}

    # Detect literature search provider if not passed in
    if literature_provider is None:
        literature_provider = await detect_literature_provider()

    root = db.get_node(root_id)
    if root is None:
        return {"status": "error", "message": f"Root node {root_id} not found"}
    exploration_id = root.exploration_id

    completed_before = db.count_completed_expansions(exploration_id)
    remaining = budget - completed_before

    if remaining <= 0:
        return {"status": "budget_exhausted", "iterations": 0}

    db.reset_stale_expanding(exploration_id=exploration_id)

    selection_lock = asyncio.Lock()
    dedup_lock = asyncio.Lock()
    dedup_checkpoint = DedupCheckpoint()
    counter = AtomicCounter(remaining)
    shutdown = asyncio.Event()

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: shutdown.set())
    except NotImplementedError:
        # Signal handlers are not supported on some platforms (e.g. Windows)
        pass

    workers = [
        asyncio.create_task(
            worker_loop(db, exploration_dir, selection_lock, counter,
                        dedup_lock, dedup_checkpoint, shutdown, c_explore, config,
                        root_id, domain, providers, literature_provider)
        )
        for _ in range(concurrency)
    ]

    worker_results = await asyncio.gather(*workers, return_exceptions=True)
    worker_errors = [result for result in worker_results if isinstance(result, BaseException)]
    if worker_errors:
        messages = [_format_worker_error(err) for err in worker_errors]
        logger.error("Exploration aborted due to worker failure(s): %s", "; ".join(messages))
        return {
            "status": "error",
            "message": f"Worker failure(s): {'; '.join(messages)}",
            "worker_errors": messages,
            "iterations": db.count_completed_expansions(exploration_id) - completed_before,
            "nodes_total": db.count_nodes(exploration_id),
            "surprisals_found": db.count_surprisals(exploration_id),
        }

    total_completed = db.count_completed_expansions(exploration_id)
    surprisals = db.count_surprisals(exploration_id)

    return {
        "status": "completed" if not shutdown.is_set() else "interrupted",
        "iterations": total_completed - completed_before,
        "nodes_total": db.count_nodes(exploration_id),
        "surprisals_found": surprisals,
    }


async def worker_loop(
    db: Database,
    exploration_dir: Path,
    selection_lock: asyncio.Lock,
    counter: AtomicCounter,
    dedup_lock: asyncio.Lock,
    dedup_checkpoint: DedupCheckpoint,
    shutdown: asyncio.Event,
    c_explore: float,
    config: AutoDiscoveryConfig,
    root_id: str,
    domain: str,
    providers: ProviderStatus | None = None,
    literature_provider: LiteratureStatus | None = None,
) -> None:
    """Per-worker loop: select -> expand -> execute -> backpropagate."""
    import uuid

    while not shutdown.is_set():
        if not await counter.decrement():
            return

        # Atomic selection + child creation
        async with selection_lock:
            parent_id = select_node(
                db, root_id, c_explore,
                k=config.mcts.k_progressive,
                alpha=config.mcts.alpha_progressive,
                max_depth=config.mcts.max_depth,
            )
            parent = db.get_node(parent_id)

            # Create child node
            child_id = uuid.uuid4().hex[:12]
            existing_children = db.get_children(parent_id, exclude_pruned=True)
            is_branch_divergence = len(existing_children) > 0
            branch_id = assign_branch_id(
                parent.branch_id or "root",
                parent_has_other_children=is_branch_divergence,
            )

            child = Node(
                id=child_id,
                exploration_id=parent.exploration_id,
                hypothesis="(pending expansion)",
                parent_id=parent_id,
                depth=parent.depth + 1,
                status="expanding",
                branch_id=branch_id,
                virtual_loss=config.mcts.virtual_loss,
            )
            db.insert_node(child)

        # Set up workspace
        workspaces_dir = exploration_dir / "workspaces"
        ws = create_workspace(workspaces_dir, branch_id)
        if is_branch_divergence and parent.branch_id and branch_id != parent.branch_id:
            parent_workspace = workspaces_dir / parent.branch_id
            if parent_workspace.exists():
                copy_parent_memory(parent_workspace, ws)
        branch_path = db.get_path_to_root(child_id)
        write_branch_context(ws, branch_path)
        write_claude_md(ws, domain, branch_path)

        # Run full FSM with real agent calls
        logger.info(f"Worker expanding node {child_id} at depth {child.depth}")
        try:
            from surprisal.fsm_runner import run_live_fsm
            success = await run_live_fsm(
                node_id=child_id,
                db=db,
                config=config,
                workspace=ws,
                domain=domain,
                branch_path=branch_path,
                providers=providers,
                literature_provider=literature_provider,
            )
        except Exception as e:
            logger.error(f"FSM error for node {child_id}: {e}")
            db.update_node(child_id, status="failed", virtual_loss=0)
            success = False

        # Backpropagate surprisal
        node = db.get_node(child_id)
        surprisal_value = 1 if node.belief_shifted else 0
        backpropagate(db, child_id, surprisal_value)
        logger.info(
            f"Node {child_id} completed: success={success}, "
            f"surprisal={surprisal_value}, BS={node.bayesian_surprise}"
        )

        if config.mcts.dedup_interval > 0:
            completed = db.count_completed_expansions(parent.exploration_id)
            if completed > 0 and completed % config.mcts.dedup_interval == 0:
                async with dedup_lock:
                    completed = db.count_completed_expansions(parent.exploration_id)
                    should_dedup = (
                        completed > 0
                        and completed % config.mcts.dedup_interval == 0
                        and completed != dedup_checkpoint.last_completed
                    )
                    if should_dedup:
                        summary = deduplicate(db, exploration_id=parent.exploration_id)
                        dedup_checkpoint.last_completed = completed
                        logger.info(
                            "Deduplication after %s expansions: %s clusters, %s duplicates",
                            completed,
                            summary["clusters"],
                            summary["duplicates"],
                        )
