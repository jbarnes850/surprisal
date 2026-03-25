import asyncio
import signal
import logging
from pathlib import Path
from autodiscovery.config import AutoDiscoveryConfig
from autodiscovery.db import Database
from autodiscovery.mcts import select_node, backpropagate, max_children
from autodiscovery.models import Node
from autodiscovery.workspace import create_workspace, assign_branch_id, write_branch_context, write_claude_md

logger = logging.getLogger("autodiscovery")


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


async def run_exploration(
    db: Database,
    exploration_dir: Path,
    budget: int,
    concurrency: int,
    c_explore: float,
    config: AutoDiscoveryConfig,
    root_id: str,
    domain: str,
) -> dict:
    """Main exploration loop -- spawns worker pool, handles shutdown."""
    # Count completed nodes for any exploration
    all_nodes = db.execute("SELECT COUNT(*) FROM nodes WHERE status IN ('verified', 'failed')").fetchone()[0]
    remaining = budget - all_nodes

    if remaining <= 0:
        return {"status": "budget_exhausted", "iterations": 0}

    db.reset_stale_expanding()

    selection_lock = asyncio.Lock()
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
                        shutdown, c_explore, config, root_id, domain)
        )
        for _ in range(concurrency)
    ]

    await asyncio.gather(*workers, return_exceptions=True)

    total_completed = db.execute("SELECT COUNT(*) FROM nodes WHERE status IN ('verified', 'failed')").fetchone()[0]
    surprisals = db.execute("SELECT COUNT(*) FROM nodes WHERE belief_shifted = 1").fetchone()[0]

    return {
        "status": "completed" if not shutdown.is_set() else "interrupted",
        "iterations": total_completed - all_nodes,
        "nodes_total": db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "surprisals_found": surprisals,
    }


async def worker_loop(
    db: Database,
    exploration_dir: Path,
    selection_lock: asyncio.Lock,
    counter: AtomicCounter,
    shutdown: asyncio.Event,
    c_explore: float,
    config: AutoDiscoveryConfig,
    root_id: str,
    domain: str,
) -> None:
    """Per-worker loop: select -> expand -> execute -> backpropagate."""
    import uuid
    from autodiscovery.fsm import select_next_state, FSMResponse

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
            branch_id = assign_branch_id(
                parent.branch_id or "root",
                parent_has_other_children=len(existing_children) > 0,
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
        branch_path = db.get_path_to_root(child_id)
        write_branch_context(ws, branch_path)
        write_claude_md(ws, domain, branch_path)

        # Run full FSM with real agent calls
        logger.info(f"Worker expanding node {child_id} at depth {child.depth}")
        try:
            from autodiscovery.fsm_runner import run_live_fsm
            success = await run_live_fsm(
                node_id=child_id,
                db=db,
                config=config,
                workspace=ws,
                domain=domain,
                branch_path=branch_path,
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
