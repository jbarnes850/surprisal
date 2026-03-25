import pytest
import asyncio
from unittest.mock import MagicMock
from surprisal.orchestrator import AtomicCounter


@pytest.mark.asyncio
async def test_atomic_counter_decrements():
    counter = AtomicCounter(3)
    assert await counter.decrement() is True
    assert await counter.decrement() is True
    assert await counter.decrement() is True
    assert await counter.decrement() is False


@pytest.mark.asyncio
async def test_atomic_counter_zero():
    counter = AtomicCounter(0)
    assert await counter.decrement() is False
