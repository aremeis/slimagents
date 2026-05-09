"""
Type checking tests for Agent generics.

Validates that mypy correctly infers types based on response_format.
The actual type assertions live in this file but are only checked by mypy (not executed at runtime).
Run standalone: mypy tests/type_test.py --ignore-missing-imports
"""

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel
from slimagents import Agent
from slimagents.core import Response


def test_mypy_type_checking():
    """Run mypy on this file to validate generic type inference."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", __file__, "--ignore-missing-imports", "--no-incremental"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"mypy failed:\n{result.stdout}\n{result.stderr}"


# ============================================================================
# Everything below is only analyzed by mypy, never executed at runtime.
# ============================================================================

if TYPE_CHECKING:

    class MyModel(BaseModel):
        name: str
        score: float

    # --- Agent[str] (default, no response_format) ---

    def _check_str_agent() -> None:
        agent = Agent(instructions="test")
        _: Agent[str] = agent

    async def _check_str_run() -> None:
        agent = Agent(instructions="test")
        response = await agent.run("hello")
        val: str = response.value
        _ = val.upper()

    async def _check_str_call() -> None:
        agent = Agent(instructions="test")
        val = await agent("hello")
        _: str = val
        _ = val.upper()

    # --- Agent[T] with BaseModel response_format ---

    def _check_model_agent() -> None:
        agent = Agent(instructions="test", response_format=MyModel)
        _: Agent[MyModel] = agent

    async def _check_model_run() -> None:
        agent = Agent(instructions="test", response_format=MyModel)
        response = await agent.run("hello")
        val: MyModel = response.value
        _name: str = val.name
        _score: float = val.score

    async def _check_model_call() -> None:
        agent: Agent[MyModel] = Agent(instructions="test", response_format=MyModel)
        val = await agent("hello")
        _m: MyModel = val
        _n: str = val.name

    # --- Response type parameter ---

    def _check_response_type(r: Response[str], r2: Response[MyModel]) -> None:
        _: str = r.value
        _2: MyModel = r2.value

    # --- apply() return type ---

    def _check_apply_return() -> None:
        agent = Agent(instructions="test")
        val = agent.apply("hello")
        _: str = val

        agent2: Agent[MyModel] = Agent(instructions="test", response_format=MyModel)
        val2 = agent2.apply("hello")
        _2: MyModel = val2

    # --- run_sync() return type ---

    def _check_run_sync_return() -> None:
        agent = Agent(instructions="test")
        response = agent.run_sync("hello")
        _: str = response.value

        agent2: Agent[MyModel] = Agent(instructions="test", response_format=MyModel)
        response2 = agent2.run_sync("hello")
        _2: MyModel = response2.value
