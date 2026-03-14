# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

slimagents is a lightweight Python agent framework built on LiteLLM. It provides a generic `Agent[T]` class that supports tool calling, streaming, agent handoffs, multi-modal inputs, and structured responses (JSON/Pydantic). Default model is `gpt-4.1`. Python 3.10+ required.

## Commands

```bash
# Install for development
pip install -e .

# Run all tests
python -m pytest

# Run a single test
python -m pytest tests/core_test.py -k test_agent_init

# Build and release (uses release.sh which tags, builds, and uploads to PyPI)
./release.sh
```

## Architecture

The codebase is small — four source modules under `slimagents/`:

- **core.py** — The `Agent[T]` class and all supporting dataclasses (`Response[T]`, `ToolResult`, `FileContent`, `ResponseMetadata`, `MessageDelimiter`). Contains the full run loop: LLM calls via LiteLLM, tool execution (sync and async, parallel), streaming, agent handoffs, and response parsing. This is the bulk of the codebase.
- **util.py** — Schema conversion utilities: `function_to_json()` converts Python callables to OpenAI tool schemas, `type_to_response_format()` handles structured output formatting, and helpers for JSON schema flattening and MIME type detection.
- **config.py** — Global settings: `caching`, logger configuration, debug flags.
- **repl/repl.py** — Interactive CLI loop (`run_demo_loop`) for testing agents with streaming output.

### Key patterns

- `Agent` is generic over its response type `T`, determined by `response_format`: `None`→`str`, `dict`→`dict`, `BaseModel` subclass→that class.
- `run()` is the main async entry point; `run_sync()` wraps it for synchronous use; `apply()` is a convenience that returns just the value.
- Tools can return an `Agent` instance to trigger a handoff (control transfer with shared memory).
- Streaming supports multiple granularities: tokens, delimiters, tool calls, and final response objects.
- Tests use disk-cached LLM responses (`tests/llm_cache/`) to avoid live API calls.
