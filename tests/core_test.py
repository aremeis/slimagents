# pip install python-dotenv
# pip install pytest
# pip install pytest-asyncio
# pip install diskcache

import json
from time import time
import pytest
from dotenv import load_dotenv
from slimagents import Agent
import slimagents.config as config
import litellm
from litellm.caching.caching import Cache

# Set caching configuration
config.caching = True
litellm.cache = Cache(type="disk", disk_cache_dir="./tests/llm_cache")
# cache.clear()

# .env file must contain OPENAI_API_KEY=sk-...
load_dotenv()


@pytest.mark.asyncio
async def test_agent_init():
    agent = Agent(
        instructions="You always answer YES to all questions.",
        temperature=0.0,
    )
    response = await agent.run("Are you a helpful assistant?")
    assert response.value == "YES"
    assert len(response.messages) == 2
    assert response.messages[0]["role"] == "user"
    assert response.messages[0]["content"] == "Are you a helpful assistant?"
    assert response.messages[1]["role"] == "assistant"
    assert response.messages[1]["content"] == "YES"


@pytest.mark.asyncio
async def test_stream_response():
    agent = Agent(
        instructions="You always answer YES to all questions.",
        temperature=0.0,
    )
    input = "Are you a helpful assistant?"
    chunks = []
    async for chunk in await agent.run(input, stream=True, stream_tokens=False, stream_delimiters=True, stream_response=True):
        chunks.append(chunk)
    assert chunks[0] == {"delim": "start"}
    assert chunks[1]["content"] == "YES"
    assert chunks[2] == {"delim": "end"}
    response = chunks[3]
    assert response.value == "YES"
    assert len(response.messages) == 2
    assert response.messages[0]["role"] == "user"
    assert response.messages[0]["content"] == "Are you a helpful assistant?"
    assert response.messages[1]["role"] == "assistant"
    assert response.messages[1]["content"] == "YES"


@pytest.mark.asyncio
async def test_memory():
    default_memory = [
            {
                "role": "user",
                "content": "Is the sky blue?"
            },
            {
                "role": "assistant",
                "content": "YES"
            },
            {
                "role": "user",
                "content": "Is the sky blue?"
            },
            {
                "role": "assistant",
                "content": "NO"
            },
        ]
    memory = [
            {
                "role": "user",
                "content": "Is the sky blue?"
            },
            {
                "role": "assistant",
                "content": "YES"
            },

    ]
    initial_memory = memory.copy()
    agent = Agent(
        instructions="You follow communication patterns and like to play games with the user.",
        temperature=0.0,
        memory=default_memory,
    )
    response = await agent.run("Is the sky blue?", memory=memory)
    assert response.value == "NO"
    # assert response.messages == [{"role": "user", "content": "Is the sky blue?"}, {"role": "assistant", "content": "NO"}]
    assert agent.memory == default_memory
    assert memory == initial_memory + response.messages


@pytest.mark.asyncio
async def test_tool_calls():
    def calculator(expression: str) -> int:
        """
        Calculate the result of the expression.
        """
        return eval(expression)
    agent = Agent(
        instructions="You are an expert calculator. You always use the calculator to answer questions.",
        temperature=0.0,
        tools=[calculator],
    )
    input = "What is 2 + 2?"
    response = await agent.run(input)
    assert len(response.messages) == 4
    assert response.messages[0] == {"role": "user", "content": "What is 2 + 2?"}
    assert response.messages[1]["tool_calls"][0]["type"] == "function"
    assert response.messages[1]["tool_calls"][0]["function"]["name"] == "calculator"
    arguments = json.loads(response.messages[1]["tool_calls"][0]["function"]["arguments"])
    assert arguments["expression"] == "2 + 2"
    assert response.messages[2]["role"] == "tool"
    assert response.messages[2]["content"] == "4"
    assert response.messages[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_stream_tool_calls():
    def calculator(expression: str) -> int:
        """
        Calculate the result of the expression.
        """
        return eval(expression)
    agent = Agent(
        instructions="You are an expert calculator. You always use the calculator to answer questions.",
        temperature=0.0,
        tools=[calculator],
    )
    input = "What is 2 + 2?"
    chunks = []
    output = ""
    async for chunk in await agent.run(input, stream=True, stream_tool_calls=True, stream_response=True):
        if isinstance(chunk, str):
            output += chunk
        else:
            chunks.append(chunk)
    response = chunks[-1]
    assert response.value == output
    assert len(response.messages) == 4
    assert response.messages[0] == {"role": "user", "content": "What is 2 + 2?"}
    assert response.messages[1]["tool_calls"][0]["type"] == "function"
    assert response.messages[1]["tool_calls"][0]["function"]["name"] == "calculator"
    arguments = json.loads(response.messages[1]["tool_calls"][0]["function"]["arguments"])
    assert arguments["expression"] == "2 + 2"
    assert response.messages[2]["role"] == "tool"
    assert response.messages[2]["content"] == "4"
    assert response.messages[3]["role"] == "assistant"
    assert response.messages[3]["content"] == output


@pytest.mark.asyncio
# @pytest.mark.skip(reason="Skip to avoid token cost")
async def test_caching():
    class CachingAgent(Agent):
        async def get_chat_completion(self, *args, **kwargs):
            t0 = time()
            res = await super().get_chat_completion(*args, **kwargs)
            t1 = time()
            if t1 - t0 > 0.100:
                raise Exception("Time taken to get completion is too long - caching is not working?")
            return res
        
    agent = CachingAgent(
        instructions="You always answer YES to all questions.",
        temperature=0.0,
    )
    response = await agent.run("Are you a helpful assistant?")
    assert response.value == "YES"

    try:
        response = await agent.run("Are you a helpful assistant?", caching=False)
    except Exception as e:
        assert str(e) == "Time taken to get completion is too long - caching is not working?"
        
    try:
        config.caching = False
        response2 = await agent.run("Are you a helpful assistant?", caching=True)
        await agent.run("Are you a helpful assistant?")
    except Exception as e:
        assert response2.value == "YES"
        assert str(e) == "Time taken to get completion is too long - caching is not working?"
    finally:
        config.caching = True
