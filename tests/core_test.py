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
from pydantic import BaseModel

from slimagents.core import Result
# Set caching configuration
config.caching = True
litellm.cache = Cache(type="disk", disk_cache_dir="./tests/llm_cache")
# cache.clear()

# .env file must contain OPENAI_API_KEY=sk-...
load_dotenv()


def calculator(expression: str) -> int:
    """
    Calculate the result of the expression.
    """
    # Remove all characters except numbers, operators and parentheses (for safety)
    expression = ''.join(char for char in expression if char in '0123456789+-*/(). ')
    return eval(expression)


@pytest.mark.asyncio
async def test_agent_init():
    agent = Agent(
        instructions="You always answer YES to all questions.",
        temperature=0.0,
    )
    memory_delta = []
    value = await agent("Are you a helpful assistant?", memory_delta=memory_delta)
    assert value == "YES"
    assert len(memory_delta) == 2
    assert memory_delta[0]["role"] == "user"
    assert memory_delta[0]["content"] == "Are you a helpful assistant?"
    assert memory_delta[1]["role"] == "assistant"
    assert memory_delta[1]["content"] == "YES"


@pytest.mark.asyncio
async def test_stream_response():
    agent = Agent(
        instructions="You always answer YES to all questions.",
        temperature=0.0,
    )
    input = "Are you a helpful assistant?"
    chunks = []
    async for chunk in await agent(input, stream=True, stream_tokens=False, stream_delimiters=True, stream_response=True):
        chunks.append(chunk)
    assert chunks[0] == {"delim": "start"}
    assert chunks[1]["content"] == "YES"
    assert chunks[2] == {"delim": "end"}
    response = chunks[3]
    assert response.value == "YES"
    assert len(response.memory_delta) == 2
    assert response.memory_delta[0]["role"] == "user"
    assert response.memory_delta[0]["content"] == "Are you a helpful assistant?"
    assert response.memory_delta[1]["role"] == "assistant"
    assert response.memory_delta[1]["content"] == "YES"


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
    memory_delta = []
    value = await agent("Is the sky blue?", memory=memory, memory_delta=memory_delta)
    assert value == "NO"
    assert agent.memory == default_memory
    assert len(memory_delta) == 2
    assert memory_delta[0]["role"] == "user"
    assert memory_delta[0]["content"] == "Is the sky blue?"
    assert memory_delta[1]["role"] == "assistant"
    assert memory_delta[1]["content"] == "NO"
    assert memory == initial_memory + memory_delta


@pytest.mark.asyncio
async def test_tool_calls():
    agent = Agent(
        instructions="You are an expert calculator. You always use the calculator to answer questions.",
        temperature=0.0,
        tools=[calculator],
    )
    input = "What is 2 + 2?"
    response = await agent.run(input)
    assert len(response.memory_delta) == 4
    assert response.memory_delta[0] == {"role": "user", "content": "What is 2 + 2?"}
    assert response.memory_delta[1]["tool_calls"][0]["type"] == "function"
    assert response.memory_delta[1]["tool_calls"][0]["function"]["name"] == "calculator"
    arguments = json.loads(response.memory_delta[1]["tool_calls"][0]["function"]["arguments"])
    assert arguments["expression"] == "2 + 2"
    assert response.memory_delta[2]["role"] == "tool"
    assert response.memory_delta[2]["content"] == "4"
    assert response.memory_delta[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_stream_tool_calls():
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
    assert len(response.memory_delta) == 4
    assert response.memory_delta[0] == {"role": "user", "content": "What is 2 + 2?"}
    assert response.memory_delta[1]["tool_calls"][0]["type"] == "function"
    assert response.memory_delta[1]["tool_calls"][0]["function"]["name"] == "calculator"
    arguments = json.loads(response.memory_delta[1]["tool_calls"][0]["function"]["arguments"])
    assert arguments["expression"] == "2 + 2"
    assert response.memory_delta[2]["role"] == "tool"
    assert response.memory_delta[2]["content"] == "4"
    assert response.memory_delta[3]["role"] == "assistant"
    assert response.memory_delta[3]["content"] == output


@pytest.mark.asyncio
@pytest.mark.skip(reason="Skip to avoid token cost")
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
    value = await agent("Are you a helpful assistant?")
    assert value == "YES"

    try:
        value = await agent("Are you a helpful assistant?", caching=False)
    except Exception as e:
        assert str(e) == "Time taken to get completion is too long - caching is not working?"
        
    try:
        config.caching = False
        value2 = await agent("Are you a helpful assistant?", caching=True)
        await agent("Are you a helpful assistant?")
    except Exception as e:
        assert value2 == "YES"
        assert str(e) == "Time taken to get completion is too long - caching is not working?"
    finally:
        config.caching = True


@pytest.mark.asyncio
async def test_response_format():
    agent = Agent(
        instructions="You are a calculator",
        temperature=0.0,
        response_format=int,
    )
    value = await agent("What is 2 + 2?")
    assert value == 4

    agent.response_format = float
    value = await agent("What is 2 + 2.5?")
    assert value == 4.5

    agent.instructions = "You are a validator"
    agent.response_format = bool
    value = await agent("The sky is blue")
    assert value is True
    value = await agent("The sky is green")
    assert value is False

    agent.instructions = "You are a list generator"
    agent.response_format = list[int]
    value = await agent("Generate a list of numbers, from 1 to 3")
    assert value == [1, 2, 3]

    agent.instructions = "You extract entities in JSON format from text"
    agent.response_format = dict
    value = await agent("A man called John Doe lives in New York")
    assert value == {
        "Person": "John Doe",
        "Location": "New York",
    }

    class Person(BaseModel):
        first_name: str
        last_name: str

    agent.instructions = "You extract person information from text"
    agent.response_format = list[Person]
    value = await agent("Some famous people are John Doe, Jane Smith and Jim Beam")
    assert value[0].first_name == "John"
    assert value[0].last_name == "Doe"
    assert value[1].first_name == "Jane"
    assert value[1].last_name == "Smith"
    assert value[2].first_name == "Jim"
    assert value[2].last_name == "Beam"

@pytest.mark.asyncio
async def test_non_string_output():
    def calculator(input: str) -> float:
        ret = eval(input)
        return Result(value=ret, exit=True)
    memory = []
    agent = Agent(
        instructions="You always use the calculator tool to calculate mathematical expressions.",
        temperature=0.0,
        tools=[calculator],
        memory=memory,
    )
    memory_delta = []
    value = await agent("What is 2 + 2?", memory=memory, memory_delta=memory_delta)
    assert len(memory_delta) == 3
    assert value == 4
    memory_delta = []
    value = await agent("What is 3 + 3?", memory=memory, memory_delta=memory_delta)
    assert len(memory_delta) == 3
    assert value == 6