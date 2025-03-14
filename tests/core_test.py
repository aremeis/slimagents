import pytest
from dotenv import load_dotenv

from slimagents.core import Agent

# pip install python-dotenv
# pip install pytest
# pip install pytest-asyncio

# .env file must contain OPENAI_API_KEY=sk-...
load_dotenv()


async def stream_response(agent, *inputs):
    chunks = []
    async for chunk in await agent.run(*inputs, stream=True):
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_agent_init():
    agent = Agent(
        instructions="You always answer YES to all questions.",
        temperature=0.0,
    )
    response = await agent.run("Are you a helpful assistant?")
    assert response.value == "YES"
    assert len(response.messages) == 2

@pytest.mark.asyncio
async def test_stream_response():
    agent = Agent(
        instructions="You always answer YES to all questions.",
        temperature=0.0,
    )
    chunks = await stream_response(agent, "Are you a helpful assistant?")
    response = chunks[-1]["response"]
    assert response.value == "YES"
    assert len(response.messages) == 2

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

