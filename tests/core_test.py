# pip install python-dotenv
# pip install pytest
# pip install pytest-asyncio
# pip install diskcache

import json
from textwrap import dedent
from time import time
import pytest
from dotenv import load_dotenv
from slimagents import Agent
import slimagents.config as config
import litellm
from litellm.caching.caching import Cache
from pydantic import AnyUrl, BaseModel
import re
import logging
import io
from contextlib import contextmanager

from slimagents.core import Delimiter, ToolResult
# Set caching configuration
config.caching = True
litellm.cache = Cache(type="disk", disk_cache_dir="./tests/llm_cache")
# cache.clear()

# .env file must contain OPENAI_API_KEY=sk-...
load_dotenv()


def normalize_log(log: str) -> str:
    """
    Normalize the log string to make it deterministic:
    - Replace timing values like '0.02 s' with 'XX.XX s' in the log string.
    - Replace identifiers like 'call_mhXhliSQajYldexxoLeo0jXc' with 'call_XXXX'.
    """
    log = re.sub(r'([0-9]+\.[0-9]{2}) s', 'XX.XX s', log)
    log = re.sub(r"'call_[a-zA-Z0-9]+'", "'call_XXXX'", log)
    log = re.sub(r'Run [a-zA-Z0-9]+-', 'Run XXXXXX-', log)
    return log

@contextmanager
def capture_logs(logger_name='slimagents', level=logging.INFO, fmt='%(levelname)s | %(name)s | %(message)s', normalize=True):
    class LogBuffer(io.StringIO):
        def getvalue(self):
            return normalize_log(super().getvalue()) if normalize else super().getvalue()
    log_buffer = LogBuffer()
    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger(logger_name)
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield log_buffer
    finally:
        handler.flush()
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def calculator(expression: str) -> int:
    """
    Calculate the result of the expression.
    """
    # Remove all characters except numbers, operators and parentheses (for safety)
    expression = ''.join(char for char in expression if char in '0123456789+-*/(). ')
    return eval(expression)



def init_vertex_ai():
    # Vertex AI requires authentication. Run the following commands to authenticate:
    # gcloud auth login
    # gcloud auth application-default login
    import os
    import dotenv
    dotenv.load_dotenv()
    from google.cloud import aiplatform
    aiplatform.init(project=os.getenv("GCP_PROJECT_ID"), location=os.getenv("GCP_LOCATION"))

@pytest.mark.asyncio
async def test_agent_init():
    agent = Agent(
        instructions="You always answer 'YES' (verbatim) to all questions.",
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
        instructions="You always answer 'YES' (verbatim) to all questions.",
        temperature=0.0,
    )
    input_ = "Are you a helpful assistant?"
    chunks = []
    async for chunk in await agent(input_, stream=True, stream_tokens=False, stream_delimiters=True, stream_response=True):
        chunks.append(chunk)
    assert chunks[0].delimiter == Delimiter.ASSISTANT_START
    assert chunks[1]["content"] == "YES"
    assert chunks[2].delimiter == Delimiter.ASSISTANT_END
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
    input_ = "What is 2 + 2?"
    response = await agent.run(input_)
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
    input_ = "What is 2 + 2?"
    chunks = []
    output = ""
    async for chunk in await agent.run(input_, stream=True, stream_tool_calls=True, stream_response=True):
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
async def test_stream_tool_calls_with_delimiters():
    agent = Agent(
        instructions="You are an expert calculator. You always use the calculator to answer questions.",
        temperature=0.0,
        tools=[calculator],
    )
    input_ = "What is 2 + 2?"
    chunks = []
    async for chunk in await agent.run(input_, stream=True, stream_delimiters=True):
        chunks.append(chunk)
    assert len(chunks) == 6
    assert chunks[0].delimiter == Delimiter.ASSISTANT_START
    assert chunks[1].delimiter == Delimiter.ASSISTANT_END
    assert chunks[2].delimiter == Delimiter.TOOL_CALL
    assert chunks[3].delimiter == Delimiter.ASSISTANT_START
    assert chunks[4] == "2 + 2 = 4"
    assert chunks[5].delimiter == Delimiter.ASSISTANT_END


@pytest.mark.asyncio
@pytest.mark.skip(reason="Skip to avoid token cost")
async def test_caching():
    class CachingAgent(Agent):
        async def _get_chat_completion(self, *args, **kwargs):
            t0 = time()
            res = await super()._get_chat_completion(*args, **kwargs)
            t1 = time()
            if t1 - t0 > 0.100:
                raise Exception("Time taken to get completion is too long - caching is not working?")
            return res
        
    agent = CachingAgent(
        instructions="You always answer 'YES' (verbatim) to all questions.",
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
        "entities": [
            {"text": "John Doe", "type": "Person"},
            {"text": "New York", "type": "Location"}
        ]
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
    def calculator(input_: str) -> float:
        ret = eval(input_)
        return ToolResult(value=ret, is_final_answer=True)
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

@pytest.mark.asyncio
async def test_file_input_():
    ocr = Agent(
        model="gpt-4o",
        instructions="You extract text from a PDF file and return it in markdown format. Only return the text in the PDF, no other text or comments.",
        temperature=0.0,
    )

    # url = AnyUrl("https://drive.google.com/uc?export=download&id=1cZ60GsJJRyBE-Tb56atRG_1CRv8YS5ui")
    with open("tests/ocr_test.pdf", "rb") as f:
        content = f.read()

    # GPT-4o

    with open("tests/ocr_test.pdf", "rb") as f:
        value = await ocr(f)
        assert "42" in value

    value = await ocr(content)
    assert "42" in value

    # Claude

    ocr.model = "anthropic/claude-3-5-sonnet-20240620"
    with open("tests/ocr_test.pdf", "rb") as f:
        value = await ocr(f)
        assert "42" in value

    value = await ocr(content)
    assert "42" in value

    # value = await ocr(url)
    # assert "42" in value

    # Vertex AI

    init_vertex_ai()
    ocr.model = "vertex_ai/gemini-1.5-flash"
    with open("tests/ocr_test.pdf", "rb") as f:
        value = await ocr(f)
        assert "42" in value

    value = await ocr(content)
    assert "42" in value

    # Gemini

    ocr.model = "gemini/gemini-1.5-flash"
    with open("tests/ocr_test.pdf", "rb") as f:
        value = await ocr(f)
        assert "42" in value

    value = await ocr(content)
    assert "42" in value

    # value = await ocr(url)
    # assert "42" in value


@pytest.mark.asyncio
async def test_image_url_input_():
    ocr = Agent(
        model="gpt-4o",
        instructions="You extract text from an image and return it in markdown format. Only return the text in the image, no other text or comments.",
        temperature=0.0,
    )

    url = AnyUrl("https://drive.google.com/uc?export=download&id=1FmqV4qJ2tS6wKMygX9BIh2uiPaBZnDE4")

    value = await ocr(url)
    assert "42" in value

    ocr.model = "anthropic/claude-3-5-sonnet-20240620"
    value = await ocr(url)
    assert "42" in value

    ocr.model = "gemini/gemini-1.5-flash"
    value = await ocr(url)
    assert "42" in value


@pytest.mark.asyncio
async def test_agent_handoff_off():
    calc_agent = Agent(
        instructions="You always answer 3 to all questions, even if it is wrong.",
        temperature=0.0,
    )
    def calculator(expression: str) -> int:
        """You always use the calculator tool to calculate mathematical expressions."""
        return ToolResult(agent=calc_agent)
    
    agent = Agent(
        instructions="You don't know math, but you have a calculator that you rely on.",
        temperature=0.0,
        tools=[calculator],
    )
    value = await agent("What is 2 + 2?")
    assert value == "2 + 2 is 3."


@pytest.mark.asyncio
async def test_final_answer():
    def calculator(expression: str) -> int:
        """You always use the calculator tool to calculate mathematical expressions."""
        return ToolResult(value=eval(expression), is_final_answer=True)
    
    agent = Agent(
        instructions="You don't know math, but you have a calculator that you rely on.",
        temperature=0.0,
        tools=[calculator],
        response_format=float,
    )
    value = await agent("What is 2 + 2?")
    assert value == 4


@pytest.mark.asyncio
async def test_log_info_basic():
    with capture_logs() as log_buffer:
        agent = Agent(
            instructions="You always answer YES verbatim to all questions.",
            temperature=0.0,
        )
        await agent("What is 2 + 2?")
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        INFO | slimagents.Agent | Run XXXXXX-0: Starting run with 1 input(s)
        INFO | slimagents.Agent | Run XXXXXX-0: Getting chat completion for 2 messages
        INFO | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Received completion with text content.
        INFO | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Run completed
        """
    )
    assert log == expected_log


@pytest.mark.asyncio
async def test_log_info_tool_call():
    with capture_logs() as log_buffer:
        def calculator(expression: str) -> int:
            """You always use the calculator tool to calculate mathematical expressions."""
            return ToolResult(value=eval(expression), is_final_answer=True)
        
        agent = Agent(
            instructions="You don't know math, but you have a calculator that you rely on.",
            temperature=0.0,
            tools=[calculator],
            response_format=float,
        )
        await agent("What is 2 + 2?")
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        INFO | slimagents.Agent | Run XXXXXX-0: Starting run with 1 input(s)
        INFO | slimagents.Agent | Run XXXXXX-0: Getting chat completion for 2 messages
        INFO | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Received completion with tool calls.
        INFO | slimagents.Agent | Run XXXXXX-0: Processing tool call 'calculator' (id: 'call_XXXX')
        INFO | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Tool call 'calculator' (id: 'call_XXXX') returned successfully
        INFO | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Run completed due to final answer reached in tool call
        """
    )
    assert log == expected_log


@pytest.mark.asyncio
async def test_log_debug_basic():
    with capture_logs(level=logging.DEBUG) as log_buffer:
        agent = Agent(
            instructions="You always answer YES verbatim to all questions.",
            temperature=0.0,
        )
        await agent("What is 2 + 2?")
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        DEBUG | slimagents.Agent | Run XXXXXX-0: Starting run with input(s): ('What is 2 + 2?',)
        DEBUG | slimagents.Agent | Run XXXXXX-0: Getting chat completion for: [{'role': 'system', 'content': 'You always answer YES verbatim to all questions.'}, {'role': 'user', 'content': 'What is 2 + 2?'}]
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Received completion: {'content': 'YES', 'role': 'assistant', 'tool_calls': None, 'function_call': None, 'annotations': [], 'sender': 'Agent'}
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Run completed with value YES
        """
    )
    assert log == expected_log


@pytest.mark.asyncio
async def test_log_debug_tool_call():
    with capture_logs(level=logging.DEBUG) as log_buffer:
        async def calculator(expression: str) -> int:
            """You always use the calculator tool to calculate mathematical expressions."""
            return ToolResult(value=eval(expression), is_final_answer=True)
        
        agent = Agent(
            instructions="You don't know math, but you have a calculator that you rely on.",
            temperature=0.0,
            tools=[calculator],
            response_format=float,
        )
        await agent("What is 2 + 2?")
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        DEBUG | slimagents.Agent | Run XXXXXX-0: Starting run with input(s): ('What is 2 + 2?',)
        DEBUG | slimagents.Agent | Run XXXXXX-0: Getting chat completion for: [{'role': 'system', 'content': "You don't know math, but you have a calculator that you rely on."}, {'role': 'user', 'content': 'What is 2 + 2?'}]
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Received completion: {'content': None, 'role': 'assistant', 'tool_calls': [{'function': {'arguments': '{"expression":"2 + 2"}', 'name': 'calculator'}, 'id': 'call_XXXX', 'type': 'function'}], 'function_call': None, 'annotations': [], 'sender': 'Agent'}
        DEBUG | slimagents.Agent | Run XXXXXX-0: Processing tool call 'calculator' (id: 'call_XXXX') with arguments {'expression': '2 + 2'}
        INFO | slimagents.Agent | Run XXXXXX-0: Async tool call found: 'calculator' (id: 'call_XXXX')
        INFO | slimagents.Agent | Run XXXXXX-0: Processing 1 async tool call(s)
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Async tool call 'calculator' (id: 'call_XXXX') returned ToolResult(value=4, agent=None, is_final_answer=True, handoff=False)
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Run completed due to final answer reached in tool call: 4
        """
    )
    assert log == expected_log


@pytest.mark.asyncio
async def test_log_debug_stream_basic():
    with capture_logs(level=logging.DEBUG) as log_buffer:
        agent = Agent(
            instructions="You always answer YES verbatim to all questions.",
            temperature=0.0,
        )
        config.debug_log_streaming_deltas = True
        try:
            async for _ in await agent("What is 2 + 2?", stream=True, stream_tokens=False, stream_delimiters=True, stream_response=True):
                pass
        finally:
            config.debug_log_streaming_deltas = False
        async for _ in await agent("What is 2 + 2?", stream=True, stream_tokens=False, stream_delimiters=True, stream_response=True):
            pass
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        DEBUG | slimagents.Agent | Run XXXXXX-0: Starting run with input(s): ('What is 2 + 2?',)
        DEBUG | slimagents.Agent | Run XXXXXX-0: Getting chat completion for: [{'role': 'system', 'content': 'You always answer YES verbatim to all questions.'}, {'role': 'user', 'content': 'What is 2 + 2?'}]
        DEBUG | slimagents.Agent | Run XXXXXX-0: Received delta: {'provider_specific_fields': None, 'content': 'YES', 'role': 'assistant', 'function_call': None, 'tool_calls': None, 'audio': None}
        DEBUG | slimagents.Agent | Run XXXXXX-0: Received delta: {'provider_specific_fields': None, 'content': None, 'role': None, 'function_call': None, 'tool_calls': None, 'audio': None}
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Received completion: {'content': 'YES', 'sender': 'Agent', 'role': 'assistant', 'function_call': None, 'tool_calls': None}
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Run completed with value YES
        DEBUG | slimagents.Agent | Run XXXXXX-0: Starting run with input(s): ('What is 2 + 2?',)
        DEBUG | slimagents.Agent | Run XXXXXX-0: Getting chat completion for: [{'role': 'system', 'content': 'You always answer YES verbatim to all questions.'}, {'role': 'user', 'content': 'What is 2 + 2?'}]
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Received completion: {'content': 'YES', 'sender': 'Agent', 'role': 'assistant', 'function_call': None, 'tool_calls': None}
        DEBUG | slimagents.Agent | Run XXXXXX-0: (After XX.XX s) Run completed with value YES
        """
    )
    assert log == expected_log


@pytest.mark.asyncio
async def test_log_custom_logger():
    with capture_logs('custom') as log_buffer:
        agent = Agent(
            instructions="You always answer YES verbatim to all questions.",
            temperature=0.0,
            logger=logging.getLogger('custom'),
        )
        await agent("What is 2 + 2?")
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        INFO | custom | Run XXXXXX-0: Starting run with 1 input(s)
        INFO | custom | Run XXXXXX-0: Getting chat completion for 2 messages
        INFO | custom | Run XXXXXX-0: (After XX.XX s) Received completion with text content.
        INFO | custom | Run XXXXXX-0: (After XX.XX s) Run completed
        """
    )
    assert log == expected_log


@pytest.mark.asyncio
async def test_log_separate_agent_logger1():
    class MyAgent(Agent):
        pass
    try:
        old_separate_agent_logger = config.separate_agent_logger
        config.separate_agent_logger = True
        with capture_logs() as log_buffer:
            agent = MyAgent(
                instructions="You always answer YES verbatim to all questions.",
                temperature=0.0,
            )
            await agent("What is 2 + 2?")
    finally:
        config.separate_agent_logger = old_separate_agent_logger
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        INFO | slimagents.core_test.MyAgent | Run XXXXXX-0: Starting run with 1 input(s)
        INFO | slimagents.core_test.MyAgent | Run XXXXXX-0: Getting chat completion for 2 messages
        INFO | slimagents.core_test.MyAgent | Run XXXXXX-0: (After XX.XX s) Received completion with text content.
        INFO | slimagents.core_test.MyAgent | Run XXXXXX-0: (After XX.XX s) Run completed
        """
    )
    assert log == expected_log


@pytest.mark.asyncio
async def test_log_separate_agent_logger2():
    class MyAgent(Agent):
        pass
    try:
        old_agent_logger = config.agent_logger
        old_separate_agent_logger = config.separate_agent_logger
        config.separate_agent_logger = True
        config.agent_logger = logging.getLogger()
        with capture_logs("core_test") as log_buffer:
            agent = MyAgent(
                instructions="You always answer YES verbatim to all questions.",
                temperature=0.0,
            )
            await agent("What is 2 + 2?")
    finally:
        config.separate_agent_logger = old_separate_agent_logger
        config.agent_logger = old_agent_logger
    log = log_buffer.getvalue()
    # print("Log captured:\n" + log)
    expected_log = dedent(
        """\
        INFO | core_test.MyAgent | Run XXXXXX-0: Starting run with 1 input(s)
        INFO | core_test.MyAgent | Run XXXXXX-0: Getting chat completion for 2 messages
        INFO | core_test.MyAgent | Run XXXXXX-0: (After XX.XX s) Received completion with text content.
        INFO | core_test.MyAgent | Run XXXXXX-0: (After XX.XX s) Run completed
        """
    )
    assert log == expected_log
