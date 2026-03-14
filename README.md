# SlimAgents

A lightweight and developer-friendly library for building and orchestrating AI agents.

SlimAgents wraps any LLM (via [LiteLLM](https://github.com/BerriAI/litellm)) with a simple `Agent` class that handles tool calling, streaming, structured outputs, multi-modal inputs, and agent handoffs — all in under 1200 lines of code.

## Install

Requires Python 3.10+

```shell
pip install slimagents
```

Or install the latest development version:

```shell
pip install git+https://github.com/aremeis/slimagents.git
```

## Quick start

```python
from slimagents import Agent

def calculator(expression: str) -> str:
    """Evaluate a Python expression."""
    return str(eval(expression))

agent = Agent(
    instructions="You are a helpful assistant. Use the calculator tool for math.",
    tools=[calculator],
)

value = agent.apply("What is 1234 * 5678?")
print(value)  # "1234 * 5678 = 7,006,652."
```

`apply()` is a synchronous convenience method that returns just the response value. For async code, call the agent directly:

```python
value = await agent("What is 1234 * 5678?")
```

## Tools

A tool is just a Python function. The function name, docstring, and type annotations are automatically converted to the LLM's tool schema — no decorators or registration needed.

```python
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"22 degrees {unit} in {city}"

agent = Agent(tools=[get_weather])
```

### Async tools

Both sync and async tools are supported. When the LLM generates multiple tool calls, async tools run concurrently:

```python
import httpx

async def fetch_url(url: str) -> str:
    """Fetch the content of a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

agent = Agent(tools=[fetch_url])
```

### Tools as methods

Tools can be methods on an `Agent` subclass, which allows you to encapsulate state and logic:

```python
import python_weather
from slimagents import Agent

class WeatherAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant who answers questions about the weather.",
            tools=[self.get_temperature],
        )

    async def get_temperature(self, location: str) -> float:
        """Get the current temperature in a given location, in degrees Celsius."""
        async with python_weather.Client(unit=python_weather.METRIC) as client:
            weather = await client.get(location)
            return weather.temperature

agent = WeatherAgent()
value = agent.apply("What is the temperature difference between London and Paris?")
print(value)
```

```
The temperature difference between London and Paris is 1°C, with London being warmer.
```

Since `get_temperature` is async, both calls run in parallel when the LLM requests them simultaneously.

## LLM support

SlimAgents uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, so you can use virtually any LLM. The default model is `gpt-4.1`. Specify any model string that LiteLLM supports:

```python
# OpenAI
agent = Agent(model="gpt-4.1-mini")

# Anthropic
agent = Agent(model="anthropic/claude-sonnet-4-20250514")

# Google Gemini
agent = Agent(model="gemini/gemini-2.5-flash")

# Azure, AWS Bedrock, Ollama, etc. — see LiteLLM docs
```

Any extra keyword arguments are passed through to LiteLLM:

```python
agent = Agent(model="gpt-4.1", api_key="sk-...", base_url="https://my-proxy.com")
```

## Instructions

Instructions become the system message. They can be a string or a callable for dynamic instructions:

```python
# Static instructions
agent = Agent(instructions="You are a helpful assistant.")

# Dynamic instructions via callable
agent = Agent(instructions=lambda: f"Today's date is {date.today()}")
```

You can also override the `instructions` property in a subclass for full control:

```python
class StrictAgent(Agent):
    def __init__(self, max_responses: int):
        super().__init__(tools=[self.decrement])
        self._answers_left = max_responses

    @property
    def instructions(self) -> str:
        if self._answers_left > 0:
            return f"You have {self._answers_left} responses left. Call `decrement` before each response."
        return "You always answer 'I can't answer that.'."

    def decrement(self):
        """Call this before every response."""
        self._answers_left -= 1
        return "OK"
```

## Memory

Memory is a list of message dicts in OpenAI chat format. There are two levels:

- **Default memory** (`agent.memory`): always included in every call, set at construction or via the property.
- **Per-call memory**: passed to `run()` / `apply()` and tracks the conversation for that call.

```python
agent = Agent(instructions="You are a helpful assistant.")

# Maintain a conversation across multiple calls
memory = []
agent.apply("My name is Alice.", memory=memory)
value = agent.apply("What's my name?", memory=memory)
print(value)  # "Your name is Alice."
```

Use `memory_delta` to capture only the new messages added during a call:

```python
delta = []
agent.apply("Hello!", memory=memory, memory_delta=delta)
print(len(delta))  # Number of new messages (user message + assistant response + any tool calls)
```

## Handoffs

A tool can return an `Agent` instance to transfer control to a different agent. The new agent inherits the conversation memory:

```python
sales_agent = Agent(
    name="Sales Agent",
    instructions="You are a sales agent. Help the customer with purchases.",
)

support_agent = Agent(
    name="Support Agent",
    instructions="You are a support agent. Help with technical issues.",
)

def transfer_to_sales():
    """Transfer the customer to the sales team."""
    return sales_agent

def transfer_to_support():
    """Transfer the customer to the support team."""
    return support_agent

triage = Agent(
    name="Triage",
    instructions="Route the customer to the right department.",
    tools=[transfer_to_sales, transfer_to_support],
)

response = triage.run_sync("I want to buy a new laptop.")
print(response.agent.name)  # "Sales Agent"
```

### Nested agent calls (non-handoff)

If you want an agent to process a sub-task and return the result as a tool output (without transferring control), use `ToolResult`:

```python
from slimagents import ToolResult

researcher = Agent(instructions="You are a research assistant.")

def research(topic: str):
    """Research a topic using a specialized agent."""
    return ToolResult(agent=researcher, handoff=False)

agent = Agent(tools=[research])
# The researcher processes the topic, and its response becomes the tool result.
# Control stays with `agent`.
```

## Structured outputs

Use `response_format` to get typed responses instead of plain strings.

### Pydantic models

```python
from pydantic import BaseModel
from slimagents import Agent

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

agent = Agent[MovieReview](
    instructions="You are a movie critic.",
    response_format=MovieReview,
)

review = agent.apply("Review The Matrix")
print(review.title)    # "The Matrix"
print(review.rating)   # 9.0
print(review.summary)  # "A groundbreaking sci-fi film..."
```

### JSON mode

Pass `dict` to get a parsed JSON dictionary:

```python
agent = Agent[dict](response_format=dict)
data = agent.apply("Return a JSON object with fields: name, age")
print(data["name"])  # str
```

### Primitive types

You can also use `int`, `float`, `bool`, or `list` as the response format:

```python
agent = Agent[int](response_format=int)
count = agent.apply("How many continents are there?")
print(count)  # 7 (int, not str)
```

## Multi-modal inputs

Pass file-like objects, bytes, `FileContent`, or URLs alongside text. The agent handles base64 encoding and MIME type detection automatically:

```python
from slimagents import Agent

agent = Agent(
    model="gemini/gemini-2.0-flash",
    instructions="Describe the contents of the provided files.",
)

# File object
with open("photo.jpg", "rb") as f:
    description = agent.apply("What's in this image?", f)

# Multiple inputs
with open("report.pdf", "rb") as pdf:
    summary = agent.apply("Summarize this document", pdf)
```

For programmatic file content, use `FileContent`:

```python
from slimagents.core import FileContent

content = FileContent(
    content=image_bytes,
    filename="chart.png",
    mime_type="image/png",
)
description = agent.apply("Describe this chart", content)
```

## Streaming

Enable streaming to receive tokens as they arrive:

```python
response = await agent.run("Tell me a story", stream=True)

async for chunk in response:
    if isinstance(chunk, str):
        print(chunk, end="", flush=True)
```

Fine-tune what gets streamed:

```python
response = await agent.run(
    "Tell me a story",
    stream=True,
    stream_tokens=True,        # Yield individual tokens as strings (default: True)
    stream_delimiters=True,    # Yield MessageDelimiter events for message boundaries
    stream_tool_calls=True,    # Yield tool call deltas as they arrive
    stream_response=True,      # Yield the final Response object at the end of the stream
)
```

When `stream_response=True`, the final item in the stream is a `Response` object:

```python
from slimagents import Response

async for chunk in response:
    if isinstance(chunk, Response):
        print(f"\nTokens used: {chunk.metadata.total_tokens}")
    elif isinstance(chunk, str):
        print(chunk, end="")
```

## The Response object

`run()` and `run_sync()` return a `Response[T]` with:

```python
response = agent.run_sync("Hello!")

response.value          # The response content (str, dict, or BaseModel depending on response_format)
response.memory_delta   # List of messages added during this call
response.agent          # The agent that produced the response (may differ from original if handoff occurred)
response.metadata       # ResponseMetadata with token counts and cost
```

`ResponseMetadata` tracks usage across all turns:

```python
meta = response.metadata
meta.input_tokens       # Total input tokens
meta.output_tokens      # Total output tokens
meta.total_tokens       # Total tokens
meta.cost               # Total cost (USD)
```

## Interactive CLI

Use `run_demo_loop` to quickly test an agent in your terminal:

```python
from slimagents import Agent, run_demo_loop

agent = Agent(instructions="You are a helpful assistant.")
run_demo_loop(agent, stream=True)
```

```
Starting SlimAgents CLI 🪶
User: Hello!
Agent: Hi there! How can I help you today?
User:
```

## Logging

SlimAgents uses Python's standard `logging` module:

```python
import logging
from slimagents import logger

logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)  # Verbose agent logs
```

## Origin

SlimAgents started as a fork of OpenAI's [Swarm](https://github.com/openai/swarm) framework. Major differences:

- Works with any LLM (not just OpenAI)
- Designed for subclassing `Agent` to encapsulate behavior
- Async-native with concurrent tool execution
- Multi-modal input support
- Structured outputs with Pydantic
- Proper Python logging

## License

MIT
