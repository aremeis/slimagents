# SlimAgents

A lightweight and developer-friendly library for building and orchestrating AI agents


## Install

Requires Python 3.10+

Latest stable release:

```shell
pip install slimagents
```

Latest development version:

```shell
pip install git+ssh://git@github.com/aremeis/slimagents.git
```

or

```shell
pip install git+https://github.com/aremeis/slimagents.git
```

## Documentation

In SlimAgents, an Agent is simply a wrapper around a large language model, textual instructions, and a set of tools. 
Based on the inputs, the agent selects tool calls, executes them, and adds the result to its memory. 
This process is repeated until the LLM does not generate any more tool calls, in which case the agent returns the last 
message content generated from the LLM.

Here's a simple example:

```python
from slimagents import Agent

def python_evaluator(expression: str) -> str:
    """Evaluate a Python expression. Always use this tool for calculations and other complex operations."""
    print(f"--- Evaluating {expression}")
    # Obviously not secure, but for the sake of this example we'll just eval the expression.
    ret = str(eval(expression))
    print(f"--> {ret}")
    return ret

agent = Agent(
    instructions="You are a helpful assistant. When given a task you always try to solve it by using tools, never rely on your own knowledge.",
    tools=[python_evaluator],
)

prompt = "How many R's are in the word 'STRAWBERRY'?"
print(f"User: {prompt}")
response = agent.run_sync(prompt)
print(f"Agent: {response.value}")
```

Result:
```
User: How many R's are in the word 'STRAWBERRY'?
--- Evaluating 'STRAWBERRY'.count('R')
--> 3
Agent: There are 3 'R's in the word 'STRAWBERRY'.
```

### Tools

As you can see from the example above, a tool is simply a normal Python function! This means that it is very easy to integrate 
existing Python libraries with your agents. Use the tool's docstring to describe the tool and its arguments to the LLM.

SlimAgents supports both synchronous and asynchronous tool calls. If the LLM generates several async tool calls, they will be 
executed in parallel. 

NOTE: The method `run_sync` is used in the examples in this document. In async applications, you should use the `run` method instead.

Tools can also be implemented as methods. This allows for encapsulation of the agent's settings and logic into an `Agent` subclass:

```python
# !pip install python-weather

from slimagents import Agent
import python_weather

class WeatherAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant who answers questions about the weather.",
            tools=[self.get_temperature],
        )

    async def get_temperature(self, location: str) -> float:
        """Get the current temperature in a given location, in degrees Celsius."""
        async with python_weather.Client(unit=python_weather.METRIC) as client:
            print(f"--- Getting temperature for {location}")
            weather = await client.get(location)
            print(f"--> Temperature in {location}: {weather.temperature}")
            return weather.temperature

agent = WeatherAgent()
prompt = "What is the temperature difference between London and Paris?"
print(f"User: {prompt}")
response = agent.run_sync(prompt)
print(f"Agent: {response.value}")
```

```
User: What is the temperature difference between London and Paris?
--- Getting temperature for London
--- Getting temperature for Paris
--> Temperature in London: 4
--> Temperature in Paris: 3
Agent: The temperature difference between London and Paris is 1°C, with London being warmer.
```


### LLMs

SlimAgents uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, which means that you can use virtually any LLM to power your agents! 
OpenAI's `gpt-4o` is used by default, but this example shows how to use Google's Gemini 1.5 Pro instead. See LiteLLM's 
[documentation](https://github.com/BerriAI/litellm?tab=readme-ov-file#supported-providers-docs) 
for more information about model support and how to specify models.

```python
from slimagents import Agent

agent = Agent(
    model="gemini/gemini-1.5-pro",
)

response = agent.run_sync("Who are you?")
print(response.value)
```

```
I am a large language model, trained by Google.
```


### Instructions

Instructions are passed to the LLM as the `system` message. They are used to guide the LLM's behavior and to provide context for the tools.
SlimAgents does not come with pre-defined instructions, so your agent's behavior is entirely controlled by the information you provide as
instructions and in the tool documentation. 

Instructions can be dynamic, i.e. generated based on the agent's state. A typical use case is when you want the instructions to include 
information that change based on previous tool calls. To accomplish this, simply override the `instructions` property of the agent:

```python
from slimagents import Agent, run_demo_loop

class StrictAgent(Agent):
    def __init__(self, max_responses: int):
        super().__init__(
            tools=[self.update_responses_left],
        )
        self._answers_left = max_responses

    @property
    def instructions(self) -> str:
        if self._answers_left >= 0:
            return f"""You are a helpful assistant. 
You currently have {self._answers_left} responses left.
ALWAYS call the `update_responses_left` tool before you respond."""
        else:
            return "You always answer 'I can't answer that.'."

    def update_responses_left(self):
        """IMPORTANT! You ALWAYS call this tool before you respond, no matter what the user says."""
        self._answers_left -= 1
        return "Good! You may now answer the question."

agent = StrictAgent(2) # This agent will only respond 2 times.
run_demo_loop(agent)
```

```
Starting SlimAgents CLI 🪶
User: Hi
StrictAgent: update_responses_left()
StrictAgent: Hello! How can I assist you today?
User: How many answers left?
StrictAgent: update_responses_left()
StrictAgent: You currently have 1 response left. How else may I assist you?
User: 2 + 2?                        
StrictAgent: update_responses_left()
StrictAgent: I can't answer that.
User: Why not?
StrictAgent: update_responses_left()
StrictAgent: I can't answer that.
```

This example also illustrates the `run_demo_loop` function. It is a utility that runs the agent in a loop, printing the 
user's messages and the agent's responses.


### Memory

The memory of an agent is simply the history of the agent's messages, using the same format as the chat history in the OpenAI API
(but without the 'system' or 'developer' message). Tool selection and tool call results are added to the memory, as well as the 
LLM's response when the agent is done.


### Handoffs

Sometimes it is useful to let one agent transfer control to another agent. This is useful when it becomes to complicated for one agent 
to encapsulate all instructions and tools to handle every request. To accomplish such handoffs, simply return an `Agent` from a tool call:

```python
sales_agent = Agent(name="Sales Agent")

def transfer_to_sales():
   return sales_agent

agent = Agent(functions=[transfer_to_sales])

response = agent.run("Transfer me to sales.")
print(response.agent.name)
```

```
Sales Agent
```

Note: When using handoffs, the memory of the original agent will be shared with the new agent. This means that the new agent will have 
access to the original agent's memory, and any changes to the memory will be reflected in both agents.

If you think this feature looks like it is borrowed from OpenAI's [Swarm](https://github.com/openai/swarm) framework, you are right! In fact, 
SlimAgents started out as a fork of Swarm, so big shoutout to OpenAI and the Swarm team for the inspiration!

Major changes from Swarm:
- Supports virtually any LLM
- Designed for subclassing `Agent` to encapsulate agent behavior
- Supports async, concurrent tool calls
- Uses proper Python logging instead of print statements
- Supports multi modal inputs (see below)
- Supports structured outputs with Pydantic (see below)


### Multi modal inputs

SlimAgents makes it easy to use multi modal inputs like images, videos, audio files and PDF files (as long as these types are supported by the LLM).
Here's an example:

```python
from slimagents import Agent

pdf_converter = Agent(
    model="gemini/gemini-2.0-flash", # 👈 Gemini 2.0 Flash supports PDF files as input
)

with open("annual_report.pdf", "rb") as pdf_file:
    response = pdf_converter.run_sync(pdf_file, "Convert this PDF to Markdown.")
    print(response.value)
```


### Structured outputs

WIP


### The response object

WIP


### Response type

WIP


### Handoff vs tool call

WIP
