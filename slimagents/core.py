# Standard library imports
import base64
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import mimetypes
import random
import string
import time
from typing import AsyncGenerator, Callable, Generic, Literal, Optional, TypeVar, Union, Coroutine, Any, cast, overload
import inspect
import asyncio
import logging

# Package/library imports
from litellm import Usage, acompletion

from pydantic import AnyUrl, BaseModel, ValidationError

# Local imports
from .util import PrimitiveResult, function_to_json, get_mime_type_from_content, get_mime_type_from_file_like_object, get_pydantic_type, merge_chunk, type_to_response_format
import slimagents.config as config

# Types
T = TypeVar('T')
AgentFunction = Callable[..., Union[str, "Agent", dict, Coroutine[Any, Any, Union[str, "Agent", dict]]]]
StreamResponse = AsyncGenerator[Union[str, dict, "MessageDelimiter", "Response[T]"], None]

@dataclass
class ResponseMetadata():
    """
    Represents metadata about a response from an agent.

    Attributes:
        cost (Optional[float]): The total cost of the response.
        input_tokens (Optional[int]): The total number of input tokens used in the response.
        output_tokens (Optional[int]): The total number of output tokens used in the response.
        total_tokens (Optional[int]): The total number of tokens used in the response.
        litellm_usage (Optional[list[Usage]]): Additional usage information about the response as provided by litellm for each turn of the response.
        litellm_hidden_params (Optional[list[dict]]): Additional metadata about the response as provided by litellm for each turn of the response.
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    litellm_usage: Optional[list[Usage]] = None
    litellm_hidden_params: Optional[list[dict]] = None

@dataclass
class Response(Generic[T]):
    """
    Represents a response from an agent.

    Attributes:
        value (T): The response value. The type depends on the agent's response_format:
            str when no response_format is set, dict for JSON mode, or a BaseModel subclass for structured output.
        memory_delta (list[dict]): The list of messages that were added to the memory during this response.
        agent (Agent): The agent instance that generated this response.
        metadata (Optional[ResponseMetadata]): Additional metadata about the response.
    """
    value: T
    memory_delta: list[dict]
    agent: "Agent[Any]"
    metadata: Optional[ResponseMetadata] = None

@dataclass
class ToolResult():
    """
    Encapsulates the possible return values for an agent tool call.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        is_final_answer (bool): Whether to exit the current agent and return the result as the final answer. Defaults to False.
        handoff (bool): Only used if an agent is provided. If true, the control of the conversation is transferred to the
                       provided agent. If false, the inputs are processed by the provided agent and the result is returned
                       as the tool call result.
    """
    value: str = ""
    agent: Optional["Agent"] = None
    is_final_answer: bool = False
    handoff: bool = False

@dataclass
class HandleToolCallResult():
    """
    Represents the result of handling tool calls in an agent.

    Attributes:
        messages (list[dict]): List of messages generated during tool call handling.
        agent (Optional[Agent]): The agent instance, if a handoff occurred.
        filtered_tool_calls (list[dict]): List of tool calls that were processed.
        result (Optional[ToolResult]): The final result of the tool call handling, if any.
    """
    messages: list[dict]
    agent: Optional["Agent"] = None
    filtered_tool_calls: list[dict] = field(default_factory=list)
    result: Optional[ToolResult] = None

class DelimiterType(Enum):
    ASSISTANT_START = "assistant_start"
    ASSISTANT_END = "assistant_end"
    TOOL_CALL = "tool_call"

@dataclass
class MessageDelimiter():
    """
    A delimiter for the message stream that marks special events in the conversation flow.

    Attributes:
        delimiter_type (DelimiterType): The type of delimiter (ASSISTANT_START, ASSISTANT_END, or TOOL_CALL).
        message (dict): The associated message data for this delimiter event.
    """
    delimiter_type: DelimiterType
    message: dict

@dataclass
class FileContent():
    """
    Represents the content of a file.

    Attributes:
        content (bytes): The content of the file.
        filename (Optional[str]): The name of the file.
        mime_type (Optional[str]): The MIME type of the file.
    """
    content: bytes
    filename: Optional[str] = None
    mime_type: Optional[str] = None

# Agent class

DEFAULT_MODEL = "gpt-4.1"

class Agent(Generic[T]):
    """
    A conversational agent that can process inputs, use tools, and generate responses.

    The agent maintains a conversation history (memory) and can use various tools to process
    inputs and generate responses. It supports streaming responses, tool calls, and agent handoffs.

    Attributes:
        logger (logging.Logger): Logger instance for the agent.
    """

    logger = config.agent_logger.getChild("Agent")

    @overload
    def __init__(
            self: "Agent[str]",
            name: Optional[str] = ...,
            model: Optional[str] = ...,
            instructions: Optional[Union[str, Callable[[], str]]] = ...,
            memory: Optional[list[dict]] = ...,
            tools: Optional[list[AgentFunction]] = ...,
            tool_choice: Optional[Union[str, dict]] = ...,
            parallel_tool_calls: Optional[bool] = ...,
            response_format: None = ...,
            temperature: Optional[float] = ...,
            logger: Optional[logging.Logger] = ...,
            response_format_retries: int = ...,
            **lite_llm_args: Any,
    ) -> None: ...

    @overload
    def __init__(
            self,
            name: Optional[str] = ...,
            model: Optional[str] = ...,
            instructions: Optional[Union[str, Callable[[], str]]] = ...,
            memory: Optional[list[dict]] = ...,
            tools: Optional[list[AgentFunction]] = ...,
            tool_choice: Optional[Union[str, dict]] = ...,
            parallel_tool_calls: Optional[bool] = ...,
            response_format: Union[dict, type[T]] = ...,
            temperature: Optional[float] = ...,
            logger: Optional[logging.Logger] = ...,
            response_format_retries: int = ...,
            **lite_llm_args: Any,
    ) -> None: ...

    def __init__(
            self,
            name: Optional[str] = None,
            model: Optional[str] = None,
            instructions: Optional[Union[str, Callable[[], str]]] = None,
            memory: Optional[list[dict]] = None,
            tools: Optional[list[AgentFunction]] = None,
            tool_choice: Optional[Union[str, dict]] = None,
            parallel_tool_calls: Optional[bool] = None,
            response_format: Optional[Union[dict, type]] = None,
            temperature: Optional[float] = None,
            logger: Optional[logging.Logger] = None,
            response_format_retries: int = 1,
            **lite_llm_args: Any,
    ):
        """
        Initialize a new Agent instance.

        Args:
            name (Optional[str]): Name of the agent. Defaults to class name.
            model (Optional[str]): LLM model to use. Defaults to DEFAULT_MODEL.
            instructions (Optional[Union[str, Callable[[], str]]]): System instructions or a callable that returns them.
            memory (Optional[list[dict]]): Initial conversation memory.
            tools (Optional[list[AgentFunction]]): List of functions the agent can use.
            tool_choice (Optional[Union[str, dict]]): Control over tool selection.
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
            response_format (Optional[Union[dict, type]]): Format for response parsing.
                Determines the type parameter T: None → str, dict → dict, BaseModel subclass → that subclass.
            temperature (Optional[float]): Temperature for response generation.
            logger (Optional[logging.Logger]): Custom logger instance.
            response_format_retries (int): Number of times to retry the LLM call when the response cannot
                be parsed against the configured response_format (e.g. invalid JSON or Pydantic validation
                failure). Only meaningful when response_format is set. Defaults to 1.
            **lite_llm_args: Additional arguments passed to the LLM (e.g. num_retries, api_base,
                fallbacks, context_window_fallback_dict). See LiteLLM docs for all options.
        """
        self._name = name or self.__class__.__name__
        self._model = model or DEFAULT_MODEL
        self._instructions = instructions
        self._memory = memory or []
        self._tools = tools or []
        self._tool_choice = tool_choice
        self._parallel_tool_calls = parallel_tool_calls
        self._response_format = get_pydantic_type(response_format)
        self._temperature = temperature
        self._lite_llm_args = lite_llm_args
        self._response_format_retries = response_format_retries

        # Set up logging
        if logger:
            self.logger = logger
        elif config.separate_agent_logger and self.__class__ != Agent:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self.logger = config.agent_logger.getChild(logger_name)
        else:
            # Use the class level logger
            pass

        # Cache related
        self.__tools: Optional[list[AgentFunction]] = None
        self.__json_tools: Optional[list[dict]] = None
        self.__json_response_format: Optional[dict] = None
        self.__all_chat_completion_params: Optional[dict[str, Any]] = None
        self.__function_map: Optional[dict[str, Callable]] = None

    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, value: str) -> None:
        self._name = value
    
    @property
    def model(self) -> str:
        return self._model
    @model.setter
    def model(self, value: str) -> None:
        if value != self._model:
            self.__all_chat_completion_params = None
            self._model = value

    @property
    def instructions(self) -> Optional[Union[str, Callable[[], str]]]:
        return self._instructions
    @instructions.setter
    def instructions(self, value: Optional[Union[str, Callable[[], str]]]) -> None:
        if value != self._instructions:
            self.__all_chat_completion_params = None
            self._instructions = value

    @property
    def memory(self) -> list[dict]:
        """
        The "default" memory of the agent that will always be included for each chat completion.
        """
        return self._memory
    @memory.setter
    def memory(self, value: list[dict]) -> None:
        self._memory = value

    @property
    def tools(self) -> list[AgentFunction]:
        return self._tools
    @tools.setter
    def tools(self, value: list[AgentFunction]) -> None:
        if value != self._tools:
            self.__all_chat_completion_params = None
            self.__json_tools = None
            self.__function_map = None
            self._tools = value 

    @property
    def tool_choice(self) -> Optional[Union[str, dict]]:
        return self._tool_choice
    @tool_choice.setter
    def tool_choice(self, value: Optional[Union[str, dict]]) -> None:
        if value != self._tool_choice:
            self.__all_chat_completion_params = None
            self._tool_choice = value

    @property
    def parallel_tool_calls(self) -> Optional[bool]:
        return self._parallel_tool_calls
    @parallel_tool_calls.setter
    def parallel_tool_calls(self, value: Optional[bool]) -> None:
        if value != self._parallel_tool_calls:
            self.__all_chat_completion_params = None
            self._parallel_tool_calls = value

    @property
    def response_format(self) -> Optional[Union[dict, type[BaseModel]]]:
        return self._response_format
    @response_format.setter
    def response_format(self, value: Optional[Union[dict, type[BaseModel]]]) -> None:
        if value != self._response_format:
            self.__all_chat_completion_params = None
            self.__json_response_format = None
            self._response_format = get_pydantic_type(value)

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature
    @temperature.setter
    def temperature(self, value: Optional[float]) -> None:
        if value != self._temperature:
            self.__all_chat_completion_params = None
            self._temperature = value

    @property
    def response_format_retries(self) -> int:
        return self._response_format_retries
    @response_format_retries.setter
    def response_format_retries(self, value: int) -> None:
        self._response_format_retries = value

    @property
    def lite_llm_args(self) -> dict:
        return self._lite_llm_args
    @lite_llm_args.setter
    def lite_llm_args(self, value: dict) -> None:
        if value != self._lite_llm_args:
            self.__all_chat_completion_params = None
            self._lite_llm_args = value

    
    def __get_function_map(self) -> dict[str, Callable]:
        if self.__function_map is not None and self.__tools == self.tools:
            # Use cached function map
            return self.__function_map

        def sync_wrapper(f: Callable) -> Callable:
            """Wraps a synchronous function in an async function."""
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return f(*args, **kwargs)
            return wrapper
        
        function_map = {}
        for f in self.tools:
            if inspect.iscoroutinefunction(f):
                function_map[f.__name__] = f
            else:
                function_map[f.__name__] = sync_wrapper(f)
        self.__function_map = function_map
        return function_map


    def __get_all_chat_completion_params(self) -> dict[str, Any]:
        if self.__all_chat_completion_params is not None:
            if self.__tools == self.tools:
                # It's safe to return the cached params
                return self.__all_chat_completion_params
            else:
                # Tools list has changed from the "outside". Make sure to update the cache afterwards.
                self.__tools = None
                self.__json_tools = None
        if self.__tools != self.tools:
            # Tools list has changed from the "outside"
            self.__tools = self.tools
            if self.tools:
                self.__json_tools = [function_to_json(f) for f in self.tools]
            else:
                self.__json_tools = None
        if self.__json_response_format is None:
            # Response format is updated, so we need to update the cached JSON response format
            self.__json_response_format = type_to_response_format(self.response_format)
        params = {}
        if self._lite_llm_args:
            params.update(self._lite_llm_args)
        params.update({
            "model": self.model,
            "temperature": self.temperature,
        })
        if self.__json_tools:
            params.update({
                "tools": self.__json_tools,
                "tool_choice": self.tool_choice,
                "parallel_tool_calls": self.parallel_tool_calls,
            })
        if self.__json_response_format:
            params["response_format"] = self.__json_response_format
        self.__all_chat_completion_params = params
        return params


    async def _get_chat_completion(self, run_id: str, turns: int, memory: list[dict], memory_delta: list[dict], stream: bool = False, caching: bool = False) -> Any:
        if self.instructions:
            messages = [{"role": "system", "content": self.instructions}]
        else:
            messages = []
        # self.memory is the "default" memory that will always be included for each chat completion
        messages.extend(self.memory)
        # Add the memory added by the user
        messages.extend(memory)
        # Add the memory added by the agent during the current call
        messages.extend(memory_delta)
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug("Run %s-%d: Getting chat completion for: %s", run_id, turns, messages)
        else:
            self.logger.info("Run %s-%d: Getting chat completion for %d messages", run_id, turns, len(messages))

        create_params = self.__get_all_chat_completion_params().copy()
        create_params["messages"] = messages
        create_params["stream"] = stream
        create_params["caching"] = caching
        if stream:
            create_params["stream_options"] = {"include_usage": True}
        return await acompletion(**create_params)


    async def _handle_function_result(self, run_id: str, result: Any, memory: list[dict], memory_delta: list[dict], caching: bool) -> ToolResult:
        if isinstance(result, ToolResult):
            if result.agent and not result.handoff:
                response = await result.agent._run(run_id, memory=memory.copy(), memory_delta=memory_delta.copy(), caching=caching)
                result.value = response.value
                result.agent = None
                return result
            else:
                return result
        elif isinstance(result, Agent):
            return ToolResult(
                value=json.dumps({"assistant": result.name}),
                agent=result,
            )
        else:
            try:
                return ToolResult(value=str(result))
            except Exception as e:
                error_message = "Failed to cast response to string: %s. Make sure agent functions return a string, Result object, or coroutine. Error: %s"
                self.logger.error(error_message, result, str(e))
                raise TypeError(error_message % (result, str(e)))
            

    _RESPONSE_FORMAT_RETRY_MESSAGE = (
        "Your previous response could not be parsed as the required structured output.\n"
        "Error: {error}\n"
        "Please reply again with a valid response that matches the schema. "
        "Return only the structured output, no prose."
    )

    def _try_parse_value(self, content: str) -> tuple[bool, Any]:
        """
        Attempt to parse content into the agent's response_format.
        Returns (True, value) on success, (False, exception) on parse failure.
        Only catches JSON / Pydantic validation errors; other exceptions propagate.
        """
        try:
            return True, self._get_value(content)
        except (ValidationError, json.JSONDecodeError) as e:
            return False, e

    def _get_value(self, content: str) -> T:
        if self.response_format:
            if self.response_format is dict or isinstance(self.response_format, dict):
                return cast(T, json.loads(content))
            elif issubclass(self.response_format, BaseModel):
                ret = self.response_format.model_validate_json(content)
                if isinstance(ret, PrimitiveResult):
                    return cast(T, ret.result)
                else:
                    return cast(T, ret)
            else:
                raise ValueError(f"Unsupported response_format: {self.response_format}")
        else:
            return cast(T, content)


    def _update_partial_response(
            self, 
            partial_response: HandleToolCallResult, 
            tool_call: dict,
            result: ToolResult
    ) -> dict:
        partial_response.filtered_tool_calls.append(tool_call)
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "tool_name": tool_call["function"]["name"],
            "content": str(result.value),
        }
        partial_response.messages.append(tool_message)
        if result.agent:
            partial_response.agent = result.agent
        if result.is_final_answer:
            partial_response.result = result
        return tool_message


    def _before_chat_completion(self) -> None:
        pass

    async def _handle_tool_calls(
            self,
            run_id: str,
            turn: int,
            tool_calls: list[dict],
            memory: list[dict],
            memory_delta: list[dict],
            caching: bool,
    ) -> AsyncGenerator[tuple[dict, ToolResult], None]:
        function_map = self.__get_function_map()

        async def tool_call_wrapper(tool_call: dict) -> tuple[dict, ToolResult]:
            function = tool_call["function"]
            name = function["name"]
            tool_id = tool_call["id"]

            if name not in function_map:
                self.logger.warning("Run %s-%d: Tool '%s' (id: '%s') not found in function map.", run_id, turn, name, tool_id)
                return tool_call, ToolResult(value=f"Error: Tool {name} not found.")
            
            args = json.loads(function["arguments"])
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.logger.debug("Run %s-%d: Processing tool call '%s' (id: '%s') with arguments %s", run_id, turn, name, tool_id, args)
            else:
                self.logger.info("Run %s-%d: Processing tool call '%s' (id: '%s')", run_id, turn, name, tool_id)
            
            func = function_map[name]            
            t0 = time.time()
            ret = await func(**args)
            delta_t = time.time() - t0
            
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.logger.debug("Run %s-%d: (After %.2f s) Tool call '%s' (id: '%s') returned %s", run_id, turn, delta_t, name, tool_id, ret)
            else:
                self.logger.info("Run %s-%d: (After %.2f s) Tool call '%s' (id: '%s') returned successfully", run_id, turn, delta_t, name, tool_id)
            
            result = await self._handle_function_result(run_id, ret, memory, memory_delta, caching)
            return tool_call, result

        # Create tasks for all tool calls
        tasks = [tool_call_wrapper(tool_call) for tool_call in tool_calls]
        
        # Process tool calls as they complete
        for task in asyncio.as_completed(tasks):
            tool_call, result = await task
            yield tool_call, result


    def _handle_partial_response(self, run_id: str, turns: int, t0_run: float, partial_response: HandleToolCallResult, message: dict, memory: list[dict], memory_delta: list[dict], metadata: Optional[ResponseMetadata]) -> Optional[Response[T]]:
        if partial_response.filtered_tool_calls:
            # Only add tool calls to memory if there are any left after filtering
            memory_delta.append(message)
            memory_delta.extend(partial_response.messages)
        if partial_response.result:
            t_run_delta = time.time() - t0_run
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.logger.debug("Run %s-%d: (After %.2f s) Run completed due to final answer reached in tool call: %s", run_id, turns, t_run_delta, partial_response.result.value)
            else:
                self.logger.info("Run %s-%d: (After %.2f s) Run completed due to final answer reached in tool call", run_id, turns, t_run_delta)
            memory.extend(memory_delta)
            return Response(
                value=cast(T, partial_response.result.value),
                memory_delta=memory_delta,
                agent=self,
                metadata=metadata,
            )
        return None

    def _get_response(self, run_id: str, turns: int, t0_run: float, memory: list[dict], memory_delta: list[dict], metadata: Optional[ResponseMetadata]) -> Response[T]:
        memory.extend(memory_delta) # FIXME? Is this really a good idea? 
        value = self._get_value(memory[-1]["content"])
        t_run_delta = time.time() - t0_run
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug("Run %s-%d: (After %.2f s) Run completed with value %s", run_id, turns, t_run_delta, value)
        else:
            self.logger.info("Run %s-%d: (After %.2f s) Run completed", run_id, turns, t_run_delta)
        return Response(
            value=value,
            memory_delta=memory_delta,
            agent=self,
            metadata=metadata,
        )


    def _get_user_message(self, inputs: tuple) -> dict:
        def user_message_part(input: Any) -> dict:
            if isinstance(input, str):
                return {
                    "type": "text",
                    "text": input,
                }
            elif isinstance(input, dict):
                return input
            elif isinstance(input, AnyUrl):
                # Assume image.
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": str(input),
                    },
                }
            elif isinstance(input, FileContent):
                file_name = input.filename or None
                mime_type = input.mime_type
                content = input.content
            elif hasattr(input, 'read'):  # is file-like object
                file_name = input.name or None
                mime_type = None
                content = input.read()
            elif isinstance(input, bytes):
                # file_name = "temp_file"
                # mime_type = get_mime_type_from_content(input)
                file_name = None
                mime_type = None
                content = input
            else:
                raise ValueError(f"Unsupported element type: {type(input)}")
 
            if not mime_type and file_name:
                # Try to guess the MIME type from the file name
                mime_type, _ = mimetypes.guess_type(file_name)
            if not mime_type:
                # MIME type still not set, try to guess the MIME type from the content
                mime_type = get_mime_type_from_content(content[:2048])
            if not file_name:
                file_name = "temp_file"
            base64_content = base64.b64encode(content).decode('utf-8')
            return {
                "type": "file",
                "file": {
                    "filename": file_name,
                    "file_data": f"data:{mime_type};base64,{base64_content}",
                },
            }

        if len(inputs) == 1 and isinstance(inputs[0], str):
            # Keep it simple if there's only one string input
            return {"role": "user", "content": inputs[0]}
        else:
            return {"role": "user", "content": [user_message_part(input) for input in inputs]} 


    def _log_completion(self, run_id: str, turns: int, t0: float, message: dict) -> None:
        delta_t = time.time() - t0
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug("Run %s-%d: (After %.2f s) Received completion: %s", run_id, turns, delta_t, message)
        else:
            if message["tool_calls"] and message["content"]:
                self.logger.info("Run %s-%d: (After %.2f s) Received completion with tool calls and text content.", run_id, turns, delta_t)
            elif message["tool_calls"]:
                self.logger.info("Run %s-%d: (After %.2f s) Received completion with tool calls.", run_id, turns, delta_t)
            elif message["content"]:
                self.logger.info("Run %s-%d: (After %.2f s) Received completion with text content.", run_id, turns, delta_t)


    def _update_metadata(self, metadata: Optional[ResponseMetadata], completion: Any) -> Optional[ResponseMetadata]:
        usage = getattr(completion, "usage", None)
        hidden_params = getattr(completion, "_hidden_params", None)
        if usage or hidden_params:
            metadata = metadata or ResponseMetadata()
            if usage:
                metadata.litellm_usage = metadata.litellm_usage or []
                metadata.litellm_usage.append(usage)
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                if prompt_tokens:
                    metadata.input_tokens = (metadata.input_tokens or 0) + prompt_tokens
                completion_tokens = getattr(usage, "completion_tokens", 0)
                if completion_tokens:
                    metadata.output_tokens = (metadata.output_tokens or 0) + completion_tokens
                total_tokens = getattr(usage, "total_tokens", 0)
                if total_tokens:
                    metadata.total_tokens = (metadata.total_tokens or 0) + total_tokens
            if hidden_params:
                metadata.litellm_hidden_params = metadata.litellm_hidden_params or []
                metadata.litellm_hidden_params.append(hidden_params)
                response_cost = hidden_params.get("response_cost", 0)
                if response_cost:
                    metadata.cost = (metadata.cost or 0) + response_cost
        return metadata


    async def _run_and_stream(
            self,
            run_id: str,
            memory: list[dict],
            memory_delta: list[dict],
            stream_tokens: bool,
            stream_delimiters: bool,
            stream_tool_calls: bool,
            stream_response: bool,
            max_turns: float,
            execute_tools: bool,
            caching: bool,
            response_format_retries: int = 0,
    ) -> StreamResponse:
        t0_run = time.time()
        active_agent = self
        turns = 0
        metadata = None
        retries_left = response_format_retries

        while turns < max_turns:
            active_agent._before_chat_completion()
            message: dict[str, Any] = {
                "content": "",
                "sender": active_agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            t0 = time.time()
            # get completion with current history, agent
            completion = await active_agent._get_chat_completion(run_id, turns, memory, memory_delta, stream=True, caching=caching)

            if stream_delimiters:
                yield MessageDelimiter(delimiter_type=DelimiterType.ASSISTANT_START, message=message)

            async for chunk in completion:
                if getattr(chunk, "usage", None):
                    metadata = self._update_metadata(metadata, chunk)
                    continue
                delta = chunk.choices[0].delta.model_dump()
                if config.debug_log_streaming_deltas:
                    self.logger.debug("Run %s-%d: Received delta: %s", run_id, turns, delta)
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                if "content" in delta and delta["content"]:
                    if stream_tokens:
                        yield delta["content"]
                    else:
                        yield delta
                elif "tool_calls" in delta and delta["tool_calls"]:
                    if stream_tool_calls:
                        yield delta
                else:
                    # In theory, the check for "content" and "tool_calls" should be enough, so
                    # this should never happen. However, LiteLLM seems to send some additional
                    # empty chunks. We ignore them for now.
                    pass
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            
            # Convert tool calls dictionary to list (or None if empty)
            message["tool_calls"] = list(message.get("tool_calls", {}).values()) or None

            active_agent._log_completion(run_id, turns, t0, message)

            if stream_delimiters:
                yield MessageDelimiter(delimiter_type=DelimiterType.ASSISTANT_END, message=message)

            if not message["tool_calls"] or not execute_tools:
                memory_delta.append(message)
                if active_agent.response_format and execute_tools and retries_left > 0:
                    ok, value_or_err = active_agent._try_parse_value(message["content"] or "")
                    if not ok:
                        active_agent.logger.warning(
                            "Run %s-%d: Failed to parse response against response_format (%d retries left): %s",
                            run_id, turns, retries_left, value_or_err,
                        )
                        memory_delta.append({
                            "role": "user",
                            "content": active_agent._RESPONSE_FORMAT_RETRY_MESSAGE.format(error=value_or_err),
                        })
                        retries_left -= 1
                        continue
                break

            # handle function calls and switching agents
            partial_response = HandleToolCallResult(messages=[], agent=None, filtered_tool_calls=[])
            async for tool_call, result in active_agent._handle_tool_calls(run_id, turns, message["tool_calls"], memory, memory_delta, caching):
                tool_message = active_agent._update_partial_response(partial_response, tool_call, result)
                if stream_delimiters:
                    yield MessageDelimiter(delimiter_type=DelimiterType.TOOL_CALL, message=tool_message)
                if partial_response.result:
                    break
            
            response = active_agent._handle_partial_response(run_id, turns, t0_run, partial_response, message, memory, memory_delta, metadata)
            if response:
                if stream_response:
                    yield response
                else:
                    return

            if partial_response.agent:
                active_agent = partial_response.agent
            
            turns += 1

        memory.extend(memory_delta)
        if stream_response:
            yield active_agent._get_response(run_id, turns, t0_run, memory, memory_delta, metadata)
        else:
            active_agent.logger.info("Run %s-%d: (After %.2f s) Run completed", run_id, turns, time.time() - t0_run)


    async def _run(
            self,
            run_id: str,
            memory: list[dict],
            memory_delta: list[dict],
            max_turns: float = float("inf"),
            execute_tools: bool = True,
            caching: bool = False,
            response_format_retries: int = 0,
    ) -> Response[T]:
        t0_run = time.time()
        active_agent = self
        turns = 0
        metadata = None
        retries_left = response_format_retries

        while turns < max_turns and active_agent:
            active_agent._before_chat_completion()
            # get completion with current history, agent
            t0 = time.time()
            completion = await active_agent._get_chat_completion(run_id, turns, memory, memory_delta, caching=caching)
            message = completion.choices[0].message.model_dump()
            message["sender"] = active_agent.name
            metadata = self._update_metadata(metadata, completion)

            active_agent._log_completion(run_id, turns, t0, message)

            if not message["tool_calls"] or not execute_tools:
                memory_delta.append(message)
                if active_agent.response_format and execute_tools and retries_left > 0:
                    ok, value_or_err = active_agent._try_parse_value(message["content"] or "")
                    if not ok:
                        active_agent.logger.warning(
                            "Run %s-%d: Failed to parse response against response_format (%d retries left): %s",
                            run_id, turns, retries_left, value_or_err,
                        )
                        memory_delta.append({
                            "role": "user",
                            "content": active_agent._RESPONSE_FORMAT_RETRY_MESSAGE.format(error=value_or_err),
                        })
                        retries_left -= 1
                        continue
                break

            # handle function calls and switching agents
            partial_response = HandleToolCallResult(messages=[], agent=None, filtered_tool_calls=[])
            async for tool_call, result in active_agent._handle_tool_calls(run_id, turns, message["tool_calls"], memory, memory_delta, caching):
                active_agent._update_partial_response(partial_response, tool_call, result)
                if partial_response.result:
                    break

            response = active_agent._handle_partial_response(run_id, turns, t0_run, partial_response, message, memory, memory_delta, metadata)
            if response:
                return response

            if partial_response.agent:
                active_agent = partial_response.agent

            turns += 1

        return active_agent._get_response(run_id, turns, t0_run, memory, memory_delta, metadata)


    def _get_run_id(self) -> str:
        # 6 random alphanumeric characters
        return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

    
    @overload
    async def run(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[False] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> Response[T]: ...

    @overload
    async def run(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[True] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> StreamResponse: ...

    async def run(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: Optional[bool] = False,
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: float = float("inf"),
            execute_tools: bool = True,
            caching: Optional[bool] = None,
            response_format_retries: Optional[int] = None,
    ) -> Union[Response[T], StreamResponse]:
        """
        Run the agent with the given inputs and return a response.

        This is the main method for interacting with the agent. It processes inputs,
        maintains conversation history, and can stream responses in various formats.

        Args:
            *inputs: Variable length input arguments to process.
            memory (Optional[list[dict]]): Conversation history to use. Defaults to empty list.
            memory_delta (Optional[list[dict]]): Additional messages generated by the agent will be added to this list. Must be empty if provided.
            stream (Optional[bool]): Whether to stream the response. Defaults to False.
            stream_tokens (bool): Whether to stream individual tokens. Defaults to True.
            stream_delimiters (bool): Whether to stream message delimiters. Defaults to False.
            stream_tool_calls (bool): Whether to stream tool calls. Defaults to False.
            stream_response (bool): Whether to stream the final response. Defaults to False.
            max_turns (Optional[int]): Maximum number of conversation turns. Defaults to infinity.
            execute_tools (Optional[bool]): Whether to execute tool calls. Defaults to True.
            caching (Optional[bool]): Whether to use response caching. Defaults to config.caching.

        Returns:
            Union[Response, AsyncGenerator[Response, None]]: The response or a generator of response chunks if streaming.

        Raises:
            ValueError: If memory_delta is provided and not empty.
        """
        if memory is None:
            memory = []
        if caching is None:
            caching = config.caching
        if response_format_retries is None:
            response_format_retries = self._response_format_retries

        if memory_delta is None:
            memory_delta = []
        elif memory_delta:
            raise ValueError("memory_delta must be an empty list if provided as a parameter")

        if inputs:
            memory_delta.append(self._get_user_message(inputs))

        run_id = self._get_run_id()

        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug("Run %s-0: Starting run with input(s): %s", run_id, inputs)
        else:
            self.logger.info("Run %s-0: Starting run with %d input(s)", run_id, len(inputs))

        if stream:
            return self._run_and_stream(
                run_id=run_id,
                memory=memory,
                memory_delta=memory_delta,
                stream_tokens=stream_tokens,
                stream_delimiters=stream_delimiters,
                stream_tool_calls=stream_tool_calls,
                stream_response=stream_response,
                max_turns=max_turns,
                execute_tools=execute_tools,
                caching=caching,
                response_format_retries=response_format_retries,
            )
        else:
            return await self._run(
                run_id=run_id,
                memory=memory,
                memory_delta=memory_delta,
                max_turns=max_turns,
                execute_tools=execute_tools,
                caching=caching,
                response_format_retries=response_format_retries,
            )


    @overload
    def run_sync(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[False] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> Response[T]: ...

    @overload
    def run_sync(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[True] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> StreamResponse: ...

    def run_sync(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: bool = False,
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: float = float("inf"),
            execute_tools: bool = True,
            caching: Optional[bool] = None,
            response_format_retries: Optional[int] = None,
    ) -> Union[Response[T], StreamResponse]:
        """
        Synchronously run the agent with the given inputs and return a response.

        This is a synchronous wrapper around the async run method. It creates a new event loop
        if one doesn't exist and runs the agent in that loop.

        Args:
            *inputs: Variable length input arguments to process.
            memory (Optional[list[dict]]): Conversation history to use. Defaults to empty list.
            memory_delta (Optional[list[dict]]): Additional messages to add to memory. Must be empty if provided.
            stream (bool): Whether to stream the response. Defaults to False.
            stream_tokens (bool): Whether to stream actual tokens instead of delta chunks. Defaults to True.
            stream_delimiters (bool): Whether to stream message delimiters. Defaults to False.
            stream_tool_calls (bool): Whether to stream tool call delta chunks. Defaults to False.
            stream_response (bool): Whether to stream the final response object. Defaults to False.
            max_turns (int): Maximum number of conversation turns. Defaults to infinity.
            execute_tools (bool): Whether to execute tool calls. Defaults to True.
            caching (bool): Whether to use response caching. Defaults to config.caching.

        Returns:
            Response: The response from the agent.

        Raises:
            ValueError: If memory_delta is provided and not empty.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.run(  # type: ignore[call-overload]
                *inputs,
                memory=memory,
                memory_delta=memory_delta,
                stream=stream,
                stream_tokens=stream_tokens,
                stream_delimiters=stream_delimiters,
                stream_tool_calls=stream_tool_calls,
                stream_response=stream_response,
                max_turns=max_turns,
                execute_tools=execute_tools,
                caching=caching,
                response_format_retries=response_format_retries,
            )
        )


    @overload
    def apply(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[False] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> T: ...

    @overload
    def apply(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[True] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> StreamResponse: ...

    def apply(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: bool = False,
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: float = float("inf"),
            execute_tools: bool = True,
            caching: Optional[bool] = None,
            response_format_retries: Optional[int] = None,
    ) -> Union[T, StreamResponse]:
        """
        Synchronously apply the agent to the inputs and return the response value.

        This is a convenience method that wraps run_sync and returns just the response value.
        It's useful for simple synchronous interactions with the agent.

        Args:
            *inputs: Variable length input arguments to process.
            memory (Optional[list[dict]]): Conversation history to use. Defaults to empty list.
            memory_delta (Optional[list[dict]]): Messages generated by the agent will be added to this list. Must be empty if provided.
            stream (bool): Whether to stream the response. Defaults to False.
            stream_tokens (bool): Whether to stream actual tokens instead of delta chunks. Defaults to True.
            stream_delimiters (bool): Whether to stream message delimiters. Defaults to False.
            stream_tool_calls (bool): Whether to stream tool call delta chunks. Defaults to False.
            stream_response (bool): Whether to stream the final response object. Defaults to False.
            max_turns (int): Maximum number of conversation turns. Defaults to infinity.
            execute_tools (bool): Whether to execute tool calls. Defaults to True.
            caching (bool): Whether to use response caching. Defaults to config.caching.

        Returns:
            Response: The response value from the agent.

        Raises:
            ValueError: If memory_delta is provided and not empty.
        """
        response = self.run_sync(  # type: ignore[call-overload]
            *inputs,
            memory=memory,
            memory_delta=memory_delta,
            stream=stream,
            stream_tokens=stream_tokens,
            stream_delimiters=stream_delimiters,
            stream_tool_calls=stream_tool_calls,
            stream_response=stream_response,
            max_turns=max_turns,
            execute_tools=execute_tools,
            caching=caching,
            response_format_retries=response_format_retries,
        )
        if stream:
            return cast(StreamResponse, response)
        else:
            assert isinstance(response, Response)
            return response.value
        

    @overload
    async def __call__(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[False] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> T: ...

    @overload
    async def __call__(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = ...,
            memory_delta: Optional[list[dict]] = ...,
            stream: Literal[True] = ...,
            stream_tokens: bool = ...,
            stream_delimiters: bool = ...,
            stream_tool_calls: bool = ...,
            stream_response: bool = ...,
            max_turns: float = ...,
            execute_tools: bool = ...,
            caching: Optional[bool] = ...,
            response_format_retries: Optional[int] = ...,
    ) -> StreamResponse: ...

    async def __call__(
            self,
            *inputs: Any,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: Optional[bool] = False,
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: float = float("inf"),
            execute_tools: bool = True,
            caching: Optional[bool] = None,
            response_format_retries: Optional[int] = None,
    ) -> Union[T, StreamResponse]:
        """
        Asynchronously apply the agent to the inputs and return the response value.

        This method allows the agent to be called like a function, making it easy to use
        in async contexts. It wraps the run method and returns just the response value.

        Args:
            *inputs: Variable length input arguments to process.
            memory (Optional[list[dict]]): Conversation history to use. Defaults to empty list.
            memory_delta (Optional[list[dict]]): Messages generated by the agent will be added to this list. Must be empty if provided.
            stream (Optional[bool]): Whether to stream the response. Defaults to False.
            stream_tokens (bool): Whether to stream actual tokens instead of delta chunks. Defaults to True.
            stream_delimiters (bool): Whether to stream message delimiters. Defaults to False.
            stream_tool_calls (bool): Whether to stream tool call delta chunks. Defaults to False.
            stream_response (bool): Whether to stream the final response object. Defaults to False.
            max_turns (Optional[int]): Maximum number of conversation turns. Defaults to infinity.
            execute_tools (Optional[bool]): Whether to execute tool calls. Defaults to True.
            caching (Optional[bool]): Whether to use response caching. Defaults to config.caching.

        Returns:
            Response: The response value from the agent.

        Raises:
            ValueError: If memory_delta is provided and not empty.
        """
        response = await self.run(  # type: ignore[call-overload, misc]
            *inputs,
            memory=memory,
            memory_delta=memory_delta,
            stream=stream,
            stream_tokens=stream_tokens,
            stream_delimiters=stream_delimiters,
            stream_tool_calls=stream_tool_calls,
            stream_response=stream_response,
            max_turns=max_turns,
            execute_tools=execute_tools,
            caching=caching,
            response_format_retries=response_format_retries,
        )
        if stream:
            return cast(StreamResponse, response)
        else:
            assert isinstance(response, Response)
            return response.value
        

__all__ = ["Agent", "Response", "ToolResult", "MessageDelimiter", "DelimiterType"]