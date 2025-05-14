# Standard library imports
import base64
from dataclasses import dataclass, field
import json
from collections import defaultdict
import time
from typing import Callable, Optional, Union, Coroutine, Any
import inspect
import asyncio
import logging

# Package/library imports
from litellm import acompletion
from litellm.types.completion import ChatCompletionMessageToolCallParam, Function
from pydantic import AnyUrl, BaseModel

# Local imports
from .util import PrimitiveResult, function_to_json, get_mime_type_from_content, get_mime_type_from_file_like_object, get_pydantic_type, merge_chunk, type_to_response_format
import slimagents.config as config

# Types
AgentFunction = Callable[..., Union[str, "Agent", dict, Coroutine[Any, Any, Union[str, "Agent", dict]]]]

@dataclass
class Response():
    value: Any
    memory_delta: list
    agent: "Agent"

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
    messages: list
    agent: Optional["Agent"] = None
    filtered_tool_calls: list[ChatCompletionMessageToolCallParam] = field(default_factory=list)
    result: Optional[ToolResult] = None

# Agent class

DEFAULT_MODEL = "gpt-4.1"

class Agent:

    logger = config.agent_logger.getChild("Agent")

    def __init__(
            self, 
            name: Optional[str] = None, 
            model: Optional[str] = None,
            instructions: Optional[Union[str, Callable[[], str]]] = None, 
            memory: Optional[list[dict]] = None,
            tools: Optional[list[AgentFunction]] = None, 
            tool_choice: Optional[Union[str, dict]] = None, 
            parallel_tool_calls: Optional[bool] = None, 
            response_format: Optional[Union[dict, type[BaseModel]]] = None,
            temperature: Optional[float] = None,
            logger: Optional[logging.Logger] = None,
            **lite_llm_args
    ):
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

        # Set up logging
        if logger:
            self.logger = logger
        elif config.separate_agent_logger and self.__class__ != Agent:
            self.logger = config.agent_logger.getChild(f"{__name__}.{self.__class__.__name__}")
        else:
            # Use the class level logger
            pass

        # Cache related
        self.__tools = None
        self.__json_tools = None
        self.__json_response_format = None
        self.__all_chat_completion_params = None
        
    @property 
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value
    
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        if value != self._model:
            self.__all_chat_completion_params = None
            self._model = value

    @property
    def instructions(self):
        return self._instructions
    @instructions.setter
    def instructions(self, value):
        if value != self._instructions:
            self.__all_chat_completion_params = None
            self._instructions = value

    @property
    def memory(self):
        """
        The "default" memory of the agent that will always be included for each chat completion.
        """
        return self._memory
    @memory.setter
    def memory(self, value):
        self._memory = value

    @property
    def tools(self):
        return self._tools
    @tools.setter
    def tools(self, value):
        if value != self._tools:
            self.__all_chat_completion_params = None
            self.__json_tools = None
            self._tools = value 

    @property
    def tool_choice(self):
        return self._tool_choice
    @tool_choice.setter
    def tool_choice(self, value):
        if value != self._tool_choice:
            self.__all_chat_completion_params = None
            self._tool_choice = value

    @property
    def parallel_tool_calls(self):
        return self._parallel_tool_calls
    @parallel_tool_calls.setter
    def parallel_tool_calls(self, value):
        if value != self._parallel_tool_calls:
            self.__all_chat_completion_params = None
            self._parallel_tool_calls = value

    @property
    def response_format(self):
        return self._response_format
    @response_format.setter
    def response_format(self, value):
        if value != self._response_format:
            self.__all_chat_completion_params = None
            self.__json_response_format = None
            self._response_format = get_pydantic_type(value)

    @property
    def temperature(self):
        return self._temperature
    @temperature.setter
    def temperature(self, value):
        if value != self._temperature:
            self.__all_chat_completion_params = None
            self._temperature = value

    @property
    def lite_llm_args(self):
        return self._lite_llm_args
    @lite_llm_args.setter
    def lite_llm_args(self, value):
        if value != self._lite_llm_args:
            self.__all_chat_completion_params = None
            self._lite_llm_args = value


    def __get_all_chat_completion_params(self):
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
        return params


    async def _get_chat_completion(self, turns: int, memory: list[dict], memory_delta: list[dict], stream: bool = False, caching: bool = False):
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
            self.logger.debug("Turn %d: Getting chat completion for: %s", turns, messages)
        else:
            self.logger.info("Turn %d: Getting chat completion for %d messages", turns, len(messages))

        create_params = self.__get_all_chat_completion_params().copy()
        create_params["messages"] = messages
        create_params["stream"] = stream
        create_params["caching"] = caching
        return await acompletion(**create_params)


    async def _handle_function_result(self, result, memory: list[dict], memory_delta: list[dict], caching: bool) -> ToolResult:
        if isinstance(result, ToolResult):
            if result.agent and not result.handoff:
                response = await result.agent._run(memory=memory.copy(), memory_delta=memory_delta.copy(), caching=caching)
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
            

    def _get_value(self, content: str):
        if self.response_format:
            if self.response_format is dict or isinstance(self.response_format, dict):
                return json.loads(content)
            elif issubclass(self.response_format, BaseModel):
                ret = self.response_format.model_validate_json(content)
                if isinstance(ret, PrimitiveResult):
                    return ret.result
                else:
                    return ret
            else:
                raise ValueError(f"Unsupported response_format: {self.response_format}")
        else:
            return content


    def _update_partial_response(
            self, 
            partial_response: HandleToolCallResult, 
            tool_call: ChatCompletionMessageToolCallParam, 
            result: ToolResult
    ) -> None:
        partial_response.filtered_tool_calls.append(tool_call)
        partial_response.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "tool_name": tool_call["function"]["name"],
                "content": str(result.value),
            }
        )
        if result.agent:
            partial_response.agent = result.agent
        if result.is_final_answer:
            partial_response.result = result


    def _before_chat_completion(self) -> None:
        pass

    async def _handle_tool_calls(
            self,
            turn: int,
            tool_calls: list[ChatCompletionMessageToolCallParam],
            memory: list[dict],
            memory_delta: list[dict],
            caching: bool,
    ) -> HandleToolCallResult:
        function_map = {f.__name__: f for f in self.tools}
        partial_response = HandleToolCallResult(messages=[], agent=None, filtered_tool_calls=[])

        async_tasks = []
        for tool_call in tool_calls:
            function = tool_call["function"]
            name = function["name"]
            if name not in function_map:
                self.logger.warning("Turn %d: Tool %s not found in function map.", turn, name)
                self._update_partial_response(partial_response, tool_call, ToolResult(value=f"Error: Tool {name} not found."))
                continue            
            
            args = json.loads(function["arguments"])

            func = function_map[name]
            
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.logger.debug("Turn %d: Processing tool call: %s with arguments %s", turn, name, args)
            else:
                self.logger.info("Turn %d: Processing tool call: %s", turn, name)
            t0 = time.time()
            raw_result = func(**args)
            delta_t = time.time() - t0
            if inspect.iscoroutine(raw_result):
                # Store coroutine with its metadata for parallel execution
                self.logger.info("Turn %d: Async tool call found: %s", turn, name)
                async def tool_call_wrapper(raw_result):
                    t0 = time.time()
                    ret = await raw_result
                    delta_t = time.time() - t0
                    if self.logger.getEffectiveLevel() <= logging.DEBUG:
                        self.logger.debug("Turn %d: (After %.2f s) Async tool call %s returned %s", turn, delta_t, name, ret)
                    else:
                        self.logger.info("Turn %d: (After %.2f s) Async tool call %s returned successfully", turn, delta_t, name)
                    return ret
                async_tasks.append((tool_call, tool_call_wrapper(raw_result)))
            else:
                # Handle synchronous results immediately
                if self.logger.getEffectiveLevel() <= logging.DEBUG:
                    self.logger.debug("Turn %d: (After %.2f s) Tool call %s returned %s", turn, delta_t, name, raw_result)
                else:
                    self.logger.info("Turn %d: (After %.2f s) Tool call %s returned successfully", turn, delta_t, name)
                result = await self._handle_function_result(raw_result, memory, memory_delta, caching)
                self._update_partial_response(partial_response, tool_call, result)
                if partial_response.result:
                    break

        # Execute async tasks in parallel if any exist
        if async_tasks:
            self.logger.info("Turn %d: Processing %d async tool call(s)", turn, len(async_tasks))
            raw_results = await asyncio.gather(*(task[1] for task in async_tasks))
            for (tool_call, _), raw_result in zip(async_tasks, raw_results):
                result = await self._handle_function_result(raw_result, memory, memory_delta, caching)
                self._update_partial_response(partial_response, tool_call, result)
                if partial_response.result:
                    break

        # TODO: Cancel all pending async tasks if the result is a final answer

        return partial_response


    def _handle_partial_response(self, turns: int, t0_run: float, partial_response: HandleToolCallResult, message: dict, memory: list[dict], memory_delta: list[dict]) -> Optional[Response]:
        if partial_response.filtered_tool_calls:
            # Only add tool calls to memory if there are any left after filtering
            memory_delta.append(message)
            memory_delta.extend(partial_response.messages)
        if partial_response.result:
            t_run_delta = time.time() - t0_run
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.logger.debug("Turn %d: (After %.2f s) Run completed due to final answer reached in tool call: %s", turns, t_run_delta, partial_response.result.value)
            else:
                self.logger.info("Turn %d: (After %.2f s) Run completed due to final answer reached in tool call", turns, t_run_delta)
            memory.extend(memory_delta)
            return Response(
                value=partial_response.result.value,
                memory_delta=memory_delta,
                agent=self,
            )
        

    def _get_response(self, turns: int, t0_run: float, memory: list[dict], memory_delta: list[dict]):
        memory.extend(memory_delta)
        value = self._get_value(memory[-1]["content"])
        t_run_delta = time.time() - t0_run
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug("Turn %d: (After %.2f s) Run completed with value %s", turns, t_run_delta, value)
        else:
            self.logger.info("Turn %d: (After %.2f s) Run completed", turns, t_run_delta)
        return Response(
            value=value,
            memory_delta=memory_delta,
            agent=self,
        )


    def _get_user_message(self, inputs: tuple, model: str) -> dict:
        def user_message_part(input):
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
            else:
                if hasattr(input, 'read'):  # is file-like object
                    file_name = input.name if input.name else None
                    mime_type = get_mime_type_from_file_like_object(input, file_name)
                    file_name = file_name or "temp_file"
                    content = input.read()
                elif isinstance(input, bytes):
                    file_name = "temp_file"
                    mime_type = get_mime_type_from_content(input)
                    content = input
                else:
                    raise ValueError(f"Unsupported element type: {type(input)}")
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


    def _log_completion(self, turns: int, t0: float, message: dict):
        delta_t = time.time() - t0
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug("Turn %d: (After %.2f s) Received completion: %s", turns, delta_t, message)
        else:
            if message["tool_calls"] and message["content"]:
                self.logger.info("Turn %d: (After %.2f s) Received completion with tool calls and text content.", turns, delta_t)
            elif message["tool_calls"]:
                self.logger.info("Turn %d: (After %.2f s) Received completion with tool calls.", turns, delta_t)
            elif message["content"]:
                self.logger.info("Turn %d: (After %.2f s) Received completion with text content.", turns, delta_t)


    async def _run_and_stream(
            self,
            memory: list[dict],
            memory_delta: list[dict],
            stream_tokens: bool,
            stream_delimiters: bool,
            stream_tool_calls: bool,
            stream_response: bool,
            max_turns: int,
            execute_tools: bool,
            caching: bool,
    ):
        t0_run = time.time()
        active_agent = self
        turns = 0

        while turns < max_turns:
            self._before_chat_completion()
            message = {
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
            completion = await active_agent._get_chat_completion(turns, memory, memory_delta, stream=True, caching=caching)

            if stream_delimiters:
                yield {"delim": "start"}
            async for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                self.logger.debug("Turn %d: Received delta: %s", turns, delta)
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
            if stream_delimiters:
                yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None

            active_agent._log_completion(turns, t0, message)

            if not message["tool_calls"] or not execute_tools:
                active_agent.logger.debug("Turn %d: Ending turn.", turns)
                memory_delta.append(message)
                break

            # convert tool_calls from dict to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCallParam(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls and switching agents
            partial_response = await active_agent._handle_tool_calls(turns, tool_calls, memory, memory_delta, caching)
            response = active_agent._handle_partial_response(turns, t0_run, partial_response, message, memory, memory_delta)
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
            yield active_agent._get_response(turns, t0_run, memory, memory_delta)
        else:
            active_agent.logger.info("Turn %d: (After %.2f s) Run completed", turns, time.time() - t0_run)


    async def _run(
            self,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            max_turns: Optional[int] = float("inf"),
            execute_tools: Optional[bool] = True,
            caching: Optional[bool] = None,
    ) -> Response:
        t0_run = time.time()
        active_agent = self
        turns = 0

        while turns < max_turns and active_agent:
            active_agent._before_chat_completion()
            # get completion with current history, agent
            t0 = time.time()
            completion = await active_agent._get_chat_completion(turns, memory, memory_delta, caching=caching)
            message = completion.choices[0].message.model_dump()
            message["sender"] = active_agent.name

            active_agent._log_completion(turns, t0, message)

            if not message["tool_calls"] or not execute_tools:
                memory_delta.append(message)
                break

            # handle function calls and switching agents
            partial_response = await active_agent._handle_tool_calls(turns, message["tool_calls"], memory, memory_delta, caching)
            response = active_agent._handle_partial_response(turns, t0_run, partial_response, message, memory, memory_delta)
            if response:
                return response
            
            if partial_response.agent:
                active_agent = partial_response.agent

            turns += 1

        return active_agent._get_response(turns, t0_run, memory, memory_delta)

    
    async def run(
            self,
            *inputs,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: Optional[bool] = False,
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: Optional[int] = float("inf"),
            execute_tools: Optional[bool] = True,
            caching: Optional[bool] = None,
    ) -> Response:
        if memory is None:
            memory = []
        if caching is None:
            caching = config.caching

        if memory_delta is None:
            memory_delta = []
        elif memory_delta:
            raise ValueError("memory_delta must be an empty list if provided as a parameter")
        
        if inputs:
            memory_delta.append(self._get_user_message(inputs, self.model))

        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.logger.debug("Starting run with input(s): %s", inputs)
        else:
            self.logger.info("Starting run with %d input(s)", len(inputs))

        if stream:
            return self._run_and_stream(
                memory=memory,
                memory_delta=memory_delta,
                stream_tokens=stream_tokens,
                stream_delimiters=stream_delimiters,
                stream_tool_calls=stream_tool_calls,
                stream_response=stream_response,
                max_turns=max_turns,
                execute_tools=execute_tools,
                caching=caching,
            )
        
        return await self._run(
            memory=memory,
            memory_delta=memory_delta,
            max_turns=max_turns,
            execute_tools=execute_tools,
            caching=caching,
        )


    def run_sync(
            self, 
            *inputs,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: bool = False, 
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: int = float("inf"), 
            execute_tools: bool = True,
            caching: bool = None,
    ) -> Response:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.run(
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
            )
        )


    def apply(
            self, 
            *inputs,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: bool = False, 
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: int = float("inf"), 
            execute_tools: bool = True,
            caching: bool = None,
    ) -> Response:
        """
        Synchronously apply the agent to the inputs and return the response value.
        """
        response = self.run_sync(
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
        )
        if stream:
            return response
        else:
            return response.value
        

    async def __call__(
            self,
            *inputs,
            memory: Optional[list[dict]] = None,
            memory_delta: Optional[list[dict]] = None,
            stream: Optional[bool] = False,
            stream_tokens: bool = True,
            stream_delimiters: bool = False,
            stream_tool_calls: bool = False,
            stream_response: bool = False,
            max_turns: Optional[int] = float("inf"),
            execute_tools: Optional[bool] = True,
            caching: Optional[bool] = None,
    ) -> Response:
        """
        Asynchronously apply the agent to the inputs and return the response value.
        """
        response = await self.run(
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
        )
        if stream:
            return response
        else:
            return response.value
        

__all__ = ["Agent", "Response", "ToolResult"]