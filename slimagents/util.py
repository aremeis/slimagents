import copy
import inspect
from datetime import datetime
from typing import Union
import jsonref
from pydantic import BaseModel


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        tool_call = tool_calls[0]
        index = tool_call.pop("index")
        type_ = tool_call.pop("type", None)
        final_tool_call = final_response["tool_calls"][index]
        merge_fields(final_tool_call, tool_call)
        if type_ and not final_tool_call.get("type"):
            # type = "function" is always returned by LiteLLM in the delta. Bug?
            # This ensures that the type is only set once.
            final_tool_call["type"] = type_


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def fix_schema(schema):
    if "title" in schema:
        # Title is not very useful, and also not supported by Gemini
        del schema["title"]
    if "default" in schema:
        # Default is not supported by OpenI or Gemini
        del schema["default"]
    _type = schema.get("type")
    if _type == "array":
        fix_schema(schema["items"])
    elif _type == "object":
        # OpenAI requires additionalProperties to be false
        schema["additionalProperties"] = False
        required = []
        for propery, property_type in schema["properties"].items():
            fix_schema(property_type)
            # OpenAI requires all properties to be required
            required.append(propery)
        schema["required"] = required
    return schema


def flatten_schema(schema):
    if "$defs" in schema:
        schema = jsonref.replace_refs(schema, lazy_load=False)
        del schema["$defs"]
        # Convert non-JSON-serializable types to serializable types
        schema = copy.deepcopy(schema)
    return schema


def type_to_response_format(type_: Union[dict, type[BaseModel]]) -> dict:
    if type_ is None:
        return None
    elif isinstance(type_, dict):
        return type_
    elif issubclass(type_, BaseModel):
        schema = type_.model_json_schema(mode="serialization")
        # LLMs typically don't support $defs, so we need to remove them
        schema = flatten_schema(schema)
        schema = fix_schema(schema)

        ret = {
            "type": "json_schema",
            "json_schema": {
                "name": type_.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        return ret
    else:
        raise ValueError(f"Unsupported type for response_format: {type_}")
    