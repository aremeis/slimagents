import copy
import inspect
import logging
import re
from typing import Any, Generic, Optional, TypeVar, get_args, get_origin
import jsonref
from pydantic import BaseModel
import mimetypes

logger = logging.getLogger(__name__)

_has_magic = False
try:
    from magic import Magic
    _has_magic = True
except ImportError as e:
    logger.warning(f"python-magic is not properly installed, content based MIME type detection will be disabled. Cause: {e}")
    pass


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


T = TypeVar('T')

class PrimitiveResult(BaseModel):
    result: Any

class IntResult(PrimitiveResult):
    result: int

class FloatResult(PrimitiveResult):
    result: float

class BoolResult(PrimitiveResult):
    result: bool

class ListResult(PrimitiveResult, Generic[T]):
    result: list[T]

TYPE_MAP = {
    int: IntResult,
    float: FloatResult,
    bool: BoolResult,
    list: ListResult,
}


def get_pydantic_type(t: type) -> type:
    _t = get_origin(t) or t
    ret = TYPE_MAP.get(_t, t)
    if ret is ListResult:
        t_args = get_args(t)
        if len(t_args) == 0:
            return ret[Any]
        elif len(t_args) == 1:
            return ret[t_args[0]]
        else:
            raise ValueError(f"Unsupported list type: {t}")
    return ret


JSON_MODE = {
    "type": "json_object",
}

def type_to_response_format(type_: Optional[type]) -> dict:
    if type_ is None:
        return None
    elif isinstance(type_, dict):
        return type_
    elif type_ is dict:
        return JSON_MODE
    elif issubclass(type_, BaseModel):
        schema = type_.model_json_schema(mode="serialization")
        # LLMs typically don't support $defs, so we need to remove them
        schema = flatten_schema(schema)
        schema = fix_schema(schema)

        ret = {
            "type": "json_schema",
            "json_schema": {
                "name": re.sub(r'[^a-zA-Z0-9-_]', '_', type_.__name__),
                "schema": schema,
                "strict": True,
            },
        }
        return ret
    else:
        raise ValueError(f"Unsupported type for response_format: {type_}")
    

def get_mime_type_from_file_like_object(file_like_object:str, filename:str = None):
    """
    Determines the MIME type of a file-like object by examining its name (if available)
    or its content as a fallback.
    
    Args:
        file_like_object: A file-like object with read and seek methods
        filename: The name of the file to get the MIME type from
    Returns:
        str: The detected MIME type of the content, or 'application/octet-stream' if detection fails
    """
    # First try to get MIME type from filename if available
    if filename:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
    
    # Try content-based detection if python-magic is available
    if _has_magic:
        current_pos = file_like_object.tell()
        try:
            # Read the first 2048 bytes for MIME detection
            content = file_like_object.read(2048)
            return get_mime_type_from_content(content)
        finally:
            # Restore original position
            file_like_object.seek(current_pos)
    return 'application/octet-stream'


def get_mime_type_from_content(content: bytes) -> str:
    """
    Determines the MIME type of a bytes object by examining its content.

    Args:
        content: A bytes object - usually the first 2048 bytes of a file

    Returns:
        str: The detected MIME type of the content, or 'application/octet-stream' if detection fails
    """
    # Try content-based detection if python-magic is available
    if _has_magic:
        try:
            # Create a Magic instance for MIME type detection
            mime = Magic(mime=True)
            
            # Detect MIME type
            if isinstance(content, str):
                mime_type = mime.from_buffer(content.encode('utf-8'))
            else:
                mime_type = mime.from_buffer(content)
                
            return mime_type
        except Exception:
            pass
            
    # If python-magic is not available or fails, return a default mime type
    return 'application/octet-stream'
