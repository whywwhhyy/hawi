"""Utility functions for tool schema generation and validation."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from jsonschema import ValidationError, validate


def build_parameters_schema(
    run_method: Callable[..., Any] | None,
    type_converter: Callable[[type], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build JSON Schema from run method type hints.

    Args:
        run_method: The run method to extract parameters from.
        type_converter: Optional custom type converter function.

    Returns:
        JSON Schema object for the run method parameters.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    if run_method is None:
        return schema

    try:
        hints = get_type_hints(run_method)
    except Exception:
        hints = {}

    sig = inspect.signature(run_method)

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = hints.get(param_name, str)
        if type_converter is not None:
            param_schema = type_converter(param_type)
        else:
            param_schema = type_to_json_schema(param_type)

        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            schema["required"].append(param_name)

        schema["properties"][param_name] = param_schema

    return schema


def type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert Python type to JSON Schema type.

    Args:
        python_type: Python type to convert.

    Returns:
        JSON Schema type definition.
    """
    origin = getattr(python_type, "__origin__", None)
    args = getattr(python_type, "__args__", ())

    if origin is not None:
        # Handle Optional[T] = Union[T, None]
        if origin is type or str(origin) == "typing.Union":
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return type_to_json_schema(non_none_types[0])
        # Handle list[T]
        if origin is list and args:
            return {
                "type": "array",
                "items": type_to_json_schema(args[0]),
            }
        # Handle dict[str, T]
        if origin is dict and len(args) >= 2:
            return {
                "type": "object",
                "additionalProperties": type_to_json_schema(args[1]),
            }

    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        Any: {},
    }

    return type_map.get(python_type, {"type": "string"})


def validate_parameters(
    parameters: dict[str, Any],
    parameters_schema: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Validate parameters against JSON Schema.

    Args:
        parameters: Parameters to validate.
        parameters_schema: JSON Schema for validation.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    if parameters_schema is None:
        return True, []

    try:
        validate(instance=parameters, schema=parameters_schema)
        return True, []
    except ValidationError as e:
        errors = [f"{e.json_path}: {e.message}"]
        if e.context:
            for error in e.context:
                errors.append(f"{error.json_path}: {error.message}")
        return False, errors
    except Exception as e:
        return False, [str(e)]
