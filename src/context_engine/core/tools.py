"""
Tool Calling System for Agents

Implements automatic tool detection and calling based on user input.
The LLM sees tool definitions and decides when to call them.

Supports OpenAI-style function calling format.
"""

from typing import Callable, Dict, Any, List, Optional, get_type_hints
from pydantic import BaseModel, Field, create_model
from enum import Enum
import inspect
import json


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None  # For string enums


class ToolDefinition(BaseModel):
    """
    Definition of a tool that can be called by the agent.

    Compatible with OpenAI function calling format.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    function: Optional[Callable] = None  # The actual function to call

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format.

        Returns:
            Dict compatible with OpenAI's function calling API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """
        Convert to Anthropic tool use format.

        Returns:
            Dict compatible with Anthropic's tool use API
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

    def call(self, **kwargs) -> Any:
        """
        Call the tool with given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool result
        """
        if not self.function:
            raise ValueError(f"Tool '{self.name}' has no function attached")

        return self.function(**kwargs)


class ToolCall(BaseModel):
    """Represents a tool call made by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    """Result from executing a tool."""
    tool_call_id: str
    name: str
    content: str  # JSON string of result


def python_type_to_json_type(py_type: type) -> str:
    """Convert Python type to JSON Schema type."""
    if py_type == str:
        return "string"
    elif py_type in (int, float):
        return "number"
    elif py_type == bool:
        return "boolean"
    elif py_type == list:
        return "array"
    elif py_type == dict:
        return "object"
    else:
        return "string"  # Default fallback


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable:
    """
    Decorator to convert a Python function into a tool.

    Automatically extracts parameter information from function signature.

    Example:
        ```python
        @tool(description="Get the current weather")
        def get_weather(city: str, units: str = "celsius") -> dict:
            '''Get weather for a city.'''
            return {"temp": 20, "condition": "sunny"}

        # Tool definition is automatically created
        tool_def = get_weather.tool_definition
        ```

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)

    Returns:
        Decorated function with .tool_definition attribute
    """
    def decorator(func: Callable) -> Callable:
        # Get function info
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()

        # Extract parameters from function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            # Get type hint
            param_type = type_hints.get(param_name, str)
            json_type = python_type_to_json_type(param_type)

            # Build parameter definition
            param_def = {
                "type": json_type,
                "description": f"Parameter: {param_name}"
            }

            parameters_schema["properties"][param_name] = param_def

            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                parameters_schema["required"].append(param_name)

        # Create tool definition
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters_schema,
            function=func
        )

        # Attach to function
        func.tool_definition = tool_def

        return func

    return decorator


class ToolRegistry:
    """
    Registry for managing tools available to an agent.

    Handles tool registration, lookup, and execution.
    """

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> ToolDefinition:
        """
        Register a function as a tool.

        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description

        Returns:
            ToolDefinition for the registered tool
        """
        # Check if function already has tool definition (from decorator)
        if hasattr(func, 'tool_definition'):
            tool_def = func.tool_definition
        else:
            # Create tool definition manually
            tool_name = name or func.__name__
            tool_description = description or (func.__doc__ or "").strip()

            # Extract parameters
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)

            parameters_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }

            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, str)
                json_type = python_type_to_json_type(param_type)

                parameters_schema["properties"][param_name] = {
                    "type": json_type,
                    "description": f"Parameter: {param_name}"
                }

                if param.default == inspect.Parameter.empty:
                    parameters_schema["required"].append(param_name)

            tool_def = ToolDefinition(
                name=tool_name,
                description=tool_description,
                parameters=parameters_schema,
                function=func
            )

        self.tools[tool_def.name] = tool_def
        return tool_def

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Get all registered tools."""
        return list(self.tools.values())

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def to_anthropic_format(self) -> List[Dict[str, Any]]:
        """Convert all tools to Anthropic format."""
        return [tool.to_anthropic_format() for tool in self.tools.values()]

    def call_tool(self, name: str, **kwargs) -> Any:
        """
        Call a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool parameters

        Returns:
            Tool result
        """
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")

        return tool.call(**kwargs)

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self.tools
