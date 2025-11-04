"""
Context Engine: A state-of-the-art framework for context engineering in agentic AI systems.

This framework implements the core patterns from modern context engineering research:
- Context Isolation via multi-agent systems
- Layered Memory Architecture (STM/LTM)
- Context Abstraction (Self-Baking)
- Smart Context Selection
- Structured Agent Communication
- Automatic Tool Calling

Based on the principles of entropy reduction and semantic operating systems.
"""

__version__ = "0.1.0"

from .core.agent import Agent, AgentConfig
from .core.llm_client import LLMClient, LLMConfig
from .core.memory import LayeredMemory, MemoryConfig
from .core.context_manager import ContextManager, ContextConfig
from .core.tools import tool, ToolDefinition, ToolRegistry

__all__ = [
    "Agent",
    "AgentConfig",
    "LLMClient",
    "LLMConfig",
    "LayeredMemory",
    "MemoryConfig",
    "ContextManager",
    "ContextConfig",
    "tool",
    "ToolDefinition",
    "ToolRegistry",
]
