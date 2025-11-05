"""
Multi-Agent POC Framework

A modular, production-ready framework for creating agentic AI systems
following the principles from framework.md and using the AGENTS.md standard.

Key Features:
- Context is Code: Agent instructions externalized in AGENTS.md files
- Configurable Strategies: ReAct, ToT, Auto reasoning modes
- Multi-Agent Collaboration: Coordinate specialized agents
- OpenRouter Integration: Support for multiple LLM providers
"""

__version__ = "0.1.0"

from .core.agent import Agent
from .core.config import AgentConfig
from .core.llm_client import OpenRouterClient
from .strategies.react import ReActStrategy

__all__ = [
    "Agent",
    "AgentConfig",
    "OpenRouterClient",
    "ReActStrategy",
]
