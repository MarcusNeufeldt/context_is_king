"""
Base Reasoning Strategy Module

Defines the abstract base class for reasoning strategies as outlined
in framework.md Section III.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ReasoningStrategy(ABC):
    """
    Abstract base class for reasoning strategies.

    Implementations:
    - ReActStrategy: Fast, linear reasoning for 80% of tasks
    - ToTStrategy: Tree of Thoughts for complex, non-linear problems
    - AutoStrategy: Dynamically selects based on task complexity
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def reason(
        self,
        task: str,
        context: Dict[str, Any],
        llm_client,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute reasoning strategy for the given task.

        Args:
            task: The user's task/query
            context: Agent context including instructions, memory, tools
            llm_client: LLM client for API calls
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dict with 'response', 'reasoning_trace', 'tool_calls', etc.
        """
        pass

    def format_system_prompt(self, agent_instructions: str, context: Dict[str, Any]) -> str:
        """
        Format the system prompt combining agent instructions with context.

        This implements the "Context is Code" principle from framework.md.
        """
        system_prompt = f"{agent_instructions}\n\n"

        # Add available tools
        if context.get('available_tools'):
            system_prompt += "\n## Available Tools\n"
            for tool in context['available_tools']:
                system_prompt += f"- {tool}\n"

        # Add memory context if available
        if context.get('relevant_memories'):
            system_prompt += "\n## Relevant Context from Memory\n"
            system_prompt += context['relevant_memories'] + "\n"

        return system_prompt
