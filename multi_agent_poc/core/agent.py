"""
Core Agent Module

Implements the main Agent class following the modular, production-ready
architecture described in framework.md.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json

from .config import AgentConfig
from .agents_md_loader import AgentsMDLoader
from .llm_client import OpenRouterClient
from ..strategies.base import ReasoningStrategy
from ..strategies.react import ReActStrategy


class Agent:
    """
    Core Agent class implementing the configurable architecture from framework.md.

    Key principles:
    1. Context is Code: Instructions loaded from AGENTS.md
    2. Configurable Strategies: Select reasoning, memory, self-improvement
    3. Agentic Components: Intelligent processes, not passive utilities
    4. Open Standards: Built on interoperable protocols
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Optional[OpenRouterClient] = None
    ):
        """
        Initialize an agent with the given configuration.

        Args:
            config: AgentConfig object defining the agent's capabilities
            llm_client: Optional custom LLM client (defaults to OpenRouter)
        """
        self.config = config
        self.llm_client = llm_client or OpenRouterClient()
        self.loader = AgentsMDLoader()

        # Load agent instructions from AGENTS.md
        self.instructions = self._load_instructions()

        # Initialize reasoning strategy
        self.reasoning_strategy = self._init_reasoning_strategy()

        # Initialize memory (if enabled)
        self.memory = None
        if config.memory_enabled:
            self.memory = self._init_memory()

        # Execution history
        self.history = []

    def _load_instructions(self) -> str:
        """
        Load agent instructions from AGENTS.md file.

        Implements "Context is Code" principle from framework.md.
        """
        if not self.config.instructions_path.exists():
            raise FileNotFoundError(
                f"AGENTS.md not found at {self.config.instructions_path}"
            )

        return self.loader.get_agent_instructions(self.config.instructions_path)

    def _init_reasoning_strategy(self) -> ReasoningStrategy:
        """
        Initialize the reasoning strategy based on config.

        Supports: 'react', 'tot', 'auto' as defined in framework.md Section III.
        """
        strategy_name = self.config.reasoning_strategy.lower()

        if strategy_name == "react":
            return ReActStrategy()
        elif strategy_name == "tot":
            # ToT strategy would be imported here
            # from ..strategies.tot import ToTStrategy
            # return ToTStrategy()
            raise NotImplementedError("ToT strategy not yet implemented")
        elif strategy_name == "auto":
            # Auto strategy would assess complexity and route
            # from ..strategies.auto import AutoStrategy
            # return AutoStrategy()
            raise NotImplementedError("Auto strategy not yet implemented")
        else:
            raise ValueError(f"Unknown reasoning strategy: {strategy_name}")

    def _init_memory(self):
        """
        Initialize memory backend if enabled.

        Would implement hierarchical memory from framework.md Section III.
        """
        # Placeholder for memory implementation
        # from ..memory.hierarchical import HierarchicalMemory
        # return HierarchicalMemory(self.config.memory_path)
        return None

    def run(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the agent's reasoning strategy.

        Args:
            task: The task or query to process
            **kwargs: Additional parameters for the reasoning strategy

        Returns:
            Dict with 'response', 'success', 'reasoning_trace', etc.
        """
        print(f"\n{'='*60}")
        print(f"Agent: {self.config.agent_name} ({self.config.agent_role})")
        print(f"Strategy: {self.reasoning_strategy.name}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        # Build context for reasoning
        context = self._build_context(task, **kwargs)

        # Execute reasoning strategy
        result = self.reasoning_strategy.reason(
            task=task,
            context=context,
            llm_client=self.llm_client,
            **kwargs
        )

        # Store in history
        self.history.append({
            'task': task,
            'result': result,
            'context': context
        })

        # Update memory if enabled
        if self.memory:
            self._update_memory(task, result)

        return result

    def _build_context(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Build the context dict for reasoning strategy.

        Includes agent instructions, available tools, memory, etc.
        """
        context = {
            'agent_instructions': self.instructions,
            'agent_name': self.config.agent_name,
            'agent_role': self.config.agent_role,
            'available_tools': self.config.available_tools,
            'model': self.config.model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }

        # Add memory context if available
        if self.memory:
            context['relevant_memories'] = self._retrieve_relevant_memories(task)

        # Add any additional context from kwargs
        context.update(kwargs)

        return context

    def _retrieve_relevant_memories(self, task: str) -> str:
        """Retrieve relevant memories for the task (placeholder)"""
        # Would implement semantic search over memory
        return ""

    def _update_memory(self, task: str, result: Dict[str, Any]):
        """Update memory with task and result (placeholder)"""
        # Would store interaction in memory backend
        pass

    def get_history(self) -> list:
        """Get the agent's execution history"""
        return self.history

    def save_state(self, path: Path):
        """Save agent state to disk"""
        state = {
            'config': {
                'agent_name': self.config.agent_name,
                'agent_role': self.config.agent_role,
                'reasoning_strategy': self.config.reasoning_strategy,
                'model': self.config.model,
            },
            'history': self.history
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def __repr__(self):
        return f"Agent(name='{self.config.agent_name}', role='{self.config.agent_role}', strategy='{self.reasoning_strategy.name}')"
