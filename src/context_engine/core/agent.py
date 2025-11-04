"""
Base Agent class with context management.

Integrates all core components:
- LLM client for completions
- Layered memory (STM/LTM)
- Context manager for optimization
- Tool calling interface
"""

from typing import List, Dict, Any, Optional, Callable
import uuid
from pydantic import BaseModel, Field

from .llm_client import LLMClient, LLMConfig
from .memory import LayeredMemory, MemoryConfig
from .context_manager import ContextManager, ContextConfig
from ..utils.schemas import Message, MessageRole, AgentMetadata, ToolCall


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = "Agent"
    role: str = "assistant"
    system_prompt: str = "You are a helpful AI assistant."

    # Component configs
    llm_config: Optional[LLMConfig] = None
    memory_config: Optional[MemoryConfig] = None
    context_config: Optional[ContextConfig] = None

    # Behavior
    enable_streaming: bool = False
    enable_memory_consolidation: bool = True
    enable_context_optimization: bool = True

    # Goal tracking (implements "Recite Your Goals" pattern)
    track_goals: bool = True


class Agent:
    """
    Base agent with full context engineering capabilities.

    Features:
    - Layered memory (STM/LTM)
    - Smart context selection
    - Tool calling
    - Goal tracking and recitation
    - Subagent spawning (for isolation pattern)

    Example:
        ```python
        agent = Agent(AgentConfig(name="Assistant"))
        response = agent.process_message("Hello, how are you?")
        print(response)
        ```
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize agent with configuration."""
        self.config = config or AgentConfig()
        self.id = str(uuid.uuid4())

        # Initialize components
        self.llm = LLMClient(self.config.llm_config)
        self.memory = LayeredMemory(self.config.memory_config)
        self.context_manager = ContextManager(self.config.context_config)

        # Agent metadata
        self.metadata = AgentMetadata(
            id=self.id,
            name=self.config.name,
            role=self.config.role
        )

        # System prompt as initial message
        self.system_message = Message(
            role=MessageRole.SYSTEM,
            content=self.config.system_prompt
        )

        # Tool registry
        self.tools: Dict[str, Callable] = {}

        # Goal tracking (implements "Recite Your Goals" pattern from paper)
        self.active_goals: List[str] = []

        # Statistics
        self.total_interactions = 0

    def process_message(
        self,
        user_message: str,
        stream: Optional[bool] = None
    ) -> str:
        """
        Process a user message and return response.

        This is the main interaction method.

        Args:
            user_message: User's input message
            stream: If True, stream response (overrides config)

        Returns:
            Assistant's response
        """
        # Add user message to memory
        msg = Message(role=MessageRole.USER, content=user_message)
        self.memory.add_message(msg)

        # Get response
        response = self._generate_response(stream=stream)

        # Add response to memory
        response_msg = Message(role=MessageRole.ASSISTANT, content=response)
        self.memory.add_message(response_msg)

        # Update statistics
        self.total_interactions += 1
        self.metadata.total_messages += 2

        return response

    def _generate_response(self, stream: Optional[bool] = None) -> str:
        """Generate response using LLM."""
        use_streaming = stream if stream is not None else self.config.enable_streaming

        # Build context
        context = self._build_context()

        # Check context limits
        fits, token_count = self.llm.check_context_limit(context)
        if not fits:
            print(f"Warning: Context approaching limit ({token_count} tokens)")

        # Add goal recitation if enabled (keeps goals in attention)
        if self.config.track_goals and self.active_goals:
            goals_reminder = self._create_goals_reminder()
            context.append(Message(
                role=MessageRole.SYSTEM,
                content=goals_reminder,
                metadata={"type": "goal_recitation"}
            ))

        # Generate response
        if use_streaming:
            # Streaming response
            chunks = []
            for chunk in self.llm.complete_stream(context):
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            print()  # Newline after streaming
            return "".join(chunks)
        else:
            # Non-streaming response
            response, usage = self.llm.complete(context)
            self.metadata.total_tokens += usage["total_tokens"]
            return response

    def _build_context(self) -> List[Message]:
        """
        Build optimal context for LLM.

        Combines:
        1. System message (stable prefix for KV cache)
        2. Relevant LTM entries
        3. Optimized STM messages
        """
        # Get current query (last user message)
        current_query = None
        if self.memory.stm:
            user_messages = [m for m in self.memory.stm if m.role == MessageRole.USER]
            if user_messages:
                current_query = user_messages[-1].content

        # Get full context from memory (STM + relevant LTM)
        all_messages = self.memory.get_all_context(current_query)

        # Optimize context selection if enabled
        if self.config.enable_context_optimization:
            optimized = self.context_manager.select_context(
                all_messages=all_messages,
                current_goal=self._get_current_goal(),
                token_counter=self.llm.count_tokens
            )
            messages = optimized
        else:
            messages = all_messages

        # Ensure system message is first (KV cache optimization)
        final_context = [self.system_message]

        # Add other messages
        for msg in messages:
            if msg.role != MessageRole.SYSTEM or msg != self.system_message:
                final_context.append(msg)

        return final_context

    def add_tool(self, name: str, function: Callable, description: str = "") -> None:
        """
        Register a tool/function for the agent to call.

        Args:
            name: Tool name
            function: Callable function
            description: Tool description for LLM
        """
        self.tools[name] = function

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a registered tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered")

        tool = self.tools[tool_name]
        try:
            result = tool(**kwargs)
            return result
        except Exception as e:
            # Keep errors in context (paper recommendation)
            error_msg = f"Tool '{tool_name}' failed: {str(e)}"
            return {"error": error_msg}

    def add_goal(self, goal: str) -> None:
        """
        Add a goal to track.

        Implements "Recite Your Goals" pattern: keep goals in attention.
        """
        if goal not in self.active_goals:
            self.active_goals.append(goal)

    def complete_goal(self, goal: str) -> bool:
        """Mark a goal as completed."""
        if goal in self.active_goals:
            self.active_goals.remove(goal)
            self.metadata.tasks_completed += 1
            return True
        return False

    def _get_current_goal(self) -> Optional[str]:
        """Get the current (most recent) goal."""
        return self.active_goals[-1] if self.active_goals else None

    def _create_goals_reminder(self) -> str:
        """
        Create a reminder of active goals.

        Implements "Recite Your Goals" pattern from the paper.
        The model recites the full list of remaining goals in natural language,
        keeping primary objectives in its recent attention.
        """
        if not self.active_goals:
            return ""

        reminder = "Current active goals:\n"
        for i, goal in enumerate(self.active_goals, 1):
            reminder += f"{i}. {goal}\n"

        return reminder

    def consolidate_memory(self, force: bool = False) -> None:
        """
        Manually trigger memory consolidation (STM → LTM).

        Args:
            force: Force consolidation even if threshold not reached
        """
        self.memory.ftransfer(force=force)

    def clear_short_term_memory(self) -> None:
        """Clear short-term memory (useful for new tasks)."""
        self.memory.clear_stm()

    def save_state(self) -> None:
        """Save agent state (LTM and structured state) to disk."""
        self.memory.save()

    def load_state(self) -> None:
        """Load agent state from disk."""
        self.memory.load()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.id,
            "name": self.config.name,
            "total_interactions": self.total_interactions,
            "total_messages": self.metadata.total_messages,
            "total_tokens": self.metadata.total_tokens,
            "tasks_completed": self.metadata.tasks_completed,
            "stm_size": len(self.memory.stm),
            "ltm_size": len(self.memory.ltm),
            "active_goals": len(self.active_goals),
            "consolidations": self.memory.total_consolidations
        }

    def create_subagent(
        self,
        name: str,
        role: str,
        system_prompt: str,
        config: Optional[AgentConfig] = None
    ) -> "Agent":
        """
        Create a subagent with isolated context.

        Implements Context Isolation pattern from the paper.
        Subagents have their own context and system prompt.

        Args:
            name: Subagent name
            role: Subagent role/specialization
            system_prompt: System prompt for subagent
            config: Optional full config (will override name, role, prompt)

        Returns:
            New Agent instance
        """
        if config is None:
            config = AgentConfig(
                name=name,
                role=role,
                system_prompt=system_prompt,
                llm_config=self.config.llm_config,  # Inherit LLM config
            )
        else:
            config.name = name
            config.role = role
            config.system_prompt = system_prompt

        subagent = Agent(config)
        subagent.metadata.parent_id = self.id
        subagent.metadata.specialization = role

        return subagent

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"Agent(name='{self.config.name}', id='{self.id[:8]}...', role='{self.config.role}')"
