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
import json
from pydantic import BaseModel, Field

from .llm_client import LLMClient, LLMConfig
from .memory import LayeredMemory, MemoryConfig
from .context_manager import ContextManager, ContextConfig
from .tools import ToolRegistry, ToolDefinition
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
    enable_tools: bool = False  # Enable automatic tool calling
    max_tool_iterations: int = 5  # Max tool calling loops to prevent infinite loops

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

        # Tool registry (new system for automatic tool calling)
        self.tool_registry = ToolRegistry()

        # Legacy tool dict (for backwards compatibility)
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
        """
        Generate response using LLM.

        If tools are enabled, implements automatic tool calling loop:
        1. LLM decides if it needs to call tools
        2. Tools are executed automatically
        3. Results are fed back to LLM
        4. Loop continues until LLM provides final response
        """
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

        # Tool calling loop
        if self.config.enable_tools and len(self.tool_registry) > 0:
            return self._generate_with_tools(context)

        # Standard response (no tools)
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
            response, usage, _ = self.llm.complete(context)
            self.metadata.total_tokens += usage["total_tokens"]
            return response

    def _generate_with_tools(self, initial_context: List[Message]) -> str:
        """
        Generate response with automatic tool calling.

        Implements agentic loop:
        - LLM generates response (possibly with tool calls)
        - Execute any tool calls
        - Add results to context
        - Repeat until LLM provides final text response
        """
        context = initial_context.copy()
        tools_spec = self.tool_registry.to_openai_format()

        iteration = 0
        max_iterations = self.config.max_tool_iterations

        while iteration < max_iterations:
            iteration += 1

            # Generate response with tools
            response, usage, tool_calls = self.llm.complete(context, tools=tools_spec)
            self.metadata.total_tokens += usage["total_tokens"]

            # If no tool calls, we have final response
            if not tool_calls:
                return response or "I apologize, but I don't have a response."

            # Execute tool calls
            print(f"\n[Tool Calling - Iteration {iteration}]")
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                print(f"  → Calling: {function_name}({arguments})")

                # Execute tool
                try:
                    result = self.tool_registry.call_tool(function_name, **arguments)
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    print(f"  ✓ Result: {result_str[:100]}...")
                except Exception as e:
                    result_str = f"Error: {str(e)}"
                    print(f"  ✗ Error: {str(e)}")

                # Add tool result to context
                # Note: OpenAI expects tool results in a specific format
                context.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=f"Called {function_name} with {arguments}",
                    metadata={"tool_call_id": tool_call["id"]}
                ))
                context.append(Message(
                    role=MessageRole.TOOL,
                    content=result_str,
                    name=function_name,
                    metadata={"tool_call_id": tool_call["id"]}
                ))

        # Max iterations reached
        return "I apologize, but I've reached the maximum number of tool calls. Please try rephrasing your request."

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

        Legacy method - for backwards compatibility.
        For automatic tool calling, use add_tool_function() or the @tool decorator.

        Args:
            name: Tool name
            function: Callable function
            description: Tool description for LLM
        """
        self.tools[name] = function
        # Also add to tool registry for automatic calling
        self.tool_registry.register(function, name=name, description=description)

    def add_tool_function(self, func: Callable) -> None:
        """
        Register a function as a tool for automatic calling.

        The function should either:
        1. Be decorated with @tool decorator
        2. Have a docstring that will be used as description

        Example:
            ```python
            @tool(description="Get current weather")
            def get_weather(city: str) -> dict:
                return {"temp": 20, "condition": "sunny"}

            agent.add_tool_function(get_weather)
            ```

        Args:
            func: Function to register as tool
        """
        self.tool_registry.register(func)

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
