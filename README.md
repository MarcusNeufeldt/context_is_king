# Context Engine: State-of-the-Art Context Engineering Framework

A production-ready Python framework implementing cutting-edge context engineering patterns for agentic AI systems, based on the latest research in the field.

## Overview

Context Engineering is the fundamental discipline of managing how AI agents process, store, and utilize information. This framework implements all core patterns from modern research:

- **Layered Memory Architecture** (STM/LTM)
- **Context Isolation** via multi-agent systems
- **Context Abstraction** (Self-Baking)
- **Smart Context Selection** ("Attention Before Attention")
- **Structured Communication** (Blackboard + MessageBus)

Think of this as **"the operating system for agent memory"** - managing context as a core cognitive resource.

## Key Concepts

### The Four Eras of AI Interaction

1. **Era 1.0 (Pre-2020)**: Humans translate intent into structured input (GUIs, CLIs)
2. **Era 2.0 (Today)**: Humans provide instructions and context to agents (LLMs)
3. **Era 3.0 (Future)**: AI as reliable collaborator, handling high-entropy context

**This framework helps you build Era 2.0 systems and prepares for Era 3.0.**

### Context Engineering = Entropy Reduction

The fundamental job of context engineering is reducing the entropy (messiness, ambiguity) of the real world into low-entropy formats that models can understand effectively.

## Features

### Core Components

- **LLM Client**: OpenRouter/Gemini integration with streaming, token counting, KV cache optimization
- **Base Agent**: Full-featured agent with context window management, tool calling, goal tracking
- **Layered Memory**: Short-term (context window) + long-term (vector store) with automatic consolidation
- **Context Manager**: Multi-faceted filtering, keeps context <50% full, optimizes for performance
- **Vector Store**: Semantic search with recency/frequency weighting

### Patterns

- **Isolation Pattern**: Multi-agent systems with isolated contexts (prevents pollution)
- **Self-Baking Pattern**: Convert raw experience into structured knowledge (3 strategies)
- **Communication Pattern**: Structured messages and blackboard for agent collaboration

## Installation

### Prerequisites

- Python 3.11+
- OpenRouter API key (for LLM access)

### Setup

1. Clone or copy this project

2. Install dependencies:

```bash
cd we_explore_context
pip install -r requirements.txt
```

3. Configure environment variables (`.env`):

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=YourApp
```

## Quick Start

### Basic Agent

```python
from src.context_engine import Agent, AgentConfig, LLMConfig

# Configure
llm_config = LLMConfig(model="google/gemini-2.5-flash")
agent_config = AgentConfig(
    name="Assistant",
    system_prompt="You are a helpful AI assistant."
    llm_config=llm_config
)

# Create agent
agent = Agent(agent_config)

# Use it
response = agent.process_message("Hello, how are you?")
print(response)
```

### With Layered Memory

```python
from src.context_engine.core.memory import MemoryConfig

memory_config = MemoryConfig(
    stm_max_messages=20,
    ltm_storage_path="./memory",
    auto_consolidate=True,
    consolidate_threshold=15
)

agent = Agent(AgentConfig(
    name="MemoryAgent",
    memory_config=memory_config
))

# Agent now has persistent memory that consolidates automatically
```

### With Automatic Tool Calling

```python
from src.context_engine import Agent, AgentConfig, tool

# Define tools with decorator
@tool(description="Get current weather for a city")
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get weather information."""
    return {"temp": 20, "condition": "sunny", "city": city}

@tool(description="Perform mathematical calculations")
def calculator(a: float, b: float, operation: str = "add") -> dict:
    """Calculate a op b."""
    ops = {"add": a + b, "multiply": a * b}
    return {"result": ops.get(operation, 0)}

# Create agent with tools enabled
agent = Agent(AgentConfig(
    name="ToolAgent",
    enable_tools=True,  # Enable automatic tool calling
    system_prompt="You have access to tools. Use them when needed."
))

# Register tools
agent.add_tool_function(get_weather)
agent.add_tool_function(calculator)

# Agent automatically detects when to use tools!
response = agent.process_message("What's the weather in Paris?")
# → Agent sees question, decides to call get_weather("Paris")
# → Executes tool, gets result
# → Responds: "The weather in Paris is 20°C and sunny"

response = agent.process_message("Calculate 25 * 17")
# → Agent calls calculator(25, 17, "multiply")
# → Returns: "The result of 25 * 17 is 425"
```

### Multi-Agent System

```python
from src.context_engine.patterns.isolation import MultiAgentSystem

# Create planning agent
planner = Agent(AgentConfig(name="Planner", ...))

# Create multi-agent system
system = MultiAgentSystem(planner)

# Create specialized subagents
researcher = system.create_subagent(
    name="Researcher",
    role="research specialist",
    system_prompt="You research and find information."
)

# Delegate tasks (isolated contexts!)
result = system.delegate_task(
    subagent_id=researcher.agent.id,
    task_description="Research quantum computing basics"
)
```

### Self-Baking (Knowledge Extraction)

```python
from src.context_engine.patterns.self_baking import (
    SelfBakingAgent,
    SummaryBaker
)

# Create agent with self-baking
baker = SummaryBaker()
agent = SelfBakingAgent(config, baking_strategy=baker)

# Have conversations...
agent.process_message("...")

# Bake experience into knowledge
summary = agent.bake_experience()  # Converts raw history → summary
```

## Examples

The `examples/` directory contains 6 comprehensive examples:

1. **01_basic_agent.py** - Simple conversational agent
2. **02_layered_memory.py** - STM/LTM with consolidation
3. **03_multi_agent.py** - Context isolation demo
4. **04_self_baking.py** - All three self-baking strategies
5. **05_research_assistant.py** - Full system combining all patterns
6. **06_tool_calling_agent.py** - **NEW!** Automatic tool detection and calling

### Running Examples

```bash
# Basic agent
python examples/01_basic_agent.py

# Layered memory
python examples/02_layered_memory.py

# Multi-agent system
python examples/03_multi_agent.py

# Self-baking
python examples/04_self_baking.py

# Full research assistant
python examples/05_research_assistant.py

# Automatic tool calling
python examples/06_tool_calling_agent.py
```

## Architecture

### Layered Memory System

```
┌─────────────────────────────────────────┐
│     Short-Term Memory (STM)             │
│  Recent messages in context window      │
│  Fast, volatile, limited                │
└──────────────┬──────────────────────────┘
               │ ftransfer()
               │ (consolidation)
               ↓
┌─────────────────────────────────────────┐
│     Long-Term Memory (LTM)              │
│  Vector store + structured storage      │
│  Persistent, large, semantic search     │
└─────────────────────────────────────────┘
```

### Multi-Agent Architecture

```
┌──────────────────────────────────────┐
│      Planning Agent                  │
│  - High-level orchestration          │
│  - Receives summaries only           │
└───────────┬──────────────────────────┘
            │ delegates
            │
   ┌────────┼────────┐
   │        │        │
   ↓        ↓        ↓
┌─────┐ ┌─────┐ ┌─────┐
│Sub  │ │Sub  │ │Sub  │  Each has:
│Agent│ │Agent│ │Agent│  - Isolated context
│  1  │ │  2  │ │  3  │  - Specialized role
└─────┘ └─────┘ └─────┘  - Own tools
```

### Context Selection ("Attention Before Attention")

```
All Messages → Filtering → Selected Context
                 │
                 ├─ Semantic Relevance
                 ├─ Logical Dependency
                 ├─ Recency Weighting
                 ├─ Frequency Weighting
                 └─ Redundancy Removal

Keep context <50% full (paper recommendation)
```

## API Reference

### Core Classes

#### Agent

```python
class Agent:
    def process_message(self, message: str) -> str
    def add_goal(self, goal: str) -> None
    def complete_goal(self, goal: str) -> bool

    # Tool methods
    def add_tool(self, name: str, function: Callable) -> None  # Legacy
    def add_tool_function(self, func: Callable) -> None  # For automatic calling

    def consolidate_memory(self, force: bool = False) -> None
    def create_subagent(self, name: str, role: str, prompt: str) -> Agent
    def get_stats(self) -> Dict[str, Any]
```

#### Tool Calling

```python
# Decorator for defining tools
@tool(description="Tool description")
def my_tool(param1: str, param2: int = 0) -> dict:
    """Tool implementation."""
    return {"result": "value"}

# Tool registry
class ToolRegistry:
    def register(self, func: Callable) -> ToolDefinition
    def get(self, name: str) -> Optional[ToolDefinition]
    def call_tool(self, name: str, **kwargs) -> Any
    def to_openai_format() -> List[Dict]
```

#### LayeredMemory

```python
class LayeredMemory:
    def add_message(self, message: Message) -> None
    def ftransfer(self, force: bool = False) -> None  # STM → LTM
    def retrieve_relevant(self, query: str, k: int = 5) -> List[MemoryEntry]
    def get_all_context(self, query: Optional[str]) -> List[Message]
```

#### MultiAgentSystem

```python
class MultiAgentSystem:
    def create_subagent(self, name: str, role: str, prompt: str) -> SubAgent
    def delegate_task(self, subagent_id: str, task: str) -> Any
    def get_system_stats(self) -> Dict[str, Any]
```

### Configuration

#### LLMConfig

```python
LLMConfig(
    model="google/gemini-2.5-flash",
    api_key=None,  # Uses env var
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=None,
    context_window=128000,
    max_context_usage=0.5  # Keep <50% full
)
```

#### MemoryConfig

```python
MemoryConfig(
    stm_max_messages=20,
    ltm_storage_path=None,  # Path for persistence
    auto_consolidate=True,
    consolidate_threshold=15,
    ltm_retrieval_k=5
)
```

#### AgentConfig

```python
AgentConfig(
    name="Agent",
    role="assistant",
    system_prompt="You are a helpful AI assistant.",
    llm_config=LLMConfig(),
    memory_config=MemoryConfig(),
    context_config=ContextConfig(),
    enable_streaming=False,
    enable_tools=False,  # Enable automatic tool calling
    max_tool_iterations=5,  # Max tool calling loops
    track_goals=True
)
```

## Best Practices

### From the Research Paper

1. **Keep context <50% full** - Performance degrades beyond this
2. **Optimize for KV caching** - Stable system prompts at the beginning
3. **Keep <30 tools** - Too many tools confuse the model
4. **Keep errors in context** - Allows self-correction
5. **Recite your goals** - Maintain focus in long tasks
6. **Use context isolation** - Prevent pollution in multi-agent systems
7. **Bake your experience** - Convert raw history → learned knowledge

### Implementation Tips

```python
# ✓ Good: Isolated subagents
researcher = system.create_subagent(...)
result = system.delegate_task(researcher.agent.id, task)

# ✗ Bad: One agent with massive context
agent.process_message(task1 + task2 + task3 + ...)

# ✓ Good: Auto-consolidation
memory_config = MemoryConfig(auto_consolidate=True, threshold=15)

# ✗ Bad: Let STM grow indefinitely
# (context will overflow)

# ✓ Good: Structured schemas
findings = SchemaBaker(ProjectInfo).bake(messages, llm)

# ✗ Bad: Just concatenate everything
# (no structure, hard to query)
```

## Project Structure

```
we_explore_context/
├── .env                          # Environment config
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── docs/
│   └── context_engineering.md   # Research paper summary
├── src/
│   └── context_engine/          # Main framework
│       ├── core/                # Core components
│       │   ├── llm_client.py
│       │   ├── agent.py
│       │   ├── memory.py
│       │   └── context_manager.py
│       ├── patterns/            # Context patterns
│       │   ├── isolation.py
│       │   ├── self_baking.py
│       │   └── communication.py
│       └── utils/               # Utilities
│           ├── vector_store.py
│           └── schemas.py
└── examples/                    # Working examples
    ├── 01_basic_agent.py
    ├── 02_layered_memory.py
    ├── 03_multi_agent.py
    ├── 04_self_baking.py
    └── 05_research_assistant.py
```

## Theory: The Semantic Operating System

This framework is a step toward building a **"Semantic OS"** - an operating system that manages context as a core cognitive resource, similar to how traditional OSes manage memory, processes, and I/O.

### Key Parallels

| Traditional OS | Semantic OS (Context Engine) |
|----------------|------------------------------|
| RAM | Short-Term Memory (STM) |
| Hard Drive | Long-Term Memory (LTM) |
| Page Swapping | Memory Consolidation (ftransfer) |
| Process Isolation | Context Isolation (Subagents) |
| IPC | Structured Communication |
| Cache | KV Cache Optimization |

### The Vision

As models evolve toward Era 3.0 (Human-Level Intelligence), they'll handle higher entropy input with less preprocessing. But until then, **context engineering is your most important lever for agent performance.**

## Contributing

This framework is designed to be extended. Key extension points:

- **New baking strategies** in `patterns/self_baking.py`
- **New communication patterns** in `patterns/communication.py`
- **Custom vector stores** implementing the `VectorStore` interface
- **Tool libraries** registered with `agent.add_tool()`

## License

Open source. Use freely for research and production.

## References

Based on the research paper "Context Engineering: The New Frontier of AI Development" (November 2025) and best practices from production agentic systems.

## Credits

Built with:
- OpenRouter for LLM access
- Sentence Transformers for embeddings
- Pydantic for schemas
- Claude Code for development assistance

---

**Context is the new code. Manage it well.**
