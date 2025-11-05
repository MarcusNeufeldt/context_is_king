# Multi-Agent POC Framework

A modular, production-ready framework for creating agentic AI systems based on:
- **framework.md**: SOTA-2026 architectural principles
- **agents_md_framework.md**: AGENTS.md standard for agent configuration

## Key Features

### 1. Context is Code
All agent instructions externalized in version-controlled `AGENTS.md` files. No hard-coded prompts.

### 2. Configurable Strategies
- **ReAct**: Fast, linear reasoning for 80% of tasks (implemented)
- **Tree of Thoughts**: Complex problem-solving (planned)
- **Auto**: Dynamic strategy selection (planned)

### 3. Multi-Agent Collaboration
Specialized agents coordinate to solve complex tasks requiring different expertise.

### 4. OpenRouter Integration
Support for multiple LLM providers through OpenRouter API.

## Architecture

```
multi_agent_poc/
├── core/
│   ├── agent.py              # Core Agent class
│   ├── config.py             # AgentConfig dataclass
│   ├── llm_client.py         # OpenRouter integration
│   └── agents_md_loader.py   # AGENTS.md parser
├── strategies/
│   ├── base.py               # Abstract reasoning strategy
│   └── react.py              # ReAct implementation
├── agents/
│   ├── researcher_AGENTS.md  # Research agent config
│   ├── writer_AGENTS.md      # Writer agent config
│   └── coordinator_AGENTS.md # Coordinator agent config
├── memory/                   # Memory backends (planned)
├── tools/                    # Agent tools (planned)
└── demo.py                   # Demonstration script
```

## Quick Start

### 1. Environment Setup

Ensure Python 3.11+ is available and OpenRouter API key is configured:

```bash
# In Firebase Studio environment
export PATH="$HOME/.nix-profile/bin:$PATH"
export PYTHONPATH="$HOME/.nix-profile/lib/python3.11/site-packages:$PYTHONPATH"

# Verify .env file exists with OPENROUTER_API_KEY
cat ../.env | grep OPENROUTER_API_KEY
```

### 2. Run the Demo

```bash
cd /home/user/test/we_explore_context

# Run the demonstration
PATH="$HOME/.nix-profile/bin:$PATH" PYTHONPATH="$HOME/.nix-profile/lib/python3.11/site-packages:$PYTHONPATH" python3 multi_agent_poc/demo.py
```

### 3. Create Your Own Agent

**Step 1**: Create an AGENTS.md file

```markdown
# AGENTS.md: My Custom Agent

## Agent Overview
**Agent Name:** MyAgent
**Agent Role:** Custom Specialist
**Purpose:** [What this agent does]

## Agent Instructions
[Detailed instructions for the agent...]

## Conventions & Patterns
[How the agent should behave and format outputs...]

## Available Tools
- [Tool 1]
- [Tool 2]

## Example Tasks
[Sample tasks the agent should handle...]
```

**Step 2**: Configure and initialize

```python
from multi_agent_poc import Agent, AgentConfig

config = AgentConfig(
    agent_name="MyAgent",
    agent_role="Custom Specialist",
    instructions_path="path/to/my_agent_AGENTS.md",
    reasoning_strategy="react",
    model="minimax/minimax-m2"
)

agent = Agent(config=config)
result = agent.run("Your task here")
print(result['response'])
```

## Example Usage

### Single Agent

```python
from multi_agent_poc import Agent, AgentConfig
from pathlib import Path

# Configure agent
config = AgentConfig(
    agent_name="ResearchAgent",
    agent_role="Research Specialist",
    instructions_path=Path("agents/researcher_AGENTS.md"),
    reasoning_strategy="react"
)

# Initialize and run
agent = Agent(config=config)
result = agent.run("Research the benefits of AGENTS.md standard")

if result['success']:
    print(result['response'])
```

### Multi-Agent Collaboration

```python
# Create specialized agents
researcher = Agent(config=research_config)
writer = Agent(config=writer_config)
coordinator = Agent(config=coordinator_config)

# Coordinator plans the work
plan = coordinator.run("Create a blog post about AGENTS.md")

# Researcher gathers information
research = researcher.run("Research AGENTS.md benefits")

# Writer creates content
article = writer.run(f"Write blog post using: {research['response']}")

print(article['response'])
```

## Configuration Options

### AgentConfig Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | str | Unique agent identifier |
| `agent_role` | str | Agent's role/specialization |
| `instructions_path` | Path | Path to AGENTS.md file |
| `reasoning_strategy` | str | "react", "tot", or "auto" |
| `model` | str | LLM model identifier (default: "minimax/minimax-m2") |
| `temperature` | float | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | Maximum response tokens |
| `memory_enabled` | bool | Enable memory system |
| `self_improvement_enabled` | bool | Enable ACE Loop |
| `available_tools` | list | List of available tools |

## Framework Principles

### 1. Context is Code
Agent instructions are externalized in AGENTS.md files, not hard-coded. This enables:
- Version control of agent behavior
- Easy modification without code changes
- Sharing agent configurations across projects

### 2. Evolving Intelligence
Optional ACE Loop for self-improvement (planned):
- Agents learn from interactions
- Build domain-specific playbooks
- Prevent context collapse in long-running agents

### 3. Agentic Components
Core capabilities implemented as intelligent processes:
- Memory as agentic retrieval (planned)
- Tools as agentic actions (planned)
- Reasoning as pluggable strategies

### 4. Interoperability
Built on open standards:
- AGENTS.md standard for configuration
- OpenRouter for multi-provider LLM access
- MCP protocol support (planned)

### 5. Deliberate Reasoning
Multiple reasoning modes:
- **ReAct**: Fast, linear (implemented)
- **Tree of Thoughts**: Complex, deliberate (planned)
- **Auto**: Dynamic selection (planned)

## Demos Included

### Demo 1: Single Agent Execution
Shows basic agent initialization and execution with AGENTS.md configuration.

### Demo 2: Multi-Agent Coordination
Demonstrates Coordinator orchestrating Researcher and Writer agents to create a blog post.

### Demo 3: AGENTS.md Discovery
Shows the discovery hierarchy and parsing of AGENTS.md files.

## Roadmap

### Current (v0.1.0)
- ✅ Core Agent class
- ✅ AgentConfig system
- ✅ AGENTS.md loader and parser
- ✅ ReAct reasoning strategy
- ✅ OpenRouter integration
- ✅ Multi-agent coordination (basic)

### Next (v0.2.0)
- ⏳ Tree of Thoughts strategy
- ⏳ Auto strategy with complexity assessment
- ⏳ Hierarchical memory system
- ⏳ Tool integration framework
- ⏳ ACE Loop for self-improvement

### Future (v0.3.0+)
- Agent-to-Agent (A2A) protocol
- MCP (Model Context Protocol) support
- Streaming responses
- Async agent execution
- Production deployment templates

## Contributing

This is a POC demonstrating architectural principles. To extend:

1. **Add new reasoning strategies**: Implement `ReasoningStrategy` base class
2. **Create new agents**: Write AGENTS.md files following the standard
3. **Add tools**: Implement tool interfaces in `tools/` directory
4. **Enhance memory**: Build memory backends in `memory/` directory

## References

- **framework.md**: Architectural principles and design patterns
- **agents_md_framework.md**: AGENTS.md standard specification
- **OpenRouter**: https://openrouter.ai/docs
- **AGENTS.md Standard**: https://agents.md/

## License

MIT License - See LICENSE file for details

## Credits

Built following principles from:
- framework.md (SOTA-2026 Agentic Framework)
- agents_md_framework.md (AGENTS.md Standard)
- OpenRouter API for multi-provider LLM access
