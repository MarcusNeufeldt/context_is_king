#!/usr/bin/env python3
"""
Multi-Agent POC Demonstration

This demo showcases the multi-agent framework following principles from:
- framework.md: Modular, configurable agent architecture
- agents_md_framework.md: AGENTS.md standard for agent configuration

Demonstrates:
1. Single agent execution
2. Multi-agent collaboration
3. Context is Code principle (AGENTS.md)
4. Configurable reasoning strategies
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_poc.core.agent import Agent
from multi_agent_poc.core.config import AgentConfig
from multi_agent_poc.core.llm_client import OpenRouterClient


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_single_agent():
    """
    Demo 1: Single Agent Execution

    Shows how to create and run a single agent configured via AGENTS.md
    """
    print_section("DEMO 1: Single Agent Execution")

    # Create ResearchAgent configuration
    research_config = AgentConfig(
        agent_name="ResearchAgent",
        agent_role="Research Specialist",
        instructions_path=Path(__file__).parent / "agents" / "researcher_AGENTS.md",
        reasoning_strategy="react",
        model="minimax/minimax-m2",
        temperature=0.7,
        max_tokens=1500
    )

    # Initialize agent
    print("Initializing ResearchAgent...")
    researcher = Agent(config=research_config)
    print(f"✓ Agent initialized: {researcher}\n")

    # Run a simple research task
    task = "Explain the key benefits of using AGENTS.md standard for AI development in 3-4 bullet points."

    print(f"Task: {task}\n")
    result = researcher.run(task)

    # Display results
    if result['success']:
        print("\n--- Agent Response ---")
        print(result['response'])
        print("\n--- Metadata ---")
        print(f"Strategy: {result['strategy']}")
        print(f"Tokens used: {result['metadata'].get('tokens_used', 'N/A')}")
    else:
        print(f"\n✗ Error: {result['error']}")

    return result


def demo_multi_agent_coordination():
    """
    Demo 2: Multi-Agent Coordination

    Shows how multiple specialized agents can collaborate on a complex task.
    The Coordinator agent orchestrates ResearchAgent and WriterAgent.
    """
    print_section("DEMO 2: Multi-Agent Coordination")

    # Create agents
    print("Initializing multi-agent system...\n")

    # 1. Research Agent
    research_config = AgentConfig(
        agent_name="ResearchAgent",
        agent_role="Research Specialist",
        instructions_path=Path(__file__).parent / "agents" / "researcher_AGENTS.md",
        reasoning_strategy="react",
        temperature=0.7
    )
    researcher = Agent(config=research_config)
    print(f"✓ {researcher}")

    # 2. Writer Agent
    writer_config = AgentConfig(
        agent_name="WriterAgent",
        agent_role="Content Creation Specialist",
        instructions_path=Path(__file__).parent / "agents" / "writer_AGENTS.md",
        reasoning_strategy="react",
        temperature=0.7
    )
    writer = Agent(config=writer_config)
    print(f"✓ {writer}")

    # 3. Coordinator Agent
    coordinator_config = AgentConfig(
        agent_name="CoordinatorAgent",
        agent_role="Multi-Agent Orchestrator",
        instructions_path=Path(__file__).parent / "agents" / "coordinator_AGENTS.md",
        reasoning_strategy="react",
        temperature=0.7
    )
    coordinator = Agent(config=coordinator_config)
    print(f"✓ {coordinator}\n")

    # Complex task requiring coordination
    complex_task = """
Create a brief technical blog post (200-300 words) about the AGENTS.md standard.

The post should:
1. Explain what AGENTS.md is
2. List 3 key benefits
3. Include a simple getting started guide
"""

    print(f"Complex Task:\n{complex_task}\n")

    # Step 1: Coordinator plans the work
    print("\n--- Step 1: Coordinator Planning ---")
    coordination_plan = coordinator.run(
        "Break down this task into subtasks for ResearchAgent and WriterAgent:\n" + complex_task
    )

    if coordination_plan['success']:
        print(coordination_plan['response'][:500] + "...\n")

    # Step 2: ResearchAgent gathers information
    print("\n--- Step 2: Research Phase ---")
    research_result = researcher.run(
        "Research the AGENTS.md standard: what it is, key benefits, and how to get started. Be concise."
    )

    if research_result['success']:
        research_output = research_result['response']
        print(research_output[:500] + "...\n")

    # Step 3: WriterAgent creates content
    print("\n--- Step 3: Writing Phase ---")
    writer_task = f"""
Using this research, write a 200-300 word technical blog post about AGENTS.md:

{research_output[:1000]}

Make it engaging and practical with a getting started section.
"""

    writing_result = writer.run(writer_task)

    if writing_result['success']:
        print("\n--- Final Blog Post ---")
        print(writing_result['response'])
        print("\n--- Metadata ---")
        print(f"Total agents involved: 3 (Coordinator, Researcher, Writer)")
        print(f"Workflow: Planning → Research → Writing")
    else:
        print(f"\n✗ Writing Error: {writing_result['error']}")

    return {
        'coordination': coordination_plan,
        'research': research_result,
        'writing': writing_result
    }


def demo_agents_md_discovery():
    """
    Demo 3: AGENTS.md Discovery

    Shows how the framework discovers and loads AGENTS.md files
    """
    print_section("DEMO 3: AGENTS.md Discovery & Loading")

    from multi_agent_poc.core.agents_md_loader import AgentsMDLoader

    loader = AgentsMDLoader()

    # List available AGENTS.md files
    agents_dir = Path(__file__).parent / "agents"
    agents_md_files = list(agents_dir.glob("*_AGENTS.md"))

    print(f"Found {len(agents_md_files)} AGENTS.md files:\n")

    for md_file in agents_md_files:
        print(f"  📄 {md_file.name}")

        # Load and parse
        parsed = loader.load_agents_md(md_file)

        # Show summary
        sections = parsed['sections']
        print(f"     Sections: {list(sections.keys())[:3]}...")
        print(f"     Tools: {parsed['tools'][:2] if parsed['tools'] else 'None specified'}")
        print()


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  MULTI-AGENT POC DEMONSTRATION")
    print("  Framework based on framework.md + AGENTS.md standard")
    print("="*70)

    try:
        # Demo 1: Single Agent
        demo_single_agent()

        # Demo 2: Multi-Agent Coordination
        input("\n\nPress Enter to continue to Multi-Agent Demo...")
        demo_multi_agent_coordination()

        # Demo 3: AGENTS.md Discovery
        input("\n\nPress Enter to see AGENTS.md Discovery...")
        demo_agents_md_discovery()

        print_section("DEMO COMPLETE")
        print("✓ All demos executed successfully!")
        print("\nKey Takeaways:")
        print("  1. Agents configured via external AGENTS.md files (Context is Code)")
        print("  2. Multiple agents can collaborate on complex tasks")
        print("  3. Configurable reasoning strategies (ReAct shown)")
        print("  4. OpenRouter integration for multi-provider LLM support")
        print("\nNext Steps:")
        print("  - Implement Tree of Thoughts (ToT) strategy")
        print("  - Add hierarchical memory system")
        print("  - Build ACE Loop for self-improvement")
        print("  - Add tool integration (web search, file ops, etc.)")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Demo error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
