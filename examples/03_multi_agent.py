"""
Example 3: Multi-Agent Context Isolation

Demonstrates:
- Creating specialized subagents with isolated contexts
- Task delegation from planning agent to subagents
- Context isolation (subagent details don't pollute parent context)
- Structured communication via summaries

This shows how to prevent "context pollution" in complex multi-agent systems.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.context_engine import Agent, AgentConfig, LLMConfig
from src.context_engine.patterns.isolation import MultiAgentSystem

# Load environment variables
load_dotenv()


def main():
    """Run multi-agent isolation example."""
    print("=" * 60)
    print("Example 3: Multi-Agent Context Isolation")
    print("=" * 60)
    print()

    # Configure LLM
    llm_config = LLMConfig(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
    )

    # Create planning agent (orchestrator)
    planning_config = AgentConfig(
        name="PlanningAgent",
        role="task orchestrator",
        system_prompt=(
            "You are a planning agent that coordinates specialized subagents. "
            "Break down complex tasks and delegate to specialized agents. "
            "Synthesize their results into a coherent response."
        ),
        llm_config=llm_config,
    )

    print("Creating planning agent...")
    planning_agent = Agent(planning_config)
    print(f"✓ Planning agent created: {planning_agent}")
    print()

    # Create multi-agent system
    print("Creating multi-agent system...")
    system = MultiAgentSystem(planning_agent)
    print("✓ Multi-agent system created")
    print()

    # Create specialized subagents
    print("Creating specialized subagents...")
    print()

    # Subagent 1: Data Analyzer
    analyzer = system.create_subagent(
        name="DataAnalyzer",
        role="data analysis specialist",
        system_prompt=(
            "You are a data analysis specialist. Analyze data, find patterns, "
            "and provide statistical insights. Be precise and quantitative."
        )
    )
    print(f"✓ Created: {analyzer.agent.config.name} ({analyzer.specialization})")

    # Subagent 2: Researcher
    researcher = system.create_subagent(
        name="Researcher",
        role="research specialist",
        system_prompt=(
            "You are a research specialist. Provide detailed explanations, "
            "background information, and context. Cite key concepts."
        )
    )
    print(f"✓ Created: {researcher.agent.config.name} ({researcher.specialization})")

    # Subagent 3: Summarizer
    summarizer = system.create_subagent(
        name="Summarizer",
        role="summarization specialist",
        system_prompt=(
            "You are a summarization specialist. Take complex information "
            "and distill it into clear, concise summaries. Focus on key points."
        )
    )
    print(f"✓ Created: {summarizer.agent.config.name} ({summarizer.specialization})")

    print()
    print("-" * 60)
    print()

    # Task 1: Delegate to researcher
    print("Task 1: Delegating research task")
    print("-" * 60)
    task1_desc = "Explain the concept of 'context window' in large language models"
    print(f"Task: {task1_desc}")
    print()

    result1 = system.delegate_task(
        subagent_id=researcher.agent.id,
        task_description=task1_desc
    )

    print(f"Result (first 200 chars):")
    print(f"  {result1[:200]}...")
    print()

    # Show that planning agent only received summary, not full context
    print("Planning agent's perspective (what it knows):")
    # Get the last message in planning agent's memory (should be summary)
    if planning_agent.memory.stm:
        last_msg = planning_agent.memory.stm[-1]
        print(f"  {last_msg.content[:150]}...")
        print(f"  (This is a SUMMARY, not the full subagent conversation)")
    print()
    print("-" * 60)
    print()

    # Task 2: Delegate to analyzer
    print("Task 2: Delegating analysis task")
    print("-" * 60)
    task2_desc = "If a model has 128K context window and you have 50K tokens of conversation, what percentage is used? Is this optimal?"
    print(f"Task: {task2_desc}")
    print()

    result2 = system.delegate_task(
        subagent_id=analyzer.agent.id,
        task_description=task2_desc
    )

    print(f"Result:")
    print(f"  {result2[:300]}...")
    print()
    print("-" * 60)
    print()

    # Task 3: Delegate to summarizer
    print("Task 3: Delegating summarization task")
    print("-" * 60)
    task3_desc = f"Summarize these two pieces of information: 1) {result1[:200]} 2) {result2[:200]}"
    print(f"Task: Summarize the previous results")
    print()

    result3 = system.delegate_task(
        subagent_id=summarizer.agent.id,
        task_description=task3_desc
    )

    print(f"Result:")
    print(f"  {result3}")
    print()
    print("-" * 60)
    print()

    # Show context isolation
    print("Context Isolation Demonstration:")
    print("-" * 60)
    print()

    print(f"Planning Agent:")
    print(f"  STM size: {len(planning_agent.memory.stm)} messages")
    print(f"  (Only receives summaries from subagents)")
    print()

    print(f"Researcher Subagent:")
    print(f"  STM size: {len(researcher.agent.memory.stm)} messages")
    print(f"  (Has full context of its own tasks)")
    print()

    print(f"Analyzer Subagent:")
    print(f"  STM size: {len(analyzer.agent.memory.stm)} messages")
    print()

    print(f"Summarizer Subagent:")
    print(f"  STM size: {len(summarizer.agent.memory.stm)} messages")
    print()

    print("✓ Each subagent has ISOLATED context")
    print("✓ Planning agent's context is NOT polluted with subagent details")
    print()

    # Show system statistics
    print("-" * 60)
    print("System Statistics:")
    print("-" * 60)
    stats = system.get_system_stats()

    print(f"Total subagents: {stats['num_subagents']}")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Completed tasks: {stats['completed_tasks']}")
    print()

    print("Subagent summaries:")
    for summary in stats['subagent_summaries']:
        print(f"  - {summary['specialization']}")
        print(f"    Tasks completed: {summary['tasks_completed']}")
        print(f"    Tasks failed: {summary['tasks_failed']}")
    print()

    print("=" * 60)
    print("Key Takeaways:")
    print("- Each subagent has isolated context (prevents pollution)")
    print("- Planning agent only receives summaries, not full details")
    print("- Specialized agents can focus on their domain")
    print("- Scales better than single agent with massive context")
    print("=" * 60)


if __name__ == "__main__":
    main()
