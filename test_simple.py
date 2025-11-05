#!/usr/bin/env python3
"""
Simple test to verify the multi-agent POC works end-to-end
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_agent_poc.core.agent import Agent
from multi_agent_poc.core.config import AgentConfig


def main():
    print("\n" + "="*60)
    print("  SIMPLE POC TEST")
    print("="*60 + "\n")

    try:
        # Test 1: Create and configure an agent
        print("TEST 1: Agent Configuration")
        print("-" * 60)

        config = AgentConfig(
            agent_name="TestResearcher",
            agent_role="Research Specialist",
            instructions_path=Path("multi_agent_poc/agents/researcher_AGENTS.md"),
            reasoning_strategy="react",
            model="minimax/minimax-m2",
            temperature=0.7,
            max_tokens=500
        )

        print(f"✓ AgentConfig created")
        print(f"  - Name: {config.agent_name}")
        print(f"  - Role: {config.agent_role}")
        print(f"  - Strategy: {config.reasoning_strategy}")
        print(f"  - Model: {config.model}")

        # Test 2: Initialize agent
        print("\nTEST 2: Agent Initialization")
        print("-" * 60)

        agent = Agent(config=config)
        print(f"✓ Agent initialized: {agent}")
        print(f"  - Instructions loaded: {len(agent.instructions)} chars")

        # Test 3: Run a simple task
        print("\nTEST 3: Agent Execution")
        print("-" * 60)

        task = "List 3 benefits of using AGENTS.md files for AI agents. Be very brief, maximum 3 sentences total."

        print(f"Task: {task}\n")
        print("Calling OpenRouter API...")

        result = agent.run(task)

        # Display results
        print("\nRESULTS:")
        print("-" * 60)

        if result['success']:
            print("✓ SUCCESS\n")
            print("Response:")
            print(result['response'])
            print("\nMetadata:")
            print(f"  - Strategy: {result['strategy']}")
            print(f"  - Model: {result['metadata'].get('model')}")
            print(f"  - Tokens: {result['metadata'].get('tokens_used')}")
        else:
            print("✗ FAILED\n")
            print(f"Error: {result['error']}")
            return 1

        # Test 4: Verify history tracking
        print("\nTEST 4: History Tracking")
        print("-" * 60)

        history = agent.get_history()
        print(f"✓ History tracked: {len(history)} interactions")

        print("\n" + "="*60)
        print("  ALL TESTS PASSED!")
        print("="*60 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
