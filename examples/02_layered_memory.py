"""
Example 2: Layered Memory System

Demonstrates:
- Short-Term Memory (STM) - recent messages in context
- Long-Term Memory (LTM) - persistent vector store
- Memory consolidation (ftransfer)
- Retrieval of relevant past context

This shows how agents can maintain knowledge across long conversations.
"""

import sys
import os
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.context_engine import Agent, AgentConfig, LLMConfig
from src.context_engine.core.memory import MemoryConfig

# Load environment variables
load_dotenv()


def main():
    """Run layered memory example."""
    print("=" * 60)
    print("Example 2: Layered Memory System")
    print("=" * 60)
    print()

    # Create temporary directory for LTM storage
    ltm_dir = tempfile.mkdtemp()
    ltm_path = os.path.join(ltm_dir, "agent_memory")

    print(f"LTM storage path: {ltm_path}")
    print()

    # Configure LLM
    llm_config = LLMConfig(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
    )

    # Configure memory with consolidation
    memory_config = MemoryConfig(
        stm_max_messages=10,
        ltm_storage_path=ltm_path,
        auto_consolidate=True,
        consolidate_threshold=5,  # Consolidate after 5 messages
        ltm_retrieval_k=3,
    )

    # Configure agent
    agent_config = AgentConfig(
        name="MemoryAgent",
        role="assistant with memory",
        system_prompt="You are a helpful assistant with memory. Remember past conversations.",
        llm_config=llm_config,
        memory_config=memory_config,
        enable_memory_consolidation=True,
    )

    # Create agent
    print("Creating agent with layered memory...")
    agent = Agent(agent_config)
    print(f"✓ Agent created")
    print()

    # Phase 1: Build up conversation (will trigger consolidation)
    print("Phase 1: Building conversation history")
    print("-" * 60)

    phase1_messages = [
        "Hi! My name is Alex and I'm working on a project about renewable energy.",
        "Specifically, I'm researching solar panel efficiency in different climates.",
        "I'm particularly interested in how temperature affects output.",
        "Can you help me understand the physics behind this?",
        "Also, I prefer data-driven explanations with examples.",
    ]

    for msg in phase1_messages:
        print(f"User: {msg}")
        response = agent.process_message(msg)
        print(f"Assistant: {response[:100]}...")  # Truncate for display
        print()

        # Show memory stats after each message
        print(f"  STM size: {len(agent.memory.stm)}, LTM size: {len(agent.memory.ltm)}")

        # Check if consolidation happened
        if agent.memory.total_consolidations > 0:
            print(f"  ✓ Memory consolidated! ({agent.memory.total_consolidations} times)")

        print()

    print("-" * 60)
    print()

    # Phase 2: New conversation that references past context
    print("Phase 2: New conversation with memory retrieval")
    print("-" * 60)
    print()

    # Clear short-term memory to simulate new session
    print("Clearing short-term memory (simulating new session)...")
    agent.clear_short_term_memory()
    print(f"STM cleared. STM size: {len(agent.memory.stm)}")
    print()

    # Save LTM to disk
    print("Saving long-term memory to disk...")
    agent.save_state()
    print("✓ LTM saved")
    print()

    # Ask question that requires past context
    new_message = "What was I researching? And what's my name?"

    print(f"User: {new_message}")
    print()

    # The agent should retrieve relevant context from LTM
    relevant_memories = agent.memory.retrieve_relevant(new_message)
    print(f"Retrieved {len(relevant_memories)} relevant memories from LTM:")
    for i, memory in enumerate(relevant_memories, 1):
        print(f"  {i}. {memory.content[:80]}...")
    print()

    response = agent.process_message(new_message)
    print(f"Assistant: {response}")
    print()

    print("-" * 60)
    print()

    # Show final statistics
    print("Final Memory Statistics:")
    print("-" * 60)
    print(f"Short-Term Memory (STM):")
    print(f"  Messages: {len(agent.memory.stm)}")
    print(f"  Max capacity: {agent.memory.config.stm_max_messages}")
    print()
    print(f"Long-Term Memory (LTM):")
    print(f"  Entries: {len(agent.memory.ltm)}")
    print(f"  Storage: {ltm_path}")
    print()
    print(f"Consolidations: {agent.memory.total_consolidations}")
    print()

    # Show LTM contents
    print("LTM Contents:")
    for entry_id, entry in list(agent.memory.ltm.entries.items())[:5]:
        print(f"  - {entry.content[:80]}...")
        print(f"    (type: {entry.metadata.get('type', 'unknown')}, "
              f"importance: {entry.importance:.2f})")
    print()

    print("=" * 60)
    print("Key Takeaways:")
    print("- STM holds recent messages (fast, volatile)")
    print("- LTM stores consolidated summaries (persistent)")
    print("- ftransfer() automatically consolidates when threshold reached")
    print("- Agent can retrieve relevant past context from LTM")
    print("- Memory persists across sessions (saved to disk)")
    print("=" * 60)


if __name__ == "__main__":
    main()
