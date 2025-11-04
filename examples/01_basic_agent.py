"""
Example 1: Basic Conversational Agent

Demonstrates:
- Creating a simple agent with OpenRouter/Gemini
- Basic conversation flow
- Context window management
- Token tracking

This is the foundation - a simple agent without advanced patterns.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.context_engine import Agent, AgentConfig, LLMConfig

# Load environment variables
load_dotenv()


def main():
    """Run basic agent example."""
    print("=" * 60)
    print("Example 1: Basic Conversational Agent")
    print("=" * 60)
    print()

    # Configure LLM
    llm_config = LLMConfig(
        model="google/gemini-2.5-flash",  # Using Gemini 2.5 Flash from .env
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.7,
    )

    # Configure agent
    agent_config = AgentConfig(
        name="BasicAssistant",
        role="helpful assistant",
        system_prompt="You are a helpful AI assistant. Be concise but informative.",
        llm_config=llm_config,
        enable_streaming=False,
    )

    # Create agent
    print("Creating agent...")
    agent = Agent(agent_config)
    print(f"✓ Agent created: {agent}")
    print()

    # Conversation examples
    conversations = [
        "Hello! What can you help me with?",
        "I'm learning about context engineering in AI. Can you explain what that means?",
        "What are the key challenges in managing context for AI agents?",
    ]

    for i, message in enumerate(conversations, 1):
        print(f"User: {message}")
        print()

        # Process message
        response = agent.process_message(message)

        print(f"Assistant: {response}")
        print()
        print("-" * 60)
        print()

    # Show statistics
    print("Agent Statistics:")
    print("-" * 60)
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    print()

    # Show context window usage
    context = agent._build_context()
    token_count = agent.llm.count_tokens(context)
    max_tokens = agent.context_manager.config.max_tokens
    usage_pct = (token_count / max_tokens) * 100

    print(f"\nContext Usage:")
    print(f"  Tokens: {token_count} / {max_tokens} ({usage_pct:.1f}%)")
    print(f"  Messages in context: {len(context)}")

    # Check if context is within recommended limits (<50%)
    if usage_pct < 50:
        print(f"  Status: ✓ Within recommended limit (<50%)")
    else:
        print(f"  Status: ⚠ Approaching limit (>50%)")


if __name__ == "__main__":
    main()
