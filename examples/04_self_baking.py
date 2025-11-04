"""
Example 4: Self-Baking (Context Abstraction)

Demonstrates:
- Natural language summaries of experience
- Schema extraction (structured knowledge)
- Progressive compression
- Converting raw history into learned knowledge

This shows how agents can "learn" rather than just "recall".
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict

from src.context_engine import Agent, AgentConfig, LLMConfig
from src.context_engine.patterns.self_baking import (
    SelfBakingAgent,
    SummaryBaker,
    SchemaBaker,
    ProgressiveBaker
)

# Load environment variables
load_dotenv()


# Define a schema for structured extraction
class ProjectInfo(BaseModel):
    """Schema for extracting project information."""
    project_name: str = ""
    goals: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    decisions: Dict[str, str] = Field(default_factory=dict)


def demo_summary_baking():
    """Demonstrate natural language summary baking."""
    print("=" * 60)
    print("Strategy 1: Natural Language Summaries")
    print("=" * 60)
    print()

    # Configure agent with summary baking
    llm_config = LLMConfig(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
    )

    agent_config = AgentConfig(
        name="SummaryAgent",
        role="assistant",
        system_prompt="You are a helpful assistant.",
        llm_config=llm_config,
    )

    baker = SummaryBaker(max_summary_length=300)
    agent = SelfBakingAgent(agent_config, baking_strategy=baker)

    print("Having a conversation...")
    print()

    # Have a conversation
    conversations = [
        "I'm working on a web application for task management.",
        "We decided to use React for the frontend and Node.js for the backend.",
        "The main challenge is handling real-time updates across multiple users.",
        "We're considering WebSockets or Server-Sent Events.",
    ]

    for msg in conversations:
        print(f"User: {msg}")
        response = agent.process_message(msg)
        print(f"Assistant: {response[:100]}...")
        print()

    # Now bake the experience into a summary
    print("-" * 60)
    print("Baking experience into summary...")
    print()

    summary = agent.bake_experience()

    print("Generated Summary:")
    print(f"  {summary}")
    print()

    print("✓ Raw conversation converted to concise summary")
    print("✓ Summary stored in LTM for future reference")
    print()


def demo_schema_baking():
    """Demonstrate schema-based extraction."""
    print("=" * 60)
    print("Strategy 2: Schema Extraction")
    print("=" * 60)
    print()

    # Configure agent with schema baking
    llm_config = LLMConfig(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.1,  # Lower temperature for extraction
    )

    agent_config = AgentConfig(
        name="SchemaAgent",
        role="assistant",
        system_prompt="You are a helpful assistant.",
        llm_config=llm_config,
    )

    baker = SchemaBaker(schema_class=ProjectInfo)
    agent = SelfBakingAgent(agent_config, baking_strategy=baker)

    print("Having a conversation...")
    print()

    # Have a conversation with structured information
    conversations = [
        "I'm building an app called 'TaskFlow' for project management.",
        "The main goals are: 1) Real-time collaboration, 2) Offline support, 3) Cross-platform sync.",
        "We're using React, Node.js, PostgreSQL, and Redis.",
        "The biggest challenge is conflict resolution for offline edits.",
        "We decided to use operational transformation for conflict resolution.",
    ]

    for msg in conversations:
        print(f"User: {msg}")
        response = agent.process_message(msg)
        print(f"Assistant: {response[:80]}...")
        print()

    # Now bake into structured schema
    print("-" * 60)
    print("Extracting structured information...")
    print()

    structured_info = agent.bake_experience()

    print("Extracted ProjectInfo:")
    print(f"  Project: {structured_info.project_name}")
    print(f"  Goals: {structured_info.goals}")
    print(f"  Technologies: {structured_info.technologies}")
    print(f"  Challenges: {structured_info.challenges}")
    print(f"  Decisions: {structured_info.decisions}")
    print()

    print("✓ Unstructured conversation converted to structured data")
    print("✓ Can be queried programmatically")
    print()


def demo_progressive_baking():
    """Demonstrate progressive compression."""
    print("=" * 60)
    print("Strategy 3: Progressive Compression")
    print("=" * 60)
    print()

    # Configure agent
    llm_config = LLMConfig(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.5,
    )

    agent_config = AgentConfig(
        name="ProgressiveAgent",
        role="assistant",
        system_prompt="You are a helpful assistant.",
        llm_config=llm_config,
    )

    baker = ProgressiveBaker(compression_ratio=0.5, levels=2)
    agent = SelfBakingAgent(agent_config, baking_strategy=baker)

    print("Having multiple conversations...")
    print()

    # Have several exchanges
    conversations = [
        "Let's discuss API design principles.",
        "What's the difference between REST and GraphQL?",
        "How do you handle authentication in APIs?",
        "What about rate limiting?",
        "Tell me about API versioning strategies.",
        "How do you document APIs effectively?",
    ]

    for msg in conversations:
        print(f"User: {msg}")
        response = agent.process_message(msg)
        print(f"Assistant: {response[:60]}...")
        print()

    # Apply progressive compression
    print("-" * 60)
    print("Applying progressive compression...")
    print()

    compressed = agent.bake_experience()

    print("Hierarchical Compressed Memory:")
    for level, content in compressed.items():
        print(f"\n{level.upper()}:")
        if isinstance(content, list):
            for i, item in enumerate(content, 1):
                print(f"  {i}. {item[:100]}...")
        print(f"  (Total items: {len(content)})")

    print()
    print("✓ Created hierarchical memory structure")
    print("✓ Abstract summaries at each level")
    print("✓ Can retrieve at appropriate granularity")
    print()


def main():
    """Run all self-baking demos."""
    print("\n" + "=" * 60)
    print("Example 4: Self-Baking (Context Abstraction)")
    print("=" * 60)
    print()
    print("Demonstrating three self-baking strategies:")
    print("1. Natural Language Summaries")
    print("2. Schema Extraction")
    print("3. Progressive Compression")
    print()

    # Run demos
    demo_summary_baking()
    print("\n" + "=" * 60 + "\n")

    demo_schema_baking()
    print("\n" + "=" * 60 + "\n")

    demo_progressive_baking()

    # Final summary
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("-" * 60)
    print("• Summary Baking: Quick, natural language consolidation")
    print("• Schema Baking: Structured, queryable knowledge")
    print("• Progressive Baking: Hierarchical compression for scale")
    print()
    print("All strategies convert RAW EXPERIENCE → LEARNED KNOWLEDGE")
    print("This is what separates 'remembering' from 'learning'")
    print("=" * 60)


if __name__ == "__main__":
    main()
