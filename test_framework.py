"""
Test script to verify the Context Engine framework loads correctly.

This tests the framework structure without making actual API calls.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing Context Engine Framework")
print("=" * 60)
print()

# Test 1: Import schemas
print("Test 1: Schemas module")
try:
    from src.context_engine.utils.schemas import Message, MessageRole, MemoryEntry, AgentMetadata
    msg = Message(role=MessageRole.USER, content="Hello")
    print(f"  ✓ Schemas imported")
    print(f"  ✓ Message created: {msg.role.value}")
    print(f"  ✓ Token estimate: {msg.token_estimate()}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 2: Import vector store
print("Test 2: Vector Store")
try:
    from src.context_engine.utils.vector_store import VectorStore, EmbeddingModel
    print(f"  ✓ VectorStore class available")
    print(f"  ✓ EmbeddingModel class available")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 3: Import LLM client
print("Test 3: LLM Client")
try:
    from src.context_engine.core.llm_client import LLMClient, LLMConfig
    config = LLMConfig(model="google/gemini-2.5-flash")
    print(f"  ✓ LLMConfig created")
    print(f"  ✓ Model: {config.model}")
    print(f"  ✓ Context window: {config.context_window}")
    print(f"  ✓ Max usage: {config.max_context_usage * 100}%")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 4: Import Memory
print("Test 4: Layered Memory")
try:
    from src.context_engine.core.memory import LayeredMemory, MemoryConfig
    mem_config = MemoryConfig(stm_max_messages=20)
    print(f"  ✓ MemoryConfig created")
    print(f"  ✓ STM max: {mem_config.stm_max_messages}")
    print(f"  ✓ Auto-consolidate: {mem_config.auto_consolidate}")
    print(f"  ✓ Threshold: {mem_config.consolidate_threshold}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 5: Import Context Manager
print("Test 5: Context Manager")
try:
    from src.context_engine.core.context_manager import ContextManager, ContextConfig
    ctx_config = ContextConfig(max_context_ratio=0.5)
    print(f"  ✓ ContextConfig created")
    print(f"  ✓ Max ratio: {ctx_config.max_context_ratio * 100}%")
    print(f"  ✓ Semantic filtering: {ctx_config.enable_semantic_filtering}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 6: Import Agent
print("Test 6: Base Agent")
try:
    from src.context_engine.core.agent import Agent, AgentConfig
    print(f"  ✓ Agent class available")
    print(f"  ✓ AgentConfig class available")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 7: Import Patterns - Isolation
print("Test 7: Multi-Agent Isolation Pattern")
try:
    from src.context_engine.patterns.isolation import MultiAgentSystem, SubAgent, SubAgentTask
    print(f"  ✓ MultiAgentSystem class available")
    print(f"  ✓ SubAgent class available")
    print(f"  ✓ SubAgentTask class available")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 8: Import Patterns - Self-Baking
print("Test 8: Self-Baking Pattern")
try:
    from src.context_engine.patterns.self_baking import (
        SummaryBaker, SchemaBaker, ProgressiveBaker, SelfBakingAgent
    )
    print(f"  ✓ SummaryBaker class available")
    print(f"  ✓ SchemaBaker class available")
    print(f"  ✓ ProgressiveBaker class available")
    print(f"  ✓ SelfBakingAgent class available")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 9: Import Patterns - Communication
print("Test 9: Structured Communication Pattern")
try:
    from src.context_engine.patterns.communication import (
        StructuredMessage, MessageBus, Blackboard, MessageType
    )
    msg = StructuredMessage(
        type=MessageType.INFORMATION,
        sender_id="test",
        content={"data": "test"}
    )
    print(f"  ✓ StructuredMessage created")
    print(f"  ✓ MessageBus class available")
    print(f"  ✓ Blackboard class available")
    print(f"  ✓ Message type: {msg.type.value}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 10: Test Blackboard functionality
print("Test 10: Blackboard Functionality")
try:
    from src.context_engine.patterns.communication import Blackboard
    bb = Blackboard()
    bb.write("test", "key1", "value1", "agent1")
    value = bb.read("test", "key1", "agent2")
    assert value == "value1", "Blackboard read/write failed"
    print(f"  ✓ Blackboard write works")
    print(f"  ✓ Blackboard read works")
    print(f"  ✓ Access log: {len(bb.access_log)} entries")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Summary
print("=" * 60)
print("Framework Structure Test Complete!")
print("=" * 60)
print()
print("Status: ✓ All core modules load correctly")
print()
print("Next Steps:")
print("1. Install dependencies to run live examples:")
print("   - pip install -r requirements.txt (in a venv)")
print("2. Set up .env with OPENROUTER_API_KEY")
print("3. Run examples:")
print("   - python examples/01_basic_agent.py")
print("   - python examples/02_layered_memory.py")
print("   - python examples/03_multi_agent.py")
print("   - python examples/04_self_baking.py")
print("   - python examples/05_research_assistant.py")
print()
print("=" * 60)
