"""
Self-Baking Pattern: Context Abstraction

Implements context abstraction ("self-baking") from the paper.
Agents that only store raw history don't learn; they just recall.
Agents should actively process their experience into knowledge.

Three approaches:
1. Natural Language Summaries
2. Extract to Fixed Schema
3. Progressive Compression

This module provides tools for all three approaches.
"""

from typing import List, Dict, Any, Optional, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel
import json

from ..core.agent import Agent
from ..core.llm_client import LLMClient
from ..utils.schemas import Message, MemoryEntry


class SelfBaker(ABC):
    """
    Abstract base class for self-baking strategies.

    Self-baking = converting raw experience into structured knowledge.
    """

    @abstractmethod
    def bake(self, messages: List[Message], llm: LLMClient) -> Any:
        """
        Process raw messages into structured knowledge.

        Args:
            messages: Raw conversation messages
            llm: LLM client for processing

        Returns:
            Structured knowledge (format depends on strategy)
        """
        pass


class SummaryBaker(SelfBaker):
    """
    Strategy 1: Natural Language Summaries

    Periodically use a model to summarize the last N tokens of
    conversation/activity into a concise paragraph.

    Example:
        ```python
        baker = SummaryBaker()
        summary = baker.bake(recent_messages, llm)
        # "The user requested a feature to export data. I explained
        # # three approaches and implemented the CSV export option."
        ```
    """

    def __init__(self, max_summary_length: int = 500):
        """Initialize summary baker."""
        self.max_summary_length = max_summary_length

    def bake(self, messages: List[Message], llm: LLMClient) -> str:
        """
        Create a natural language summary of messages.

        Args:
            messages: Messages to summarize
            llm: LLM client for generating summary

        Returns:
            Concise summary string
        """
        if not messages:
            return ""

        # Build conversation text
        conversation = self._format_messages(messages)

        # Create summarization prompt
        system_msg = Message(
            role="system",
            content="You are a concise summarizer. Create a brief summary of the following conversation, "
                    "focusing on key decisions, important information, and outcomes. "
                    f"Keep it under {self.max_summary_length} characters."
        )

        user_msg = Message(
            role="user",
            content=f"Summarize this conversation:\n\n{conversation}"
        )

        # Generate summary
        summary, _ = llm.complete([system_msg, user_msg], temperature=0.3)

        return summary.strip()

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages as readable conversation."""
        lines = []
        for msg in messages:
            role = msg.role.value.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)


class SchemaBaker(SelfBaker):
    """
    Strategy 2: Extract to Fixed Schema

    Define a JSON schema for key entities (e.g., files, goals, user preferences).
    After an interaction, extract relevant information from the raw context
    and update this structured state object.

    Example:
        ```python
        class ProjectState(BaseModel):
            files: List[str]
            goals: List[str]
            decisions: Dict[str, str]

        baker = SchemaBaker(ProjectState)
        state = baker.bake(messages, llm)
        # ProjectState(files=["main.py", "utils.py"], goals=["Add logging"], ...)
        ```
    """

    def __init__(self, schema_class: Type[BaseModel]):
        """
        Initialize schema baker.

        Args:
            schema_class: Pydantic model class defining the schema
        """
        self.schema_class = schema_class
        self.schema_json = schema_class.model_json_schema()

    def bake(self, messages: List[Message], llm: LLMClient) -> BaseModel:
        """
        Extract structured data according to schema.

        Args:
            messages: Messages to extract from
            llm: LLM client for extraction

        Returns:
            Instance of schema_class with extracted data
        """
        if not messages:
            return self.schema_class()

        # Build conversation text
        conversation = self._format_messages(messages)

        # Create extraction prompt
        system_msg = Message(
            role="system",
            content="You are a precise information extractor. Extract structured information "
                    "from the conversation according to the provided schema. Return ONLY valid JSON."
        )

        user_msg = Message(
            role="user",
            content=f"Extract information according to this schema:\n{json.dumps(self.schema_json, indent=2)}\n\n"
                    f"From this conversation:\n{conversation}\n\n"
                    f"Return only the JSON object, nothing else."
        )

        # Generate extraction
        response, _ = llm.complete([system_msg, user_msg], temperature=0.1)

        # Parse JSON
        try:
            # Extract JSON from response (handle code blocks)
            json_str = response.strip()
            if json_str.startswith("```"):
                # Remove code blocks
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1])

            data = json.loads(json_str)
            return self.schema_class(**data)
        except Exception as e:
            print(f"Failed to parse schema: {e}")
            return self.schema_class()

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages as readable conversation."""
        lines = []
        for msg in messages:
            role = msg.role.value.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)


class ProgressiveBaker(SelfBaker):
    """
    Strategy 3: Progressive Compression

    Convert context into embeddings and periodically summarize older
    embeddings into more abstract representations.

    This is the most advanced approach, creating a hierarchical
    memory structure.
    """

    def __init__(
        self,
        compression_ratio: float = 0.5,
        levels: int = 3
    ):
        """
        Initialize progressive baker.

        Args:
            compression_ratio: How much to compress at each level (0-1)
            levels: Number of compression levels
        """
        self.compression_ratio = compression_ratio
        self.levels = levels

    def bake(self, messages: List[Message], llm: LLMClient) -> Dict[str, Any]:
        """
        Progressively compress messages into hierarchical representation.

        Args:
            messages: Messages to compress
            llm: LLM client for compression

        Returns:
            Hierarchical compressed representation
        """
        if not messages:
            return {"level_0": []}

        # Level 0: Original messages (most detailed)
        compressed = {"level_0": [msg.content for msg in messages]}

        # Progressively compress
        current_content = compressed["level_0"]

        for level in range(1, self.levels + 1):
            # How many items to compress into one
            chunk_size = int(1.0 / self.compression_ratio)

            compressed_level = []

            # Process in chunks
            for i in range(0, len(current_content), chunk_size):
                chunk = current_content[i:i + chunk_size]

                # Compress chunk
                summary = self._compress_chunk(chunk, llm, level)
                compressed_level.append(summary)

            compressed[f"level_{level}"] = compressed_level
            current_content = compressed_level

            # Stop if we've compressed to a single item
            if len(current_content) <= 1:
                break

        return compressed

    def _compress_chunk(self, chunk: List[str], llm: LLMClient, level: int) -> str:
        """Compress a chunk of text into a summary."""
        if len(chunk) == 1:
            return chunk[0]

        combined = " ".join(chunk)

        system_msg = Message(
            role="system",
            content=f"Compress the following text into a concise summary (compression level {level}). "
                    "Preserve the most important information."
        )

        user_msg = Message(
            role="user",
            content=f"Compress this:\n\n{combined}"
        )

        summary, _ = llm.complete([system_msg, user_msg], temperature=0.3)
        return summary.strip()


class SelfBakingAgent(Agent):
    """
    Extended Agent with self-baking capabilities.

    Automatically applies self-baking strategies to convert
    raw experience into structured knowledge.
    """

    def __init__(self, config, baking_strategy: Optional[SelfBaker] = None):
        """
        Initialize self-baking agent.

        Args:
            config: Agent configuration
            baking_strategy: Self-baking strategy to use (default: SummaryBaker)
        """
        super().__init__(config)
        self.baking_strategy = baking_strategy or SummaryBaker()
        self.baked_knowledge: List[Any] = []

    def bake_experience(self, messages: Optional[List[Message]] = None) -> Any:
        """
        Apply self-baking to recent experience.

        Args:
            messages: Messages to bake (defaults to current STM)

        Returns:
            Baked knowledge
        """
        messages = messages or self.memory.stm

        if not messages:
            return None

        # Apply baking strategy
        knowledge = self.baking_strategy.bake(messages, self.llm)

        # Store baked knowledge
        self.baked_knowledge.append(knowledge)

        # Add to LTM as a memory entry
        import uuid
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=str(knowledge),
            metadata={
                "type": "baked_knowledge",
                "strategy": self.baking_strategy.__class__.__name__,
                "source_messages": len(messages)
            },
            importance=1.5  # Baked knowledge is important
        )
        self.memory.ltm.add(entry)

        return knowledge

    def auto_bake(self, threshold: int = 10) -> None:
        """
        Automatically bake experience when STM reaches threshold.

        Args:
            threshold: Number of messages before auto-baking
        """
        if len(self.memory.stm) >= threshold:
            self.bake_experience()
            # Keep only recent messages in STM
            self.memory.stm = self.memory.stm[-5:]

    def get_baked_knowledge(self) -> List[Any]:
        """Get all baked knowledge."""
        return self.baked_knowledge
