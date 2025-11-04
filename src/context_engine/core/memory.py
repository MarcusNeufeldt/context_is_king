"""
Layered Memory Architecture for agents.

Implements the two-layer memory pattern from the paper:
- Short-Term Memory (STM): Recent messages in context window
- Long-Term Memory (LTM): Persistent vector store + structured storage

Includes ftransfer function for memory consolidation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

from ..utils.schemas import Message, MemoryEntry
from ..utils.vector_store import VectorStore, EmbeddingModel


class MemoryConfig(BaseModel):
    """Configuration for layered memory system."""
    stm_max_messages: int = 20  # Max messages in short-term memory
    ltm_storage_path: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"

    # Consolidation settings
    auto_consolidate: bool = True
    consolidate_threshold: int = 15  # Consolidate when STM reaches this size

    # Retrieval settings
    ltm_retrieval_k: int = 5  # Number of LTM entries to retrieve


class LayeredMemory:
    """
    Layered memory system for agents.

    Pattern: Think like an OS designer
    - STM = Fast RAM (volatile, limited)
    - LTM = Hard drive (persistent, large)
    - ftransfer = Memory consolidation function

    Features:
    - Automatic consolidation of STM → LTM
    - Smart retrieval from LTM based on context
    - Importance weighting and access tracking
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize layered memory system."""
        self.config = config or MemoryConfig()

        # Short-Term Memory (in context window)
        self.stm: List[Message] = []

        # Long-Term Memory (persistent vector store)
        self.ltm = VectorStore(
            embedding_model=EmbeddingModel(self.config.embedding_model),
            storage_path=self.config.ltm_storage_path
        )

        # Structured state (extracted knowledge)
        self.structured_state: Dict[str, Any] = {}

        # Statistics
        self.total_consolidations = 0

    def add_message(self, message: Message, auto_consolidate: bool = True) -> None:
        """
        Add a message to short-term memory.

        Args:
            message: Message to add
            auto_consolidate: If True, automatically consolidate when threshold reached
        """
        self.stm.append(message)

        # Auto-consolidate if threshold reached
        if auto_consolidate and self.config.auto_consolidate:
            if len(self.stm) >= self.config.consolidate_threshold:
                self.ftransfer()

    def ftransfer(self, force: bool = False) -> None:
        """
        Transfer and consolidate memories from STM to LTM.

        This is the key "memory consolidation" function from the paper.
        Processes recent experiences into structured knowledge.

        Args:
            force: If True, consolidate even if threshold not reached
        """
        if not force and len(self.stm) < self.config.consolidate_threshold:
            return

        # Take messages for consolidation (keep most recent few in STM)
        keep_in_stm = 5  # Keep last 5 messages in STM
        to_consolidate = self.stm[:-keep_in_stm] if len(self.stm) > keep_in_stm else []

        if not to_consolidate:
            return

        # Strategy 1: Create summary of consolidated messages
        summary = self._create_summary(to_consolidate)

        # Store summary in LTM
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=summary,
            metadata={
                "type": "consolidated_summary",
                "message_count": len(to_consolidate),
                "time_range": {
                    "start": to_consolidate[0].timestamp.isoformat(),
                    "end": to_consolidate[-1].timestamp.isoformat()
                }
            },
            importance=self._calculate_importance(to_consolidate)
        )
        self.ltm.add(entry)

        # Strategy 2: Extract key facts and decisions
        key_facts = self._extract_key_facts(to_consolidate)
        for fact in key_facts:
            fact_entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=fact,
                metadata={"type": "extracted_fact"},
                importance=1.5  # Facts are important
            )
            self.ltm.add(fact_entry)

        # Update STM (keep only recent messages)
        self.stm = self.stm[-keep_in_stm:]
        self.total_consolidations += 1

    def retrieve_relevant(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories from LTM based on query.

        Implements "Attention Before Attention" pattern:
        - Semantic relevance via vector search
        - Recency and frequency weighting
        - Importance filtering

        Args:
            query: Query to search for
            k: Number of results (defaults to config value)

        Returns:
            List of relevant memory entries
        """
        k = k or self.config.ltm_retrieval_k

        # Search LTM with reranking
        results = self.ltm.search(
            query=query,
            k=k,
            rerank_by_recency=True,
            rerank_by_frequency=True
        )

        # Return entries only (without scores)
        return [entry for entry, score in results]

    def get_all_context(
        self,
        current_query: Optional[str] = None
    ) -> List[Message]:
        """
        Get full context for agent: STM + relevant LTM.

        Args:
            current_query: If provided, retrieve relevant LTM entries

        Returns:
            Combined context (LTM summaries + STM messages)
        """
        context_messages = []

        # Retrieve relevant LTM if query provided
        if current_query:
            ltm_entries = self.retrieve_relevant(current_query)

            # Convert LTM entries to messages
            if ltm_entries:
                ltm_summary = "Relevant past context:\n\n"
                for entry in ltm_entries:
                    ltm_summary += f"- {entry.content}\n"

                from ..utils.schemas import MessageRole
                context_messages.append(Message(
                    role=MessageRole.SYSTEM,
                    content=ltm_summary,
                    metadata={"source": "ltm"}
                ))

        # Add STM messages
        context_messages.extend(self.stm)

        return context_messages

    def add_to_structured_state(self, key: str, value: Any) -> None:
        """
        Add to structured knowledge state.

        This is for extracted, structured information (e.g., user preferences,
        key decisions, entities) that should be readily accessible.
        """
        self.structured_state[key] = value

    def save(self) -> None:
        """Persist LTM and structured state to disk."""
        if self.config.ltm_storage_path:
            self.ltm.save()
            # TODO: Save structured state

    def load(self) -> None:
        """Load LTM and structured state from disk."""
        if self.config.ltm_storage_path:
            self.ltm.load()
            # TODO: Load structured state

    # Helper methods for consolidation

    def _create_summary(self, messages: List[Message]) -> str:
        """
        Create a natural language summary of messages.

        Note: In production, this should use an LLM to generate the summary.
        For now, this is a simple concatenation.
        """
        # Simple summary (in production, use LLM)
        summary_parts = []

        for msg in messages:
            if msg.role.value == "user":
                summary_parts.append(f"User: {msg.content[:100]}")
            elif msg.role.value == "assistant":
                summary_parts.append(f"Assistant: {msg.content[:100]}")

        return " | ".join(summary_parts)

    def _extract_key_facts(self, messages: List[Message]) -> List[str]:
        """
        Extract key facts from messages.

        Note: In production, use an LLM with structured extraction.
        For now, this is a placeholder.
        """
        # Placeholder: in production, use LLM to extract key facts
        facts = []

        # Look for important keywords
        keywords = ["decided", "important", "remember", "key point", "note that"]

        for msg in messages:
            content_lower = msg.content.lower()
            for keyword in keywords:
                if keyword in content_lower:
                    # Extract sentence containing keyword
                    sentences = msg.content.split(".")
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            facts.append(sentence.strip())

        return facts[:5]  # Limit to top 5 facts

    def _calculate_importance(self, messages: List[Message]) -> float:
        """
        Calculate importance score for a set of messages.

        Factors:
        - Length (longer = potentially more important)
        - Keywords indicating importance
        - User vs assistant messages
        """
        importance = 1.0

        # Length factor
        total_length = sum(len(msg.content) for msg in messages)
        if total_length > 1000:
            importance += 0.3

        # Keyword factor
        important_keywords = ["important", "critical", "key", "remember", "must"]
        for msg in messages:
            content_lower = msg.content.lower()
            for keyword in important_keywords:
                if keyword in content_lower:
                    importance += 0.2
                    break

        return min(importance, 2.0)  # Cap at 2.0

    def clear_stm(self) -> None:
        """Clear short-term memory."""
        self.stm.clear()

    def __len__(self) -> int:
        """Return total memory entries (STM + LTM)."""
        return len(self.stm) + len(self.ltm)
