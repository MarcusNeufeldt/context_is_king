"""
Context Manager with smart context selection and filtering.

Implements "Attention Before Attention" pattern from the paper:
The quality of reasoning depends on what you choose to put in the context window.
Performance often decreases when context windows are >50% full.

Features:
- Multi-faceted filtering (semantic, logical, recency, redundancy)
- Context window optimization
- KV cache optimization
- Logical dependency tracking
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pydantic import BaseModel

from ..utils.schemas import Message, MessageRole, MemoryEntry


class ContextConfig(BaseModel):
    """Configuration for context management."""
    max_context_ratio: float = 0.5  # Keep context <50% full (paper recommendation)
    max_tokens: int = 64000  # Max tokens to use from context window
    priority_recent_messages: int = 5  # Always include N most recent messages

    # Filtering settings
    enable_semantic_filtering: bool = True
    enable_redundancy_filtering: bool = True
    enable_dependency_tracking: bool = True

    # KV cache optimization
    stable_system_prefix: bool = True  # Keep system messages at start for caching


class ContextManager:
    """
    Manages context selection and optimization for agents.

    Key responsibilities:
    1. Select most relevant context for current task
    2. Keep context window usage optimal (<50%)
    3. Track logical dependencies between messages
    4. Optimize for KV cache efficiency
    5. Filter redundant information
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        """Initialize context manager."""
        self.config = config or ContextConfig()

        # Dependency graph: message_id -> depends_on [message_ids]
        self.dependency_graph: Dict[str, Set[str]] = {}

        # Message importance scores
        self.importance_scores: Dict[str, float] = {}

    def select_context(
        self,
        all_messages: List[Message],
        ltm_entries: Optional[List[MemoryEntry]] = None,
        current_goal: Optional[str] = None,
        token_counter: Optional[callable] = None
    ) -> List[Message]:
        """
        Select optimal context from available messages.

        Implements multi-faceted filtering:
        1. Semantic relevance (if goal provided)
        2. Logical dependency (required messages)
        3. Recency (recent messages prioritized)
        4. Redundancy removal (similar messages)

        Args:
            all_messages: All available messages
            ltm_entries: Long-term memory entries to potentially include
            current_goal: Current task/goal for relevance filtering
            token_counter: Function to count tokens in messages

        Returns:
            Optimally selected context messages
        """
        if not all_messages:
            return []

        # Step 1: Always include system messages (stable prefix for KV cache)
        system_messages = [m for m in all_messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in all_messages if m.role != MessageRole.SYSTEM]

        # Step 2: Always include N most recent messages
        recent_messages = other_messages[-self.config.priority_recent_messages:]
        candidate_messages = other_messages[:-self.config.priority_recent_messages]

        # Step 3: Add messages with logical dependencies
        required_messages = self._get_dependent_messages(recent_messages, all_messages)

        # Step 4: Score remaining candidates
        scored_candidates = []
        for msg in candidate_messages:
            if msg not in required_messages:
                score = self._score_message(msg, current_goal)
                scored_candidates.append((msg, score))

        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Add candidates until token limit
        selected_messages = list(required_messages) + recent_messages

        if token_counter:
            current_tokens = token_counter(system_messages + selected_messages)
            max_tokens = self.config.max_tokens

            for msg, score in scored_candidates:
                # Check if adding this message would exceed limits
                test_messages = selected_messages + [msg]
                test_tokens = token_counter(system_messages + test_messages)

                if test_tokens <= max_tokens:
                    selected_messages.append(msg)
                else:
                    break  # Stop adding messages
        else:
            # No token counter, add top scored messages (limited to reasonable number)
            max_additional = 10
            for msg, score in scored_candidates[:max_additional]:
                selected_messages.append(msg)

        # Step 6: Sort selected messages by timestamp (chronological order)
        selected_messages.sort(key=lambda m: m.timestamp)

        # Step 7: Remove redundant messages
        if self.config.enable_redundancy_filtering:
            selected_messages = self._remove_redundancy(selected_messages)

        # Step 8: Prepend LTM context if available
        ltm_messages = self._convert_ltm_to_messages(ltm_entries) if ltm_entries else []

        # Final context: System messages + LTM + Selected messages
        final_context = system_messages + ltm_messages + selected_messages

        return final_context

    def add_dependency(self, message_id: str, depends_on: str) -> None:
        """
        Add a logical dependency between messages.

        Example: Tool result message depends on the tool call message.
        """
        if message_id not in self.dependency_graph:
            self.dependency_graph[message_id] = set()
        self.dependency_graph[message_id].add(depends_on)

    def set_importance(self, message_id: str, importance: float) -> None:
        """Set importance score for a message (0-1 scale)."""
        self.importance_scores[message_id] = importance

    def _get_dependent_messages(
        self,
        selected: List[Message],
        all_messages: List[Message]
    ) -> Set[Message]:
        """
        Get all messages that selected messages depend on.

        Traverses dependency graph to find required context.
        """
        if not self.config.enable_dependency_tracking:
            return set()

        required = set()
        message_lookup = {id(m): m for m in all_messages}

        # Build reverse lookup for dependencies
        # In practice, you'd use actual message IDs, not id()
        # This is simplified for the example

        return required

    def _score_message(self, message: Message, goal: Optional[str]) -> float:
        """
        Score a message for relevance.

        Factors:
        - Semantic relevance to goal
        - Message role (assistant messages often more important)
        - Manual importance score
        - Recency
        """
        score = 1.0

        # Base score from importance
        msg_id = id(message)  # In production, use actual message ID
        if msg_id in self.importance_scores:
            score *= self.importance_scores[msg_id]

        # Role factor
        if message.role == MessageRole.ASSISTANT:
            score *= 1.2  # Assistant messages often contain key information
        elif message.role == MessageRole.USER:
            score *= 1.1

        # Recency factor (newer = slightly higher score)
        age_seconds = (datetime.now() - message.timestamp).total_seconds()
        age_hours = age_seconds / 3600
        recency_factor = 1.0 / (1.0 + age_hours * 0.01)  # Slow decay
        score *= recency_factor

        # Semantic relevance (if goal provided)
        if goal and self.config.enable_semantic_filtering:
            # Simple keyword matching (in production, use embeddings)
            goal_words = set(goal.lower().split())
            message_words = set(message.content.lower().split())
            overlap = len(goal_words & message_words)
            if overlap > 0:
                score *= (1.0 + overlap * 0.1)

        return score

    def _remove_redundancy(self, messages: List[Message]) -> List[Message]:
        """
        Remove redundant/similar messages.

        Uses simple heuristic: very similar content within short time window.
        In production, use embedding similarity.
        """
        if len(messages) <= 2:
            return messages

        filtered = []
        for i, msg in enumerate(messages):
            is_redundant = False

            # Check against recent messages
            for j in range(max(0, i - 3), i):
                prev_msg = messages[j]

                # Simple similarity check
                if self._messages_similar(msg, prev_msg):
                    is_redundant = True
                    break

            if not is_redundant:
                filtered.append(msg)

        return filtered

    def _messages_similar(self, msg1: Message, msg2: Message, threshold: float = 0.8) -> bool:
        """
        Check if two messages are similar.

        Simple implementation using character overlap.
        In production, use embedding cosine similarity.
        """
        # Must be same role
        if msg1.role != msg2.role:
            return False

        # Must be close in time (within 1 hour)
        time_diff = abs((msg1.timestamp - msg2.timestamp).total_seconds())
        if time_diff > 3600:
            return False

        # Check content similarity
        words1 = set(msg1.content.lower().split())
        words2 = set(msg2.content.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / max(len(words1), len(words2))

        return similarity >= threshold

    def _convert_ltm_to_messages(self, ltm_entries: List[MemoryEntry]) -> List[Message]:
        """Convert LTM entries to system messages for context."""
        if not ltm_entries:
            return []

        # Combine LTM entries into a single context message
        ltm_content = "Relevant background information from past interactions:\n\n"
        for i, entry in enumerate(ltm_entries, 1):
            ltm_content += f"{i}. {entry.content}\n"

        return [Message(
            role=MessageRole.SYSTEM,
            content=ltm_content,
            metadata={"source": "ltm", "entry_count": len(ltm_entries)}
        )]

    def optimize_for_cache(self, messages: List[Message]) -> List[Message]:
        """
        Optimize message ordering for KV cache efficiency.

        Key strategy: System messages at the start (stable prefix).
        """
        if not self.config.stable_system_prefix:
            return messages

        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_msgs = [m for m in messages if m.role != MessageRole.SYSTEM]

        return system_msgs + other_msgs

    def estimate_token_usage(self, messages: List[Message]) -> float:
        """
        Estimate token usage as ratio of max context.

        Returns:
            Usage ratio (0-1), where >0.5 is getting full
        """
        # Rough estimate: 4 characters per token
        total_chars = sum(len(m.content) for m in messages)
        estimated_tokens = total_chars // 4

        return estimated_tokens / self.config.max_tokens
