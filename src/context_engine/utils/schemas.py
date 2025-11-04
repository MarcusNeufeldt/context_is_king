"""
Core data schemas for the context engine.

These Pydantic models define the structure of messages, memory entries,
and metadata throughout the system.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Standard message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None  # For tool/function messages
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def token_estimate(self) -> int:
        """Rough token estimate (4 chars per token average)."""
        return len(self.content) // 4

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        msg = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg


class MemoryEntry(BaseModel):
    """An entry in long-term memory."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    # Semantic categories
    tags: List[str] = Field(default_factory=list)
    importance: float = 1.0  # 0-1 scale

    def mark_accessed(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class AgentMetadata(BaseModel):
    """Metadata about an agent."""
    id: str
    name: str
    role: str
    created_at: datetime = Field(default_factory=datetime.now)
    parent_id: Optional[str] = None  # For subagents
    specialization: Optional[str] = None
    context_window_size: int = 128000  # Default for Gemini 2.5 Flash

    # Statistics
    total_messages: int = 0
    total_tokens: int = 0
    tasks_completed: int = 0


class ToolCall(BaseModel):
    """A tool/function call made by the agent."""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ContextSnapshot(BaseModel):
    """A snapshot of agent context at a point in time."""
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    messages: List[Message]
    token_count: int
    memory_entries: List[str] = Field(default_factory=list)  # IDs
    active_goals: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
