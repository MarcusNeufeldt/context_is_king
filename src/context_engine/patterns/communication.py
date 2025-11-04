"""
Structured Communication Pattern

Implements structured multi-agent communication patterns from the paper.
Having agents pass blobs of text to each other is brittle.

Two main approaches:
1. Structured Messages: Predefined schema (JSON/Pydantic)
2. Shared Memory / Blackboard: Agents communicate via centralized memory

This enables reliable, scalable agent collaboration.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    """Types of structured messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION = "information"
    QUERY = "query"
    UPDATE = "update"
    ERROR = "error"


class StructuredMessage(BaseModel):
    """
    A structured message between agents.

    Uses Pydantic for validation and type safety.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender_id: str
    receiver_id: Optional[str] = None  # None = broadcast
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    in_reply_to: Optional[str] = None  # ID of message this replies to

    def is_reply_to(self, message_id: str) -> bool:
        """Check if this is a reply to a specific message."""
        return self.in_reply_to == message_id


class MessageBus:
    """
    Message bus for structured agent communication.

    Agents publish and subscribe to messages via the bus.
    Supports:
    - Direct messages (agent-to-agent)
    - Broadcast messages (one-to-many)
    - Message history and replay
    - Subscriptions by message type
    """

    def __init__(self):
        """Initialize message bus."""
        self.messages: List[StructuredMessage] = []
        self.subscriptions: Dict[str, Set[MessageType]] = {}  # agent_id -> message types

    def publish(self, message: StructuredMessage) -> None:
        """
        Publish a message to the bus.

        Args:
            message: Structured message to publish
        """
        self.messages.append(message)

    def subscribe(self, agent_id: str, message_types: List[MessageType]) -> None:
        """
        Subscribe an agent to specific message types.

        Args:
            agent_id: Agent ID
            message_types: List of message types to subscribe to
        """
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = set()
        self.subscriptions[agent_id].update(message_types)

    def get_messages_for(
        self,
        agent_id: str,
        since: Optional[datetime] = None,
        message_type: Optional[MessageType] = None
    ) -> List[StructuredMessage]:
        """
        Get messages for a specific agent.

        Args:
            agent_id: Agent ID
            since: Only messages after this time
            message_type: Filter by message type

        Returns:
            List of messages for the agent
        """
        messages = []

        for msg in self.messages:
            # Check if message is for this agent
            is_for_agent = (
                msg.receiver_id == agent_id or
                msg.receiver_id is None  # Broadcast
            )

            # Check if agent is subscribed to this type
            if agent_id in self.subscriptions:
                is_subscribed = msg.type in self.subscriptions[agent_id]
            else:
                is_subscribed = False

            if not (is_for_agent or is_subscribed):
                continue

            # Apply filters
            if since and msg.timestamp < since:
                continue

            if message_type and msg.type != message_type:
                continue

            messages.append(msg)

        return messages

    def get_conversation(self, initial_message_id: str) -> List[StructuredMessage]:
        """
        Get a conversation thread starting from a message.

        Args:
            initial_message_id: Starting message ID

        Returns:
            List of messages in the conversation thread
        """
        thread = []
        message_map = {msg.id: msg for msg in self.messages}

        # Find initial message
        if initial_message_id not in message_map:
            return []

        # Build thread
        current_id = initial_message_id
        visited = set()

        while current_id and current_id not in visited:
            if current_id in message_map:
                thread.append(message_map[current_id])
                visited.add(current_id)

                # Find replies
                replies = [msg for msg in self.messages if msg.in_reply_to == current_id]
                if replies:
                    current_id = replies[0].id  # Follow first reply
                else:
                    break
            else:
                break

        return thread

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()


class Blackboard:
    """
    Blackboard pattern for shared memory between agents.

    Agents communicate indirectly by reading/writing to a
    centralized, structured memory space.

    Better for asynchronous collaboration where agents don't
    need to directly message each other.

    Example:
        ```python
        blackboard = Blackboard()

        # Agent 1 writes
        blackboard.write("research", "completed", {
            "findings": ["fact1", "fact2"],
            "confidence": 0.85
        })

        # Agent 2 reads
        research = blackboard.read("research", "completed")
        ```
    """

    def __init__(self):
        """Initialize blackboard."""
        # Hierarchical storage: namespace -> key -> value
        self.storage: Dict[str, Dict[str, Any]] = {}

        # Access log for debugging and analysis
        self.access_log: List[Dict[str, Any]] = []

    def write(
        self,
        namespace: str,
        key: str,
        value: Any,
        agent_id: Optional[str] = None
    ) -> None:
        """
        Write data to the blackboard.

        Args:
            namespace: Category/namespace (e.g., "research", "tasks")
            key: Specific key
            value: Data to write
            agent_id: ID of agent writing (for logging)
        """
        if namespace not in self.storage:
            self.storage[namespace] = {}

        self.storage[namespace][key] = value

        # Log access
        self.access_log.append({
            "action": "write",
            "namespace": namespace,
            "key": key,
            "agent_id": agent_id,
            "timestamp": datetime.now()
        })

    def read(
        self,
        namespace: str,
        key: str,
        agent_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Read data from the blackboard.

        Args:
            namespace: Category/namespace
            key: Specific key
            agent_id: ID of agent reading (for logging)

        Returns:
            Value or None if not found
        """
        # Log access
        self.access_log.append({
            "action": "read",
            "namespace": namespace,
            "key": key,
            "agent_id": agent_id,
            "timestamp": datetime.now()
        })

        if namespace in self.storage:
            return self.storage[namespace].get(key)

        return None

    def read_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Read all data in a namespace.

        Args:
            namespace: Category/namespace

        Returns:
            Dict of all key-value pairs in namespace
        """
        return self.storage.get(namespace, {}).copy()

    def exists(self, namespace: str, key: str) -> bool:
        """Check if a key exists."""
        return namespace in self.storage and key in self.storage[namespace]

    def delete(self, namespace: str, key: str, agent_id: Optional[str] = None) -> bool:
        """
        Delete data from the blackboard.

        Args:
            namespace: Category/namespace
            key: Specific key
            agent_id: ID of agent deleting (for logging)

        Returns:
            True if deleted, False if not found
        """
        if not self.exists(namespace, key):
            return False

        del self.storage[namespace][key]

        # Log access
        self.access_log.append({
            "action": "delete",
            "namespace": namespace,
            "key": key,
            "agent_id": agent_id,
            "timestamp": datetime.now()
        })

        return True

    def get_access_log(
        self,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get access log, optionally filtered.

        Args:
            agent_id: Filter by agent
            namespace: Filter by namespace

        Returns:
            List of access log entries
        """
        log = self.access_log

        if agent_id:
            log = [entry for entry in log if entry.get("agent_id") == agent_id]

        if namespace:
            log = [entry for entry in log if entry.get("namespace") == namespace]

        return log

    def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear the blackboard.

        Args:
            namespace: If provided, clear only this namespace. Otherwise clear all.
        """
        if namespace:
            if namespace in self.storage:
                self.storage[namespace].clear()
        else:
            self.storage.clear()


# Helper functions for creating common message types

def create_task_request(
    sender_id: str,
    receiver_id: str,
    task_description: str,
    parameters: Optional[Dict[str, Any]] = None
) -> StructuredMessage:
    """Create a task request message."""
    return StructuredMessage(
        type=MessageType.TASK_REQUEST,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content={
            "task": task_description,
            "parameters": parameters or {}
        }
    )


def create_task_response(
    sender_id: str,
    receiver_id: str,
    result: Any,
    success: bool = True,
    in_reply_to: Optional[str] = None
) -> StructuredMessage:
    """Create a task response message."""
    return StructuredMessage(
        type=MessageType.TASK_RESPONSE,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content={
            "result": result,
            "success": success
        },
        in_reply_to=in_reply_to
    )


def create_information(
    sender_id: str,
    info_type: str,
    data: Dict[str, Any],
    receiver_id: Optional[str] = None
) -> StructuredMessage:
    """Create an information message (can be broadcast)."""
    return StructuredMessage(
        type=MessageType.INFORMATION,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content={
            "info_type": info_type,
            "data": data
        }
    )


def create_error(
    sender_id: str,
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None,
    receiver_id: Optional[str] = None
) -> StructuredMessage:
    """Create an error message."""
    return StructuredMessage(
        type=MessageType.ERROR,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content={
            "error": error_message,
            "details": error_details or {}
        }
    )
