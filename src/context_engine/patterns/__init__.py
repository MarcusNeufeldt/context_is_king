"""Context engineering patterns."""

from .isolation import MultiAgentSystem, SubAgent
from .self_baking import SelfBaker, SummaryBaker, SchemaBaker
from .communication import StructuredMessage, Blackboard, MessageBus

__all__ = [
    "MultiAgentSystem",
    "SubAgent",
    "SelfBaker",
    "SummaryBaker",
    "SchemaBaker",
    "StructuredMessage",
    "Blackboard",
    "MessageBus",
]
