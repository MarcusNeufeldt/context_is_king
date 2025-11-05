"""
Agent Configuration Module

Defines the configuration structure for agents following the
"Context is Code" principle from framework.md
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class AgentConfig:
    """
    Configuration for an Agent instance.

    This allows developers to explicitly select the right "cognitive architecture"
    for their task, as outlined in framework.md Section III.
    """

    # Agent identity
    agent_name: str
    agent_role: str

    # AGENTS.md file path (Context is Code principle)
    instructions_path: Path

    # Reasoning strategy
    reasoning_strategy: str = "react"  # Options: "react", "tot", "auto"

    # Memory configuration
    memory_enabled: bool = False
    memory_path: Optional[Path] = None

    # Self-improvement (ACE Loop)
    self_improvement_enabled: bool = False
    playbook_path: Optional[Path] = None

    # LLM configuration
    model: str = "minimax/minimax-m2"
    temperature: float = 0.7
    max_tokens: int = 2000

    # Tools and permissions
    available_tools: list = field(default_factory=list)
    tool_permissions: Dict[str, bool] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization"""
        if isinstance(self.instructions_path, str):
            self.instructions_path = Path(self.instructions_path)

        if self.memory_path and isinstance(self.memory_path, str):
            self.memory_path = Path(self.memory_path)

        if self.playbook_path and isinstance(self.playbook_path, str):
            self.playbook_path = Path(self.playbook_path)

    @classmethod
    def from_agents_md(cls, agents_md_path: Path, agent_name: str) -> 'AgentConfig':
        """
        Factory method to create AgentConfig from an AGENTS.md file.

        This implements the discovery hierarchy from agents_md_framework.md
        """
        # This will be implemented to parse AGENTS.md and extract config
        return cls(
            agent_name=agent_name,
            agent_role="default",
            instructions_path=agents_md_path
        )
