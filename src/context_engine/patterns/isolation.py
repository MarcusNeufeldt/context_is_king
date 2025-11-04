"""
Context Isolation Pattern: Multi-Agent Systems

Implements the Context Isolation via Subagents pattern from the paper.
Break down complex tasks into specialized subagents, each with its own
isolated context, system prompt, and tools.

Benefits:
- Prevents "context pollution"
- Improves performance and scalability
- Clear separation of concerns
- Parent agent only needs summaries, not full subagent context
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel

from ..core.agent import Agent, AgentConfig


class SubAgentStatus(str, Enum):
    """Status of a subagent task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SubAgentTask(BaseModel):
    """A task assigned to a subagent."""
    id: str
    agent_id: str
    description: str
    status: SubAgentStatus = SubAgentStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None


class SubAgent:
    """
    Wrapper for a subagent with task management.

    Each subagent has:
    - Isolated context (separate from parent)
    - Specialized role and system prompt
    - Own tool set
    - Task queue
    """

    def __init__(self, agent: Agent, specialization: str):
        """Initialize subagent wrapper."""
        self.agent = agent
        self.specialization = specialization
        self.tasks: List[SubAgentTask] = []

    def assign_task(self, task: SubAgentTask) -> None:
        """Assign a task to this subagent."""
        self.tasks.append(task)

    def execute_task(self, task: SubAgentTask) -> Any:
        """Execute a task and return result."""
        task.status = SubAgentStatus.RUNNING

        try:
            # Add task as goal
            self.agent.add_goal(task.description)

            # Process task
            result = self.agent.process_message(task.description)

            # Mark complete
            task.status = SubAgentStatus.COMPLETED
            task.result = result
            self.agent.complete_goal(task.description)

            return result

        except Exception as e:
            task.status = SubAgentStatus.FAILED
            task.error = str(e)
            return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of subagent's work.

        Parent agent receives this summary, not the full context.
        This is the key to context isolation.
        """
        return {
            "agent_id": self.agent.id,
            "specialization": self.specialization,
            "tasks_completed": len([t for t in self.tasks if t.status == SubAgentStatus.COMPLETED]),
            "tasks_failed": len([t for t in self.tasks if t.status == SubAgentStatus.FAILED]),
            "recent_results": [
                {
                    "task": t.description,
                    "status": t.status.value,
                    "result": str(t.result)[:200] if t.result else None  # Truncated
                }
                for t in self.tasks[-3:]  # Last 3 tasks only
            ]
        }


class MultiAgentSystem:
    """
    Multi-agent system with context isolation.

    Implements the architecture pattern:
    - Planning Agent: High-level orchestration
    - Execution Subagents: Specialized tasks with isolated contexts
    - Structured Communication: Summaries, not full contexts

    Example:
        ```python
        system = MultiAgentSystem(planning_agent)

        # Create specialized subagents
        code_agent = system.create_subagent(
            name="CodeExecutor",
            role="Code execution specialist",
            prompt="You execute and analyze code."
        )

        search_agent = system.create_subagent(
            name="Searcher",
            role="Information retrieval specialist",
            prompt="You search and summarize information."
        )

        # Delegate tasks
        result = system.delegate_task(
            subagent_id=code_agent.agent.id,
            task="Run this Python script and report errors"
        )
        ```
    """

    def __init__(self, planning_agent: Agent):
        """
        Initialize multi-agent system.

        Args:
            planning_agent: The main planning/orchestration agent
        """
        self.planning_agent = planning_agent
        self.subagents: Dict[str, SubAgent] = {}
        self.task_history: List[SubAgentTask] = []

    def create_subagent(
        self,
        name: str,
        role: str,
        system_prompt: str,
        config: Optional[AgentConfig] = None
    ) -> SubAgent:
        """
        Create a specialized subagent with isolated context.

        Args:
            name: Subagent name
            role: Specialization/role
            system_prompt: System prompt for subagent
            config: Optional full config

        Returns:
            SubAgent instance
        """
        # Create agent using planning agent's create_subagent method
        agent = self.planning_agent.create_subagent(name, role, system_prompt, config)

        # Wrap in SubAgent
        subagent = SubAgent(agent, role)
        self.subagents[agent.id] = subagent

        return subagent

    def delegate_task(
        self,
        subagent_id: str,
        task_description: str,
        task_id: Optional[str] = None
    ) -> Any:
        """
        Delegate a task to a subagent.

        Args:
            subagent_id: ID of target subagent
            task_description: Task to perform
            task_id: Optional task ID

        Returns:
            Task result
        """
        if subagent_id not in self.subagents:
            raise ValueError(f"Subagent {subagent_id} not found")

        subagent = self.subagents[subagent_id]

        # Create task
        import uuid
        task = SubAgentTask(
            id=task_id or str(uuid.uuid4()),
            agent_id=subagent_id,
            description=task_description
        )

        # Assign and execute
        subagent.assign_task(task)
        result = subagent.execute_task(task)

        # Record in history
        self.task_history.append(task)

        # Planning agent receives summary, not full context
        summary = subagent.get_summary()

        # Add summary to planning agent's memory (not full subagent context!)
        from ..utils.schemas import Message, MessageRole
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"Subagent '{subagent.specialization}' completed task. Summary: {summary}",
            metadata={"source": "subagent_summary", "subagent_id": subagent_id}
        )
        self.planning_agent.memory.add_message(summary_msg)

        return result

    def get_subagent(self, subagent_id: str) -> Optional[SubAgent]:
        """Get subagent by ID."""
        return self.subagents.get(subagent_id)

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries from all subagents."""
        return [subagent.get_summary() for subagent in self.subagents.values()]

    def parallel_delegate(
        self,
        tasks: List[tuple[str, str]]  # [(subagent_id, task_description), ...]
    ) -> List[Any]:
        """
        Delegate multiple tasks in parallel (conceptually).

        Note: Actual parallel execution would require async/threading.
        This is a sequential implementation for simplicity.

        Args:
            tasks: List of (subagent_id, task_description) tuples

        Returns:
            List of results
        """
        results = []
        for subagent_id, task_description in tasks:
            result = self.delegate_task(subagent_id, task_description)
            results.append(result)
        return results

    def shutdown_subagent(self, subagent_id: str) -> bool:
        """
        Shutdown a subagent and consolidate its memory.

        Args:
            subagent_id: ID of subagent to shutdown

        Returns:
            True if successful
        """
        if subagent_id not in self.subagents:
            return False

        subagent = self.subagents[subagent_id]

        # Consolidate subagent's memory
        subagent.agent.consolidate_memory(force=True)

        # Get final summary for planning agent
        final_summary = subagent.get_summary()

        # Add to planning agent's long-term memory
        from ..utils.schemas import MemoryEntry
        import uuid
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=f"Subagent '{subagent.specialization}' final summary: {final_summary}",
            metadata={
                "type": "subagent_final_summary",
                "subagent_id": subagent_id,
                "specialization": subagent.specialization
            },
            importance=1.5  # Summaries are important
        )
        self.planning_agent.memory.ltm.add(entry)

        # Remove from active subagents
        del self.subagents[subagent_id]

        return True

    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire multi-agent system."""
        return {
            "planning_agent": self.planning_agent.get_stats(),
            "num_subagents": len(self.subagents),
            "total_tasks": len(self.task_history),
            "completed_tasks": len([t for t in self.task_history if t.status == SubAgentStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.task_history if t.status == SubAgentStatus.FAILED]),
            "subagent_summaries": self.get_all_summaries()
        }
