"""Reviewer abstract base and review decision types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from hawi_plugins.workflow_plugin.models import (
    NodeExecution,
    ReviewDecision,
    WorkflowNode,
    WorkflowRun,
)

if TYPE_CHECKING:
    from hawi.agent import HawiAgent


class Reviewer(ABC):
    """Pluggable reviewer interface.

    Each reviewer implementation provides a different strategy for
    evaluating node output:

    - HumanReviewer: pauses execution, waits for a human via GUI/CLI.
    - SubAgentReviewer: spawns a cloned HawiAgent to evaluate the output.
    - LoggerReviewer: auto-approves while keeping a full audit log.

    Subclasses must implement ``review()`` and expose ``identity``.
    """

    @property
    @abstractmethod
    def identity(self) -> str:
        """Human-readable identity for audit trails (e.g. 'human', 'sub_agent:gpt-4')."""
        ...

    @abstractmethod
    async def review(
        self,
        node: WorkflowNode,
        execution: NodeExecution,
        run: WorkflowRun,
        agent: "HawiAgent",
    ) -> ReviewDecision:
        """Evaluate *execution.output* and return a decision.

        Args:
            node: The workflow node definition (contains prompt, review config).
            execution: Current node execution state (contains agent output).
            run: The full workflow run for context.
            agent: The HawiAgent instance (for sub-agent creation etc.).

        Returns:
            A ReviewDecision indicating approved/rejected and optional feedback.
        """
        ...
