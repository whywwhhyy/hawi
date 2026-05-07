"""LoggerReviewer — auto-approves while keeping a full audit trail."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hawi_plugins.workflow_plugin.models import (
    NodeExecution,
    ReviewDecision,
    WorkflowNode,
    WorkflowRun,
)
from hawi_plugins.workflow_plugin.reviewers.base import Reviewer

if TYPE_CHECKING:
    from hawi.agent import HawiAgent

logger = logging.getLogger(__name__)


class LoggerReviewer(Reviewer):
    """A reviewer that automatically approves every node.

    All decisions are logged (both via Python logging and via plugin events
    when bound to a plugin).  This is appropriate for nodes whose output
    does not need real-time approval but must be auditable later.
    """

    @property
    def identity(self) -> str:
        return "logger"

    async def review(
        self,
        node: WorkflowNode,
        execution: NodeExecution,
        run: WorkflowRun,
        agent: "HawiAgent",
    ) -> ReviewDecision:
        output_preview = (execution.output or "")[:200]
        logger.info(
            "Workflow '%s' node '%s' auto-approved (attempt %d, %d chars).",
            run.workflow_id,
            node.id,
            execution.attempt_count,
            len(execution.output or ""),
        )
        return ReviewDecision(approved=True)
