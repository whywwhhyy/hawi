"""HumanReviewer — pauses workflow execution until a human approves or rejects."""

from __future__ import annotations

import asyncio
import uuid
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
    from hawi_plugins.workflow_plugin.plugin import WorkflowPlugin


class HumanReviewer(Reviewer):
    """A reviewer that pauses execution and waits for a human decision.

    The review request is emitted as a ``plugin.event`` so the GUI/CLI can
    present it.  The human calls ``approve_workflow_node`` or
    ``reject_workflow_node`` (tools on WorkflowPlugin) to unblock.

    Internally this uses an ``asyncio.Future`` keyed by *review_id* stored
    on the plugin instance.
    """

    def __init__(self, plugin: "WorkflowPlugin"):
        """
        Args:
            plugin: The owning WorkflowPlugin instance (needed to access
                the pending-reviews future dict).
        """
        self._plugin = plugin

    @property
    def identity(self) -> str:
        return "human"

    async def review(
        self,
        node: WorkflowNode,
        execution: NodeExecution,
        run: WorkflowRun,
        agent: "HawiAgent",
    ) -> ReviewDecision:
        review_id = str(uuid.uuid4())[:8]

        # Emit event so GUI can show a review dialog
        self._plugin.emit_plugin_event(
            "plugin.event",
            {
                "event_name": "workflow.review.requested",
                "review_id": review_id,
                "workflow_name": getattr(
                    getattr(self._plugin, "_workflow", None), "name", ""
                ),
                "workflow_id": run.workflow_id,
                "node_id": node.id,
                "node_name": node.name,
                "node_prompt": node.prompt,
                "output": execution.output,
                "output_preview": (execution.output or "")[:500],
                "selected_next_node_id": execution.selected_next_node_id,
                "routing_reason": execution.routing_reason,
                "attempt": execution.attempt_count,
                "max_retries": node.max_retries,
                "title": f"Review required: {node.name}",
                "message": (
                    f"Node '{node.name}' (attempt {execution.attempt_count}/"
                    f"{node.max_retries}) is awaiting human review."
                ),
            },
        )

        # Create a future and wait for the human to resolve it
        future: asyncio.Future[ReviewDecision] = asyncio.get_event_loop().create_future()
        self._plugin._pending_human_reviews[review_id] = future

        try:
            decision = await future
            return decision
        finally:
            self._plugin._pending_human_reviews.pop(review_id, None)
