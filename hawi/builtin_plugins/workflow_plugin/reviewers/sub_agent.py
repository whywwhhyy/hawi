"""SubAgentReviewer — uses a cloned HawiAgent to review node output."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from hawi.builtin_plugins.workflow_plugin.models import (
    NodeExecution,
    ReviewDecision,
    WorkflowNode,
    WorkflowRun,
)
from hawi.builtin_plugins.workflow_plugin.reviewers.base import Reviewer

if TYPE_CHECKING:
    from hawi.agent import HawiAgent

logger = logging.getLogger(__name__)

_REVIEW_SYSTEM_PROMPT = """\
You are a QUALITY GATE reviewer. Your job is to evaluate whether an agent's
output on a workflow node meets the required standards.

You MUST respond with a single JSON object and nothing else:
{"approved": true/false, "feedback": "specific, actionable feedback"}

If approved is false, the agent will receive your feedback and must revise.
Be specific about what is wrong and how to fix it."""


class SubAgentReviewer(Reviewer):
    """Review node output by asking a cloned (or different) agent to evaluate it.

    This creates a fresh sub-agent via ``agent.clone()``, optionally switches
    the model, and asks it to judge the output against the review criteria.

    This is a "separation of concerns" pattern: the agent that *produces* the
    output is not the one that *judges* it.  This reduces the risk of
    self-confirmation bias.
    """

    def __init__(self, review_prompt: str, model: str | None = None):
        """
        Args:
            review_prompt: Criteria for evaluation (injected into the review
                agent's user message).
            model: Optional model override for the review sub-agent (e.g.
                a more capable or cheaper model for review tasks).
        """
        self._review_prompt = review_prompt
        self._model = model

    @property
    def identity(self) -> str:
        base = "sub_agent"
        if self._model:
            base += f":{self._model}"
        return base

    async def review(
        self,
        node: WorkflowNode,
        execution: NodeExecution,
        run: WorkflowRun,
        agent: "HawiAgent",
    ) -> ReviewDecision:
        # Clone the agent for isolation
        sub = agent.clone()

        # Optionally switch the review agent's model
        if self._model:
            sub.set_model(self._model)

        # Build upstream context
        upstream_context = self._build_upstream_context(run, node)
        routing_context = "(no route selected)"
        if execution.selected_next_node_id:
            routing_context = (
                f"{execution.selected_next_node_id}\n"
                f"Reason: {execution.routing_reason or '(none provided)'}"
            )

        review_message = f"""Review the following workflow node output.

## Workflow: {run.workflow_id}
## Gate: {node.name} ({node.id})
## Task given to the agent:
{node.prompt}

## Previous gates' outputs (context):
{upstream_context}

## Agent's output to review:
---
{execution.output or "(empty)"}
---

## Agent's selected next gate:
{routing_context}

## Review criteria:
{self._review_prompt}

Remember: respond ONLY with JSON: {{"approved": true/false, "feedback": "..."}}"""

        try:
            result = await sub.arun(review_message)
            return self._parse_decision(result.text)
        except Exception as exc:
            logger.exception("Sub-agent review failed for node '%s': %s", node.id, exc)
            return ReviewDecision(
                approved=False,
                feedback=(
                    f"Review infrastructure error: {exc}. "
                    "Please retry or escalate to a human reviewer."
                ),
            )

    @staticmethod
    def _build_upstream_context(
        run: WorkflowRun,
        current_node: WorkflowNode,
    ) -> str:
        """Build a summary of upstream node outputs for the reviewer."""
        # Collect all completed nodes (simple approach: anything with output)
        parts: list[str] = []
        for nid, execution in run.node_executions.items():
            if nid == current_node.id:
                continue
            if execution.output and execution.status == "completed":
                parts.append(
                    f"### {nid}\n{execution.output[:1500]}\n"
                )
        if not parts:
            return "(no previous gates)"
        return "\n".join(parts)

    @staticmethod
    def _parse_decision(text: str) -> ReviewDecision:
        """Extract a JSON decision from the review agent's response."""
        # Try to find JSON block first
        match = re.search(r'\{[^{}]*"approved"[^{}]*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            # Fallback: try the whole text as JSON
            json_str = text.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Best-effort heuristic: look for approved/not approved
            lowered = text.lower()
            if "approved" in lowered and "not approved" not in lowered:
                return ReviewDecision(
                    approved=True,
                    feedback=text[:500],
                )
            return ReviewDecision(
                approved=False,
                feedback=f"Reviewer output could not be parsed as JSON. "
                f"Raw response: {text[:500]}",
            )

        approved = bool(data.get("approved", False))
        feedback = str(data.get("feedback", ""))
        return ReviewDecision(approved=approved, feedback=feedback)
