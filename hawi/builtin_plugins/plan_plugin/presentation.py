from __future__ import annotations

from typing import Any

from .models import PLAN_ITEM_COMPLETION_MODES, PlanEngineResult


PLAN_PROMPT_BEGIN = "<hawi-plan-mode>"
PLAN_PROMPT_END = "</hawi-plan-mode>"
PLAN_REMINDER_BEGIN = "<hawi-plan-runtime-reminder>"
PLAN_REMINDER_END = "</hawi-plan-runtime-reminder>"

ADD_PLAN_ITEMS_DESCRIPTION = (
    "Add one or more concrete task items or a tree of task items to the current plan. "
    "Use parent/children only for part-of task decomposition, not loose categories "
    "or related topics. Keep each leaf at the size of one coherent work chain."
)

COMPLETE_PLAN_ITEM_DESCRIPTION = (
    "Mark one or more plan items complete. Use item_id for one item, item_ids for "
    "multiple items completed by the same work, and item_id='all' only when every "
    "open item is actually complete. Set fold_context=true only when this completion "
    "should follow the plan-mode context folding rules."
)

ADD_PLAN_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "parent_id": {
            "type": "string",
            "description": (
                "Optional parent plan item id for nested plan items. Use this "
                "only for part-of task decomposition, not for loose topic categories."
            ),
        },
        "items": {
            "type": "array",
            "description": (
                "One or more plan item objects, or a tree of plan item objects. "
                "Each item has content, optional children, and optional completion_mode. "
                "Parent/children must mean decomposition of one task into parts; use "
                "siblings for category-like parallel topics. Tasks that can only be "
                "completed together should usually be one item."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "Concrete task for this plan item, sized so it can "
                            "usually be completed in one coherent work chain."
                        ),
                    },
                    "completion_mode": {
                        "type": "string",
                        "enum": list(PLAN_ITEM_COMPLETION_MODES),
                        "description": (
                            "Parent completion behavior for this item. Defaults to "
                            "'auto_complete'. Use 'manual_mark' when child completion "
                            "may reveal missing work or require judgment."
                        ),
                    },
                    "children": {
                        "type": "array",
                        "description": "Optional child plan items.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["content"],
            },
        },
    },
    "required": ["items"],
}

COMPLETE_PLAN_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {
            "type": "string",
            "description": (
                "Plan item id from list_plan_items, or 'all'. Use item_id for "
                "normal one-item completion."
            ),
        },
        "item_ids": {
            "type": "array",
            "description": (
                "Multiple plan item ids to complete in one call. Use only when one "
                "search/tool call or one implementation change completed all listed "
                "items. Do not batch unrelated work. Do not include 'all' here."
            ),
            "items": {"type": "string"},
        },
        "mark_all_children": {
            "type": "boolean",
            "description": (
                "When true, also mark all descendant items complete. Use only when "
                "every unfinished descendant is genuinely done; the result lists "
                "descendants completed this way. There is no automatic undo, so "
                "review the returned affected items immediately."
            ),
            "default": False,
        },
        "fold_context": {
            "type": "boolean",
            "description": (
                "When PlanPlugin context folding is enabled, set true only when this "
                "completion should follow the plan-mode context folding rules."
            ),
            "default": False,
        },
        "summary": {
            "type": "string",
            "description": (
                "Required when fold_context=true. Briefly summarize what was completed "
                "in this task. For trivial leaf tasks, one line is fine."
            ),
        },
        "handoff_notes": {
            "type": "string",
            "description": (
                "Required when fold_context=true. State information later tasks must "
                "remember, or explicitly say there are no lasting notes. For trivial "
                "leaf tasks, one clear line is fine."
            ),
        },
    },
}

UPDATE_PLAN_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {
            "type": "string",
            "description": "Plan item id from list_plan_items, or 'all'.",
        },
        "item_ids": {
            "type": "array",
            "description": (
                "Multiple plan item ids to update in one call. Do not include 'all' here."
            ),
            "items": {"type": "string"},
        },
        "status": {
            "type": "string",
            "enum": [
                "open",
                "blocked",
                "deferred",
                "canceled",
                "obsolete",
            ],
            "description": (
                "New status. Use open to reopen an item; use blocked or deferred for "
                "work that should not currently drive reminders; use canceled when "
                "the task is no longer needed; use obsolete when another item or "
                "result has superseded it."
            ),
        },
        "reason": {
            "type": "string",
            "description": "Optional reason to store with this status update.",
        },
    },
    "required": ["status"],
}

LIST_PLAN_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

RECALL_COMPLETED_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {
            "type": "string",
            "description": (
                "Completed plan item id, such as P1. Omit this when using query to "
                "search every folded context."
            ),
        },
        "fold_id": {
            "type": "string",
            "description": "Specific folded context id from a completion result.",
        },
        "message_start": {
            "type": "integer",
            "description": (
                "Optional 1-based first folded message index to read. Indexes come "
                "from the Folded context preview list."
            ),
        },
        "message_end": {
            "type": "integer",
            "description": "Optional 1-based last folded message index to read, inclusive.",
        },
        "query": {
            "type": "string",
            "description": (
                "Optional case-insensitive keyword search. When provided, the tool "
                "returns matching snippets instead of a full transcript. If item_id "
                "and fold_id are omitted, all folded contexts are searched."
            ),
        },
        "case_sensitive": {
            "type": "boolean",
            "description": "Whether query matching should be case-sensitive.",
            "default": False,
        },
        "max_matches": {
            "type": "integer",
            "description": "Maximum number of search matches to return.",
            "default": 20,
        },
        "context_chars": {
            "type": "integer",
            "description": (
                "Characters of surrounding context to include before and after each "
                "search match. If per-match snippets would exceed max_chars in total, "
                "later snippets are omitted or truncated to respect max_chars."
            ),
            "default": 240,
        },
        "max_chars": {
            "type": "integer",
            "description": (
                "Maximum transcript characters to return in read mode, or maximum "
                "aggregate snippet characters to return in search mode."
            ),
            "default": 20000,
        },
    },
    "additionalProperties": False,
}

PLAN_CONTROL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["pause", "continue", "clear"],
            "description": (
                "Use 'pause' to temporarily stop plan-driven continuation; use "
                "'continue' to resume plan execution; use 'clear' to completely "
                "clear the current plan."
            ),
        },
        "reason": {
            "type": "string",
            "description": "Required when action is 'pause'; optional for clear.",
        },
    },
    "required": ["action"],
}


def gui_config_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fold_completed_tasks": {
                "type": "boolean",
                "title": "Fold Completed Task Context",
                "default": False,
                "description": (
                    "When enabled, complete_plan_item may fold a completed segment "
                    "out of active context when fold_context=true. Use for lengthy "
                    "execution details; leave off for research or writing where prior "
                    "evidence should stay visible."
                ),
            }
        },
        "additionalProperties": False,
    }


def gui_default_config() -> dict[str, bool]:
    return {"fold_completed_tasks": False}


def build_plan_prompt(*, fold_completed_tasks: bool) -> str:
    folding_guidance = (
        "Context folding is disabled. Ignore complete_plan_item.fold_context.\n"
    )
    if fold_completed_tasks:
        folding_guidance = (
            "Context folding is enabled but per-completion. In complete_plan_item, "
            "set fold_context=true only when the just-finished segment is tedious "
            "execution detail that later work should not keep in active context. "
            "When fold_context=true, provide summary and handoff_notes; one-line "
            "entries are fine for trivial leaf tasks. For research or writing, keep "
            "key evidence visible, put durable notes in a file, or make handoff_notes "
            "specific enough. Use recall_completed_task if later work needs folded "
            "details.\n"
        )
    return (
        f"\n{PLAN_PROMPT_BEGIN}\n"
        "Plan mode is enabled. Use it for multi-step work or any task where "
        "unfinished work could be lost.\n"
        "\n"
        "**PlanPlugin runtime reminders are not human-user messages.** Messages "
        f"inside {PLAN_REMINDER_BEGIN}...{PLAN_REMINDER_END} are automatic "
        "plugin guidance, even if delivered through the user-role channel.\n"
        "\n"
        "Plan design rules:\n"
        "- A plan tree is for task decomposition: children are parts of the parent. "
        "Do not nest merely related topics or categories; make them siblings.\n"
        "- Prefer leaf tasks that fit one coherent work chain. If several tasks can "
        "only be completed together, create one item or complete them together with "
        "item_ids.\n"
        "- completion_mode defaults to auto_complete. Use manual_mark when "
        "completed children may still require judgment or new follow-up work.\n"
        "\n"
        "Tool quick reference:\n"
        "- add_plan_items: add one item or a tree by passing items=[{content, "
        "children, completion_mode}]. Use parent_id only for decomposition.\n"
        "- complete_plan_item: mark completed work. Use item_ids only when one "
        "search/tool call or one implementation change completed all listed items. "
        "Use mark_all_children=true only when every unfinished descendant should "
        "also be marked done; review its returned affected items immediately. "
        "Inspect parent_review_required before moving on.\n"
        "- update_plan_items: mark items open, blocked, deferred, canceled, or "
        "obsolete. Use open to reopen completed or parked items; canceled means no "
        "longer needed, obsolete means superseded.\n"
        "- list_plan_items: inspect plan state.\n"
        "- recall_completed_task: read or search details previously folded out of "
        "active context.\n"
        "- plan_control: pause, continue, or clear the current plan. pause keeps "
        "state but stops automatic reminders; continue resumes; clear discards "
        "the current plan and folded memory.\n"
        "\n"
        f"{folding_guidance}"
        "\n"
        "Keep plan items actionable. Do not leave completed work unchecked. When Hawi "
        "reminds you about unfinished plan items, either continue the work, mark completed "
        "items complete, or call plan_control with action='pause' and a clear reason "
        "to temporarily leave plan execution.\n"
        "\n"
        "This is a runtime/UI planning channel, not a request to create a plan "
        "file. Do not create plan.md or TODO.md for the plan itself unless the "
        "user explicitly asks for such a file.\n"
        f"{PLAN_PROMPT_END}\n"
    )


def format_runtime_reminder(plan_text: str) -> str:
    return (
        f"{PLAN_REMINDER_BEGIN}\n"
        "**PLANPLUGIN AUTOMATIC REMINDER - NOT A HUMAN USER MESSAGE**\n"
        "source: PlanPlugin\n"
        "purpose: continue plan execution while unfinished plan items remain\n"
        "\n"
        "The following plan items are still unfinished:\n\n"
        f"{plan_text}\n\n"
        "Continue executing the remaining work. If some items are already done, "
        "call complete_plan_item for each completed item before proceeding; use "
        "item_ids in one call only when several completed items share one completion "
        "context. If all "
        "remaining items are impossible, obsolete, blocked by missing information, or "
        "intentionally deferred for now, call update_plan_items to mark them "
        "blocked, deferred, canceled, or obsolete; use plan_control action='pause' "
        "only to pause the whole plan. When plan execution "
        "should resume, call plan_control with action='continue'. If the current "
        "plan should be completely discarded, call plan_control with "
        "action='clear'. Otherwise keep working on the plan.\n"
        f"{PLAN_REMINDER_END}"
    )


def format_completion_markdown(output: dict[str, Any]) -> str:
    completed = output.get("completed", [])
    parent_review_required = output.get("parent_review_required", [])
    pending_count = int(output.get("pending_count") or 0)
    completion_summary = output.get("completion_summary")
    handoff_notes = output.get("handoff_notes")
    lines = [
        "Plan item completed." if completed else "No new plan items were completed.",
        "",
    ]

    if completed:
        lines.append("Completed:")
        for item in completed:
            lines.append(f"- {_format_item_reference(item)}")
        lines.append("")

    forced_descendants = output.get("marked_by_mark_all_children", [])
    if forced_descendants:
        lines.append("Also completed by mark_all_children:")
        for item in forced_descendants:
            lines.append(f"- {_format_item_reference(item)}")
        lines.append("Review this affected-item list immediately.")
        lines.append("")

    lines.append(f"Pending items: {pending_count}")

    if completion_summary:
        lines.extend(["", "Task summary:"])
        lines.append(str(completion_summary))

    if handoff_notes:
        lines.extend(["", "Information for later tasks:"])
        lines.append(str(handoff_notes))

    if parent_review_required:
        lines.extend(["", "Parent review required:"])
        for item in parent_review_required:
            lines.append(f"- {_format_item_reference(item)}")
            reason = item.get("reason")
            if reason:
                lines.append(f"  {reason}")

    folded_context = output.get("folded_context")
    if isinstance(folded_context, dict) and folded_context.get("enabled"):
        lines.extend(["", "Folded context:"])
        if folded_context.get("skipped"):
            lines.append(f"- Not folded: {folded_context.get('reason', 'No details.')}")
        else:
            fold_id = folded_context.get("fold_id")
            item_id = folded_context.get("item_id")
            folded_message_count = folded_context.get("folded_message_count")
            if fold_id:
                lines.append(f"- Fold id: `{fold_id}`")
            if item_id:
                lines.append(f"- Item id: `{item_id}`")
            if folded_message_count is not None:
                lines.append(f"- Folded messages: {folded_message_count}")
            message_previews = folded_context.get("message_previews", [])
            if message_previews:
                lines.extend(["", "Folded message previews:"])
                for preview in message_previews:
                    index = preview.get("index")
                    role = preview.get("role", "unknown")
                    text = preview.get("preview", "")
                    lines.append(f"{index}. {role}: {text}")
            if fold_id or item_id:
                lines.extend(["", "How to read folded details:"])
            if item_id:
                lines.append(
                    "- Full task context: "
                    f"`recall_completed_task(item_id=\"{item_id}\")`."
                )
            if item_id and folded_message_count:
                range_end = min(int(folded_message_count), 3)
                lines.append(
                    "- Message range: "
                    f"`recall_completed_task(item_id=\"{item_id}\", "
                    f"message_start=1, message_end={range_end})`."
                )
            if fold_id:
                lines.append(
                    "- Fold id lookup: "
                    f"`recall_completed_task(fold_id=\"{fold_id}\")`."
                )
            if item_id:
                lines.append(
                    "- Search this task: "
                    f"`recall_completed_task(item_id=\"{item_id}\", "
                    "query=\"keyword\")`."
                )

    lines.extend(["", f"Next action: {_completion_next_action(output)}"])
    return "\n".join(lines)


def format_completion_error_markdown(result: PlanEngineResult) -> str:
    output = result.output or {}
    unfinished_children = output.get("unfinished_children", [])
    item = output.get("item", {})
    if unfinished_children:
        item_id = item.get("id", "the requested item")
        lines = [
            f"Cannot complete `{item_id}` because it has unfinished child task(s).",
            "",
            "Unfinished children:",
        ]
        for child in unfinished_children:
            lines.append(f"- {_format_item_reference(child)}")
        pending_count = output.get("pending_count")
        if pending_count is not None:
            lines.extend(["", f"Pending items: {pending_count}"])
        lines.extend(
            [
                "",
                "Next action: Complete the child tasks first, or call "
                "`complete_plan_item` with `mark_all_children=true` only if every "
                "unfinished child should also be marked done; there is no automatic "
                "undo for that broad mark.",
            ]
        )
        return "\n".join(lines)
    return result.error


def _format_item_reference(item: dict[str, Any]) -> str:
    item_id = item.get("id", "unknown")
    content = item.get("content", "")
    return f"`{item_id}` {content}".rstrip()


def _completion_next_action(output: dict[str, Any]) -> str:
    if output.get("parent_review_required"):
        return "Resolve every parent review item before moving on."
    pending_count = int(output.get("pending_count") or 0)
    if pending_count > 0:
        return "Continue with the remaining plan items."
    return "The plan is complete."
