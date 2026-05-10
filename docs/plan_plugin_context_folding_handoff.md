# Plan Plugin Context Folding Handoff

This document captures the current discussion state so future Codex context can
be compacted and restored without losing the design thread.

## Latest Committed Work

Commit:

```text
6361b42 Improve plan plugin task completion UX
```

That commit intentionally made breaking changes. No backward compatibility was
kept for the renamed interfaces.

Implemented:

- Renamed `add_plan_item` to `add_plan_items`; the old tool name is gone.
- Added `complete_plan_item(item_ids=[...])` so multiple plan items can be
  completed in one call and share one folded backup.
- Removed the old `complete_children` alias; only `mark_all_children` remains.
- Renamed framework/tool UI metadata from `tool_call_description` to
  `tool_call_purpose`; old field parsing was removed.
- Updated GUI injection text for `tool_call_purpose` to say repeated values are
  allowed.
- Improved plan tool descriptions based on DeepSeek feedback:
  - task granularity should fit one coherent work chain;
  - parent/child relationships should represent part-of decomposition, not
    category grouping;
  - use `determinate` when child completion necessarily completes the parent;
  - use `exploratory` when judgment or follow-up may be needed;
  - for information-gathering tasks, persist key findings/source references to a
    file before completion if details will matter later.
- Added a cache point after each successful `complete_plan_item` result and
  removed previous cache points created by `complete_plan_item`, so globally
  only one plan-completion cache point remains.

Verification already run:

```bash
uv run pytest test/unit/plugins/test_plan_plugin.py test/unit/engine/test_event_mapper.py test/unit/engine/test_runtime_and_transports.py test/unit/test_anthropic_tool_result_conversion.py test/unit/test_snapshot_round_trip.py
npm test
npm run build
git diff --check
```

All passed.

## Current Uncommitted Follow-Up Work

After the latest discussion, the working tree now implements the next PlanPlugin
iteration:

- Renamed item completion behavior from `kind` to `completion_mode`.
- Replaced `exploratory`/`determinate` with `manual_mark`/`auto_complete`; the
  default is now `auto_complete`.
- Replaced `read_completed_task_context` with `recall_completed_task`; removed
  `task_id` and `message_range` aliases.
- Replaced `plan_control(action="abandon")` with `plan_control(action="clear")`.
- Kept `fold_completed_tasks` as a boolean config, but folding is now opt-in per
  completion through `complete_plan_item(fold_context=true)`.
- `summary` and `handoff_notes` are required only when `fold_context=true`.
- Consecutive folded completions with no intervening messages now reference the
  previous fold instead of creating an empty fold.
- Added `update_plan_items` for `open`, `blocked`, `deferred`, `canceled`, and
  `obsolete` item statuses. `open` reopens parked or completed items.
- Runtime reminders now use a bold explicit marker that they are automatic
  PlanPlugin reminders, not human-user messages.
- The prompt was shortened and reorganized around plan decomposition, tool quick
  reference, and concise folding guidance.

Verification for this uncommitted work:

```bash
uv run pytest test/unit/plugins/test_plan_plugin.py test/unit/engine/test_event_mapper.py test/unit/engine/test_runtime_and_transports.py test/unit/test_anthropic_tool_result_conversion.py test/unit/test_snapshot_round_trip.py test/unit/test_plugin_gui_schema.py
git diff --check
```

Both passed.

## DeepSeek Feedback That Drove The Changes

DeepSeek tested PlanPlugin on a research-heavy task and reported:

- Requiring a unique `tool_call_description` was painful. The current design now
  uses `tool_call_purpose`, and repeated values are allowed.
- Parent review chains were easy to forget. Some parents should have been
  `determinate`; tool descriptions now make this more explicit.
- Initial plans were often too fine-grained or semantically mixed. New guidance
  explains task size and parent/child decomposition.
- Context folding helped with context pressure but hurt later research/writing
  when raw evidence was needed. Current guidance recommends persisting important
  findings to files or writing richer `handoff_notes`.
- Runtime plan reminders appearing as user messages can be visually confusing.
  This has not been fixed yet.

## Researched But Not Implemented Yet

Dynamic runtime tool descriptions:

- Tool descriptions are gathered from plugin-decorated methods into
  `FunctionAgentTool`.
- `PluginManager.get_tool_definitions()` caches the resulting tool definitions.
- The agent refreshes `context.tool_definitions` before execution, but it reads
  the cached definitions unless the plugin manager cache is invalidated.
- A dynamic description wrapper around the relevant plan tools is feasible, but
  if folding mode can change at runtime, the plugin manager needs a clean cache
  invalidation path.

Consecutive `complete_plan_item` calls with no intervening work:

- Current folding locates the active completion tool call and folds messages
  between the previous completion boundary and the current one.
- If a model calls `complete_plan_item` repeatedly without any intervening
  message/work, the fold slice can be empty.
- Desired behavior: from the second call onward, return a result saying to refer
  to the first completion's folded context rather than creating a new empty fold.
- This is feasible, including multiple tool calls in the same assistant turn,
  because `before_tool_calling` tracks the active `complete_plan_item` tool call
  id.

## Current Design Question

The active question is whether the "completed-task context folding" switch should
be left for the model to decide based on task type.

Current favored direction:

- Prefer a three-state config rather than a single boolean:
  - `off`: never fold completed task context.
  - `on`: always fold completed task context.
  - `auto`: let the model decide, but require an explicit reason.
- Do not let folding silently change on every completion. Better flow:
  - decide before the first major completion boundary;
  - allow explicit switching later through `plan_control`;
  - show the model's reason in state/UI.

Suggested implementation shape:

```python
fold_completed_tasks: Literal["off", "on", "auto"] = "auto"
```

Possible control action:

```text
plan_control(action="set_context_folding", enabled=true/false, reason="...")
```

## Guidance To Put Near The Folding Switch

The key point from the latest discussion: all folding-related guidance should be
near the switch itself, not scattered across tool descriptions. The switch is not
just a token optimization; it changes how later tasks retrieve information.

Draft copy:

```text
Completed-task Context Folding

When enabled, completing a plan item folds detailed messages since the previous
complete_plan_item boundary out of active context and stores them in PlanPlugin
memory. The active context keeps only the completion marker, summary,
handoff_notes, and read-back instructions.

Choose this based on task shape.

Enable for:
- Long multi-step execution where completed details are unlikely to be needed
  verbatim.
- Coding, refactor, debug, or test tasks where durable state lives in files,
  diffs, tests, or tool results.
- Work with many independent plan items where keeping all intermediate messages
  would distract later steps.

Disable for:
- Information-gathering, research, comparison, or writing tasks where earlier raw
  evidence may be repeatedly reused.
- Tasks where source excerpts, webpage details, papers, logs, or user-provided
  context must stay immediately visible.
- Short plans where context pressure is low.

If enabled for research-like work:
- Before completing a task, persist key findings and source references to a file,
  or make handoff_notes detailed enough for later tasks.
- Use read_completed_task_context when later work needs folded details.

For auto mode:
- Decide before the first major completion boundary.
- Prefer disabling when unsure for information-heavy tasks.
- Prefer enabling when the task is execution-heavy and state is persisted outside
  chat.
- If multiple items can only be completed together, create one item or complete
  them with item_ids in a single call so they share one folded backup.
```

## Open Next Steps

- Decide whether to implement boolean-to-tristate migration now.
- Decide where the model records its `auto` choice and reason:
  - plugin state only;
  - plan artifact metadata;
  - GUI-visible plugin event;
  - all of the above.
- Decide whether `plan_control` should own context folding changes or whether a
  dedicated tool is cleaner.
- Improve plan runtime reminder visibility. Low-cost option: stronger visual
  text boundary/prefix in the reminder. Higher-cost option: represent reminders
  as a distinct message/event type instead of user-role text.
- Revisit the consecutive completion/fold-reference behavior after the folding
  mode design settles.

## DeepSeek Harsh Review Of Current Tool Instructions

A later DeepSeek review was explicitly asked to be strict about the current plan
tool descriptions. The main conclusion: the design is powerful, but the prompt
surface is too long and too cognitively expensive.

### High-Level Critique

- `<hawi-plan-mode>` is too long and visually dense. It mixes core concepts,
  tool reference, detailed rules, context folding, and message-origin caveats in
  one large text block. Suggested structure:
  - core idea in about three lines;
  - quick reference with one line per tool;
  - detailed rules in clearly separated sections.
- The "do not create plan.md/TODO.md" warning may be necessary, but it currently
  appears too early and feels defensive. Consider moving it lower unless this is
  still a frequent model failure.
- The runtime reminder origin rule is important but buried near the end. It
  should move near the top and be visually prominent, because reminder messages
  may appear as user-role messages and models can mistake them for human input.
- Folding guidance is repeated in multiple places. Prefer one authoritative
  folding section near the folding switch/config, then lighter references from
  individual tool descriptions.

### Naming Problems

DeepSeek found the current `kind` names misleading:

- `exploratory` sounds like "research/exploration", but its actual behavior is
  "parent completion needs manual review".
- `determinate` sounds abstract; its actual behavior is "auto-complete parent
  when all children complete".

Suggested rename candidates:

| Current | Candidate | Rationale |
| --- | --- | --- |
| `exploratory` | `manual_review`, `pending_judgment`, or `review` | Names the required action directly. |
| `determinate` | `auto_complete`, `mechanical`, or `auto` | Names the automatic behavior directly. |

If we keep current names, the descriptions need to be much stronger and much
shorter. DeepSeek's strongest recommendation was to rename them.

### Default Kind Concern

Current default is the manual-review behavior (`exploratory`). DeepSeek observed
that this creates many avoidable `parent_review_required` interruptions because
models often omit `kind` when creating plans. In its research plan, many parent
tasks should likely have been automatic, but defaulted to manual review.

Possible directions:

- Change the default to `auto_complete`/`determinate`.
- Keep the safer manual-review default but add stronger guidance:
  - most closed-scope parent tasks should use automatic completion;
  - use manual review only when child completion may reveal missing work or when
    the child set is not known to be complete.

Important caveat from the review: information-gathering often is not truly
closed-scope. Even a parent like "search official docs, papers, and release
notes" may discover new leads. If discovery can create new sub-tasks, manual
review may still be appropriate.

### `complete_plan_item` Critique

- `item_ids` wording was unclear. "share one completion context" is too abstract.
  Add concrete examples if the parameter stays:
  - use it when one search/tool call produced the evidence for multiple items;
  - use it when one implementation change completed multiple checklist items;
  - do not use it merely to batch unrelated completed work.
- `mark_all_children=true` is risky because it can mark unfinished descendants
  complete. Suggested improvements:
  - make the result clearly list which descendant ids were completed because of
    `mark_all_children`;
  - add a sharper warning in the parameter description.
- In folding mode, `summary` and `handoff_notes` are necessary but high-friction
  for small leaf tasks. The latest idea is not to remove the requirement yet, but
  to make guidance more nuanced:
  - for small information tasks, detailed `handoff_notes` can be enough;
  - writing to a file should be recommended when notes would be too large or
    will be reused heavily, not for every simple lookup.

### `read_completed_task_context` Critique

DeepSeek found this tool over-parameterized.

Parameters that feel redundant:

- `task_id` is an alias for `item_id`; suggested removal: keep only `item_id`.
- `message_range` duplicates `message_start`/`message_end`; suggested removal:
  keep explicit start/end. If shorthand is desired later, consider accepting a
  `"2-4"` string in one existing field rather than adding a third parameter.

The bigger point: this tool is powerful but currently invites choice paralysis.
It should have one obvious read path and one obvious search path.

### `plan_control` Critique

- `abandon` sounds emotionally loaded and maybe destructive. Suggested action
  name: `cancel` or `clear`.
- `pause` needs clearer semantics:
  - it preserves all plan state;
  - while paused, PlanPlugin does not proactively remind about unfinished items;
  - `continue` resumes reminders and plan-driven continuation;
  - no timeout behavior is implied unless implemented.

### Folding Workflow Critique

The current file-persistence advice is directionally right but too heavy if
presented as universal advice. Better distinction:

- `handoff_notes` can serve as the durable handoff for small or medium findings,
  especially if it includes source URLs and key facts.
- Persist to a file when evidence is too large, too detailed, or likely to be
  quoted/reused later.
- For long-context models, the cost/benefit of folding should be decided by task
  shape, not reflexively enabled.

This reinforces the earlier idea that folding guidance belongs near the folding
mode switch and should explain when to enable or disable it.

### Top Three Fixes According To DeepSeek

If only three changes are made next:

1. Rename `exploratory`/`determinate` to behavior-first names such as
   `manual_review`/`auto_complete`.
2. Remove redundant `read_completed_task_context` parameters (`task_id`,
   `message_range`).
3. Revisit the default parent kind or make the guidance much stronger so models
   choose automatic completion for closed-scope parent tasks.

### Review Score Summary

DeepSeek's rough ratings:

| Dimension | Rating | Notes |
| --- | --- | --- |
| Concept completeness | 4/5 | The parent completion model is smart. |
| Readability | 2/5 | Too long and insufficiently structured. |
| Naming intuitiveness | 2/5 | `exploratory`, `determinate`, and `abandon` are confusing. |
| Parameter consistency | 3/5 | `task_id` and `message_range` add redundancy. |
| Default behavior | 2/5 | Manual-review default causes many interruptions. |
| Edge-case coverage | 4/5 | Better than most system prompts. |
| Overall usability | 3/5 | Powerful but cognitively heavy. |
