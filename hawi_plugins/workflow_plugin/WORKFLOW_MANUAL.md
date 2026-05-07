# Workflow Plugin for Hawi — Manual

## Overview

A **workflow** is a directed acyclic graph (DAG) of **gates** (nodes).
The agent must pass through each gate in order, submitting its output for review.
Only after the reviewer approves does the workflow advance to the next gate.

This is NOT a suggestion system (like Skills). Workflows **enforce** quality gates.

---

## Concepts

| Term | Meaning |
|------|---------|
| **Workflow** | A complete DAG of gates. Defined as a YAML file. |
| **Gate (Node)** | A single checkpoint. Agent works on the task, calls `complete_workflow_node`, output is reviewed. |
| **Edge** | Directed connection: `from` gate → `to` gate. Defines the flow. |
| **Review** | Each gate has a review configuration. Output is evaluated before advancing. |
| **Run** | One execution of a workflow. Tracks current gate, outputs, review decisions. |

---

## YAML Format

Workflows are stored as YAML files in `~/.hawi/workflows/{name}.yaml`.

```yaml
id: my_workflow_id          # Unique identifier (use snake_case)
name: My Workflow           # Human-readable name (used in list_workflows)
description: >              # Optional: what this workflow accomplishes
  A description of the workflow's purpose.
start_node_id: first_gate   # The id of the entry gate

nodes:                      # Array of gate definitions
  - id: first_gate          # Unique within this workflow (use snake_case)
    name: First Gate        # Human-readable gate name
    description: >          # Optional: what this gate does
      Detailed description.
    prompt: >               # Task instructions injected into agent's system prompt
      You are at the First Gate. Your task is to...
    review_type: logger     # How to review output: logger | sub_agent | human
    sub_agent_prompt: >     # Required if review_type is sub_agent
      Check that the output meets these criteria...
    sub_agent_model: null   # Optional: model to use for review sub-agent
    max_retries: 3          # Max review rejections before workflow fails (default 3)

edges:                      # Directed connections between gates
  - from: first_gate        # Source gate id
    to: second_gate         # Target gate id
    label: approved         # Optional human-readable label
```

---

## Review Types

### `logger`
- Automatically approves all outputs.
- Every decision is recorded in the audit trail.
- Use for: gates where human review is unnecessary but you want a record.

### `sub_agent`
- Spawns a separate agent (via `agent.clone()`) to evaluate the output.
- The review agent reads the gate's prompt, your output, and the `sub_agent_prompt`.
- It responds with `{"approved": true/false, "feedback": "..."}`.
- Use for: automated quality checks, catching hallucinations, verifying completeness.

**Important**: The `sub_agent_prompt` field is required. Be specific about what the reviewer should check. Examples:
- `"Verify the response contains no factual errors. Check all claims against the provided documents."`
- `"Ensure the code has no security vulnerabilities (OWASP Top 10)."`
- `"Check that all requested sections are present and well-written."`

### `human`
- Pauses the workflow and waits for a human to approve or reject via GUI/CLI.
- The human sees your output and can provide feedback.
- Use for: critical decisions, compliance requirements, deployment approvals.

### `none`
- Same as `logger` (auto-approve + log). Kept for backward compatibility.

---

## Tools

### Agent Tools (8)

| Tool | When to use |
|------|------------|
| `read_workflow_manual` | Read this guide before writing/loading workflow YAML |
| `list_workflows` | Discover available workflows on disk |
| `load_workflow(name)` | Load & validate a YAML file before running |
| `run_workflow(name, initial_input?)` | Start executing a workflow |
| `select_next_workflow_node(next_node_id, reason)` | Choose an immediate downstream gate and record why |
| `complete_workflow_node(output)` | Submit your output for the current gate |
| `get_workflow_status` | Check progress: which gates are done/active/pending |
| `get_pending_reviews` | Check if any gates are awaiting review |

### Human Tools (GUI/CLI only — NOT available to agent)

| Method | Purpose |
|--------|---------|
| `approve_workflow_node(review_id, feedback?, modified_output?)` | Human approves a gate |
| `reject_workflow_node(review_id, feedback)` | Human rejects with feedback |

---

## Typical Workflow

### 1. Creating a Workflow

Use filesystem tools to write a YAML file to `~/.hawi/workflows/{name}.yaml`.
Then call `load_workflow(name)` to validate it. Fix any errors and retry.

```
Agent writes YAML → load_workflow validates → fix errors → load again → ready
```

### 2. Running a Workflow

```
run_workflow("My Workflow", initial_input="Review file X")
  → Gate prompt injected into system prompt
  → Agent works on the gate's task
  → If multiple downstream gates are available, agent calls select_next_workflow_node(...)
  → Agent calls complete_workflow_node(output="...")
  → Output is reviewed (logger/sub_agent/human)
  → If approved: advances to the selected gate, the default downstream gate, or completes
  → If rejected: agent receives feedback and must revise
```

### 3. During Execution

- Use `get_workflow_status` to see which gates are done, active, or pending.
- Previous gate outputs are available as context in the current gate's prompt.
- For branching workflows, call `select_next_workflow_node(next_node_id, reason)` before `complete_workflow_node`.
- Use `get_pending_reviews` to check if a gate is waiting for human approval.

### 4. Handling Rejection

If your output is rejected:
1. Read the reviewer's feedback carefully.
2. Revise your output to address the feedback.
3. Call `complete_workflow_node` again with the improved output.
4. You have `max_retries` attempts. After that, the workflow fails.

---

## Best Practices

### Gate Design
- **Keep gates focused**: Each gate should have one clear objective.
- **Order gates logically**: Earlier gates produce context for later gates.
- **Use logger for low-risk gates**: Keeps the workflow moving.
- **Use sub_agent for quality checks**: Write clear `sub_agent_prompt` criteria.
- **Use human for critical decisions**: Deployment, compliance, legal review.

### Writing Prompts
- Be specific about what the agent should produce.
- Mention the output format if needed.
- Reference upstream gate outputs that are relevant.
- Include any constraints or requirements.

### Review Criteria
- For `sub_agent`: be specific. "Check for X, Y, Z. Reject if any are missing."
- For `human`: the human sees the output and the gate name. Make the gate name descriptive.

### Audit Trail
- Every review decision is recorded: reviewer type, approved/rejected, feedback, timestamp.
- Use `get_workflow_status` to view the complete audit trail.
- LoggerReviewer records make post-hoc audits possible even without human review.

---

## Differences from Skills

| Aspect | Skills | Workflows |
|--------|--------|-----------|
| Control | Advisory ("you should") | Enforced ("you must pass this gate") |
| Quality | Agent self-judges | Independent reviewer per gate |
| Progress | Invisible | Visible gate-by-gate |
| Audit | None | Complete review trail |
| Failure | Agent may skip steps | Gate retries → workflow fails |
| Human-in-loop | Not available | Native human review gates |
| Composition | One big SKILL.md | Reusable DAG of gates |
