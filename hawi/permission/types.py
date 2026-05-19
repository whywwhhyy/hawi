"""Core permission types for the Hawi permission system.

This module defines the fundamental types for:
- Permission identifiers (stable, namespaced)
- Permission policies (allow / deny / human_review / agent_review)
- Permission descriptors (what plugins declare)
- Permission sets (runtime permission configuration)
- Audit records (permission decision tracking)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# PermissionId — stable, namespaced identifier
# ---------------------------------------------------------------------------

class PermissionId(str):
    """A stable, namespaced permission identifier.

    Format: ``"scope:capability"``

    Examples:
        ``"filesystem:read"``, ``"shell:execute"``, ``"network:fetch"``,
        ``"subagent:spawn"``, ``"python:execute"``
    """

    __slots__ = ()

    def __new__(cls, value: str) -> "PermissionId":
        if ":" not in value:
            raise ValueError(
                f"PermissionId must use 'scope:capability' format, got: {value!r}"
            )
        return super().__new__(cls, value)

    @property
    def scope(self) -> str:
        """The scope portion of the permission id (e.g. ``"filesystem"``)."""
        return self.split(":", 1)[0]

    @property
    def capability(self) -> str:
        """The capability portion of the permission id (e.g. ``"read"``)."""
        return self.split(":", 1)[1]


# ---------------------------------------------------------------------------
# PermissionPolicy — what to do when a permission is requested
# ---------------------------------------------------------------------------

class PermissionPolicy(str, Enum):
    """Policy for a specific permission.

    .. list-table:: First-phase behaviour
       :header-rows: 1

       * - Policy
         - Phase 1 behaviour
       * - ``allow``
         - Tool appears in definitions and executes normally.
       * - ``deny``
         - Tool is hidden from model and rejected at execution.
       * - ``human_review``
         - Treated as ``deny``; audit record is marked for future approval.
       * - ``agent_review``
         - Treated as ``allow``; audit record is marked for future review.

    Future phases will implement the full human/agent review flows.
    """

    allow = "allow"
    deny = "deny"
    human_review = "human_review"
    agent_review = "agent_review"

    def effective_phase1(self) -> "PermissionPolicy":
        """Return the effective policy under first-phase semantics."""
        if self is PermissionPolicy.allow:
            return PermissionPolicy.allow
        if self is PermissionPolicy.deny:
            return PermissionPolicy.deny
        if self is PermissionPolicy.human_review:
            return PermissionPolicy.deny  # first phase: deny
        if self is PermissionPolicy.agent_review:
            return PermissionPolicy.allow  # first phase: allow
        return self


# ---------------------------------------------------------------------------
# RiskLevel — how dangerous a permission is
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Risk categorization for a permission."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# ---------------------------------------------------------------------------
# Permission — a declared permission descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Permission:
    """A permission descriptor declared by a plugin or the framework.

    Each permission has a stable *id*, a human-readable *scope* and
    *description*, a *risk_level*, and a *default_policy* that applies
    when no explicit policy is configured.
    """

    id: PermissionId
    """Stable namespaced identifier, e.g. ``"filesystem:read"``."""

    description: str = ""
    """Human-readable explanation of what this permission grants."""

    risk_level: RiskLevel = RiskLevel.medium
    """How dangerous granting this permission is."""

    default_policy: PermissionPolicy = PermissionPolicy.deny
    """Policy used when the permission is not explicitly configured."""

    tags: tuple[str, ...] = ()
    """Optional tags for categorization in GUI / tooling."""

    def __post_init__(self) -> None:
        # Ensure id is a PermissionId
        if not isinstance(self.id, PermissionId):
            object.__setattr__(self, "id", PermissionId(self.id))


# ---------------------------------------------------------------------------
# PermissionDeclared — what a plugin declares
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermissionDeclared:
    """A permission declaration from a plugin, linking to specific tools.

    Plugins return a sequence of these from their :meth:`HawiPlugin.permissions`
    property.  Each entry says "these tools require this permission".
    """

    permission: Permission
    """The permission being declared."""

    tool_names: tuple[str, ...] = ()
    """Tool names that require this permission.

    An empty tuple means this permission is a *scope-level* declaration
    that applies to all tools owned by the plugin.
    """

    def __post_init__(self) -> None:
        if isinstance(self.tool_names, list):
            object.__setattr__(self, "tool_names", tuple(self.tool_names))


# ---------------------------------------------------------------------------
# PermissionSet — mutable runtime permission configuration
# ---------------------------------------------------------------------------

@dataclass
class PermissionSet:
    """A mutable map of permission id → policy for one agent instance.

    Create with an optional initial mapping:

        ps = PermissionSet({"filesystem:read": "allow"})
        ps.allow("filesystem:write")
        ps.deny("shell:execute")
        print(ps.resolve("filesystem:read"))   # PermissionPolicy.allow
    """

    _policies: dict[PermissionId, PermissionPolicy] = field(default_factory=dict)

    # --- builders ---

    def allow(self, permission_id: str) -> "PermissionSet":
        """Set policy to *allow* for *permission_id*."""
        self._policies[PermissionId(permission_id)] = PermissionPolicy.allow
        return self

    def deny(self, permission_id: str) -> "PermissionSet":
        """Set policy to *deny* for *permission_id*."""
        self._policies[PermissionId(permission_id)] = PermissionPolicy.deny
        return self

    def human_review(self, permission_id: str) -> "PermissionSet":
        """Mark *permission_id* for human review (deny in phase 1)."""
        self._policies[PermissionId(permission_id)] = PermissionPolicy.human_review
        return self

    def agent_review(self, permission_id: str) -> "PermissionSet":
        """Mark *permission_id* for agent review (allow in phase 1)."""
        self._policies[PermissionId(permission_id)] = PermissionPolicy.agent_review
        return self

    def set_policy(self, permission_id: str, policy: PermissionPolicy | str) -> "PermissionSet":
        """Set an explicit policy for *permission_id*."""
        pid = PermissionId(permission_id)
        if isinstance(policy, str):
            policy = PermissionPolicy(policy)
        self._policies[pid] = policy
        return self

    def remove(self, permission_id: str) -> "PermissionSet":
        """Remove any explicit policy, reverting to default."""
        self._policies.pop(PermissionId(permission_id), None)
        return self

    def clear(self) -> "PermissionSet":
        """Remove all explicit policies."""
        self._policies.clear()
        return self

    # --- bulk ---

    def merge(self, other: "PermissionSet | FrozenPermissionSet") -> "PermissionSet":
        """Merge policies from another set (overwrites on conflict)."""
        for pid, policy in other.items():
            self._policies[pid] = policy
        return self

    @classmethod
    def from_dict(
        cls,
        mapping: Mapping[str, str],
    ) -> "PermissionSet":
        """Create a PermissionSet from a plain dict of ``{id: policy}``."""
        ps = cls()
        for pid, policy in mapping.items():
            ps.set_policy(pid, PermissionPolicy(policy))
        return ps

    def clone(self) -> "PermissionSet":
        """Return an independent deep copy."""
        return PermissionSet(_policies=dict(self._policies))

    # --- query ---

    def resolve(
        self,
        permission_id: str,
        declared: Permission | None = None,
    ) -> PermissionPolicy:
        """Return the effective policy for *permission_id*.

        If the id is explicitly configured, its policy is used.  Otherwise
        falls back to *declared.default_policy*, and finally to ``deny``.
        """
        pid = PermissionId(permission_id)
        if pid in self._policies:
            return self._policies[pid].effective_phase1()
        if declared is not None:
            return declared.default_policy.effective_phase1()
        return PermissionPolicy.deny

    def get(self, permission_id: str) -> PermissionPolicy | None:
        """Return the explicit policy or None if not configured."""
        return self._policies.get(PermissionId(permission_id))

    def items(self):
        """Iterate over ``(PermissionId, PermissionPolicy)`` pairs."""
        return self._policies.items()

    def has(self, permission_id: str) -> bool:
        """Return whether an explicit policy exists for *permission_id*."""
        return PermissionId(permission_id) in self._policies

    def is_allowed(
        self,
        permission_id: str,
        declared: Permission | None = None,
    ) -> bool:
        """Return True if the effective policy for *permission_id* is *allow*."""
        return self.resolve(permission_id, declared) == PermissionPolicy.allow

    def __len__(self) -> int:
        return len(self._policies)

    def __contains__(self, permission_id: str) -> bool:
        return PermissionId(permission_id) in self._policies

    def __repr__(self) -> str:
        policies = {str(k): v.value for k, v in self._policies.items()}
        return f"PermissionSet({policies})"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe dict representation."""
        return {str(k): v.value for k, v in self._policies.items()}

    def freeze(self) -> "FrozenPermissionSet":
        """Return an immutable snapshot."""
        return FrozenPermissionSet(_policies=dict(self._policies))

    # --- JSON support ---

    def to_json(self) -> dict[str, str]:
        """Alias for :meth:`to_dict`."""
        return self.to_dict()

    @classmethod
    def from_json(cls, data: dict[str, str]) -> "PermissionSet":
        """Alias for :meth:`from_dict`."""
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# FrozenPermissionSet — immutable snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenPermissionSet:
    """An immutable snapshot of a :class:`PermissionSet`.

    Created via :meth:`PermissionSet.freeze`.
    """

    _policies: dict[PermissionId, PermissionPolicy] = field(default_factory=dict)

    def resolve(
        self,
        permission_id: str,
        declared: Permission | None = None,
    ) -> PermissionPolicy:
        pid = PermissionId(permission_id)
        if pid in self._policies:
            return self._policies[pid].effective_phase1()
        if declared is not None:
            return declared.default_policy.effective_phase1()
        return PermissionPolicy.deny

    def get(self, permission_id: str) -> PermissionPolicy | None:
        return self._policies.get(PermissionId(permission_id))

    def items(self):
        return self._policies.items()

    def is_allowed(
        self,
        permission_id: str,
        declared: Permission | None = None,
    ) -> bool:
        return self.resolve(permission_id, declared) == PermissionPolicy.allow

    def to_dict(self) -> dict[str, str]:
        return {str(k): v.value for k, v in self._policies.items()}

    def to_json(self) -> dict[str, str]:
        return self.to_dict()

    def __len__(self) -> int:
        return len(self._policies)

    def __repr__(self) -> str:
        policies = {str(k): v.value for k, v in self._policies.items()}
        return f"FrozenPermissionSet({policies})"


# ---------------------------------------------------------------------------
# PermissionAuditRecord — what happened and why
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermissionAuditRecord:
    """An immutable record of a permission decision for a single tool call.

    Created by :class:`PermissionChecker` before/after tool execution and
    collected by the agent's audit sink for observability.
    """

    permission_id: PermissionId
    """The permission being checked."""

    tool_name: str
    """The tool that triggered the check."""

    effective_policy: PermissionPolicy
    """The effective policy used for the decision."""

    decision: str
    """``"allowed"``, ``"denied"``, ``"denied_human_review"``, or
    ``"allowed_agent_review"``."""

    agent_id: str = ""
    """Agent instance id (runtime only)."""

    session_id: str = ""
    """Session id (runtime only)."""

    plugin_id: str = ""
    """Plugin that declared the permission."""

    tool_call_id: str = ""
    """The provider tool-call id (runtime only)."""

    run_id: str = ""
    """The agent run id (runtime only)."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp of the record."""

    declared_policy: str = ""
    """The declared default policy at declaration time."""

    configured_policy: str = ""
    """The explicitly configured policy, if any."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "permission_id": str(self.permission_id),
            "tool_name": self.tool_name,
            "effective_policy": self.effective_policy.value,
            "decision": self.decision,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "plugin_id": self.plugin_id,
            "tool_call_id": self.tool_call_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "declared_policy": self.declared_policy,
            "configured_policy": self.configured_policy,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Predefined well-known permissions
# ---------------------------------------------------------------------------

# These are the canonical permission ids used by built-in plugins.
# External plugins SHOULD use these ids where applicable.

WELL_KNOWN_PERMISSIONS: dict[str, Permission] = {
    # --- Filesystem ---
    "filesystem:read": Permission(
        id=PermissionId("filesystem:read"),
        description="Read files and directories on the local filesystem.",
        risk_level=RiskLevel.low,
        default_policy=PermissionPolicy.allow,
        tags=("filesystem", "io"),
    ),
    "filesystem:write": Permission(
        id=PermissionId("filesystem:write"),
        description="Create, modify, edit, or delete files and directories.",
        risk_level=RiskLevel.high,
        default_policy=PermissionPolicy.deny,
        tags=("filesystem", "io", "mutation"),
    ),
    # --- Shell ---
    "shell:execute": Permission(
        id=PermissionId("shell:execute"),
        description="Execute arbitrary shell commands on the host system.",
        risk_level=RiskLevel.critical,
        default_policy=PermissionPolicy.deny,
        tags=("shell", "execution"),
    ),
    # --- Network ---
    "network:fetch": Permission(
        id=PermissionId("network:fetch"),
        description="Make HTTP/HTTPS requests to external services.",
        risk_level=RiskLevel.medium,
        default_policy=PermissionPolicy.allow,
        tags=("network", "io"),
    ),
    # --- Python interpreter ---
    "python:execute": Permission(
        id=PermissionId("python:execute"),
        description="Execute arbitrary Python code in a persistent interpreter.",
        risk_level=RiskLevel.high,
        default_policy=PermissionPolicy.deny,
        tags=("python", "execution"),
    ),
    # --- SubAgent ---
    "subagent:spawn": Permission(
        id=PermissionId("subagent:spawn"),
        description="Create sub-agents with their own runners and contexts.",
        risk_level=RiskLevel.medium,
        default_policy=PermissionPolicy.allow,
        tags=("subagent", "orchestration"),
    ),
    "subagent:send": Permission(
        id=PermissionId("subagent:send"),
        description="Send messages to running sub-agents.",
        risk_level=RiskLevel.low,
        default_policy=PermissionPolicy.allow,
        tags=("subagent", "communication"),
    ),
    "subagent:close": Permission(
        id=PermissionId("subagent:close"),
        description="Interrupt and close sub-agents.",
        risk_level=RiskLevel.low,
        default_policy=PermissionPolicy.allow,
        tags=("subagent", "lifecycle"),
    ),
    # --- MCP ---
    "mcp:connect": Permission(
        id=PermissionId("mcp:connect"),
        description="Connect to external MCP servers.",
        risk_level=RiskLevel.medium,
        default_policy=PermissionPolicy.deny,
        tags=("mcp", "network"),
    ),
    # --- Workflow ---
    "workflow:manage": Permission(
        id=PermissionId("workflow:manage"),
        description="Create, gate, and complete workflow steps.",
        risk_level=RiskLevel.low,
        default_policy=PermissionPolicy.allow,
        tags=("workflow", "orchestration"),
    ),
    # --- Skills ---
    "skills:use": Permission(
        id=PermissionId("skills:use"),
        description="Discover and invoke Claude-style skills.",
        risk_level=RiskLevel.low,
        default_policy=PermissionPolicy.allow,
        tags=("skills", "orchestration"),
    ),
}
