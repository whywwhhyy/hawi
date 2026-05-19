"""Permission checking logic for tool filtering and execution gating."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Sequence

from .types import (
    Permission,
    PermissionAuditRecord,
    PermissionDeclared,
    PermissionPolicy,
    PermissionSet,
    FrozenPermissionSet,
)

if TYPE_CHECKING:
    from hawi.tool.types import AgentTool


class PermissionChecker:
    """Central permission decision engine.

    Used by :class:`~hawi.plugin.manager.PluginManager` to filter tools and
    by :class:`~hawi.agent.tool_executor.ToolExecutor` to gate execution.
    """

    def __init__(
        self,
        *,
        permission_set: PermissionSet | FrozenPermissionSet | None = None,
    ) -> None:
        self._permission_set: PermissionSet | FrozenPermissionSet | None = permission_set
        self._audit_records: list[PermissionAuditRecord] = []

    # --- permission set management ---

    @property
    def permission_set(self) -> PermissionSet | FrozenPermissionSet | None:
        return self._permission_set

    def set_permission_set(
        self,
        ps: PermissionSet | FrozenPermissionSet | None,
    ) -> None:
        """Replace the active permission set."""
        self._permission_set = ps

    # --- tool filtering ---

    def is_tool_allowed(
        self,
        tool: "AgentTool",
        *,
        tool_permissions: dict[str, Sequence[PermissionDeclared]],
        plugin_id: str = "",
    ) -> bool:
        """Return True if *tool* passes the active permission set.

        A tool is allowed when **all** of its required permissions have
        an effective policy of ``allow`` under the active set.  If no
        permission set is configured, all tools are allowed (backwards
        compatible).

        Args:
            tool: The tool to check.
            tool_permissions: ``{tool_name: [PermissionDeclared, ...]}`` mapping
                built by the PluginManager from plugin declarations.
            plugin_id: The plugin that owns the tool (for audit context).
        """
        if self._permission_set is None:
            return True

        declared_list = tool_permissions.get(tool.name, [])
        if not declared_list:
            # Tool has no declared permissions → allowed (opt-in model)
            return True

        for declared in declared_list:
            if not self._check_single(declared.permission):
                self._record_audit(
                    declared.permission,
                    tool_name=tool.name,
                    plugin_id=plugin_id,
                    allowed=False,
                )
                return False
            self._record_audit(
                declared.permission,
                tool_name=tool.name,
                plugin_id=plugin_id,
                allowed=True,
            )
        return True

    def check_tool_permission(
        self,
        tool_name: str,
        *,
        tool_permissions: dict[str, Sequence[PermissionDeclared]],
        plugin_id: str = "",
    ) -> PermissionPolicy:
        """Return the *most restrictive* effective policy for *tool_name*.

        Uses :meth:`PermissionPolicy.effective_phase1` so that
        ``human_review`` / ``agent_review`` are mapped to ``allow``
        (tool visible to model).  Use :meth:`check_tool_permission_raw`
        when you need the unmapped original policy for execution gating.

        Returns ``allow`` when:
        - No permission set is configured, or
        - The tool has no declared permissions, or
        - All declared permissions resolve to ``allow``.

        Otherwise returns ``deny``.
        """
        if self._permission_set is None:
            return PermissionPolicy.allow

        declared_list = tool_permissions.get(tool_name, [])
        if not declared_list:
            return PermissionPolicy.allow

        for declared in declared_list:
            if not self._check_single(declared.permission):
                return PermissionPolicy.deny
        return PermissionPolicy.allow

    def check_tool_permission_raw(
        self,
        tool_name: str,
        *,
        tool_permissions: dict[str, Sequence[PermissionDeclared]],
    ) -> PermissionPolicy:
        """Return the *original* policy for *tool_name* (no phase1 mapping).

        Unlike :meth:`check_tool_permission`, this returns the unmapped
        policy so callers (e.g. :class:`ToolExecutor`) can distinguish
        ``human_review`` from ``allow`` for execution gating.
        """
        if self._permission_set is None:
            return PermissionPolicy.allow

        declared_list = tool_permissions.get(tool_name, [])
        if not declared_list:
            return PermissionPolicy.allow

        most_restrictive = PermissionPolicy.allow
        for declared in declared_list:
            original = self._permission_set.resolve_raw(
                str(declared.permission.id), declared=declared.permission
            )
            # Keep the original (unmapped) policy — human_review stays human_review
            # _rank for ordering: deny=3, human_review=2, agent_review=1, allow=0
            def _rank(p: PermissionPolicy) -> int:
                if p == PermissionPolicy.deny: return 3
                if p == PermissionPolicy.human_review: return 2
                if p == PermissionPolicy.agent_review: return 1
                return 0
            if _rank(original) > _rank(most_restrictive):
                most_restrictive = original
        return most_restrictive

    # --- audit ---

    @property
    def audit_records(self) -> list[PermissionAuditRecord]:
        """Return all audit records collected so far."""
        return list(self._audit_records)

    def clear_audit(self) -> None:
        """Clear collected audit records."""
        self._audit_records.clear()

    def pop_audit_records(self) -> list[PermissionAuditRecord]:
        """Return and clear audit records."""
        records = self._audit_records
        self._audit_records = []
        return records

    # --- internal ---

    def _check_single(self, permission: Permission) -> bool:
        """Check whether a single permission should be allowed."""
        if self._permission_set is None:
            return True
        policy = self._permission_set.resolve(str(permission.id), declared=permission)
        return policy == PermissionPolicy.allow

    def _record_audit(
        self,
        permission: Permission,
        *,
        tool_name: str,
        plugin_id: str = "",
        allowed: bool,
        tool_call_id: str = "",
        run_id: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> None:
        policy = (
            self._permission_set.resolve(str(permission.id), declared=permission)
            if self._permission_set
            else permission.default_policy
        )
        configured = (
            self._permission_set.get(str(permission.id))
            if self._permission_set
            else None
        )

        if allowed:
            if policy == PermissionPolicy.agent_review:
                decision = "allowed_agent_review"
            else:
                decision = "allowed"
        else:
            if policy == PermissionPolicy.human_review:
                decision = "denied_human_review"
            else:
                decision = "denied"

        self._audit_records.append(
            PermissionAuditRecord(
                permission_id=permission.id,
                tool_name=tool_name,
                effective_policy=policy,
                decision=decision,
                plugin_id=plugin_id,
                tool_call_id=tool_call_id,
                run_id=run_id,
                agent_id=agent_id,
                session_id=session_id,
                timestamp=time.time(),
                declared_policy=permission.default_policy.value,
                configured_policy=configured.value if configured else "",
            )
        )


# ---------------------------------------------------------------------------
# Module-level helper utilities
# ---------------------------------------------------------------------------


def build_tool_permission_map(
    all_declarations: Sequence[PermissionDeclared],
) -> dict[str, list[PermissionDeclared]]:
    """Build a ``{tool_name: [PermissionDeclared, ...]}`` map.

    Scope-level declarations (those with empty ``tool_names``) are skipped
    because they are not yet linked to specific tools.  Plugins that want
    per-tool filtering must provide tool-level declarations.
    """
    result: dict[str, list[PermissionDeclared]] = {}
    for decl in all_declarations:
        if not decl.tool_names:
            continue
        for tool_name in decl.tool_names:
            result.setdefault(tool_name, []).append(decl)
    return result


def filter_tools(
    tools: list["AgentTool"],
    checker: PermissionChecker,
    tool_permissions: dict[str, Sequence[PermissionDeclared]],
    *,
    plugin_id: str = "",
) -> list["AgentTool"]:
    """Return *tools* with permission-denied entries removed.

    When no permission set is active (``checker.permission_set is None``),
    all tools pass through unchanged.
    """
    if checker.permission_set is None:
        return tools
    return [
        t
        for t in tools
        if checker.is_tool_allowed(
            t,
            tool_permissions=tool_permissions,
            plugin_id=plugin_id,
        )
    ]
