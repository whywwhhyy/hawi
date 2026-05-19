"""Hawi permission system — first-class permission modelling.

The permission system provides:

- **Stable permission ids** (:class:`PermissionId`): namespaced identifiers
  like ``"filesystem:read"`` that plugins use to declare what they need.
- **Permission policies** (:class:`PermissionPolicy`): ``allow`` / ``deny`` /
  ``human_review`` / ``agent_review`` with first-phase fallback semantics.
- **Permission sets** (:class:`PermissionSet`): runtime configuration that
  maps permission ids to policies per agent instance.
- **Tool filtering**: :class:`PluginManager` uses the active permission set
  to hide denied tools from model tool definitions.
- **Execution gating**: :class:`ToolExecutor` performs a secondary
  permission check before running any tool.
- **Audit trails**: every permission decision is recorded for observability.

Quick start::

    from hawi.permission import PermissionSet

    ps = PermissionSet()
    ps.allow("filesystem:read").allow("network:fetch").deny("shell:execute")
    agent = HawiAgent(model=model, permission_set=ps)
"""

from .types import (
    Permission,
    PermissionAuditRecord,
    PermissionDeclared,
    PermissionId,
    PermissionPolicy,
    PermissionSet,
    FrozenPermissionSet,
    RiskLevel,
    WELL_KNOWN_PERMISSIONS,
)
from .checker import (
    PermissionChecker,
    build_tool_permission_map,
    filter_tools,
)
from .declaration import (
    collect_plugin_permissions,
    PermissionDeclarer,
)
from .audit import PermissionAuditSink

__all__ = [
    # --- types ---
    "Permission",
    "PermissionAuditRecord",
    "PermissionDeclared",
    "PermissionId",
    "PermissionPolicy",
    "PermissionSet",
    "FrozenPermissionSet",
    "RiskLevel",
    "WELL_KNOWN_PERMISSIONS",
    # --- checker ---
    "PermissionChecker",
    "build_tool_permission_map",
    "filter_tools",
    # --- declaration ---
    "collect_plugin_permissions",
    "PermissionDeclarer",
    # --- audit ---
    "PermissionAuditSink",
]
