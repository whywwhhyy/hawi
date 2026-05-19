"""Tests for the Hawi permission system (first-class citizen)."""

from __future__ import annotations

import pytest

from hawi.permission import (
    Permission,
    PermissionAuditRecord,
    PermissionAuditSink,
    PermissionChecker,
    PermissionDeclared,
    PermissionId,
    PermissionPolicy,
    PermissionSet,
    FrozenPermissionSet,
    RiskLevel,
    WELL_KNOWN_PERMISSIONS,
    build_tool_permission_map,
    collect_plugin_permissions,
    filter_tools,
)
from hawi.plugin import HawiPlugin, PluginManager, tool
from hawi.tool.types import AgentTool


# ============================================================================
# PermissionId
# ============================================================================

class TestPermissionId:
    def test_valid_formats(self):
        pid = PermissionId("filesystem:read")
        assert str(pid) == "filesystem:read"
        assert pid.scope == "filesystem"
        assert pid.capability == "read"

    def test_rejects_missing_colon(self):
        with pytest.raises(ValueError, match="scope:capability"):
            PermissionId("invalid")

    def test_equality(self):
        a = PermissionId("a:b")
        b = PermissionId("a:b")
        assert a == b
        assert hash(a) == hash(b)


# ============================================================================
# PermissionPolicy
# ============================================================================

class TestPermissionPolicy:
    def test_effective_phase1_allow(self):
        assert PermissionPolicy.allow.effective_phase1() == PermissionPolicy.allow

    def test_effective_phase1_deny(self):
        assert PermissionPolicy.deny.effective_phase1() == PermissionPolicy.deny

    def test_effective_phase1_human_review_is_deny(self):
        assert PermissionPolicy.human_review.effective_phase1() == PermissionPolicy.deny

    def test_effective_phase1_agent_review_is_allow(self):
        assert PermissionPolicy.agent_review.effective_phase1() == PermissionPolicy.allow


# ============================================================================
# PermissionSet
# ============================================================================

class TestPermissionSet:
    def test_empty_resolves_to_deny(self):
        ps = PermissionSet()
        assert ps.resolve("unknown:perm") == PermissionPolicy.deny

    def test_explicit_allow(self):
        ps = PermissionSet().allow("filesystem:read")
        assert ps.resolve("filesystem:read") == PermissionPolicy.allow

    def test_explicit_deny(self):
        ps = PermissionSet().deny("shell:execute")
        assert ps.resolve("shell:execute") == PermissionPolicy.deny

    def test_human_review_resolves_to_deny_phase1(self):
        ps = PermissionSet().human_review("python:execute")
        assert ps.resolve("python:execute") == PermissionPolicy.deny

    def test_agent_review_resolves_to_allow_phase1(self):
        ps = PermissionSet().agent_review("network:fetch")
        assert ps.resolve("network:fetch") == PermissionPolicy.allow

    def test_resolve_falls_back_to_declared_default(self):
        perm = Permission(id="test:perm", default_policy=PermissionPolicy.allow, risk_level=RiskLevel.low)
        ps = PermissionSet()
        assert ps.resolve("test:perm", declared=perm) == PermissionPolicy.allow

    def test_freeze_roundtrip(self):
        ps = PermissionSet().allow("a:b").deny("c:d")
        frozen = ps.freeze()
        assert frozen.resolve("a:b") == PermissionPolicy.allow
        assert frozen.resolve("c:d") == PermissionPolicy.deny

    def test_to_dict_from_dict_roundtrip(self):
        original = PermissionSet().allow("filesystem:read").deny("shell:execute")
        data = original.to_dict()
        restored = PermissionSet.from_dict(data)
        assert restored.resolve("filesystem:read") == PermissionPolicy.allow
        assert restored.resolve("shell:execute") == PermissionPolicy.deny

    def test_is_allowed(self):
        ps = PermissionSet().allow("a:b")
        assert ps.is_allowed("a:b")
        assert not ps.is_allowed("c:d")

    def test_merge(self):
        a = PermissionSet().allow("a:b")
        b = PermissionSet().deny("c:d")
        a.merge(b)
        assert a.resolve("a:b") == PermissionPolicy.allow
        assert a.resolve("c:d") == PermissionPolicy.deny

    def test_clone(self):
        a = PermissionSet().allow("a:b")
        b = a.clone()
        b.deny("a:b")
        assert a.resolve("a:b") == PermissionPolicy.allow  # original unchanged
        assert b.resolve("a:b") == PermissionPolicy.deny


# ============================================================================
# Permission / PermissionDeclared
# ============================================================================

class TestPermissionDeclaration:
    def test_permission_declared_creation(self):
        perm = Permission(id="filesystem:read", description="Read files", risk_level=RiskLevel.low)
        decl = PermissionDeclared(permission=perm, tool_names=["read_file", "glob"])
        assert decl.permission.id == "filesystem:read"
        assert decl.tool_names == ("read_file", "glob")

    def test_collect_plugin_permissions(self):
        class PluginA(HawiPlugin):
            name = "plugin_a"

            @property
            def permissions(self):
                return [
                    PermissionDeclared(
                        permission=WELL_KNOWN_PERMISSIONS["filesystem:read"],
                        tool_names=["read_file"],
                    ),
                ]

        class PluginB(HawiPlugin):
            name = "plugin_b"

            @property
            def permissions(self):
                return [
                    PermissionDeclared(
                        permission=WELL_KNOWN_PERMISSIONS["shell:execute"],
                        tool_names=["run_shell"],
                    ),
                ]

        plugins = [PluginA(), PluginB()]
        all_decls = collect_plugin_permissions(plugins)
        assert len(all_decls) == 2
        ids = {str(d.permission.id) for d in all_decls}
        assert ids == {"filesystem:read", "shell:execute"}


# ============================================================================
# PermissionChecker
# ============================================================================

class TestPermissionChecker:
    def test_no_permission_set_allows_all(self):
        checker = PermissionChecker()
        tool_perm_map = {"read_file": []}
        assert checker.check_tool_permission("read_file", tool_permissions=tool_perm_map) == PermissionPolicy.allow

    def test_deny_hides_tool(self):
        ps = PermissionSet().deny("filesystem:read")
        checker = PermissionChecker(permission_set=ps)
        perm = WELL_KNOWN_PERMISSIONS["filesystem:read"]
        tool_perm_map = {"read_file": [PermissionDeclared(permission=perm, tool_names=["read_file"])]}
        assert checker.check_tool_permission("read_file", tool_permissions=tool_perm_map) == PermissionPolicy.deny

    def test_allow_shows_tool(self):
        ps = PermissionSet().allow("filesystem:read")
        checker = PermissionChecker(permission_set=ps)
        perm = WELL_KNOWN_PERMISSIONS["filesystem:read"]
        tool_perm_map = {"read_file": [PermissionDeclared(permission=perm, tool_names=["read_file"])]}
        assert checker.check_tool_permission("read_file", tool_permissions=tool_perm_map) == PermissionPolicy.allow

    def test_unlisted_tool_allowed(self):
        ps = PermissionSet().deny("shell:execute")
        checker = PermissionChecker(permission_set=ps)
        # Tool without declarations → always allowed
        assert checker.check_tool_permission("unknown_tool", tool_permissions={}) == PermissionPolicy.allow

    def test_audit_records(self):
        ps = PermissionSet().deny("filesystem:write")
        checker = PermissionChecker(permission_set=ps)
        perm = WELL_KNOWN_PERMISSIONS["filesystem:write"]
        tool_perm_map = {"write_file": [PermissionDeclared(permission=perm, tool_names=["write_file"])]}
        # is_tool_allowed records audits (check_tool_permission is lightweight)
        class DummyTool(AgentTool):
            name = "write_file"
            description = "desc"
            parameters_schema = {"type": "object"}

        checker.is_tool_allowed(DummyTool(), tool_permissions=tool_perm_map)
        records = checker.pop_audit_records()
        assert len(records) == 1
        assert records[0].decision == "denied"


# ============================================================================
# PluginManager permission filtering
# ============================================================================

class WritePlugin(HawiPlugin):
    name = "test_write"
    display_name = "Test Write"

    @property
    def permissions(self):
        return [
            PermissionDeclared(
                permission=WELL_KNOWN_PERMISSIONS["filesystem:write"],
                tool_names=["write_file"],
            ),
        ]

    @tool()
    def write_file(self, path: str, content: str) -> dict:
        """Write content."""
        return {"ok": True}


class ReadPlugin(HawiPlugin):
    name = "test_read"
    display_name = "Test Read"

    @property
    def permissions(self):
        return [
            PermissionDeclared(
                permission=WELL_KNOWN_PERMISSIONS["filesystem:read"],
                tool_names=["read_file"],
            ),
        ]

    @tool()
    def read_file(self, path: str) -> dict:
        """Read content."""
        return {"ok": True}


class TestPluginManagerPermissions:
    """PluginManager integrates permission filtering in get_tools()."""

    def test_no_permission_set_all_tools_visible(self):
        pm = PluginManager(plugins=[WritePlugin(), ReadPlugin()])
        tools = pm.get_tools()
        assert len(tools) == 2

    def test_deny_hides_specific_tool(self):
        pm = PluginManager(plugins=[WritePlugin(), ReadPlugin()])
        pm.set_permission_set(PermissionSet().deny("filesystem:write").allow("filesystem:read"))
        tool_names = {t.name.split("__")[-1] for t in pm.get_tools()}
        assert "read_file" in tool_names
        assert "write_file" not in tool_names

    def test_allow_shows_tool(self):
        pm = PluginManager(plugins=[WritePlugin()])
        pm.set_permission_set(PermissionSet().allow("filesystem:write"))
        assert len(pm.get_tools()) == 1

    def test_none_clears_filter(self):
        pm = PluginManager(plugins=[WritePlugin()])
        pm.set_permission_set(PermissionSet().deny("filesystem:write"))
        assert len(pm.get_tools()) == 0
        pm.set_permission_set(None)
        assert len(pm.get_tools()) == 1

    def test_human_review_treated_as_deny_phase1(self):
        pm = PluginManager(plugins=[WritePlugin()])
        pm.set_permission_set(PermissionSet().human_review("filesystem:write"))
        assert len(pm.get_tools()) == 0  # hidden

    def test_agent_review_treated_as_allow_phase1(self):
        pm = PluginManager(plugins=[WritePlugin()])
        pm.set_permission_set(PermissionSet().agent_review("filesystem:write"))
        assert len(pm.get_tools()) == 1  # visible

    def test_check_tool_permission_returns_policy(self):
        pm = PluginManager(plugins=[WritePlugin()])
        pm.set_permission_set(PermissionSet().deny("filesystem:write"))
        tool_name = pm.get_tool_permissions_map().popitem()[0]  # the actual tool name
        assert pm.check_tool_permission(tool_name) == PermissionPolicy.deny

    def test_clone_preserves_permission_state(self):
        pm = PluginManager(plugins=[WritePlugin()])
        pm.set_permission_set(PermissionSet().deny("filesystem:write"))
        cloned = pm.clone()
        assert len(cloned.get_tools()) == 0

    def test_tool_definitions_respect_permissions(self):
        pm = PluginManager(plugins=[WritePlugin()])
        pm.set_permission_set(PermissionSet().deny("filesystem:write"))
        assert len(pm.get_tool_definitions()) == 0
        pm.set_permission_set(None)
        assert len(pm.get_tool_definitions()) == 1

    def test_plugin_with_no_declared_permissions_always_visible(self):
        class NoPermPlugin(HawiPlugin):
            name = "no_perm"

            @tool()
            def free_tool(self, x: str) -> str:
                """Free tool."""
                return x

        pm = PluginManager(plugins=[NoPermPlugin()])
        pm.set_permission_set(PermissionSet().deny("shell:execute"))
        assert len(pm.get_tools()) == 1  # not affected


# ============================================================================
# AuditSink
# ============================================================================

class TestPermissionAuditSink:
    def test_record_and_query(self):
        sink = PermissionAuditSink()
        record = PermissionAuditRecord(
            permission_id=PermissionId("filesystem:read"),
            tool_name="read_file",
            effective_policy=PermissionPolicy.allow,
            decision="allowed",
        )
        sink.record(record)
        assert len(sink) == 1
        assert sink.recent()[0].tool_name == "read_file"

    def test_to_json(self):
        sink = PermissionAuditSink()
        sink.record(
            PermissionAuditRecord(
                permission_id=PermissionId("a:b"),
                tool_name="t",
                effective_policy=PermissionPolicy.deny,
                decision="denied",
            )
        )
        data = sink.to_json()
        assert isinstance(data, list)
        assert data[0]["decision"] == "denied"

    def test_max_records_enforced(self):
        sink = PermissionAuditSink(max_records=3)
        for i in range(5):
            sink.record(
                PermissionAuditRecord(
                    permission_id=PermissionId(f"p:{i}"),
                    tool_name=f"t{i}",
                    effective_policy=PermissionPolicy.allow,
                    decision="allowed",
                )
            )
        assert len(sink) == 3


# ============================================================================
# filter_tools helper
# ============================================================================

class TestFilterTools:
    def test_no_permission_all_pass(self):
        from hawi.tool.types import AgentTool

        class DummyTool(AgentTool):
            name = "dummy"
            description = "desc"
            parameters_schema = {"type": "object"}

        tools = [DummyTool()]
        checker = PermissionChecker()
        result = filter_tools(tools, checker, {})
        assert len(result) == 1

    def test_deny_filters_out(self):
        class DummyTool(AgentTool):
            name = "dummy"
            description = "desc"
            parameters_schema = {"type": "object"}

        ps = PermissionSet().deny("filesystem:read")
        checker = PermissionChecker(permission_set=ps)
        perm = WELL_KNOWN_PERMISSIONS["filesystem:read"]
        tool_map = {"dummy": [PermissionDeclared(permission=perm, tool_names=["dummy"])]}
        result = filter_tools([DummyTool()], checker, tool_map)
        assert len(result) == 0


# ============================================================================
# WELL_KNOWN_PERMISSIONS
# ============================================================================

class TestWellKnownPermissions:
    def test_all_have_valid_ids(self):
        for pid, perm in WELL_KNOWN_PERMISSIONS.items():
            assert isinstance(perm.id, PermissionId)
            assert str(perm.id) == pid

    def test_default_policies_are_expected(self):
        # Low-risk operations default to allow
        assert WELL_KNOWN_PERMISSIONS["filesystem:read"].default_policy == PermissionPolicy.allow
        assert WELL_KNOWN_PERMISSIONS["network:fetch"].default_policy == PermissionPolicy.allow
        # High-risk operations default to deny
        assert WELL_KNOWN_PERMISSIONS["shell:execute"].default_policy == PermissionPolicy.deny
        assert WELL_KNOWN_PERMISSIONS["filesystem:write"].default_policy == PermissionPolicy.deny
        assert WELL_KNOWN_PERMISSIONS["python:execute"].default_policy == PermissionPolicy.deny
