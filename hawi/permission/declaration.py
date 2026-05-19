"""Permission declaration API for plugins.

This module provides the base class extension for :class:`HawiPlugin` so
that plugins can declare their permissions in a structured way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .types import PermissionDeclared

if TYPE_CHECKING:
    from hawi.plugin import HawiPlugin


class PermissionDeclarer:
    """Mixin / helper that plugins use to declare permissions.

    Plugins call :meth:`declare_permission` in their ``__init__`` or override
    the :attr:`permissions` property to return a static list.
    """

    def __init__(self) -> None:
        self._declared_permissions: list[PermissionDeclared] = []

    @property
    def permissions(self) -> Sequence[PermissionDeclared]:
        """Return the permissions declared by this plugin.

        Subclasses may override this property to return a static list.
        """
        return self._declared_permissions

    def declare_permission(self, decl: PermissionDeclared) -> None:
        """Add a permission declaration at runtime."""
        self._declared_permissions.append(decl)

    def declare_permissions(self, decls: Sequence[PermissionDeclared]) -> None:
        """Add multiple permission declarations at runtime."""
        self._declared_permissions.extend(decls)


def collect_plugin_permissions(
    plugins: Sequence["HawiPlugin"],
) -> list[PermissionDeclared]:
    """Collect all :class:`PermissionDeclared` from a sequence of plugins.

    This is the canonical entry point used by :class:`PluginManager` when
    building its permission-to-tool map.
    """
    all_decls: list[PermissionDeclared] = []
    for plugin in plugins:
        perms = getattr(plugin, "permissions", None)
        if perms is None:
            continue
        if callable(perms):
            perms = perms()
        all_decls.extend(perms)
    return all_decls
