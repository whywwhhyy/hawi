"""Dynamic module loader — scans a directory and loads Python modules (packages or .py files)."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")

ModulePredicate = Callable[[ModuleType], bool]
Extractor = Callable[[ModuleType], "T | None"]


class ModuleLoader(Generic[T]):
    """Scans a directory for Python modules, filters them with predicates,
    and extracts objects via an extractor.

    Supports two module forms:

    - Package: a subdirectory containing ``__init__.py``
    - Single-file: a ``.py`` file not starting with ``_``

    Example::

        loader = ModuleLoader("./plugins")
        plugin_classes = loader.load(
            predicates=[has_subclass(HawiPlugin)],
            extractor=extract_subclass(HawiPlugin),
        )
        plugins = [cls() for cls in plugin_classes]
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()

    def _iter_modules(self) -> Iterable[ModuleType]:
        """Yield all importable modules found directly inside the directory."""
        if not self.directory.exists():
            return

        for entry in sorted(self.directory.iterdir()):
            module: ModuleType | None = None

            if entry.is_dir() and (entry / "__init__.py").exists():
                module = self._load_from_path(entry / "__init__.py", entry.name)
            elif entry.is_file() and entry.suffix == ".py" and not entry.name.startswith("_"):
                module = self._load_from_path(entry, entry.stem)

            if module is not None:
                yield module

    def _load_from_path(self, path: Path, name: str) -> ModuleType | None:
        """Dynamically import a module from a file path, with caching."""
        module_key = f"_hawi_loader_{self.directory.name}_{name}"
        if module_key in sys.modules:
            return sys.modules[module_key]

        try:
            spec = importlib.util.spec_from_file_location(module_key, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            return module
        except Exception as e:
            warnings.warn(f"Failed to load module from {path}: {e}", stacklevel=3)
            return None

    def load(
        self,
        predicates: list[ModulePredicate] | None = None,
        extractor: "Extractor[T] | None" = None,
    ) -> "list[T]":
        """Scan the directory, filter modules by predicates, extract results.

        Args:
            predicates: All must pass (AND logic). ``None`` accepts every module.
            extractor: Called on each passing module to produce the result.
                       ``None`` returns the module itself. Returning ``None``
                       from an extractor skips that module.

        Returns:
            List of extracted values from all passing modules.
        """
        results: list[T] = []
        for module in self._iter_modules():
            if predicates and not all(p(module) for p in predicates):
                continue
            value = extractor(module) if extractor else module  # type: ignore[assignment]
            if value is not None:
                results.append(value)
        return results


# ---------------------------------------------------------------------------
# Subclass matching helpers
# ---------------------------------------------------------------------------

def _is_concrete_subclass(obj: object, base_class: type, use_mro: bool) -> bool:
    """Return True if *obj* is a concrete (non-abstract) subclass of *base_class*.

    Args:
        use_mro: When ``True``, matches by ``(module, qualname)`` pairs in the
                 MRO instead of ``issubclass``.  Useful when *base_class* may
                 have been loaded under different ``sys.modules`` keys (e.g. in
                 test environments or deeply nested dynamic imports).
    """
    if not inspect.isclass(obj) or inspect.isabstract(obj):
        return False
    if use_mro:
        base_key = (base_class.__module__, base_class.__qualname__)
        return any(
            (c.__module__, c.__qualname__) == base_key
            for c in obj.__mro__[1:]  # type: ignore[union-attr]
        )
    return issubclass(obj, base_class) and obj is not base_class  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Built-in predicates
# ---------------------------------------------------------------------------

def has_subclass(base_class: type, *, use_mro: bool = False) -> ModulePredicate:
    """Predicate: module contains at least one concrete (non-abstract) subclass of *base_class*.

    Args:
        use_mro: Match by ``(module, qualname)`` in the MRO instead of
                 ``issubclass``.  Set to ``True`` when *base_class* might be
                 loaded under a different ``sys.modules`` key than the one used
                 inside the scanned modules.
    """
    def check(module: ModuleType) -> bool:
        return any(
            _is_concrete_subclass(obj, base_class, use_mro)
            for obj in vars(module).values()
        )
    return check


def has_function(name: str) -> ModulePredicate:
    """Predicate: module contains a callable named *name*."""
    def check(module: ModuleType) -> bool:
        return callable(getattr(module, name, None))
    return check


def has_attribute(name: str) -> ModulePredicate:
    """Predicate: module has an attribute named *name* (any type)."""
    def check(module: ModuleType) -> bool:
        return hasattr(module, name)
    return check


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------

def extract_subclass(base_class: type[T], *, use_mro: bool = False) -> "Extractor[type[T]]":
    """Extractor: returns the first concrete subclass of *base_class* found in the module.

    Args:
        use_mro: See :func:`has_subclass`.
    """
    def extract(module: ModuleType) -> type[T] | None:
        for obj in vars(module).values():
            if _is_concrete_subclass(obj, base_class, use_mro):
                return obj  # type: ignore[return-value]
        return None
    return extract


def extract_all_subclasses(base_class: type[T], *, use_mro: bool = False) -> "Extractor[list[type[T]]]":
    """Extractor: returns all concrete subclasses of *base_class* found in the module.

    Args:
        use_mro: See :func:`has_subclass`.
    """
    def extract(module: ModuleType) -> list[type[T]] | None:
        found: list[type[T]] = [
            obj  # type: ignore[misc]
            for obj in vars(module).values()
            if _is_concrete_subclass(obj, base_class, use_mro)
        ]
        return found if found else None
    return extract


def extract_function(name: str) -> "Extractor[Callable]":  # type: ignore[type-arg]
    """Extractor: returns the callable named *name* from the module."""
    def extract(module: ModuleType) -> Callable | None:  # type: ignore[type-arg]
        return getattr(module, name, None)
    return extract
