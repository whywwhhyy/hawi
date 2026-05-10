"""Structured configuration loading helpers.

The loader reads JSON/YAML/TOML mapping files from an ordered directory chain,
deep-merges raw values, and exposes a :class:`Config` object with both raw and
substituted config trees.
"""

from .config import (
    Config,
    ConfigLoaderError,
    ConfigSubstitutionError,
    ConfigValue,
    PathLike,
    substitute_config,
)
from .loader import (
    ConfigLoader,
    deep_merge,
    load_config_file,
    load_config_from_directory_chain,
    save_config_file,
)

__all__ = [
    "Config",
    "ConfigLoader",
    "ConfigLoaderError",
    "ConfigSubstitutionError",
    "ConfigValue",
    "PathLike",
    "deep_merge",
    "load_config_file",
    "load_config_from_directory_chain",
    "save_config_file",
    "substitute_config",
]
