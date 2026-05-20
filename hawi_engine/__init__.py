"""Compatibility wrapper for :mod:`hawi.engine`."""

from hawi import engine as _engine
from hawi.engine import *  # noqa: F401,F403
from hawi.engine import __all__  # noqa: F401

__path__ = _engine.__path__
