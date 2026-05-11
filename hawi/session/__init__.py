"""Session manager for Hawi: persistent storage + crash recovery + GUI session switching.

Provides a unified on-disk representation that backs both:

1. Crash recovery — boundary-event-driven checkpoints flush to a per-session
   directory. After a crash, ``SessionManager.load_session`` rebuilds context,
   queues, runtime, and plugin state.
2. Session switching — multiple sessions co-exist on disk; the GUI lists and
   switches between them like browser tabs.

Public surface:

- :class:`SessionManager` — coordinator owned by the agent runtime.
- :class:`SessionMeta` — lightweight metadata for listing.
- :class:`SessionWriter` — daemon thread that performs atomic disk writes.
"""

from .lock import SessionLockedError
from .manager import SessionManager, SessionMeta
from .writer import SessionWriter, WriteJob

__all__ = [
    "SessionManager",
    "SessionMeta",
    "SessionLockedError",
    "SessionWriter",
    "WriteJob",
]
