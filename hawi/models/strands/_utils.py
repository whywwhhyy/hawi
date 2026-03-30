"""
Utility functions for Strands model adapter.
"""


def _map_strands_stop_reason(reason: str) -> str:
    """
    Map Strands stop_reason to Hawi format.

    Args:
        reason: Strands stop reason string

    Returns:
        Mapped Hawi stop reason string
    """
    mapping = {
        "stop": "end_turn",
        "end_turn": "end_turn",
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "content_filter": "content_filter",
        "pause_turn": "pause_turn",
    }
    return mapping.get(reason, reason)
