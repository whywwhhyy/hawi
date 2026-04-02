"""Integration test configuration.

Automatically skips tests on authentication or rate limit errors.
"""

import pytest
from _pytest.reports import TestReport


def _is_auth_or_rate_limit_error(e: Exception) -> bool:
    """Check if exception is auth (401/403) or rate limit (429) error."""
    from hawi.errors import AgentError
    
    # Unwrap AgentError if needed
    original = e
    if isinstance(e, AgentError) and e.__cause__:
        original = e.__cause__
    
    error_msg = str(original).lower()
    
    # Rate limit (429)
    if "429" in error_msg or "rate limit" in error_msg or "1302" in error_msg:
        return True
    
    # Auth errors (401/403)
    if "401" in error_msg or "403" in error_msg:
        return True
    if any(kw in error_msg for kw in ["authentication", "unauthorized", "api key", "access denied", "invalid"]):
        return True
    
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to skip tests on API auth/rate limit errors instead of failing."""
    outcome = yield
    report = outcome.get_result()
    
    # Only process test call phase failures
    if report.when == "call" and report.failed:
        exc = call.excinfo.value if call.excinfo else None
        if exc and _is_auth_or_rate_limit_error(exc):
            # Create a new skipped report
            skipped_report = TestReport(
                nodeid=report.nodeid,
                location=report.location,
                keywords=report.keywords,
                outcome="skipped",
                longrepr=(report.fspath, None, f"API auth/rate limit error: {exc}"),
                when=report.when,
                sections=report.sections,
                start=report.start,
                stop=report.stop,
            )
            # Replace the report
            outcome.force_result(skipped_report)
