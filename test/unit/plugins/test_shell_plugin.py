import pytest
from hawi_plugins.shell_plugin.plugin import ShellPlugin


class TestShellPlugin:
    @pytest.fixture
    def plugin(self):
        return ShellPlugin()

    def test_run_shell(self, plugin):
        """Test shell command execution tool."""
        result = plugin.run_shell("echo 'hello shell'")
        assert "hello shell" in result

    def test_run_shell_with_stderr(self, plugin):
        """Shell output should include stderr when present."""
        result = plugin.run_shell("printf 'oops\\n' >&2")
        assert "Stderr:" in result
        assert "oops" in result

    def test_run_shell_without_output(self, plugin):
        """Successful commands with no output should return a default message."""
        result = plugin.run_shell("true")
        assert result == "Command executed successfully with no output."
