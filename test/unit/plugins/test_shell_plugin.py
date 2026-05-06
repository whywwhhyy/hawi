import pytest
from hawi_plugins.shell_plugin.plugin import ShellPlugin


class TestShellPlugin:
    @pytest.fixture
    def plugin(self):
        return ShellPlugin()

    def test_run_shell(self, plugin):
        """Test shell command execution tool."""
        result = plugin.run_shell("echo 'hello shell'")
        assert result.success is True
        assert result.error == ""
        assert "Exit code: 0" in result.output
        assert "hello shell" in result.output

    def test_run_shell_with_stderr(self, plugin):
        """Shell output should include stderr when present."""
        result = plugin.run_shell("printf 'oops\\n' >&2")
        assert result.success is True
        assert "Exit code: 0" in result.output
        assert "Stderr:" in result.output
        assert "oops" in result.output

    def test_run_shell_without_output(self, plugin):
        """Successful commands with no output should still report process status."""
        result = plugin.run_shell("true")
        assert result.success is True
        assert "Exit code: 0" in result.output
        assert "Stdout: <empty>" in result.output
        assert "Stderr: <empty>" in result.output

    def test_run_shell_nonzero_exit_returns_process_data(self, plugin):
        """Non-zero exits should return stdout/stderr/exit code to the model."""
        result = plugin.run_shell("printf 'before\\n'; printf 'oops\\n' >&2; exit 7")

        assert result.success is False
        assert result.error == "Command exited with status 7"
        assert "Exit code: 7" in result.output
        assert "before" in result.output
        assert "oops" in result.output
