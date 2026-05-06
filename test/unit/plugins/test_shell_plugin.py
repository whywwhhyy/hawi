import pytest
import re
import time
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

    def test_run_shell_timeout_returns_command_id_and_partial_output(self, plugin):
        """Long-running commands should stay controllable after timeout."""
        result = plugin.run_shell(
            "python -c \"import time; print('started', flush=True); time.sleep(2); print('finished', flush=True)\"",
            timeout=0.1,
        )

        assert result.success is True
        assert "Status: running" in result.output
        assert "Shell command id:" in result.output
        assert "started" in result.output
        command_id = _extract_command_id(result.output)

        status = plugin.shell_control(command_id, "status")
        assert status.success is True
        assert "Status: running" in status.output
        assert "started" in status.output

        cancel = plugin.shell_control(command_id, "cancel")
        assert cancel.success is False
        assert "Shell command id:" in cancel.output
        assert "Command canceled" in cancel.error

    def test_shell_control_reports_completed_background_command(self, plugin):
        """Status should return the final output once a background command exits."""
        result = plugin.run_shell(
            "python -c \"import time; print('started', flush=True); time.sleep(0.2); print('finished', flush=True)\"",
            timeout=0.05,
        )
        command_id = _extract_command_id(result.output)

        final = None
        for _ in range(20):
            final = plugin.shell_control(command_id, "status")
            if "Status: completed" in final.output:
                break
            time.sleep(0.05)

        assert final is not None
        assert final.success is True
        assert "Status: completed" in final.output
        assert "Exit code: 0" in final.output
        assert "finished" in final.output

    def test_shell_control_unknown_id(self, plugin):
        result = plugin.shell_control("shell-missing", "status")

        assert result.success is False
        assert result.error == "Unknown shell command id: shell-missing"


def _extract_command_id(output: str) -> str:
    match = re.search(r"Shell command id: (shell-[0-9a-f]+)", output)
    assert match is not None
    return match.group(1)
