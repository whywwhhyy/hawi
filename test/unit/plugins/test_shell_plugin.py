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
