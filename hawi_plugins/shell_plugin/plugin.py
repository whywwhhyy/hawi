import subprocess

from hawi.plugin import HawiPlugin, tool


class ShellPlugin(HawiPlugin):
    """
    Shell 操作插件，提供运行 shell 命令的能力。
    """

    @tool
    def run_shell(self, command: str) -> str:
        """
        运行 shell 命令。

        Args:
            command: 要执行的 shell 命令
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
            )
            ret = []
            if result.stdout:
                ret.append(f"Stdout:\n{result.stdout}")
            if result.stderr:
                ret.append(f"Stderr:\n{result.stderr}")
            return "\n\n".join(ret) if ret else "Command executed successfully with no output."
        except Exception as e:
            return f"Error running command: {e}"
