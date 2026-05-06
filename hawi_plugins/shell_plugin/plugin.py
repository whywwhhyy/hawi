import subprocess

from hawi.plugin import HawiPlugin, tool
from hawi.tool import ToolResult


class ShellPlugin(HawiPlugin):
    """
    Shell 操作插件，提供运行 shell 命令的能力。
    """

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {}

    @tool
    def run_shell(self, command: str) -> ToolResult:
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
            output = _format_process_result(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Command exited with status {result.returncode}",
                )
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=f"Error running command: {type(e).__name__}: {e}")


def _format_process_result(command: str, returncode: int, stdout: str, stderr: str) -> str:
    parts = [
        f"Command: {command}",
        f"Exit code: {returncode}",
    ]
    if stdout:
        parts.append(f"Stdout:\n{stdout}")
    else:
        parts.append("Stdout: <empty>")
    if stderr:
        parts.append(f"Stderr:\n{stderr}")
    else:
        parts.append("Stderr: <empty>")
    return "\n\n".join(parts)
