from __future__ import annotations

from pathlib import Path

from hawi.session import layout
from hawi.session.markdown_export import (
    export_message_history_to_markdown,
    write_markdown_export_bundle,
)


def test_markdown_export_folds_only_tool_arguments_and_results(tmp_path: Path) -> None:
    long_assistant = "\n".join(f"assistant line {i}" for i in range(130))
    long_args = {f"arg_line_{i:03d}": i for i in range(130)}
    long_result = "\n".join(f"result line {i}" for i in range(130))
    history = [
        {
            "run_id": "run-1",
            "role": "assistant",
            "content": [{"type": "text", "text": long_assistant}],
        },
        {
            "run_id": "run-1",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "demo_tool",
                    "arguments": long_args,
                }
            ],
        },
        {
            "run_id": "run-1",
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": "call-1",
                    "content": [{"type": "text", "text": long_result}],
                    "is_error": False,
                }
            ],
        },
    ]

    export = export_message_history_to_markdown(
        history,
        kind="session",
        subject_id="s1",
        title="Test Export",
        system_prompt="system prompt",
        tool_line_limit=100,
    )

    assert "assistant line 129" in export.markdown
    assert "> Full arguments:" in export.markdown
    assert "> Full result:" in export.markdown
    assert "arg_line_129" not in export.markdown
    assert "result line 129" not in export.markdown
    assert len(export.references) == 2
    assert "arg_line_129" in export.references[0].content
    assert "result line 129" in export.references[1].content

    written = write_markdown_export_bundle(
        export,
        export_dir=tmp_path / "exports" / export.export_id,
        message_history=history,
    )
    assert written.session_markdown_path is not None
    assert Path(written.session_markdown_path).exists()
    assert written.session_jsonl_path is not None
    assert layout.read_jsonl(Path(written.session_jsonl_path))[0]["run_id"] == "run-1"
