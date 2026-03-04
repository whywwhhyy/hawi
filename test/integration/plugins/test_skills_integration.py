import os
import shutil
import tempfile
import pytest
from typing import Any, AsyncGenerator

from hawi.agent import HawiAgent
from hawi.models import Model
from hawi.models import (
    MessageRequest, MessageResponse, TokenUsage, ToolCallPart, TextPart,
    DeltaPart, DeltaTextPart, DeltaToolCallPart, DeltaFinishPart,
    ContentPart
)
from hawi_plugins.skills_plugin import SkillsPlugin

class MockModel(Model):
    def __init__(self, workspace: str):
        self._model_id = "mock-model"
        self.workspace = workspace
        self.call_count = 0
    
    @property
    def model_id(self) -> str:
        return self._model_id
        
    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        self.call_count += 1
        last_message = request.messages[-1]
        
        content: list[ContentPart] = []
        user_text = ""
        
        # Check if it's a tool result
        if last_message["role"] == "tool":
             # Extract text from content (handle ToolResultPart format)
             result_text = ""
             if "content" in last_message:
                 for part in last_message["content"]:
                     if part["type"] == "tool_result":
                         # Extract text from nested content in ToolResultPart
                         nested_content = part.get("content", [])
                         if isinstance(nested_content, str):
                             result_text += nested_content
                         else:
                             for nested_part in nested_content:
                                 if isinstance(nested_part, dict) and nested_part.get("type") == "text":
                                     result_text += nested_part.get("text", "")
                     elif part["type"] == "text":
                         result_text += part["text"]

             if "Instructions:" in result_text:
                  content.append(TextPart(type="text", text="Skill loaded successfully."))
             else:
                  content.append(TextPart(type="text", text="Task completed."))
             
             return MessageResponse(
                id=f"msg_{self.call_count}",
                role="assistant",
                content=content,
                usage=TokenUsage(input_tokens=10, output_tokens=10)
            )

        # Extract text from the last message (user message)
        if "content" in last_message:
            for part in last_message["content"]:
                if part["type"] == "text":
                    user_text += part["text"]
        
        # Decision logic
        if "write" in user_text.lower():
            content.append(
                ToolCallPart(
                    type="tool_call",
                    id=f"call_write_{self.call_count}",
                    name="SkillsPlugin__write_file",
                    arguments={
                        "path": os.path.join(self.workspace, "test.txt"), 
                        "content": "hello world"
                    }
                )
            )
            
        elif "read" in user_text.lower():
             content.append(
                ToolCallPart(
                    type="tool_call",
                    id=f"call_read_{self.call_count}",
                    name="SkillsPlugin__read_file",
                    arguments={"path": os.path.join(self.workspace, "test.txt")}
                )
            )
            
        elif "shell" in user_text.lower():
             content.append(
                ToolCallPart(
                    type="tool_call",
                    id=f"call_shell_{self.call_count}",
                    name="SkillsPlugin__run_shell",
                    arguments={"command": "echo 'hello shell'"}
                )
            )

        elif "use skill" in user_text.lower():
             content.append(
                ToolCallPart(
                    type="tool_call",
                    id=f"call_skill_{self.call_count}",
                    name="SkillsPlugin__use_skill",
                    arguments={"name": "test-skill"}
                )
            )
            
        else:
            content.append(TextPart(type="text", text="I don't know what to do."))

        return MessageResponse(
            id=f"msg_{self.call_count}",
            role="assistant",
            content=content,
            usage=TokenUsage(input_tokens=10, output_tokens=10)
        )

    def _prepare_request_impl(self, request):
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(
            id="dummy",
            role="assistant",
            content=[]
        )

    async def _astream_impl(self, request: MessageRequest) -> AsyncGenerator[DeltaPart, None]:
        import json
        response = self._invoke_impl(request)
        
        idx = 0
        content_list = list(response.content)
        for part in content_list:
            if part["type"] == "text":
                yield {
                    "type": "text_delta",
                    "index": idx,
                    "delta": part["text"],
                    "is_start": True,
                    "is_end": True
                }
            elif part["type"] == "tool_call":
                yield {
                    "type": "tool_call_delta",
                    "index": idx,
                    "id": part["id"],
                    "name": part["name"],
                    "arguments_delta": "",
                    "is_start": True,
                    "is_end": False
                }
                yield {
                    "type": "tool_call_delta",
                    "index": idx,
                    "id": None,
                    "name": None,
                    "arguments_delta": json.dumps(part["arguments"]),
                    "is_start": False,
                    "is_end": True
                }
            idx += 1
            
        yield {
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 10}
        }

@pytest.mark.asyncio
async def test_skills_plugin_end_to_end():
    # Setup temporary workspace and skills directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        skills_dir = os.path.join(tmp_dir, "skills")
        os.makedirs(skills_dir, exist_ok=True)
        
        # Create a test skill
        skill_path = os.path.join(skills_dir, "test-skill", "SKILL.md")
        os.makedirs(os.path.dirname(skill_path), exist_ok=True)
        with open(skill_path, "w") as f:
            f.write("---\nname: test-skill\ndescription: A test skill\n---\nInstructions for test skill.")

        # Initialize
        model = MockModel(workspace=tmp_dir)
        plugin = SkillsPlugin(skills_dir=skills_dir)
        agent = HawiAgent(model=model, plugins=[plugin])

        # Test 1: Write file
        response = await agent.arun("Please write a file")
        assert "Task completed" in response.text
        assert os.path.exists(os.path.join(tmp_dir, "test.txt"))
        
        # Test 2: Use Skill
        # Verify system prompt has skills injected
        # Note: The agent's context is updated *before* the run starts if using hooks correctly.
        # But we need to run once to trigger the hook or check if it happened.
        # Actually, @before_conversation runs at the start of agent.run()
        
        response = await agent.arun("Use skill test-skill")
        assert "Skill loaded successfully" in response.text
