import os
import re
import glob
from typing import Any, Dict

from hawi.plugin import HawiPlugin, tool, before_conversation

class SkillsPlugin(HawiPlugin):
    """
    Skills plugin implementing the "Claude Skills" architecture.

    Provides:
    1. Skill discovery: Scans for SKILL.md files in a configured directory
    2. Skill invocation: use_skill tool to load skill instructions
    3. Context injection: Injects the list of available skills into the system prompt
    """

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = os.path.abspath(skills_dir)
        self.skills_registry: Dict[str, Dict[str, str]] = {}
        # Pre-scan skills on init
        self._scan_skills()

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "skills_dir": {
                    "type": "string",
                    "title": "Skills Directory",
                    "default": ".skills",
                    "description": "Directory to scan for SKILL.md files.",
                }
            },
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {
            "skills_dir": ".skills",
        }

    def _scan_skills(self):
        """Scan for SKILL.md files and populate the registry."""
        self.skills_registry.clear()

        if not os.path.exists(self.skills_dir):
            return

        # Find all SKILL.md files recursively
        pattern = os.path.join(self.skills_dir, "**", "SKILL.md")
        skill_files = glob.glob(pattern, recursive=True)

        for skill_file in skill_files:
            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse frontmatter
                frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
                if frontmatter_match:
                    frontmatter_text = frontmatter_match.group(1)
                    metadata = self._parse_yaml_simple(frontmatter_text)

                    name = metadata.get("name")
                    if name:
                        self.skills_registry[name] = {
                            "path": skill_file,
                            "description": metadata.get("description", "No description provided."),
                            "content": content,
                        }
            except Exception as e:
                print(f"Error parsing skill file {skill_file}: {e}")

    def _parse_yaml_simple(self, text: str) -> Dict[str, str]:
        """Simple YAML parser for frontmatter (avoiding PyYAML dependency)."""
        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result

    @before_conversation(system_prompt_variability="filesystem")
    def inject_skills_context(self, agent: Any, ctx):
        """Inject the list of available skills into the agent's system prompt."""
        # Re-scan to get latest skills
        self._scan_skills()

        if not self.skills_registry:
            return

        skills_list_str = "Available Skills (use 'use_skill' tool to invoke):\n"
        for name, info in self.skills_registry.items():
            skills_list_str += f"- {name}: {info['description']}\n"

        # Inject into system prompt
        if agent.context.system_prompt is None:
            agent.context.system_prompt = []

        # Add as a system message or append to existing text
        # Assuming system_prompt is a list of ContentPart or dicts
        agent.context.system_prompt.append({
            "type": "text",
            "text": f"\n\n{skills_list_str}\n",
        })

    @tool(name="use_skill")
    def use_skill(self, name: str) -> str:
        """
        Load a skill's instructions into the context.

        Args:
            name: The name of the skill to load (as listed in available skills).
        """
        if name not in self.skills_registry:
            return f"Skill '{name}' not found. Available skills: {', '.join(self.skills_registry.keys())}"

        skill_info = self.skills_registry[name]
        content = skill_info["content"]

        # Remove frontmatter for the return value to clean it up
        # (The agent doesn't need the YAML metadata again)
        skill_path = skill_info["path"]
        skill_dir = os.path.dirname(skill_path)

        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if frontmatter_match:
            instructions = frontmatter_match.group(2)
            return f"Skill '{name}' loaded.\nLocation: {skill_dir}\n\nInstructions:\n{instructions}"

        return f"Skill '{name}' loaded.\nLocation: {skill_dir}\n\n{content}"
