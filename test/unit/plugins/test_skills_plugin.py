import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock
from hawi.plugin import HookContext
from hawi_plugins.skills_plugin.plugin import SkillsPlugin

class TestSkillsPlugin:
    @pytest.fixture
    def skills_dir(self):
        """Create a temporary directory for skills."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def plugin(self, skills_dir):
        """Create a SkillsPlugin instance."""
        return SkillsPlugin(skills_dir=skills_dir)

    def test_read_write_file(self, plugin, skills_dir):
        """Test file read and write tools."""
        file_path = os.path.join(skills_dir, "test.txt")
        content = "Hello, World!"
        
        # Write file
        result = plugin.write_file(file_path, content)
        assert "Successfully wrote" in result
        assert os.path.exists(file_path)
        
        # Read file
        read_content = plugin.read_file(file_path)
        assert read_content == content

    def test_run_shell(self, plugin):
        """Test shell command execution tool."""
        result = plugin.run_shell("echo 'hello shell'")
        assert "hello shell" in result

    def test_scan_skills(self, skills_dir):
        """Test scanning and parsing of SKILL.md files."""
        # Create a valid skill structure
        skill_path = os.path.join(skills_dir, "my-skill", "SKILL.md")
        os.makedirs(os.path.dirname(skill_path), exist_ok=True)
        
        skill_content = """---
name: my-skill
description: A custom skill for testing.
---
Instructions for the skill.
Step 1: Do something.
"""
        with open(skill_path, "w") as f:
            f.write(skill_content)
            
        # Initialize plugin to trigger scan
        plugin = SkillsPlugin(skills_dir=skills_dir)
        
        # Verify registry
        assert "my-skill" in plugin.skills_registry
        skill_info = plugin.skills_registry["my-skill"]
        assert skill_info["description"] == "A custom skill for testing."
        assert skill_info["path"] == skill_path
        assert skill_info["content"] == skill_content

    def test_scan_skills_invalid_frontmatter(self, skills_dir):
        """Test scanning a file with missing or invalid frontmatter."""
        skill_path = os.path.join(skills_dir, "bad-skill", "SKILL.md")
        os.makedirs(os.path.dirname(skill_path), exist_ok=True)
        
        # Missing frontmatter
        with open(skill_path, "w") as f:
            f.write("Just some markdown without frontmatter.")
            
        plugin = SkillsPlugin(skills_dir=skills_dir)
        assert "bad-skill" not in plugin.skills_registry

    def test_inject_skills_context(self, skills_dir):
        """Test injection of skills list into agent context."""
        # Create a skill
        skill_path = os.path.join(skills_dir, "deploy", "SKILL.md")
        os.makedirs(os.path.dirname(skill_path), exist_ok=True)
        with open(skill_path, "w") as f:
            f.write("---\nname: deploy\ndescription: Deploy app.\n---\n...")
            
        plugin = SkillsPlugin(skills_dir=skills_dir)
        
        # Mock agent
        agent = MagicMock()
        agent.context.system_prompt = []
        
        # Run injection
        ctx = HookContext(run_id="test", iteration=0)
        plugin.inject_skills_context(agent, ctx)  # type: ignore[reportCallIssue]
        
        # Verify system prompt updated
        assert len(agent.context.system_prompt) == 1
        injected_text = agent.context.system_prompt[0]["text"]
        assert "Available Skills" in injected_text
        assert "- deploy: Deploy app." in injected_text

    def test_use_skill(self, skills_dir):
        """Test loading skill instructions via use_skill tool."""
        # Create a skill
        skill_path = os.path.join(skills_dir, "test-skill", "SKILL.md")
        os.makedirs(os.path.dirname(skill_path), exist_ok=True)
        with open(skill_path, "w") as f:
            f.write("""---
name: test-skill
description: Test skill
---
These are the instructions.
""")
            
        plugin = SkillsPlugin(skills_dir=skills_dir)
        
        # Invoke use_skill
        result = plugin.use_skill("test-skill")
        
        assert "Skill 'test-skill' loaded." in result
        assert "Instructions:\nThese are the instructions." in result
        # Ensure frontmatter is stripped
        assert "name: test-skill" not in result

    def test_use_skill_not_found(self, plugin):
        """Test use_skill with a non-existent skill."""
        result = plugin.use_skill("non-existent")
        assert "Skill 'non-existent' not found" in result
