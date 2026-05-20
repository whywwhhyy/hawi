# Skills Plugin

This plugin provides essential skills for the Hawi Agent, implementing the **Claude Agent Skills** architecture. It supports file system operations, shell command execution, and a powerful skill discovery mechanism using `SKILL.md` files.

## Features

- **Core Capabilities**:
  - `read_file(path)`: Read content from a file.
  - `write_file(path, content)`: Write content to a file.
  - `run_shell(command)`: Execute shell commands.

- **Skill Discovery**:
  - Automatically scans the configured `skills_dir` (default: `./skills`) for subdirectories containing `SKILL.md` files.
  - Parses YAML frontmatter for skill metadata (`name`, `description`).

- **Progressive Disclosure**:
  - Injects a summary list of available skills before each user prompt.
  - Provides a `use_skill(name)` tool for the agent to load detailed instructions for a specific skill on-demand.

## Usage

```python
from hawi.agent import HawiAgent
from hawi.builtin_plugins.skills_plugin import SkillsPlugin

# Initialize the plugin
skills_plugin = SkillsPlugin(skills_dir="./skills")

# Add to agent
agent = HawiAgent(
    model=model,
    plugins=[skills_plugin]
)
```

## Creating Custom Skills

To add a custom skill, create a directory in your `skills_dir` and add a `SKILL.md` file inside it.

**Directory Structure:**
```
skills/
  └── deploy/
      └── SKILL.md
```

**SKILL.md Format:**
The file must start with YAML frontmatter containing `name` and `description`, followed by the instructions.

```markdown
---
name: deploy
description: Deploy the application to production.
---
To deploy the application:
1. Run tests using `run_shell("pytest")`.
2. If successful, run the deployment script: `run_shell("./deploy.sh")`.
```

## How It Works

1.  **Discovery**: The plugin scans `skills/**/SKILL.md`.
2.  **Context Injection**: The agent receives a framework-injected user-context message listing available skills:
    > Available Skills:
    > - deploy: Deploy the application to production.
3.  **Invocation**: When the user asks to "deploy the app", the agent calls `use_skill(name="deploy")`.
4.  **Execution**: The plugin returns the instructions from `SKILL.md`, guiding the agent to perform the necessary steps using core tools like `run_shell`.
