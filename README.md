# Plant Simulation CoPilot — Skills & Custom Tools Extension

This folder provides a drop-in replacement for Plant Simulation's `copilot_service.py` that adds local **Skills** (`/skills`, `/<skill-name>`) and **Custom Tool Packages** without requiring any external backend service.

## Installation (for Plant Simulation users)

1. Copy the following files and folders into your Plant Simulation `Copilot` directory (for example `C:\Tecnomatix Plant Simulation 2606\Copilot\`):
   - `copilot_service.py` (replaces the existing `copilot_service.py`)
   - `skills_registry.py`
   - `tool_registry.py`
   - `skills/`
   - `tools/`
2. Ensure `PyYAML` is installed in the bundled embedded Python environment:
   ```powershell
   .\python-3.14.5-embed\python.exe -m pip install PyYAML==6.0.3 --no-warn-script-location
   ```
3. Restart Plant Simulation (or the CoPilot service).

## Skill Discoverability

Skills are discoverable across multiple channels:

1. **Auto-Discovery by the Agent (Tool-Like Progressive Loading):**
   - The head (metadata frontmatter: name, description, tools) of each available skill is included in the agent's system prompt.
   - The agent has access to the `load_skill` tool.
   - When the LLM receives a user prompt and decides that a specialized skill is needed for the task, it calls `load_skill(skill_name="<name>")`.
   - `load_skill` reads and loads the entire `SKILL.md` file into the conversation context as a tool output.
   - Once a skill is loaded, any specialized custom tools declared in the skill's frontmatter (e.g., `tools: [example_tool]`) are automatically made visible to the model.

2. **Model / Conversational Awareness:**
   - Because the head of each skill is in the system prompt, if a user asks *"What skills do you have?"*, *"What can you do?"*, or *"Can you help with X?"*, the agent is aware of all registered skills and can explain their capabilities without needing manual slash commands.

3. **HTTP API Discoverability:**
   - `GET /skills` — Returns JSON containing `skills` (available), `unavailable_skills` (disabled), and `active_skill`. Useful for BrowserView, UI menus, and autocompletion.
   - `GET /health` — Includes `skills_count` indicating how many active skills are registered.

4. **Slash Commands in Chat (Manual Invocation):**
   - `/skills` — Lists all **Available Skills** (`enabled: true`) and **Unavailable Skills** (`enabled: false`) with their descriptions.
   - `/<skill-name>` — Activates `<skill-name>` for subsequent messages in the conversation.
   - `/<skill-name> <prompt>` — Invokes `<skill-name>` for a single prompt without changing the active skill.

If a skill has `enabled: false`, the service replies with instructions on how to enable it in `skills/<skill-name>/SKILL.md`.

## Adding a Skill

Create `skills/<skill-name>/SKILL.md` with YAML frontmatter followed by the skill's Markdown instructions:

```markdown
---
name: my-skill
enabled: true
description: "Short description shown in /skills."
triggers:
  - optional trigger phrase
tools:
  - example_tool
---
Instructions loaded into the system prompt when `my-skill` is active.
```

- `name` must be lowercase (`[a-z][a-z0-9_-]*`), match the folder name, and not collide with reserved commands (`skills`).
- `enabled` (`true` or `false`, defaults to `true`) controls whether the skill is listed under `Available Skills:` or `Unavailable Skills:`.
- `tools` optionally lists tool names from `tools/` to expose to the model whenever the skill is active.

## Adding a Tool Package

Create `tools/<package-name>/manifest.yaml` and `tools/<package-name>/tools.py`:

`tools/example/manifest.yaml`:
```yaml
name: example
description: "Example custom tool package."
triggers:
  - example tool
always_on: false
```

`tools/example/tools.py`:
```python
from langchain_core.tools import tool


@tool
def example_tool(value: str) -> str:
    """Return the supplied value."""
    return value


def get_tools():
    return [example_tool]
```
