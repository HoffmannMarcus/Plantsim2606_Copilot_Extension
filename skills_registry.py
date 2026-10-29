"""Discover service-local skills and load their instructions only when selected."""

from collections.abc import Iterable
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


SKILLS_ROOT = Path(__file__).resolve().parent / "skills"
MAX_SKILL_BYTES = 64 * 1024
MAX_FRONTMATTER_BYTES = 8192
SKILL_NAME = re.compile(r"[a-z][a-z0-9_-]*\Z")
RESERVED_COMMANDS = frozenset({"skills"})
FOLLOW_UP = re.compile(r"^(?:and\b|what about\b|how about\b|does (?:it|that)\b|can (?:it|that)\b|why\b|tell me more\b)", re.I)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: tuple[str, ...]
    tools: tuple[str, ...]
    path: Path
    enabled: bool = True
    head: str = ""


class SkillRegistry:
    def __init__(self, root: Path = SKILLS_ROOT, reserved_commands: Iterable[str] = RESERVED_COMMANDS):
        self.root = root.resolve()
        self.reserved_commands: frozenset[str] = frozenset(
            self._normalize_command(command) for command in reserved_commands if str(command).strip()
        )
        self.skills: dict[str, Skill] = {}
        if not self.root.is_dir():
            raise ValueError(f"Skills directory is missing: {self.root}")
        for path in sorted(self.root.glob("*/SKILL.md")):
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Skill path escapes skills directory: {path}")
            metadata, raw_head, _ = self._parse(path, metadata_only=True)
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid skill frontmatter: {path}")
            name = metadata.get("name")
            description = metadata.get("description")
            if not isinstance(name, str) or not SKILL_NAME.fullmatch(name) or name != path.parent.name:
                raise ValueError(f"Skill name must match its directory: {path}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Skill description is required: {path}")
            triggers = metadata.get("triggers")
            if triggers is None:
                triggers = [name.replace("-", " ").replace("_", " ")]
            tools = metadata.get("tools")
            if tools is None:
                tools = []
            enabled = metadata.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"Skill enabled must be a boolean: {path}")
            if any(not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value)
                   for value in (triggers, tools)):
                raise ValueError(f"Skill triggers/tools must be nonempty strings: {path}")
            self.register(Skill(name, description.strip(), tuple(triggers), tuple(tools), path, enabled, raw_head.strip()))

    @staticmethod
    def _normalize_command(name: str) -> str:
        return str(name).strip().lstrip("/").casefold()

    def register(self, skill: Skill) -> None:
        key = self._normalize_command(skill.name)
        if not SKILL_NAME.fullmatch(skill.name):
            raise ValueError(f"Invalid skill name: {skill.name}")
        if key in self.reserved_commands:
            raise ValueError(f"Skill name collides with a command: {skill.name}")
        if key in self.skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        self.skills[key] = skill

    def validate_names(self, reserved_commands: Iterable[str] = ()) -> None:
        reserved = self.reserved_commands | {
            self._normalize_command(command) for command in reserved_commands if str(command).strip()
        }
        seen: set[str] = set()
        for key, skill in self.skills.items():
            normalized = self._normalize_command(skill.name)
            if not SKILL_NAME.fullmatch(skill.name):
                raise ValueError(f"Invalid skill name: {skill.name}")
            if normalized in reserved:
                raise ValueError(f"Skill name collides with a command: {skill.name}")
            if normalized in seen or key != normalized:
                raise ValueError(f"Duplicate skill name: {skill.name}")
            seen.add(normalized)

    def all_skills(self) -> list[Skill]:
        return [self.skills[key] for key in sorted(self.skills)]

    def available_skills(self) -> list[Skill]:
        return [skill for skill in self.all_skills() if skill.enabled]

    def unavailable_skills(self) -> list[Skill]:
        return [skill for skill in self.all_skills() if not skill.enabled]

    def format_skills(self) -> str:
        if not self.skills:
            return "No skills available."
        available = self.available_skills()
        unavailable = self.unavailable_skills()
        sections: list[str] = []
        if available:
            lines = ["Available Skills:"]
            for skill in available:
                lines.append(f"- `/{skill.name}`: {skill.description}")
            sections.append("\n".join(lines))
        else:
            sections.append("Available Skills:\n- None")
        if unavailable:
            lines = ["Unavailable Skills:"]
            for skill in unavailable:
                lines.append(f"- `/{skill.name}`: {skill.description}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def format_skills_head(self) -> str:
        available = self.available_skills()
        unavailable = self.unavailable_skills()
        if not available and not unavailable:
            return ""
        sections: list[str] = [
            "## Skills",
            (
                "You have access to specialized skills that provide focused instructions and tools for specific domains. "
                "The head of each available skill is provided below.\n"
                "When you decide that a user request matches a skill, call the `load_skill` tool with the skill's name "
                "(e.g. `load_skill(skill_name=\"<name>\")`) to load the entire skill file and detailed instructions into context before answering."
            ),
        ]
        if available:
            sections.append("Available Skills:")
            for skill in available:
                head_str = skill.head if skill.head else f"---\nname: {skill.name}\ndescription: {skill.description}\n---"
                sections.append(f"### Skill: `{skill.name}`\n{head_str}")
        if unavailable:
            sections.append("Unavailable (Disabled) Skills:")
            for skill in unavailable:
                head_str = skill.head if skill.head else f"---\nname: {skill.name}\ndescription: {skill.description}\n---"
                sections.append(f"### Skill: `{skill.name}` (disabled; requires `enabled: true` in `skills/{skill.name}/SKILL.md`)\n{head_str}")
        return "\n\n".join(sections)

    def format_skills_catalog(self) -> str:
        return self.format_skills_head()

    def match_command(self, text: str) -> tuple[Skill, str] | None:
        stripped = str(text).strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped.split(None, 1)
        command_name = self._normalize_command(parts[0])
        skill = self.skills.get(command_name)
        if skill is None:
            return None
        remainder = parts[1].strip() if len(parts) > 1 else ""
        return skill, remainder

    def _parse(self, path: Path, metadata_only: bool = False) -> tuple[object, str, str]:
        if path.stat().st_size > MAX_SKILL_BYTES:
            raise ValueError(f"Skill exceeds size limit: {path}")
        with path.open("r", encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                raise ValueError(f"Skill must start with YAML frontmatter: {path}")
            header = []
            for line in handle:
                if line.strip() == "---":
                    break
                header.append(line)
                if sum(map(len, header)) > MAX_FRONTMATTER_BYTES:
                    raise ValueError(f"Skill frontmatter exceeds size limit: {path}")
            else:
                raise ValueError(f"Skill frontmatter is not closed: {path}")
            try:
                metadata = yaml.safe_load("".join(header))
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML frontmatter: {path}") from exc
            raw_head = f"---\n{''.join(header)}---"
            return metadata, raw_head, "" if metadata_only else handle.read().strip()

    @staticmethod
    def _matches(text: str, triggers: tuple[str, ...]) -> bool:
        return any(re.search(r"(?<!\w)" + re.escape(trigger.casefold()) + r"(?!\w)", text.casefold())
                   for trigger in triggers)

    def select(self, messages: list, forced_skill: str | None = None) -> Skill | None:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        if forced_skill:
            key = self._normalize_command(forced_skill)
            if key not in self.skills:
                raise ValueError(f"Unknown skill: {forced_skill}")
            skill = self.skills[key]
            return skill if skill.enabled else None

        # 1. Check if load_skill was called in messages (most recent first)
        for message in reversed(messages):
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                for tc in reversed(message.tool_calls):
                    if tc.get("name") in ("load_skill", "read_skill"):
                        name = tc.get("args", {}).get("skill_name") or tc.get("args", {}).get("name")
                        if name:
                            key = self._normalize_command(name)
                            if key in self.skills and self.skills[key].enabled:
                                return self.skills[key]

        # 2. Check manual slash command in latest human message
        human_indices = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
        if not human_indices:
            return None
        latest_index = human_indices[-1]
        latest_text = str(messages[latest_index].content).strip()
        manual_match = self.match_command(latest_text)
        if manual_match is not None:
            return manual_match[0] if manual_match[0].enabled else None

        # 3. Check trigger matches or follow-ups
        matches = [skill for skill in self.skills.values() if skill.enabled and self._matches(latest_text, skill.triggers)]
        if len(matches) == 1:
            return matches[0]
        if "simtalk" in latest_text.casefold():
            return None
        if matches or not FOLLOW_UP.match(latest_text) or len(human_indices) < 2 or len(latest_text) > 100:
            return None
        previous_index = human_indices[-2]
        previous_text = str(messages[previous_index].content)
        previous_manual = self.match_command(previous_text)
        if previous_manual is not None:
            return previous_manual[0] if previous_manual[0].enabled else None
        prior_tools = {message.name for message in messages[previous_index + 1:latest_index]
                       if isinstance(message, ToolMessage)}
        matches = [skill for skill in self.skills.values()
                   if skill.enabled and (self._matches(previous_text, skill.triggers) or prior_tools.intersection(skill.tools))]
        return matches[0] if len(matches) == 1 else None

    def load(self, skill: Skill) -> str:
        if skill.path.resolve().is_relative_to(self.root):
            _, _, body = self._parse(skill.path)
            if body:
                return body
        raise ValueError(f"Skill is unavailable or empty: {skill.name}")

    def read_file(self, skill: Skill) -> str:
        if skill.path.resolve().is_relative_to(self.root) and skill.path.is_file():
            return skill.path.read_text(encoding="utf-8")
        raise ValueError(f"Skill file is unavailable or missing: {skill.name}")

    def prompt(self, base_prompt: str, messages: list, forced_skill: str | None = None, include_catalog: bool = False) -> str:
        skill = self.select(messages, forced_skill=forced_skill)
        parts = [base_prompt]
        if include_catalog:
            head = self.format_skills_head()
            if head:
                parts.append(head)
        if skill is not None:
            parts.append(f"## Active skill: {skill.name}\n{self.load(skill)}")
        return "\n\n".join(parts)