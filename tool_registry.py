"""Discover trusted local tool packages and select model-visible tools per turn."""

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool


TOOLS_ROOT = Path(__file__).resolve().parent / "tools"
PACKAGE_NAME = re.compile(r"[a-z][a-z0-9_-]*\Z")
FOLLOW_UP = re.compile(r"^(?:and\b|what about\b|how about\b|does (?:it|that)\b|can (?:it|that)\b|why\b|tell me more\b)", re.I)


@dataclass(frozen=True)
class ToolPackage:
    name: str
    description: str
    triggers: tuple[str, ...]
    always_on: bool
    tools: tuple[BaseTool, ...]


class ToolRegistry:
    def __init__(self, root: Path = TOOLS_ROOT):
        self.root = root.resolve()
        self.packages: dict[str, ToolPackage] = {}
        self.owned_names: set[str] = set()
        if not self.root.is_dir():
            raise ValueError(f"Tools directory is missing: {self.root}")
        for manifest in sorted(self.root.glob("*/manifest.yaml")):
            directory = manifest.parent
            module_path = directory / "tools.py"
            if any(not path.resolve().is_relative_to(self.root) or not path.is_file()
                   for path in (manifest, module_path)):
                raise ValueError(f"Tool package escapes tools directory or is incomplete: {directory}")
            metadata = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid tool manifest: {manifest}")
            name = metadata.get("name")
            description = metadata.get("description")
            triggers = metadata.get("triggers", [])
            always_on = metadata.get("always_on", False)
            if not isinstance(name, str) or not PACKAGE_NAME.fullmatch(name) or name != directory.name:
                raise ValueError(f"Tool package name must match its directory: {manifest}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Tool package description is required: {manifest}")
            if not isinstance(always_on, bool) or not isinstance(triggers, list) or not all(
                isinstance(trigger, str) and trigger.strip() for trigger in triggers
            ) or (not always_on and not triggers):
                raise ValueError(f"Tool package requires valid triggers or always_on: {manifest}")
            if name in self.packages:
                raise ValueError(f"Duplicate tool package: {name}")
            spec = importlib.util.spec_from_file_location(f"copilot_tool_{name.replace('-', '_')}", module_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot import tool package: {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            factory = getattr(module, "get_tools", None)
            if not callable(factory):
                raise ValueError(f"Tool package must export get_tools(): {module_path}")
            tools = factory()
            if not isinstance(tools, (tuple, list)) or not tools or not all(isinstance(item, BaseTool) for item in tools):
                raise ValueError(f"Tool package must return a nonempty list of LangChain tools: {module_path}")
            names = {item.name for item in tools}
            if len(names) != len(tools) or names & self.owned_names:
                raise ValueError(f"Duplicate tool name in package: {module_path}")
            self.owned_names.update(names)
            self.packages[name] = ToolPackage(name, description, tuple(triggers), always_on, tuple(tools))

    @property
    def tools(self) -> list[BaseTool]:
        return [tool for package in self.packages.values() for tool in package.tools]

    def validate_names(self, other_tools: list) -> None:
        names = [tool.name for tool in other_tools]
        if len(names) != len(set(names)) or self.owned_names.intersection(names):
            raise ValueError("Tool name collides with an existing service or MCP tool")

    @staticmethod
    def _matches(text: str, triggers: tuple[str, ...]) -> bool:
        return any(re.search(r"(?<!\w)" + re.escape(trigger.casefold()) + r"(?!\w)", text.casefold())
                   for trigger in triggers)

    def visible_names(self, messages: list) -> set[str]:
        active = {package.name for package in self.packages.values() if package.always_on}
        human_indices = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
        if not human_indices:
            return {tool.name for name in active for tool in self.packages[name].tools}
        latest_index = human_indices[-1]
        text = str(messages[latest_index].content).strip()
        matched = {package.name for package in self.packages.values() if self._matches(text, package.triggers)}
        if matched:
            active.update(matched)
        elif "simtalk" not in text.casefold() and FOLLOW_UP.match(text) and len(human_indices) > 1 and len(text) <= 100:
            previous_index = human_indices[-2]
            previous_text = str(messages[previous_index].content)
            prior_tools = {message.name for message in messages[previous_index + 1:latest_index]
                           if isinstance(message, ToolMessage)}
            active.update(package.name for package in self.packages.values()
                          if self._matches(previous_text, package.triggers)
                          or prior_tools.intersection(tool.name for tool in package.tools))
        return {tool.name for name in active for tool in self.packages[name].tools}

    def visible_tools(self, tools: list, messages: list) -> list:
        allowed = self.visible_names(messages)
        return [tool for tool in tools if not isinstance(tool, BaseTool)
                or tool.name not in self.owned_names or tool.name in allowed]