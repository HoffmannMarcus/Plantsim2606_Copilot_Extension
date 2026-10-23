import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = AGENT_DIR.parent
for path in (str(ROOT_DIR), str(AGENT_DIR)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(AGENT_DIR))

from langchain_core.messages import AIMessage, HumanMessage

import copilot_service as service
from skills_registry import SkillRegistry
from tool_registry import ToolRegistry


class CopilotServiceSkillsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.skills_root = Path(self.directory.name) / "skills"
        self.skills_root.mkdir()

    def add_skill(self, name="station-helper", triggers="[station helper]", tools="[example_tool]", enabled=True):
        directory = self.skills_root / name
        directory.mkdir()
        path = directory / "SKILL.md"
        path.write_text(
            f"---\nname: {name}\ndescription: Use for {name}\ntriggers: {triggers}\ntools: {tools}\nenabled: {str(enabled).lower()}\n---\n"
            f"Instructions for {name}.\n",
            encoding="utf-8",
        )
        return path

    def test_bundled_skills_and_tools_load_cleanly(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()
        self.assertEqual([s.name for s in skill_registry.available_skills()], ["testskill"])
        self.assertEqual([s.name for s in skill_registry.unavailable_skills()], [])
        self.assertEqual([t.name for t in tool_registry.tools], ["example_tool"])

    def test_backend_endpoints_are_removed(self):
        route_paths = {route.path for route in service.app.routes}
        self.assertIn("/query/stream", route_paths)
        self.assertIn("/clear_memory", route_paths)
        self.assertNotIn("/conversations/finish", route_paths)
        self.assertNotIn("/conversations/{conversation_id}/review", route_paths)

    def test_skills_and_slash_commands(self):
        self.add_skill("station-helper", triggers="[station helper]", tools="[example_tool]", enabled=True)
        self.add_skill("disabled-helper", triggers="[disabled helper]", tools="[]", enabled=False)

        class PromptRecordingAgent:
            def __init__(self):
                self.calls = []

            async def astream(self, payload, stream_mode="messages", config=None):
                user_text = payload["messages"][0]["content"]
                built_prompt = service._build_skill_prompt([HumanMessage(content=user_text)])
                self.calls.append((user_text, built_prompt))
                yield AIMessage(content="Skill-guided answer."), {}

        async def collect_stream(response):
            events = [event async for event in response.body_iterator]
            self.assertEqual(events[0]["event"], "token")
            return json.loads(events[-1]["data"])

        async def check():
            agent = PromptRecordingAgent()
            registry = SkillRegistry(self.skills_root)

            with patch.object(service, "_skill_registry", registry), \
                 patch.object(service, "_tool_registry", ToolRegistry()), \
                 patch.object(service, "_agent", agent), \
                 patch.object(service, "_active_skill", None), \
                 patch.object(service, "_thread_id", "conv-skills"):

                # 1. /skills lists available and unavailable skills
                done = await collect_stream(
                    await service.query_stream(service.QueryRequest(question="/skills"))
                )
                self.assertIn("Available Skills:\n- `/station-helper`: Use for station-helper", done["answer"])
                self.assertIn("Unavailable Skills:\n- `/disabled-helper`: Use for disabled-helper", done["answer"])
                self.assertEqual([item["name"] for item in done["skills"]], ["station-helper"])
                self.assertEqual([item["name"] for item in done["unavailable_skills"]], ["disabled-helper"])
                self.assertEqual(len(agent.calls), 0)

                # 2. Disabled skill returns enablement instructions without calling agent
                for disabled_cmd in ("/disabled-helper", "/disabled-helper Check this"):
                    done = await collect_stream(
                        await service.query_stream(service.QueryRequest(question=disabled_cmd))
                    )
                    self.assertIn("is not enabled", done["answer"])
                    self.assertIn("enabled: true", done["answer"])
                    self.assertIsNone(service._active_skill)
                    self.assertEqual(len(agent.calls), 0)

                # 3. One-shot /station-helper <prompt> triggers skill for this prompt only
                done = await collect_stream(
                    await service.query_stream(
                        service.QueryRequest(question="/station-helper How do I use SimTalk here?")
                    )
                )
                self.assertIsNone(service._active_skill)
                self.assertEqual(done["active_skill"], "station-helper")
                self.assertEqual(agent.calls[-1][0], "How do I use SimTalk here?")
                self.assertIn("Active skill: station-helper", agent.calls[-1][1])

                # 4. Standalone /station-helper activates the skill until /clear_memory
                done = await collect_stream(
                    await service.query_stream(service.QueryRequest(question="/station-helper"))
                )
                self.assertEqual(service._active_skill, "station-helper")
                self.assertEqual(done["active_skill"], "station-helper")

                done = await collect_stream(
                    await service.query_stream(service.QueryRequest(question="Follow-up SimTalk question"))
                )
                self.assertEqual(done["active_skill"], "station-helper")
                self.assertIn("Active skill: station-helper", agent.calls[-1][1])

                # 5. /clear_memory rotates thread_id and resets active_skill
                cleared = await service.clear_memory()
                self.assertEqual(cleared["status"], "cleared")
                self.assertNotEqual(service._thread_id, "conv-skills")
                self.assertIsNone(service._active_skill)

        asyncio.run(check())

    def test_visible_plugin_tools_includes_active_skill_tools(self):
        registry = SkillRegistry()
        tool_registry = ToolRegistry()
        request = Mock()
        request.tools = [service.search_chunks_semantic, *tool_registry.tools]
        request.override.side_effect = lambda **changes: SimpleNamespace(tools=changes["tools"])

        async def names(model_request):
            return [item.name for item in model_request.tools]

        async def check():
            with patch.object(service, "_skill_registry", registry), \
                 patch.object(service, "_tool_registry", tool_registry):
                with patch.object(service, "_active_skill", None):
                    request.state = {"messages": [HumanMessage(content="How do I use SimTalk?")]}
                    default_tools = await service._visible_plugin_tools.awrap_model_call(request, names)
                with patch.object(service, "_active_skill", "testskill"):
                    request.state = {"messages": [HumanMessage(content="How do I use SimTalk?")]}
                    skill_tools = await service._visible_plugin_tools.awrap_model_call(request, names)
                return default_tools, skill_tools

        default_tools, skill_tools = asyncio.run(check())
        self.assertEqual(set(default_tools), {"search_chunks_semantic"})
        self.assertEqual(set(skill_tools), {"search_chunks_semantic", "example_tool"})


if __name__ == "__main__":
    unittest.main()
