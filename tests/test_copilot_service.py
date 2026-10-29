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

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
        self.assertEqual(sorted(s.name for s in skill_registry.available_skills()), ["buffer-advisor", "testskill"])
        self.assertEqual([s.name for s in skill_registry.unavailable_skills()], [])
        self.assertEqual([t.name for t in tool_registry.tools], ["example_tool"])

    def test_backend_endpoints_are_removed(self):
        route_paths = {route.path for route in service.app.routes}
        self.assertIn("/query/stream", route_paths)
        self.assertIn("/clear_memory", route_paths)
        self.assertIn("/skills", route_paths)
        self.assertIn("/health", route_paths)
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

    def test_skills_discoverability(self):
        self.add_skill("station-helper", triggers="[station helper]", tools="[example_tool]", enabled=True)
        self.add_skill("disabled-helper", triggers="[disabled helper]", tools="[]", enabled=False)

        registry = SkillRegistry(self.skills_root)

        # 1. Head of each skill is formatted into catalog/system prompt
        catalog = registry.format_skills_catalog()
        self.assertIn("## Skills", catalog)
        self.assertIn("The head of each available skill is provided below", catalog)
        self.assertIn("call the `load_skill` tool with the skill's name", catalog)
        self.assertIn("### Skill: `station-helper`", catalog)
        self.assertIn("name: station-helper", catalog)
        self.assertIn("description: Use for station-helper", catalog)
        self.assertIn("tools: [example_tool]", catalog)
        self.assertIn("Unavailable (Disabled) Skills:", catalog)
        self.assertIn("### Skill: `disabled-helper`", catalog)

        # 2. Agent system prompt without active skill includes the skills head, but NOT the full body
        with patch.object(service, "_skill_registry", registry):
            prompt_no_skill = service._build_skill_prompt([HumanMessage(content="What can you do?")])
        self.assertIn("## Skills", prompt_no_skill)
        self.assertIn("### Skill: `station-helper`", prompt_no_skill)
        self.assertNotIn("## Active skill:", prompt_no_skill)
        self.assertNotIn("Instructions for station-helper.", prompt_no_skill)

        # 3. load_skill tool loads the entire file into context
        with patch.object(service, "_skill_registry", registry), \
             patch.object(service, "_active_skill", None):
            # Loading an enabled skill returns the full file content (frontmatter + body)
            loaded = service.load_skill.invoke({"skill_name": "station-helper"})
            self.assertIn("# Loaded Skill: station-helper", loaded)
            self.assertIn("name: station-helper", loaded)
            self.assertIn("Instructions for station-helper.", loaded)
            self.assertEqual(service._active_skill, "station-helper")

            # Loading a disabled skill returns enablement guidance
            disabled_result = service.load_skill.invoke({"skill_name": "disabled-helper"})
            self.assertIn("is not enabled", disabled_result)
            self.assertIn("enabled: true", disabled_result)

            # Loading an unknown skill returns error with available skills
            missing_result = service.load_skill.invoke({"skill_name": "unknown-skill"})
            self.assertIn("not found", missing_result)
            self.assertIn("station-helper", missing_result)

        # 4. Agent auto-discovery: tool calls in messages resolve active skill and unlock tools
        tool_reg = ToolRegistry()
        request = Mock()
        request.tools = [service.search_chunks_semantic, service.load_skill, *tool_reg.tools]
        request.override.side_effect = lambda **changes: SimpleNamespace(tools=changes["tools"])

        async def names(model_request):
            return [item.name for item in model_request.tools]

        with patch.object(service, "_skill_registry", registry), \
             patch.object(service, "_tool_registry", tool_reg), \
             patch.object(service, "_active_skill", None):
            # Before calling load_skill: example_tool is not visible
            request.state = {"messages": [HumanMessage(content="How do I write SimTalk?")]}
            tools_before = asyncio.run(service._visible_plugin_tools.awrap_model_call(request, names))
            self.assertNotIn("example_tool", tools_before)

            # When LLM decides to call load_skill: tool call in messages makes example_tool visible
            from langchain_core.messages import AIMessage
            request.state = {
                "messages": [
                    HumanMessage(content="How do I write SimTalk?"),
                    AIMessage(content="", tool_calls=[{"name": "load_skill", "args": {"skill_name": "station-helper"}, "id": "call_1"}]),
                ]
            }
            tools_after = asyncio.run(service._visible_plugin_tools.awrap_model_call(request, names))
            self.assertIn("example_tool", tools_after)

        # 5. Agent system prompt with active skill includes both catalog head and active skill instructions
        with patch.object(service, "_skill_registry", registry), \
             patch.object(service, "_active_skill", "station-helper"):
            prompt_with_skill = service._build_skill_prompt([HumanMessage(content="SimTalk code")])
        self.assertIn("## Skills", prompt_with_skill)
        self.assertIn("## Active skill: station-helper", prompt_with_skill)
        self.assertIn("Instructions for station-helper.", prompt_with_skill)

        # 6. HTTP GET /skills endpoint returns discoverable skills
        async def check_api():
            with patch.object(service, "_skill_registry", registry), \
                 patch.object(service, "_active_skill", "station-helper"):
                skills_resp = await service.get_skills()
                self.assertEqual([s["name"] for s in skills_resp["skills"]], ["station-helper"])
                self.assertEqual([s["name"] for s in skills_resp["unavailable_skills"]], ["disabled-helper"])
                self.assertEqual(skills_resp["active_skill"], "station-helper")

                health_resp = await service.health()
                self.assertEqual(health_resp["skills_count"], 1)

        asyncio.run(check_api())

    def test_skill_loading_message_streamed_to_ui(self):
        self.add_skill("station-helper", triggers="[station helper]", tools="[example_tool]", enabled=True)
        registry = SkillRegistry(self.skills_root)

        class SkillCallingAgent:
            async def astream(self, payload, stream_mode="messages", config=None):
                # Step 1: Agent decides to load the skill
                yield AIMessage(
                    content="",
                    tool_calls=[{"name": "load_skill", "args": {"skill_name": "station-helper"}, "id": "call_1"}],
                ), {}
                # Step 2: Tool execution result
                yield ToolMessage(content="Skill loaded.", name="load_skill", tool_call_id="call_1"), {}
                # Step 3: Agent generates final answer
                yield AIMessage(content="Here is the skill-informed answer."), {}

        async def check():
            agent = SkillCallingAgent()
            with patch.object(service, "_skill_registry", registry), \
                 patch.object(service, "_tool_registry", ToolRegistry()), \
                 patch.object(service, "_agent", agent), \
                 patch.object(service, "_active_skill", None):
                response = await service.query_stream(service.QueryRequest(question="How do I size a buffer?"))
                tokens = []
                done_data = None
                async for event in response.body_iterator:
                    if event["event"] == "token":
                        tokens.append(event["data"])
                    elif event["event"] == "done":
                        done_data = json.loads(event["data"])

                # Verification:
                # 1. UI receives the loading message token during streaming
                self.assertTrue(any("Loading skill station-helper" in t for t in tokens))
                self.assertIn("*Loading skill station-helper...*\n\n\n", tokens)

                # 2. Final response in done event also prefixes the loading message
                self.assertIsNotNone(done_data)
                self.assertTrue(done_data["answer"].startswith("*Loading skill station-helper...*\n\n\n"))
                self.assertIn("Here is the skill-informed answer.", done_data["answer"])
                self.assertEqual(done_data["active_skill"], "station-helper")

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
