from __future__ import annotations

from living_tabletop.agents import Narrator
from living_tabletop.engine import GameEngine
from living_tabletop.keeper import Keeper
from living_tabletop.llm import LLMSettings, LLMUnavailable, OpenAICompatibleLLM, RoutedLLM
from living_tabletop.models import AgentCallRecord, ActionIntent, ActionResolution, CheckOutcome, CheckResult, LLMResult, NarrativeBeat, NarrativeSequence, PlayerVisibleMemory, SessionStatus
from living_tabletop.scenario import create_initial_state, load_scenario
from types import SimpleNamespace
import pytest


def test_base_url_is_normalized(monkeypatch):
    monkeypatch.setenv("LIVING_TABLETOP_BASE_URL", "https://example.test")
    monkeypatch.setenv("LIVING_TABLETOP_LLM_ENABLED", "false")
    monkeypatch.delenv("LIVING_TABLETOP_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = LLMSettings.from_env()
    assert settings.base_url == "https://example.test/v1"
    assert settings.model == "gpt-5.6-luna-openai-compact"
    assert settings.max_retries == 1
    assert settings.max_output_tokens == 400


def test_disabled_llm_does_not_construct_client():
    client = OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    assert client.enabled is False


def test_local_settings_default_to_ollama(monkeypatch):
    monkeypatch.delenv("LIVING_TABLETOP_LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LIVING_TABLETOP_LOCAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("LIVING_TABLETOP_LOCAL_LLM_ENABLED", raising=False)

    settings = LLMSettings.local_from_env()

    assert settings.enabled is True
    assert settings.base_url == "http://127.0.0.1:11434/v1"
    assert settings.model == "qwen3.5:9b-q4_K_M"
    assert settings.reasoning_effort == "none"
    assert settings.context_window == 8192


def test_json_parser_accepts_fenced_object():
    parsed = OpenAICompatibleLLM._parse_json('```json\n{"action_id":"wait_five_minutes"}\n```')
    assert parsed == {"action_id": "wait_five_minutes"}


def test_structured_request_passes_schema_and_local_generation_controls():
    class Completions:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                usage=None,
            )

    completions = Completions()
    client = OpenAICompatibleLLM(
        LLMSettings(
            enabled=True,
            api_key="test-only",
            temperature=0,
            reasoning_effort="low",
        )
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    client.complete_json(
        system="json",
        user_payload={},
        response_schema=schema,
        schema_name="SmokeOutput",
    )

    assert completions.request["temperature"] == 0
    assert completions.request["reasoning_effort"] == "low"
    assert completions.request["response_format"]["type"] == "json_schema"
    assert completions.request["response_format"]["json_schema"]["name"] == "SmokeOutput"
    assert completions.request["response_format"]["json_schema"]["schema"] == schema


def test_local_ollama_native_request_enforces_context_window():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": '{"ok":true}'},
                "prompt_eval_count": 12,
                "eval_count": 3,
            }

    class NativeClient:
        def __init__(self):
            self.url = None
            self.payload = None

        def post(self, url, *, json):
            self.url = url
            self.payload = json
            return Response()

    client = OpenAICompatibleLLM(
        LLMSettings(
            enabled=True,
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3.5:9b-q4_K_M",
            context_window=8192,
            reasoning_effort="none",
        ),
        provider="local",
    )
    native = NativeClient()
    client._native_client = native
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = client.complete_json(system="json", user_payload={}, response_schema=schema)

    assert result.data == {"ok": True}
    assert native.url == "http://127.0.0.1:11434/api/chat"
    assert native.payload["options"]["num_ctx"] == 8192
    assert native.payload["format"] == schema
    assert native.payload["think"] is False


class FakeLLM:
    enabled = True

    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(data=self.output, latency_ms=3, input_tokens=10, output_tokens=5)


def _player_intent(text: str) -> ActionIntent:
    return ActionIntent(content=text, goal=text, confidence=1.0, source="player_text")


def test_keeper_can_reuse_available_action(scenario, state):
    available = [action for action in scenario.actions if action.location in {None, "loc_lobby"}]
    intent = _player_intent("我先留在这里听一听")
    llm = FakeLLM({"existing_action_id": "wait_five_minutes", "confidence": 0.8})
    decision = Keeper(llm, scenario).adjudicate(state, intent, available)
    assert decision.existing_action_id == "wait_five_minutes"
    assert state.agent_calls[-1].role == "keeper"
    assert state.agent_calls[-1].validation == "accepted"
    assert len(llm.calls) == 1
    assert len(llm.calls[0]["response_schema"]["oneOf"]) == 2


def test_keeper_receives_and_honors_recent_visible_history(scenario, state):
    state.agent_calls.append(
        AgentCallRecord(
            id="agent_call_visible",
            role="narrator",
            input_state_version=state.version,
            output_digest="visible",
            structured_output={"beats": ["地板上有一份关于那家人的简报。"]},
            validation="accepted",
            latency_ms=5,
            created_at=state.world_time,
        )
    )
    llm = FakeLLM(
        {
            "existing_action_id": None,
            "confidence": 0.95,
            "open_plan": {
                "label": "拿起简报查看",
                "action_type": "EXAMINE",
                "goal": "我把地板上的简报拿起来看看",
                "duration_minutes": 1,
                "resolution": "automatic",
                "risk": "safe",
                "success_text": "你拿起简报，读起表面的文字。",
            },
        }
    )

    decision = Keeper(llm, scenario).adjudicate(
        state,
        _player_intent("我把地板上的简报拿起来看看"),
        scenario.actions,
    )

    history = llm.calls[0]["user_payload"]["recent_context"]["visible_history"]
    assert history[-1]["text"] == "地板上有一份关于那家人的简报。"
    assert history[-1]["kind"] == "soft_canon"
    assert decision.open_plan.resolution == "automatic"
    assert "不得仅因某个可见物未写入" in llm.calls[0]["system"]
    assert "表层互动通常应为 automatic" in llm.calls[0]["system"]


def test_keeper_prompt_keeps_literal_small_talk_automatic_and_on_topic(scenario, state):
    llm = FakeLLM(
        {
            "existing_action_id": None,
            "confidence": 0.99,
            "open_plan": {
                "label": "与安娜谈论天气",
                "action_type": "TALK",
                "goal": "和安娜谈论天气：今天天气还不错啊",
                "target_name": "安娜·里德",
                "target_entity_id": "npc_anna",
                "duration_minutes": 1,
                "resolution": "automatic",
                "risk": "safe",
                "success_text": "“是啊，难得有个晴天。”安娜望了一眼窗外。",
            },
        }
    )

    decision = Keeper(llm, scenario).adjudicate(
        state,
        _player_intent("和安娜谈论天气：今天天气还不错啊"),
        scenario.actions,
    )

    assert decision.open_plan.resolution == "automatic"
    system = llm.calls[0]["system"]
    assert "不得把普通寒暄" in system
    assert "不得借失败重新播放上一话题" in system
    payload = llm.calls[0]["user_payload"]
    assert "last_narrative" not in payload["recent_context"]
    assert payload["current_request_contract"]["player_text_verbatim"] == decision.open_plan.goal
    assert payload["recent_context"]["visible_history"] == []
    assert payload["known_facts"] == []
    assert payload["present_npc_knowledge"] == []
    assert payload["current_scene"]["description"] == ""
    assert payload["inventory"] == []
    assert all(action["target"] == "npc_anna" for action in payload["available_actions"])


def test_keeper_harness_repairs_a_plan_that_rewrites_current_player_goal(scenario, state):
    text = "和安娜谈论天气：今天天气还不错啊"

    class RepairingLLM:
        enabled = True

        def __init__(self):
            self.calls = []
            self.outputs = [
                {
                    "existing_action_id": None,
                    "confidence": 0.9,
                    "open_plan": {
                        "label": "沉默观察安娜",
                        "action_type": "WAIT",
                        "goal": "观察安娜是否愿意继续谈话",
                        "duration_minutes": 1,
                        "resolution": "automatic",
                        "risk": "safe",
                        "success_text": "你保持沉默。",
                    },
                },
                {
                    "existing_action_id": None,
                    "confidence": 0.98,
                    "open_plan": {
                        "label": "和安娜谈论天气",
                        "action_type": "TALK",
                        "goal": text,
                        "target_name": "安娜·里德",
                        "target_entity_id": "npc_anna",
                        "duration_minutes": 1,
                        "resolution": "automatic",
                        "risk": "safe",
                        "success_text": "“是啊，天气确实不错。”安娜望向窗外。",
                    },
                },
            ]

        def complete_json(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResult(
                data=self.outputs.pop(0),
                latency_ms=3,
                input_tokens=10,
                output_tokens=5,
            )

    llm = RepairingLLM()

    decision = Keeper(llm, scenario).adjudicate(
        state,
        _player_intent(text),
        scenario.actions,
    )

    assert len(llm.calls) == 2
    assert decision.open_plan.action_type.value == "TALK"
    assert decision.open_plan.goal == text
    assert llm.calls[1]["user_payload"]["_harness_repair"]["previous_output"]


def test_engine_never_keyword_routes_player_text(scenario, state):
    intent = GameEngine(scenario, FakeLLM({})).interpret(state, text="让我直接找到真相")
    assert intent.action_id is None
    assert intent.source == "player_text"
    assert intent.clarification is None


def test_keeper_accepts_off_script_plan(scenario, state):
    intent = _player_intent("我去火车站买票离开这座城")
    decision = Keeper(
        FakeLLM(
            {
                "existing_action_id": None,
                "confidence": 0.96,
                "open_plan": {
                    "label": "前往火车站",
                    "action_type": "MOVE",
                    "goal": "我去火车站买票离开这座城",
                    "destination_name": "波士顿火车站",
                    "destination_description": "夜班列车仍在运行的城市车站。",
                    "duration_minutes": 25,
                    "resolution": "automatic",
                    "skill": None,
                    "difficulty": "regular",
                    "risk": "safe",
                    "rest_until_hour": None,
                    "rest_day_offset": 0,
                    "success_text": "你抵达火车站，站在可以离开这座城的列车时刻表前。",
                    "failure_text": "你没能及时抵达车站。",
                },
            }
        ),
        scenario,
    ).adjudicate(state, intent, scenario.actions)
    assert decision.open_plan is not None
    assert decision.open_plan.destination_name == "波士顿火车站"
    assert decision.open_plan.action_type.value == "MOVE"


def test_keeper_honors_llm_overnight_rest_intent(scenario, state):
    text = "我想回家休息一下，然后第二天再来"
    intent = _player_intent(text)
    decision = Keeper(
        FakeLLM(
            {
                "existing_action_id": None,
                "confidence": 0.91,
                "open_plan": {
                    "label": "回家休息到次日",
                    "action_type": "REST",
                    "goal": text,
                    "destination_name": "调查员的家",
                    "duration_minutes": 30,
                    "resolution": "automatic",
                    "skill": None,
                    "difficulty": "regular",
                    "risk": "safe",
                    "rest_until_hour": 8,
                    "rest_day_offset": 1,
                    "success_text": "你回到家中。",
                    "failure_text": "你没能回去。",
                },
            }
        ),
        scenario,
    ).adjudicate(state, intent, scenario.actions)

    assert decision.open_plan is not None
    assert decision.open_plan.action_type.value == "REST"
    assert decision.open_plan.rest_until_hour == 8
    assert decision.open_plan.rest_day_offset == 1


def test_follow_up_question_cannot_be_keyword_routed_to_false_ending():
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    llm = FakeLLM(
        {
            "existing_action_id": None,
            "confidence": 0.98,
            "open_plan": {
                "label": "追问马卡里奥一家的居住时长",
                "action_type": "TALK",
                "goal": "他们在房子里住了多久？",
                "target_name": "史蒂文·诺特",
                "target_entity_id": "npc_knott",
                "duration_minutes": 2,
                "resolution": "automatic",
                "risk": "safe",
                "success_text": "“确切多久我说不准，只记得他们搬进去没过几个月，情况就开始恶化。”",
            },
        }
    )
    engine = GameEngine(scenario, llm)
    state = create_initial_state(scenario, seed=19)
    state, first = engine.play(state, action_id="cafe_question_knott")
    assert first.accepted is True

    resolved, resolution = engine.play(state, text="他们在房子里住了多久？")

    assert len(llm.calls) == 1
    assert resolution.accepted is True
    assert resolution.action_id.startswith("open__")
    assert resolved.status == SessionStatus.ACTIVE
    assert resolved.ending_id is None
    assert resolved.flags["false_report"] is False
    started = next(
        event
        for event in reversed(resolved.event_log)
        if event.type == "action_started" and event.payload.get("player_text")
    )
    assert started.payload["intent_source"] == "llm"
    false_report_context = next(
        item
        for item in llm.calls[0]["user_payload"]["available_actions"]
        if item["id"] == "cafe_false_report"
    )
    assert false_report_context["requires_explicit_intent"] is True


def test_narrator_records_structured_output(scenario, state):
    action = next(action for action in scenario.actions if action.id == "lobby_guestbook")
    resolution = ActionResolution(
        action_id=action.id,
        accepted=True,
        check=CheckResult(required=True, skill="observation", target=70, roll=20, outcome=CheckOutcome.HARD, succeeded=True),
        narrative_seed=action.success_text,
        state_version=state.version,
    )
    text = Narrator(FakeLLM({"narrative": "访客簿的夹层里，纸页发出干涩的摩擦声。"}), scenario).narrate(
        state, action, resolution
    )
    assert "访客簿" in text
    assert state.agent_calls[-1].role == "narrator"


def test_dialogue_sequence_requests_longer_first_person_performance(engine, scenario, state):
    resolved, _resolution = engine.play(state, action_id="lobby_talk_anna")
    sequence = resolved.narrative_sequence
    assert sequence is not None

    fake = FakeLLM({"beats": ["我开口。", "安娜回答。", "谈话停顿片刻。"]})
    beats = Narrator(fake, scenario).expand_sequence(resolved, sequence)

    assert beats == ["我开口。", "安娜回答。", "谈话停顿片刻。"]
    assert fake.calls[-1]["user_payload"]["action_type"] == "TALK"
    assert fake.calls[-1]["max_output_tokens"] == 900
    assert "玩家台词使用第一人称" in fake.calls[-1]["system"]
    assert "重要 NPC 的回应必须放在引号内直接说出" in fake.calls[-1]["system"]
    assert fake.calls[-1]["user_payload"]["recent_visible_history"] == []
    assert fake.calls[-1]["user_payload"]["present_entities"]
    assert "present_entities 以外的人物" in fake.calls[-1]["system"]

    continuation = sequence.model_copy(update={"continues_previous": True})
    continuation_fake = FakeLLM({"beats": ["安娜直接回答。"]})
    Narrator(continuation_fake, scenario).expand_sequence(resolved, continuation)
    assert "不要重述玩家已经说过的话" in continuation_fake.calls[-1]["system"]


def test_narrator_prioritizes_current_turn_and_filters_old_dialogue_replay(engine, scenario, state):
    resolved, _resolution = engine.play(state, action_id="lobby_talk_anna")
    sequence = resolved.narrative_sequence
    assert sequence is not None
    old_text = "安娜紧紧合上访客簿，反复强调医院深处的灯光正在闪烁。"
    resolved.visible_history.append(
        PlayerVisibleMemory(
            id="visible_old_topic",
            state_version=1,
            world_time=resolved.world_time,
            location_id="loc_lobby",
            sequence_id="narrative_old",
            kind="dialogue_claim",
            source="generated",
            action_type="TALK",
            text=old_text,
        )
    )
    fake = FakeLLM(
        {
            "beats": [
                old_text,
                "安娜望向窗外的晴光，轻声回应了你关于天气的寒暄。",
            ]
        }
    )

    beats = Narrator(fake, scenario).expand_sequence(resolved, sequence)

    assert beats == ["安娜望向窗外的晴光，轻声回应了你关于天气的寒暄。"]
    payload = fake.calls[-1]["user_payload"]
    assert payload["player_text"] == sequence.player_text
    assert all(item["text"] != sequence.beats[0].text for item in payload["recent_visible_history"])
    assert "拥有最高优先级" in fake.calls[-1]["system"]


def test_narrator_keeps_automatic_smalltalk_brief_and_self_contained(scenario, state):
    sequence = NarrativeSequence(
        id="narrative_weather",
        state_version=state.version,
        action_label="谈论天气",
        action_type="TALK",
        player_text="今天天气还不错啊",
        mechanical_result={"outcome": "AUTOMATIC"},
        canonical_seed="“是啊，今天阳光很好。”安娜望向窗外。",
        beats=[
            NarrativeBeat(
                id="weather_seed",
                text="“是啊，今天阳光很好。”安娜望向窗外。",
                source="keeper",
            )
        ],
        created_at=state.world_time,
    )
    fake = FakeLLM(
        {
            "beats": [
                "安娜笑着点点头，谈起窗外温暖的阳光。",
                "她端起杯子，补充说午后或许也会这样晴朗。",
                "她忽然把话题转回了医院深处的秘密。",
            ]
        }
    )

    beats = Narrator(fake, scenario).expand_sequence(state, sequence)

    assert len(beats) == 2
    assert all("医院深处" not in beat for beat in beats)
    assert fake.calls[-1]["user_payload"]["location_description"] == ""
    assert "轻量对话" in fake.calls[-1]["system"]


def test_compatible_gateway_retries_without_response_format():
    class UnsupportedFormat(Exception):
        status_code = 400

    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "response_format" in kwargs:
                raise UnsupportedFormat()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )

    completions = Completions()
    client = OpenAICompatibleLLM(LLMSettings(enabled=True, api_key="test-only"))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = client.complete_json(system="return json", user_payload={"ping": True})
    assert result.data == {"ok": True}
    assert len(completions.calls) == 2
    assert "response_format" not in completions.calls[-1]


def test_transient_upstream_failure_retries_once_then_succeeds():
    class UpstreamUnavailable(Exception):
        status_code = 503

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise UpstreamUnavailable()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                usage=None,
            )

    completions = Completions()
    client = OpenAICompatibleLLM(
        LLMSettings(
            enabled=True,
            api_key="test-only",
            max_retries=1,
            retry_delay_seconds=0,
        )
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = client.complete_json(system="json", user_payload={})
    assert result.data == {"ok": True}
    assert completions.calls == 2
    assert client.enabled is True


def test_upstream_failure_opens_short_circuit():
    class UpstreamUnavailable(Exception):
        status_code = 503

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            raise UpstreamUnavailable()

    completions = Completions()
    client = OpenAICompatibleLLM(
        LLMSettings(
            enabled=True,
            api_key="test-only",
            cooldown_seconds=60,
            max_retries=1,
            retry_delay_seconds=0,
        )
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with pytest.raises(UpstreamUnavailable):
        client.complete_json(system="json", user_payload={})
    assert completions.calls == 2
    assert client.enabled is False


def _routable_provider(name, *, output=None, error=None):
    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if error is not None:
                raise error
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=output or '{"ok":true}'))],
                usage=None,
            )

    completions = Completions()
    provider = OpenAICompatibleLLM(
        LLMSettings(
            enabled=True,
            api_key="test-only",
            model=f"{name}-model",
            max_retries=0,
            retry_delay_seconds=0,
            cooldown_seconds=60,
        ),
        provider=name,
    )
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_routed_llm_prefers_local_without_calling_remote():
    local, local_calls = _routable_provider("local", output='{"route":"local"}')
    remote, remote_calls = _routable_provider("remote", output='{"route":"remote"}')
    router = RoutedLLM(local=local, remote=remote, mode="auto")

    result = router.complete_json(system="json", user_payload={})

    assert result.data == {"route": "local"}
    assert local_calls.calls == 1
    assert remote_calls.calls == 0
    assert router.last_used["provider"] == "local"


def test_routed_llm_falls_back_to_remote_after_local_failure():
    class LocalUnavailable(Exception):
        status_code = 503

    local, local_calls = _routable_provider("local", error=LocalUnavailable())
    remote, remote_calls = _routable_provider("remote", output='{"route":"remote"}')
    router = RoutedLLM(local=local, remote=remote, mode="auto")

    result = router.complete_json(system="json", user_payload={})

    assert result.data == {"route": "remote"}
    assert local_calls.calls == 1
    assert remote_calls.calls == 1
    assert router.last_used["provider"] == "remote"
    assert router.last_failures["local"] == "上游服务暂时不可用（503）"


def test_routed_llm_honors_remote_only_selection():
    local, local_calls = _routable_provider("local", output='{"route":"local"}')
    remote, remote_calls = _routable_provider("remote", output='{"route":"remote"}')
    router = RoutedLLM(local=local, remote=remote, mode="remote")

    result = router.complete_json(system="json", user_payload={})

    assert result.data == {"route": "remote"}
    assert local_calls.calls == 0
    assert remote_calls.calls == 1


def test_routed_llm_reports_all_provider_failures():
    class Unavailable(Exception):
        status_code = 503

    local, _ = _routable_provider("local", error=Unavailable())
    remote, _ = _routable_provider("remote", error=Unavailable())
    router = RoutedLLM(local=local, remote=remote, mode="auto")

    with pytest.raises(LLMUnavailable) as raised:
        router.complete_json(system="json", user_payload={})

    assert set(raised.value.failures) == {"local", "remote"}
    assert "右上角的模型设置" in raised.value.public_message


def test_keeper_fails_closed_on_upstream_failure(scenario, state):
    class UpstreamUnavailable(Exception):
        status_code = 503

    class BrokenLLM:
        enabled = True
        configured = True

        def complete_json(self, **_kwargs):
            raise UpstreamUnavailable()

    intent = _player_intent("我回家休息，明天再来")
    with pytest.raises(LLMUnavailable):
        Keeper(BrokenLLM(), scenario).adjudicate(state, intent, scenario.actions)
    assert state.agent_calls[-1].validation == "rejected"
