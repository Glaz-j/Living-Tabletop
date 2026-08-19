from __future__ import annotations

from datetime import datetime

from living_tabletop.engine import GameEngine
from living_tabletop.llm import LLMSettings, OpenAICompatibleLLM
from living_tabletop.models import ActionType, LLMResult, OpenActionPlan, RuleChoice
from living_tabletop.replay import verify_replay
from living_tabletop.scenario import create_initial_state, load_scenario
from living_tabletop.storage import SQLiteRepository


class PlanLLM:
    enabled = True

    def __init__(self, output):
        self.output = output

    def complete_json(self, **_kwargs):
        return LLMResult(data=self.output, latency_ms=2, input_tokens=10, output_tokens=10)


def _haunting_engine_and_state(llm=None):
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    llm = llm or OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    engine = GameEngine(scenario, llm)
    state = create_initial_state(scenario, seed=19)
    state.entities["player"].location = "loc_house_upper"
    state.world_time = datetime.fromisoformat("1920-09-17T18:22:00")
    state.event_queue = [event for event in state.event_queue if event.time >= state.world_time]
    return scenario, engine, state


def test_go_home_rest_and_return_later_is_valid_open_world_choice():
    llm = PlanLLM(
        {
            "existing_action_id": None,
            "confidence": 0.97,
            "open_plan": {
                "label": "回家休息到次日",
                "action_type": "REST",
                "goal": "我想回家休息一下，然后第二天再来",
                "destination_name": "调查员的家",
                "destination_description": "调查员在城中的私人住处。",
                "duration_minutes": 30,
                "resolution": "automatic",
                "risk": "safe",
                "rest_until_hour": 8,
                "rest_day_offset": 1,
                "success_text": "你回家休息，并在第二天早晨醒来。",
            },
        }
    )
    _scenario, engine, state = _haunting_engine_and_state(llm)
    threat_before = state.threats["threat_corbitt_awareness"].progress
    resolved, resolution = engine.play(state, text="我想回家休息一下，然后第二天再来")
    player_location = resolved.entities["player"].location
    assert resolution.accepted is True
    assert resolution.needs_clarification is False
    assert resolved.world_time == datetime.fromisoformat("1920-09-18T08:00:00")
    assert resolved.entities[player_location].name == "调查员的家"
    assert "off_main" in resolved.entities[player_location].tags
    assert resolved.threats["threat_corbitt_awareness"].progress > threat_before
    assert any(event.type == "house_stirs" for event in resolved.event_log)
    assert resolution.interrupted is False


def test_open_world_action_replay_is_deterministic(tmp_path):
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    llm = PlanLLM(
        {
            "existing_action_id": None,
            "confidence": 0.95,
            "open_plan": {
                "label": "前往火车站",
                "action_type": "MOVE",
                "goal": "我去火车站",
                "destination_name": "火车站",
                "destination_description": "波士顿仍有列车停靠的车站。",
                "duration_minutes": 30,
                "resolution": "automatic",
                "risk": "safe",
                "success_text": "你离开当前地点，抵达火车站。",
            },
        }
    )
    engine = GameEngine(scenario, llm)
    state = create_initial_state(scenario, seed=19)
    state, resolution = engine.play(state, text="我去火车站")
    assert resolution.accepted
    repository = SQLiteRepository(tmp_path / "open-world.db")
    repository.save(state)
    verified, expected, actual = verify_replay(repository.export(state.session_id), scenario)
    assert verified, (expected, actual)


def test_recorded_move_plan_cannot_move_without_player_commitment():
    _scenario, engine, state = _haunting_engine_and_state()
    origin = state.entities["player"].location
    time_before = state.world_time
    plan = OpenActionPlan(
        label="前往罗克斯伯里疗养院",
        action_type=ActionType.MOVE,
        goal="疗养院在什么地方？",
        destination_name="罗克斯伯里疗养院",
        destination_entity_id="loc_sanitarium",
        duration_minutes=15,
        success_text="你抵达疗养院。",
    )

    resolved, resolution = engine.play(
        state,
        text="疗养院在什么地方？",
        recorded_open_plan=plan,
    )

    assert resolution.accepted is False
    assert resolution.needs_clarification is True
    assert resolved.entities["player"].location == origin
    assert resolved.world_time == time_before
    assert not any(event.type == "entity_moved" for event in resolved.event_log)


def test_open_skill_check_keeps_coc_rule_choices_and_replays(tmp_path):
    scenario = load_scenario()
    llm = OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    engine = GameEngine(scenario, llm)
    plan = OpenActionPlan(
        label="辨认陌生的墙上划痕",
        action_type=ActionType.EXAMINE,
        goal="看看墙上的陌生划痕是什么意思",
        duration_minutes=5,
        resolution="check",
        skill="observation",
        difficulty="regular",
        risk="uncertain",
        success_text="你从划痕中辨认出了一些可理解的规律。",
        failure_text="这些划痕暂时无法被可靠地解读。",
    )
    for seed in range(1, 200):
        state = create_initial_state(scenario, seed=seed)
        offered, resolution = engine.play(
            state,
            text=plan.goal,
            recorded_open_plan=plan,
            interactive_rules=True,
        )
        if resolution.awaiting_rule_choice:
            break
    else:
        raise AssertionError("No deterministic open-action pending failure found")

    assert offered.pending_check is not None
    assert offered.pending_check.dynamic_action is not None
    resolved, final = engine.play(offered, rule_choice=RuleChoice.ACCEPT_FAILURE)
    assert final.accepted is True
    assert resolved.pending_check is None

    repository = SQLiteRepository(tmp_path / "open-check.db")
    repository.save(resolved)
    verified, expected, actual = verify_replay(repository.export(resolved.session_id), scenario)
    assert verified, (expected, actual)


def test_open_travel_attempt_respects_locked_world_routes():
    scenario = load_scenario()
    scenario.location_graph = {
        "loc_lobby": {"loc_basement": 3},
        "loc_basement": {"loc_lobby": 3},
    }
    llm = OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    engine = GameEngine(scenario, llm)
    state = create_initial_state(scenario, seed=19)
    plan = OpenActionPlan(
        label="直接下到锁住的地下室",
        action_type=ActionType.MOVE,
        goal="我直接去锁住的地下室",
        destination_name="地下室",
        destination_entity_id="loc_basement",
        duration_minutes=5,
        resolution="automatic",
        risk="safe",
        success_text="你抵达地下室。",
    )

    resolved, resolution = engine.play(
        state,
        text=plan.goal,
        recorded_open_plan=plan,
    )

    assert resolution.accepted is True
    assert resolution.check and resolution.check.succeeded is False
    assert resolved.entities[resolved.player.entity_id].location == "loc_lobby"
    assert "无法抵达" in resolved.last_narrative
