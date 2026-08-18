from __future__ import annotations

from datetime import datetime

from living_tabletop.engine import GameEngine
from living_tabletop.llm import LLMSettings, OpenAICompatibleLLM
from living_tabletop.models import ActionType, LLMResult, OpenActionPlan, RuleChoice
from living_tabletop.scenario import create_initial_state, load_scenario


class IntentLLM:
    enabled = True

    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(data=self.output, latency_ms=2, input_tokens=10, output_tokens=10)


def _engine_with_intent(scenario, output):
    llm = IntentLLM(output)
    return GameEngine(scenario, llm), llm


def test_public_view_separates_first_person_dialogue_from_action_suggestions(engine, state):
    view = engine.public_view(state)
    assert len(view["suggested_actions"]) == 3
    categories = {action["category"] for action in view["suggested_actions"]}
    assert "social" not in categories
    assert len(view["dialogue_options"]) == 3
    assert all(option["action_id"] == "lobby_talk_anna" for option in view["dialogue_options"])
    assert all(option["text"].startswith("“安娜") for option in view["dialogue_options"])
    assert "director" not in view
    assert "threats" not in view


def test_public_view_projects_current_scenario_as_local_visual_scene(engine, state):
    view = engine.public_view(state)
    visual = view["scene"]["visual"]
    assert visual["location_id"] == "loc_lobby"
    assert visual["archetype"] == "hospital"
    assert visual["player"]["id"] == state.player.entity_id
    assert {actor["id"] for actor in visual["actors"]} == {"npc_anna"}
    assert {item["id"] for item in visual["objects"]} == {"item_lantern"}
    assert visual["objects"][0]["interaction"]["action_id"] == "lobby_take_lantern"
    hotspot_actions = {item["interaction"]["action_id"] for item in visual["hotspots"]}
    assert "lobby_guestbook" in hotspot_actions
    exits = {item["destination_id"]: item for item in visual["exits"]}
    assert exits["loc_ward"]["interaction"]["action_id"] == "move__loc_lobby__loc_ward"
    assert exits["loc_basement"]["available"] is False
    assert exits["loc_basement"]["interaction"] is None


def test_visual_scene_click_action_uses_existing_engine_path(engine, state):
    visual = engine.public_view(state)["scene"]["visual"]
    lantern = next(item for item in visual["objects"] if item["id"] == "item_lantern")
    next_state, resolution = engine.play(
        state,
        action_id=lantern["interaction"]["action_id"],
    )
    assert resolution.accepted is True
    assert "item_lantern" in next_state.player.inventory
    next_visual = engine.public_view(next_state)["scene"]["visual"]
    assert "item_lantern" not in {item["id"] for item in next_visual["objects"]}


def test_visual_scene_changes_with_authored_movement(engine, state):
    next_state, resolution = engine.play(state, action_id="move__loc_lobby__loc_ward")
    assert resolution.accepted is True
    visual = engine.public_view(next_state)["scene"]["visual"]
    assert visual["location_id"] == "loc_ward"
    assert "npc_anna" not in {actor["id"] for actor in visual["actors"]}


def test_full_world_graph_is_developer_only(engine, state):
    public = engine.public_view(state)
    developer = engine.developer_view(state)
    assert "world_map" not in public
    assert len(developer["world_map"]["locations"]) == 9
    assert any(item["player_here"] for item in developer["world_map"]["locations"])


def test_free_text_alias_is_interpreted_by_llm(scenario, state):
    engine, llm = _engine_with_intent(
        scenario,
        {"existing_action_id": "lobby_guestbook", "confidence": 0.98},
    )
    next_state, resolution = engine.play(state, text="我想仔细看看访客簿")
    assert resolution.action_id == "lobby_guestbook"
    assert resolution.accepted is True
    assert next_state.world_time > state.world_time
    assert len(llm.calls) == 1
    assert next_state.event_log[-2].payload["intent_source"] == "llm"


def test_free_text_scene_object_is_interpreted_by_llm(scenario, state):
    engine, _llm = _engine_with_intent(
        scenario,
        {"existing_action_id": "lobby_guestbook", "confidence": 0.96},
    )
    next_state, resolution = engine.play(state, text="我想仔细检查接待台，看看有没有里德留下的东西")
    assert resolution.action_id == "lobby_guestbook"
    assert resolution.accepted is True
    assert next_state.world_time > state.world_time


def test_social_deception_phrase_is_interpreted_by_llm(scenario, state):
    engine, _llm = _engine_with_intent(
        scenario,
        {"existing_action_id": "lobby_talk_anna", "confidence": 0.94},
    )
    _next_state, resolution = engine.play(state, text="我骗安娜说警察来了，试探她的反应")
    assert resolution.action_id == "lobby_talk_anna"
    assert resolution.accepted is True


def test_selected_dialogue_utterance_becomes_first_performance_beat(engine, state):
    utterance = "“安娜，等等。请从你亲眼看见的地方说起。”"
    next_state, resolution = engine.play(
        state,
        action_id="lobby_talk_anna",
        utterance=utterance,
    )

    assert resolution.accepted is True
    assert next_state.narrative_sequence is not None
    assert next_state.narrative_sequence.beats[0].text == utterance
    assert len(next_state.narrative_sequence.beats) >= 4
    assert any("“" in beat.text for beat in next_state.narrative_sequence.beats[1:])
    started = next(event for event in next_state.event_log if event.type == "action_started")
    assert started.payload["player_text"] == utterance


def test_key_investigation_has_rich_authored_performance_without_llm(engine, state):
    next_state, resolution = engine.play(state, action_id="lobby_guestbook")

    assert resolution.accepted is True
    assert next_state.narrative_sequence is not None
    authored = [
        beat.text
        for beat in next_state.narrative_sequence.beats
        if beat.source == "authored"
    ]
    assert len(authored) >= 2
    assert sum(map(len, authored)) >= 100
    assert [entry.text for entry in next_state.visible_history] == [
        beat.text for beat in next_state.narrative_sequence.beats
    ]
    assert all(entry.kind == "hard_canon" for entry in next_state.visible_history)


def test_unlisted_free_text_uses_llm_open_plan(scenario, state):
    engine, _llm = _engine_with_intent(
        scenario,
        {
            "existing_action_id": None,
            "confidence": 0.82,
            "open_plan": {
                "label": "处理眼前的杂事",
                "action_type": "OTHER",
                "goal": "我处理一下这里",
                "duration_minutes": 5,
                "resolution": "automatic",
                "risk": "safe",
                "success_text": "你花了几分钟处理眼前的杂事。",
            },
        },
    )
    next_state, resolution = engine.play(state, text="我处理一下这里")
    assert resolution.accepted is True
    assert resolution.needs_clarification is False
    assert resolution.action_id.startswith("open__")
    assert next_state.world_time > state.world_time
    assert next_state.version == state.version + 1
    assert next_state.flags["open_action_count"] == 1
    assert next_state.narrative_sequence.player_text == "我处理一下这里"
    assert all(
        beat.source == "keeper"
        for beat in next_state.narrative_sequence.beats
        if "处理眼前的杂事" in beat.text
    )
    assert any(entry.kind == "soft_canon" for entry in next_state.visible_history)


def test_open_impossible_attempt_is_accepted_without_granting_result(engine, state):
    plan = OpenActionPlan(
        label="徒手飞上月球",
        action_type=ActionType.MOVE,
        goal="我徒手飞上月球",
        destination_name="月球",
        duration_minutes=1,
        resolution="impossible",
        risk="safe",
        success_text="你飞上了月球。",
        failure_text="你奋力跃起，但重力没有因为意志而消失。",
    )
    next_state, resolution = engine.play(
        state,
        text=plan.goal,
        recorded_open_plan=plan,
    )
    assert resolution.accepted is True
    assert resolution.check and resolution.check.succeeded is False
    assert next_state.entities["player"].location == state.entities["player"].location
    assert "重力" in next_state.last_narrative


def test_open_route_can_be_interrupted_before_leaving_scene(engine, state):
    state.world_time = datetime.fromisoformat("1927-12-17T21:16:00")
    state.event_queue = [event for event in state.event_queue if event.time >= state.world_time]
    plan = OpenActionPlan(
        label="离开医院去车站",
        action_type=ActionType.MOVE,
        goal="我离开医院去车站",
        destination_name="火车站",
        duration_minutes=30,
        resolution="automatic",
        risk="safe",
        success_text="你离开医院，前往火车站。",
    )
    next_state, resolution = engine.play(
        state,
        text=plan.goal,
        recorded_open_plan=plan,
    )
    assert resolution.accepted is True
    assert resolution.interrupted is True
    assert next_state.entities["player"].location == "loc_lobby"
    assert "火车站" not in [entity.name for entity in next_state.entities.values()]


def test_scheduled_event_interrupts_action(engine, state):
    state.world_time = datetime.fromisoformat("1927-12-17T21:14:00")
    next_state, resolution = engine.play(state, action_id="lobby_guestbook")
    assert resolution.interrupted is True
    assert resolution.interrupting_event_id == "scheduled_power_failure"
    assert next_state.world_time == datetime.fromisoformat("1927-12-17T21:18:00")
    assert "lobby_guestbook" not in next_state.completed_actions


def test_unlock_then_move_to_basement(engine, state):
    state.player.inventory.append("item_archive_key")
    state.entities["item_archive_key"].location = "player"
    unlocked, resolution = engine.play(state, action_id="lobby_unlock_basement")
    assert resolution.accepted and unlocked.facts["f_basement_locked"].value is False
    moved, resolution = engine.play(unlocked, action_id="move__loc_lobby__loc_basement")
    assert resolution.check.succeeded
    assert moved.entities["player"].location == "loc_basement"


def test_scene_change_builds_multi_beat_authored_sequence(engine, state):
    moved, resolution = engine.play(state, action_id="move__loc_lobby__loc_ward")

    assert resolution.accepted
    assert moved.narrative_sequence is not None
    assert moved.narrative_sequence.state_version == moved.version
    assert moved.narrative_sequence.status == "ready"
    assert moved.narrative_sequence.action_type == ActionType.MOVE
    scene_beats = [
        beat for beat in moved.narrative_sequence.beats if beat.source == "scene"
    ]
    assert len(scene_beats) == 3
    assert any("消毒水" in beat.text for beat in scene_beats)


def _pending_failure(engine, scenario):
    for seed in range(1, 200):
        candidate = create_initial_state(scenario, seed=seed)
        candidate.player.luck = 99
        offered, resolution = engine.play(
            candidate,
            action_id="lobby_guestbook",
            interactive_rules=True,
        )
        if resolution.awaiting_rule_choice and offered.pending_check:
            return offered, resolution
    raise AssertionError("No deterministic pending failure found")


def _pending_dialogue_failure(engine, scenario):
    utterance = "“安娜，请把你亲眼看到的事情告诉我。”"
    for seed in range(1, 200):
        candidate = create_initial_state(scenario, seed=seed)
        candidate.player.luck = 99
        offered, resolution = engine.play(
            candidate,
            action_id="lobby_talk_anna",
            utterance=utterance,
            interactive_rules=True,
        )
        if resolution.awaiting_rule_choice and offered.pending_check:
            return offered, resolution, utterance
    raise AssertionError("No deterministic pending dialogue failure found")


def test_pending_dialogue_plays_neutral_transition_not_failure_scene(engine, scenario):
    offered, resolution, utterance = _pending_dialogue_failure(engine, scenario)
    action = next(action for action in scenario.actions if action.id == "lobby_talk_anna")
    texts = [beat.text for beat in offered.narrative_sequence.beats]

    assert resolution.awaiting_rule_choice is True
    assert texts.count(utterance) == 1
    assert action.failure_text not in texts
    assert all(text not in texts for text in action.failure_beats)
    assert any("真正的回答仍悬在" in text for text in texts)


def test_rule_choice_continues_dialogue_without_replaying_player_line(engine, scenario):
    offered, _resolution, utterance = _pending_dialogue_failure(engine, scenario)
    resolved, final = engine.play(offered, rule_choice=RuleChoice.SPEND_LUCK)
    sequence = resolved.narrative_sequence

    assert final.check and final.check.succeeded
    assert final.continues_previous_narrative is True
    assert sequence is not None and sequence.continues_previous is True
    assert utterance not in [beat.text for beat in sequence.beats]
    assert sequence.beats[0].text.startswith("安娜没有立刻回答")


def test_failed_skill_roll_can_be_converted_with_luck(engine, scenario):
    offered, resolution = _pending_failure(engine, scenario)
    assert RuleChoice.SPEND_LUCK in resolution.rule_choices
    cost = resolution.luck_cost
    luck_before = offered.player.luck

    resolved, final = engine.play(offered, rule_choice=RuleChoice.SPEND_LUCK)

    assert final.accepted and final.check and final.check.succeeded
    assert final.check.luck_spent == cost
    assert resolved.player.luck == luck_before - cost
    assert resolved.pending_check is None
    assert "f_missing_note" in resolved.player_known_fact_ids


def test_failed_skill_roll_can_be_pushed_and_replayed(engine, scenario):
    for seed in range(1, 400):
        candidate = create_initial_state(scenario, seed=seed)
        offered, first = engine.play(
            candidate,
            action_id="lobby_guestbook",
            interactive_rules=True,
        )
        if not first.awaiting_rule_choice or RuleChoice.PUSH_ROLL not in first.rule_choices:
            continue
        resolved, second = engine.play(offered, rule_choice=RuleChoice.PUSH_ROLL)
        if second.check and second.check.succeeded:
            break
    else:
        raise AssertionError("No deterministic successful pushed roll found")

    assert second.check.pushed is True
    assert resolved.pending_check is None
    assert resolved.world_time > offered.world_time
    assert any(event.type == "rule_choice_made" for event in resolved.event_log)


def test_opposed_combat_roll_does_not_offer_push_or_luck(offline_llm):
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    engine = GameEngine(scenario, offline_llm)
    state = create_initial_state(scenario, seed=9)
    state.entities["player"].location = "loc_hidden_lair"
    state.entities["creature_corbitt"].active = True
    state.player.inventory.append("item_magic_dagger")
    state.entities["item_magic_dagger"].location = "player"

    resolved, resolution = engine.play(
        state,
        action_id="lair_use_dagger",
        interactive_rules=True,
    )

    assert resolution.accepted
    assert resolution.check and resolution.check.opponent
    assert resolution.awaiting_rule_choice is False
    assert resolved.pending_check is None
