from __future__ import annotations

from copy import deepcopy

from living_tabletop.engine import GameEngine
from living_tabletop.models import LLMResult
from living_tabletop.scenario import create_initial_state, load_scenario
from living_tabletop.test_agent import PERSONAS, run_test_agents


class QueueV2LLM:
    enabled = True

    def __init__(self, *outputs: dict):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return LLMResult(
            data=deepcopy(output),
            latency_ms=2,
            input_tokens=40,
            output_tokens=40,
        )


def _open_turn(
    text: str,
    *,
    performance: list[str],
    action_type: str = "TALK",
    target_id: str | None = "npc_knott",
    target_name: str | None = "史蒂文·诺特",
    resolution: str = "automatic",
    skill: str | None = None,
    failure: list[str] | None = None,
    proposed_facts: list[dict] | None = None,
    used_fact_ids: list[str] | None = None,
) -> dict:
    return {
        "decision": {
            "existing_action_id": None,
            "open_plan": {
                "label": "完整处理玩家当前输入",
                "action_type": action_type,
                "goal": text,
                "target_name": target_name,
                "target_entity_id": target_id,
                "duration_minutes": 1,
                "resolution": resolution,
                "skill": skill,
                "risk": "uncertain" if resolution == "check" else "safe",
                "speech_act": "question" if "？" in text else "statement",
                "addressee_id": target_id if action_type in {"TALK", "DECEIVE"} else None,
            },
            "confidence": 0.95,
        },
        "performance": performance,
        "failure_performance": failure or [],
        "used_fact_ids": used_fact_ids or [],
        "proposed_facts": proposed_facts or [],
        "answered_query_parts": [text] if "？" in text else [],
        "unresolved_query_parts": [],
    }


def _haunting():
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    return scenario, create_initial_state(scenario, seed=19)


def test_v2_composer_answers_and_commits_soft_fact_in_one_foreground_call():
    scenario, state = _haunting()
    text = "罗克斯伯里疗养院具体在哪里？我只是问路，暂时不去。"
    route = "在罗克斯伯里区旧电车总站北侧，沿石墙走到第二个路口"
    llm = QueueV2LLM(
        _open_turn(
            text,
            performance=[f"“{route}。到了那里再问门房。”诺特直接回答。"],
            proposed_facts=[
                {
                    "subject_entity_id": "loc_sanitarium",
                    "predicate": "route_description",
                    "value": route,
                    "confidence": 0.91,
                }
            ],
        )
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert resolved.entities["player"].location == state.entities["player"].location
    assert len(llm.calls) == 1
    assert llm.calls[0]["schema_name"] == "TurnCompositionOutput"
    assert all(
        "dialogue_text" not in action
        for action in llm.calls[0]["user_payload"]["context"]["available_actions"]
    )
    created = next(
        fact
        for fact in resolved.facts.values()
        if fact.subject == "loc_sanitarium" and fact.predicate == "route_description"
    )
    assert created.value == route
    assert created.canon == "soft_canon"
    assert created.id in resolved.player_known_fact_ids
    assert resolved.narrative_sequence is not None
    assert route in resolved.narrative_sequence.beats[0].text
    assert resolved.turn_traces[-1]["composition"]["performance"]


def test_v2_context_keeps_verbatim_history_and_retrieves_created_soft_fact():
    scenario, state = _haunting()
    first_text = "疗养院怎么走？"
    route = "沿旧电车线向北两个街区"
    second_text = "你刚才说的路线，再确认一次。"
    llm = QueueV2LLM(
        _open_turn(
            first_text,
            performance=[f"“{route}。”诺特说。"],
            proposed_facts=[
                {
                    "subject_entity_id": "loc_sanitarium",
                    "predicate": "route_description",
                    "value": route,
                    "confidence": 0.9,
                }
            ],
        ),
        _open_turn(second_text, performance=[f"“没错，就是{route}。”诺特点头。"]),
    )
    engine = GameEngine(scenario, llm)

    state, _ = engine.play(state, text=first_text)
    state, _ = engine.play(state, text=second_text)

    second_context = llm.calls[1]["user_payload"]["context"]
    transcript = second_context["recent_visible_history"]
    assert any(first_text == item["text"] and item["source"] == "player" for item in transcript)
    assert any(route in item["text"] for item in transcript)
    assert any(
        fact["predicate"] == "route_description" and fact["value"] == route
        for fact in second_context["player_known_facts"]
    )


def test_v2_unsafe_movement_metadata_is_downgraded_but_prose_survives():
    scenario, state = _haunting()
    text = "疗养院在哪儿？我没有说要去。"
    output = _open_turn(
        text,
        action_type="MOVE",
        target_id=None,
        target_name=None,
        performance=["“在罗克斯伯里区北侧。”诺特回答。"],
    )
    output["decision"]["open_plan"].update(
        {
            "destination_name": "罗克斯伯里疗养院",
            "destination_entity_id": "loc_sanitarium",
        }
    )
    llm = QueueV2LLM(output)

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert resolved.entities["player"].location == state.entities["player"].location
    assert "罗克斯伯里区北侧" in resolved.narrative_sequence.beats[0].text
    assert resolved.agent_calls[-1].role == "turn_composer"


def test_v2_invalid_soft_metadata_does_not_erase_usable_performance():
    scenario, state = _haunting()
    text = "今天天气怎么样？"
    output = _open_turn(
        text,
        performance=["“闷得厉害，下午大概会下雨。诺特望向窗外。"],
    )
    output["proposed_facts"] = [
        {
            "subject_entity_id": "loc_cafe",
            "predicate": "forbidden_secret_predicate",
            "value": "无关的坏元数据",
            "confidence": 0.9,
        }
    ]
    llm = QueueV2LLM(output)

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert "下午大概会下雨" in resolved.narrative_sequence.beats[0].text
    assert resolved.agent_calls[-1].validation == "fallback"
    assert not any(fact.value == "无关的坏元数据" for fact in resolved.facts.values())


def test_v2_cannot_reveal_a_guessed_hidden_fact_without_an_authorized_speaker():
    scenario, state = _haunting()
    text = "我在原地整理思路。"
    hidden_fact_id = "f_macario_tragedy"
    assert hidden_fact_id not in state.player_known_fact_ids
    llm = QueueV2LLM(
        _open_turn(
            text,
            action_type="OTHER",
            target_id=None,
            target_name=None,
            performance=["你把已有的笔记重新排好，没有凭空得到新线索。"],
            used_fact_ids=[hidden_fact_id],
        )
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert hidden_fact_id not in resolved.player_known_fact_ids
    assert resolved.turn_traces[-1]["composition"]["used_fact_ids"] == [hidden_fact_id]
    committed = next(
        event for event in resolved.event_log if event.type == "open_plan_committed"
    )
    assert committed.payload["approved_fact_ids"] == []


def test_v2_mechanical_turn_uses_only_the_rule_selected_composer_branch():
    scenario, state = _haunting()
    text = "我强行掰开生锈的金属盒。"
    llm = QueueV2LLM(
        _open_turn(
            text,
            action_type="FORCE",
            target_id=None,
            target_name="金属盒",
            resolution="check",
            skill="str",
            performance=["锁舌发出脆响，盒盖终于弹开。"],
            failure=["金属边缘纹丝不动，反而在掌心压出一道红痕。"],
        )
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.check is not None and resolution.check.required is True
    visible = "\n".join(beat.text for beat in resolved.narrative_sequence.beats)
    if resolution.check.succeeded:
        assert "盒盖终于弹开" in visible
        assert "纹丝不动" not in visible
    else:
        assert "纹丝不动" in visible
        assert "盒盖终于弹开" not in visible


def test_dynamic_test_agents_cover_all_personas_without_scripts_or_duplicates():
    scenario, _state = _haunting()

    report = run_test_agents(
        scenario,
        personas=PERSONAS,
        seeds=(17,),
        turns_per_run=4,
    )

    assert len(report.runs) == len(PERSONAS)
    assert report.turn_count == len(PERSONAS) * 4
    assert report.unique_input_ratio == 1.0
    assert report.issues == []
    probes = {step.probe for run in report.runs for step in run.steps}
    assert {"location", "negotiation", "mischief", "memory"} <= probes
    assert all(step.player_text for run in report.runs for step in run.steps)
