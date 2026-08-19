from __future__ import annotations

from copy import deepcopy

import pytest

from living_tabletop.engine import GameEngine
from living_tabletop.kernel import KernelValidationError, WorldKernel
from living_tabletop.llm import LLMUnavailable
from living_tabletop.models import Effect, Entity, EntityType, LLMResult
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
    proposed_item_changes: list[dict] | None = None,
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
        "proposed_item_changes": proposed_item_changes or [],
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


def test_v2_npc_gift_creates_item_fact_and_inventory_in_one_atomic_turn():
    scenario, state = _haunting()
    text = "你能给我一件防身武器吗？"
    llm = QueueV2LLM(
        _open_turn(
            text,
            performance=[
                "诺特从外套内袋取出一把老式左轮手枪，递给你。"
                "“拿着吧，只在确实需要时使用。”"
            ],
            proposed_item_changes=[
                {
                    "operation": "acquire",
                    "item_entity_id": None,
                    "item_name": "老式左轮手枪",
                    "item_kind": "weapon",
                    "counterparty_entity_id": "npc_knott",
                    "origin": "gift",
                    "description": "枪身保养良好、握把略有磨损的旧式左轮手枪。",
                    "reason": "诺特明确把它交给玩家防身",
                }
            ],
        )
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    new_items = [
        resolved.entities[item_id]
        for item_id in resolved.player.inventory
        if item_id not in state.player.inventory
    ]
    assert len(new_items) == 1
    weapon = new_items[0]
    assert weapon.name == "老式左轮手枪"
    assert weapon.location == resolved.player.entity_id
    assert {"weapon", "dynamic", "soft_canon"} <= weapon.tags
    possession_facts = [
        fact
        for fact in resolved.facts.values()
        if fact.subject == weapon.id and fact.predicate == "possession_change"
    ]
    assert len(possession_facts) == 1
    assert possession_facts[0].id in resolved.player_known_fact_ids
    assert any(event.type == "entity_created" and event.target == weapon.id for event in resolved.event_log)
    assert any(event.type == "item_acquired" and event.target == weapon.id for event in resolved.event_log)
    assert resolved.turn_traces[-1]["state_diff"]["acquired_item_ids"] == [weapon.id]
    committed = next(
        event for event in resolved.event_log if event.type == "open_plan_committed"
    )
    assert committed.payload["world_effect_ops"] == [
        "create_entity",
        "add_inventory",
        "create_fact",
        "reveal_fact",
    ]


def test_v2_acquired_dynamic_item_is_available_in_the_next_turn_context():
    scenario, state = _haunting()
    first_text = "请给我一张可以联系你的名片。"
    second_text = "我看看刚收到的名片。"
    llm = QueueV2LLM(
        _open_turn(
            first_text,
            performance=["诺特写下一张私人名片，递给你。“有事可以按这个地址找我。”"],
            proposed_item_changes=[
                {
                    "operation": "acquire",
                    "item_entity_id": None,
                    "item_name": "诺特的私人名片",
                    "item_kind": "document",
                    "counterparty_entity_id": "npc_knott",
                    "origin": "gift",
                    "description": "写着诺特联络地址的小卡片。",
                    "reason": "诺特把名片递给玩家",
                }
            ],
        ),
        _open_turn(second_text, performance=["你重新看了一眼名片上的字迹。"], action_type="OTHER", target_id=None, target_name=None),
    )
    engine = GameEngine(scenario, llm)

    state, _ = engine.play(state, text=first_text)
    item_id = next(
        item_id for item_id in state.player.inventory if state.entities[item_id].name == "诺特的私人名片"
    )
    state, _ = engine.play(state, text=second_text)

    second_inventory = llm.calls[1]["user_payload"]["context"]["inventory"]
    assert {"id": item_id, "name": "诺特的私人名片"} in second_inventory


def test_v2_existing_scene_item_is_transferred_without_cloning_it():
    scenario, state = _haunting()
    scene_id = state.entities[state.player.entity_id].location
    state.entities["item_receipt"] = Entity(
        id="item_receipt",
        type=EntityType.ITEM,
        name="咖啡馆收据",
        location=scene_id,
        tags={"document"},
    )
    text = "我拿起桌上的咖啡馆收据。"
    llm = QueueV2LLM(
        _open_turn(
            text,
            action_type="TAKE",
            target_id="item_receipt",
            target_name="咖啡馆收据",
            performance=["你拿起桌上的咖啡馆收据，把它收进口袋。"],
            proposed_item_changes=[
                {
                    "operation": "acquire",
                    "item_entity_id": "item_receipt",
                    "item_name": "咖啡馆收据",
                    "item_kind": "document",
                    "counterparty_entity_id": None,
                    "origin": "pickup",
                    "description": "桌上原有的消费收据。",
                    "reason": "玩家明确拿起收据",
                }
            ],
        )
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert resolved.player.inventory.count("item_receipt") == 1
    assert len([entity for entity in resolved.entities.values() if entity.name == "咖啡馆收据"]) == 1
    assert not any(
        event.type == "entity_created" and event.target == "item_receipt"
        for event in resolved.event_log
    )


def test_v2_rejects_visible_item_transfer_without_a_matching_world_change():
    scenario, state = _haunting()
    initial_inventory = list(state.player.inventory)
    text = "你能给我一件武器吗？"
    llm = QueueV2LLM(
        _open_turn(
            text,
            performance=["诺特掏出一把左轮手枪递给你。“拿去防身。”"],
        )
    )

    with pytest.raises(LLMUnavailable, match="invalid item world change"):
        GameEngine(scenario, llm).play(state, text=text)

    assert state.player.inventory == initial_inventory
    assert not any("dynamic" in entity.tags for entity in state.entities.values())


def test_v2_does_not_treat_recalled_item_transfer_as_a_new_world_change():
    scenario, state = _haunting()
    text = "刚才那把枪是谁递给我的？"
    llm = QueueV2LLM(
        _open_turn(
            text,
            performance=[
                "诺特顺着你关于“刚才那把左轮手枪递给你”的复述回答："
                "“那是你先前见过的事情，不是现在又发生了一次。”"
            ],
        )
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert resolved.player.inventory == state.player.inventory
    assert not any(event.type == "item_acquired" for event in resolved.event_log)


def test_v2_rejects_item_change_from_an_inaccessible_counterparty():
    scenario, state = _haunting()
    text = "请把手枪给我。"
    llm = QueueV2LLM(
        _open_turn(
            text,
            performance=["一个不在场的人把老式左轮手枪递给你。"],
            proposed_item_changes=[
                {
                    "operation": "acquire",
                    "item_entity_id": None,
                    "item_name": "老式左轮手枪",
                    "item_kind": "weapon",
                    "counterparty_entity_id": "npc_not_present",
                    "origin": "gift",
                    "description": "不应被创建。",
                    "reason": "非法来源",
                }
            ],
        )
    )

    with pytest.raises(LLMUnavailable, match="invalid item world change"):
        GameEngine(scenario, llm).play(state, text=text)

    assert not any(entity.name == "老式左轮手枪" for entity in state.entities.values())


def test_kernel_compound_effects_are_atomic_when_a_later_item_effect_is_invalid():
    scenario, state = _haunting()
    before = deepcopy(state)
    orphan = Entity(
        id="item_should_not_survive",
        type=EntityType.ITEM,
        name="不应残留的物品",
        location=state.entities[state.player.entity_id].location,
    )
    effects = [
        Effect(op="create_entity", params={"entity": orphan.model_dump(mode="json")}),
        Effect(op="add_inventory", params={"item_id": "missing_item"}),
    ]

    with pytest.raises(KernelValidationError):
        WorldKernel(scenario).apply_effects(state, effects, source="test")

    assert state == before


def test_v2_repeated_acquisition_does_not_clone_a_dynamic_item():
    scenario, state = _haunting()
    first_text = "请给我一支铅笔。"
    second_text = "我把刚才那支铅笔重新收进口袋。"
    proposal = {
        "operation": "acquire",
        "item_entity_id": None,
        "item_name": "削好的铅笔",
        "item_kind": "tool",
        "counterparty_entity_id": "npc_knott",
        "origin": "gift",
        "description": "一支削好的普通铅笔。",
        "reason": "诺特把铅笔交给玩家",
    }
    llm = QueueV2LLM(
        _open_turn(
            first_text,
            performance=["诺特从账本旁拿起一支削好的铅笔递给你，你随手收进口袋。"],
            proposed_item_changes=[proposal],
        ),
        _open_turn(
            second_text,
            performance=["你确认那支削好的铅笔仍在手中，又把它收进口袋。"],
            proposed_item_changes=[proposal],
        ),
    )
    engine = GameEngine(scenario, llm)

    state, _ = engine.play(state, text=first_text)
    state, _ = engine.play(state, text=second_text)

    pencils = [entity for entity in state.entities.values() if entity.name == "削好的铅笔"]
    assert len(pencils) == 1
    assert state.player.inventory.count(pencils[0].id) == 1
    assert len(
        [event for event in state.event_log if event.type == "item_acquired" and event.target == pencils[0].id]
    ) == 1


def test_v2_relinquish_moves_carried_item_to_present_counterparty():
    scenario, state = _haunting()
    text = "我把科比特宅邸钥匙还给诺特。"
    assert "item_house_key" in state.player.inventory
    llm = QueueV2LLM(
        _open_turn(
            text,
            performance=["你取出科比特宅邸钥匙交还给诺特，他沉默地把钥匙收进外套。"],
            proposed_item_changes=[
                {
                    "operation": "relinquish",
                    "item_entity_id": "item_house_key",
                    "item_name": "科比特宅邸钥匙",
                    "item_kind": "key",
                    "counterparty_entity_id": "npc_knott",
                    "origin": "return",
                    "description": "委托开始时交给玩家的旧钥匙。",
                    "reason": "玩家明确把钥匙还给诺特",
                }
            ],
        )
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert "item_house_key" not in resolved.player.inventory
    assert resolved.entities["item_house_key"].location == "npc_knott"
    assert any(
        event.type == "item_removed"
        and event.target == "item_house_key"
        and event.payload["destination"] == "npc_knott"
        for event in resolved.event_log
    )
    assert resolved.turn_traces[-1]["state_diff"]["removed_item_ids"] == [
        "item_house_key"
    ]


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
