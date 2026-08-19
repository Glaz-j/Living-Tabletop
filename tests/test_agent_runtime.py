from __future__ import annotations

from copy import deepcopy

from living_tabletop.agent_runtime import (
    ContextAssembler,
    DisclosurePolicy,
    GroundingValidator,
    KnowledgeQuery,
    KnowledgeResolver,
    OutcomeEnvelope,
    PlanValidator,
    PlanValidationError,
    PlayerIntentEnvelope,
    TurnPlannerDecision,
)
import pytest
from living_tabletop.director import Director
from living_tabletop.engine import GameEngine
from living_tabletop.kernel import WorldKernel
from living_tabletop.llm import LLMSettings, LLMUnavailable, OpenAICompatibleLLM
from living_tabletop.models import (
    ActionDefinition,
    ActionType,
    CheckOutcome,
    LLMResult,
    NarrativeBeat,
    NarrativeSequence,
    RuleChoice,
    WorldState,
)
from living_tabletop.replay import verify_replay
from living_tabletop.scenario import create_initial_state, load_scenario, upgrade_world_state
from living_tabletop.storage import SQLiteRepository


class PlannerLLM:
    enabled = True

    def __init__(self, output: dict):
        self.output = output
        self.calls: list[dict] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(data=deepcopy(self.output), latency_ms=3, input_tokens=20, output_tokens=20)


def _question_output(text: str, *, target_id: str, target_name: str) -> dict:
    return {
        "existing_action_id": None,
        "confidence": 0.97,
        "open_plan": {
            "label": "直接询问",
            "action_type": "TALK",
            "goal": text,
            "target_name": target_name,
            "target_entity_id": target_id,
            "duration_minutes": 2,
            "resolution": "automatic",
            "risk": "safe",
            "speech_act": "question",
        },
    }


def _haunting_state(seed: int = 19):
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    return scenario, create_initial_state(scenario, seed=seed)


def test_context_assembler_never_exposes_hidden_npc_knowledge(scenario, state):
    envelope = PlayerIntentEnvelope(
        id="intent_test",
        source="free_text",
        text="安娜，你看见了什么？",
        actor_id=state.player.entity_id,
        scene_id="loc_lobby",
    )
    context = ContextAssembler().assemble(
        state,
        envelope,
        WorldKernel(scenario).available_actions(state),
    )
    serialized = context.model_dump_json()

    assert "f_anna_saw_transfer" not in serialized
    assert "威尔逊把三名病人" not in serialized
    assert {item["id"] for item in context.player_known_facts} == state.player_known_fact_ids


def test_planner_contract_discards_legacy_result_prose_and_effects():
    decision = TurnPlannerDecision.model_validate(
        {
            "existing_action_id": None,
            "confidence": 0.9,
            "open_plan": {
                "label": "看看桌面",
                "action_type": "EXAMINE",
                "goal": "我看看桌面",
                "success_text": "你发现了幕后真凶。",
                "failure_text": "世界毁灭。",
                "success_effects": [{"op": "reveal_fact", "params": {"fact_id": "secret"}}],
            },
        }
    )

    assert decision.open_plan is not None
    assert not hasattr(decision.open_plan, "success_text")
    materialized = PlanValidator.materialize(decision)
    assert "幕后真凶" not in materialized.success_text
    assert materialized.approved_fact_ids == []


def test_plan_validator_removes_hallucinated_referent_and_fact_ids():
    scenario, state = _haunting_state()
    text = "孩子们后来怎么样了？"
    envelope = PlayerIntentEnvelope(
        id="intent_hallucinated_ref",
        source="free_text",
        text=text,
        actor_id="player",
        scene_id="loc_cafe",
    )
    decision = TurnPlannerDecision.model_validate(
        {
            "existing_action_id": None,
            "open_plan": {
                "label": "询问孩子近况",
                "action_type": "TALK",
                "goal": text,
                "target_entity_id": "npc_knott",
                "addressee_id": "npc_knott",
                "speech_act": "question",
                "referents": [
                    {"mention": "孩子们", "entity_id": "npc_invented_children", "confidence": 1}
                ],
                "knowledge_query": {
                    "query_text": text,
                    "asker_id": "player",
                    "addressee_id": "npc_knott",
                    "subject_entity_ids": ["npc_invented_children"],
                    "explicit_fact_ids": ["f_macario_children_status"],
                },
            },
        }
    )

    validated = PlanValidator().validate(
        state,
        envelope,
        decision,
        WorldKernel(scenario).available_actions(state),
    )

    assert validated.open_plan.referents[0].entity_id is None
    assert validated.open_plan.knowledge_query.subject_entity_ids == []
    assert validated.open_plan.knowledge_query.explicit_fact_ids == []


def test_question_about_a_destination_does_not_authorize_movement():
    scenario, state = _haunting_state()
    text = "疗养院在什么地方？"
    envelope = PlayerIntentEnvelope(
        id="intent_location_question",
        source="free_text",
        text=text,
        actor_id="player",
        scene_id="loc_cafe",
    )
    decision = TurnPlannerDecision.model_validate(
        {
            "existing_action_id": None,
            "open_plan": {
                "label": "前往罗克斯伯里疗养院",
                "action_type": "MOVE",
                "goal": text,
                "destination_name": "罗克斯伯里疗养院",
                "destination_entity_id": "loc_sanitarium",
            },
        }
    )

    with pytest.raises(PlanValidationError, match="explicit movement commitment"):
        PlanValidator().validate(
            state,
            envelope,
            decision,
            WorldKernel(scenario).available_actions(state),
        )


def test_repeated_unsafe_move_plan_is_reported_as_validation_not_connectivity():
    scenario, state = _haunting_state()
    text = "疗养院在什么地方？"
    llm = PlannerLLM(
        {
            "existing_action_id": None,
            "confidence": 1,
            "open_plan": {
                "label": "前往罗克斯伯里疗养院",
                "action_type": "MOVE",
                "goal": text,
                "destination_name": "罗克斯伯里疗养院",
                "destination_entity_id": "loc_sanitarium",
            },
        }
    )
    engine = GameEngine(scenario, llm)

    with pytest.raises(LLMUnavailable) as captured:
        engine.play(state, text=text)

    assert len(llm.calls) == 2
    assert "安全校验" in captured.value.public_message
    assert "无法连接" not in captured.value.public_message
    assert state.entities["player"].location == "loc_cafe"


@pytest.mark.parametrize(
    "text",
    [
        "我现在去疗养院。",
        "再去一次疗养院",
        "沿危险楼梯下到地下室",
        "我回咖啡馆",
        "转身离开病房",
        "疗养院在哪里？告诉我地址，我现在就过去。",
    ],
)
def test_explicit_movement_commitment_authorizes_location_change(text):
    assert PlanValidator.movement_commitment_evidence(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "疗养院在哪里？",
        "疗养院怎么去？",
        "我能去疗养院吗？",
        "我想先问问怎么去疗养院。",
    ],
)
def test_location_mentions_and_permission_questions_are_not_commitments(text):
    assert PlanValidator.movement_commitment_evidence(text) is None


def test_validator_preserves_server_owned_player_text_without_rejecting_punctuation_normalization():
    scenario, state = _haunting_state()
    original = "他们 都是谁来着?"
    envelope = PlayerIntentEnvelope(
        id="intent_goal_normalization",
        source="free_text",
        text=original,
        actor_id="player",
        scene_id="loc_cafe",
    )
    decision = TurnPlannerDecision.model_validate(
        {
            "existing_action_id": None,
            "open_plan": {
                "label": "询问身份",
                "action_type": "TALK",
                "goal": "他们都是谁来着？",
                "target_entity_id": "npc_knott",
                "addressee_id": "npc_knott",
                "speech_act": "question",
            },
        }
    )

    validated = PlanValidator().validate(
        state,
        envelope,
        decision,
        WorldKernel(scenario).available_actions(state),
    )

    assert validated.open_plan.goal == original


def test_composite_addressee_is_normalized_to_a_present_referenced_npc():
    scenario, state = _haunting_state()
    state.entities["player"].location = "loc_sanitarium"
    text = "加布里埃拉和维托里奥都是谁？"
    envelope = PlayerIntentEnvelope(
        id="intent_composite_addressee",
        source="free_text",
        text=text,
        actor_id="player",
        scene_id="loc_sanitarium",
    )
    decision = TurnPlannerDecision.model_validate(
        {
            "existing_action_id": None,
            "open_plan": {
                "label": "询问两人的身份",
                "action_type": "TALK",
                "goal": text,
                "target_entity_id": "npc_gabriela,npc_vittorio",
                "addressee_id": "npc_gabriela,npc_vittorio",
                "speech_act": "question",
                "referents": [
                    {"mention": "加布里埃拉", "entity_id": "npc_gabriela", "confidence": 1},
                    {"mention": "维托里奥", "entity_id": "npc_vittorio", "confidence": 1},
                ],
            },
        }
    )

    validated = PlanValidator().validate(
        state,
        envelope,
        decision,
        WorldKernel(scenario).available_actions(state),
    )

    assert validated.open_plan.addressee_id == "npc_gabriela"
    assert validated.open_plan.target_entity_id == "npc_gabriela"


def test_structured_knowledge_retrieval_is_scoped_to_addressee_and_topic():
    _scenario, state = _haunting_state()
    resolver = KnowledgeResolver()
    children = resolver.retrieve(
        state,
        KnowledgeQuery(
            query_text="孩子们后来怎么样了？",
            asker_id="player",
            addressee_id="npc_knott",
        ),
    )
    duration = resolver.retrieve(
        state,
        KnowledgeQuery(
            query_text="他们在房子里住了多久？",
            asker_id="player",
            addressee_id="npc_knott",
        ),
    )
    duration_with_planner_hints = resolver.retrieve(
        state,
        KnowledgeQuery(
            query_text="马卡里奥一家在科比特宅邸居住的具体时长是多少？",
            asker_id="player",
            addressee_id="npc_knott",
            subject_entity_ids=["loc_house_exterior"],
            predicate_hints=["tenancy_duration", "lease_period"],
        ),
    )
    rewritten_children = resolver.retrieve(
        state,
        KnowledgeQuery(
            query_text="上一户租客（马卡里奥一家）的孩子们后来的情况如何？",
            asker_id="player",
            addressee_id="npc_knott",
            predicate_hints=["death", "fate", "disappearance"],
            max_results=1,
        ),
    )
    wrong_npc = resolver.retrieve(
        state,
        KnowledgeQuery(
            query_text="孩子们后来怎么样了？",
            asker_id="player",
            addressee_id="npc_gabriela",
        ),
    )

    assert children[0].fact_id == "f_macario_children_status"
    assert duration == []
    assert duration_with_planner_hints == []
    assert rewritten_children[0].fact_id == "f_macario_children_status"
    assert wrong_npc == []


def test_compound_duration_question_does_not_disclose_an_unrelated_status_fact():
    _scenario, state = _haunting_state()
    query = KnowledgeQuery(
        query_text=(
            "询问马卡里奥一家在科比特宅邸居住的具体时长，"
            "以及是否有其他前任住户曾遭遇过类似的精神崩溃或异常事件。"
        ),
        asker_id="player",
        addressee_id="npc_knott",
        subject_entity_ids=["loc_house_exterior", "npc_knott"],
        predicate_hints=["tenancy_duration", "previous_incidents"],
        max_results=3,
    )

    evidence = KnowledgeResolver().retrieve(state, query)
    decision = DisclosurePolicy().decide(state, query, evidence)

    assert evidence == []
    assert decision.mode == "unknown"
    assert decision.approved_evidence == []
    assert "疗养院" not in decision.canonical_success


def test_compound_question_can_answer_one_atom_and_keep_the_other_unknown():
    _scenario, state = _haunting_state()
    query = KnowledgeQuery.model_validate(
        {
            "query_text": "孩子去了哪里，他们在宅子里又住了多久？",
            "asker_id": "player",
            "addressee_id": "npc_knott",
            "atoms": [
                {
                    "id": "children",
                    "query_text": "孩子去了哪里",
                    "subject_entity_ids": ["npc_gabriela"],
                    "predicate_hints": ["children_status"],
                    "relation_types": ["family", "location"],
                },
                {
                    "id": "duration",
                    "query_text": "他们在宅子里住了多久",
                    "subject_entity_ids": ["loc_house_exterior"],
                    "predicate_hints": ["tenancy_duration"],
                    "relation_types": ["duration"],
                },
            ],
            "max_results": 3,
        }
    )

    evidence = KnowledgeResolver().retrieve(state, query)
    decision = DisclosurePolicy().decide(state, query, evidence)

    assert [item.fact_id for item in decision.approved_evidence] == ["f_macario_children_status"]
    assert decision.answered_atom_ids == ["children"]
    assert decision.unanswered_atom_ids == ["duration"]
    assert "其余部分" in decision.canonical_success


def test_disclosure_policy_distinguishes_automatic_check_and_unknown(scenario, state):
    resolver = KnowledgeResolver()
    policy = DisclosurePolicy()
    automatic_query = KnowledgeQuery(
        query_text="这些针孔说明什么？",
        asker_id="player",
        addressee_id="npc_anna",
    )
    concealed_query = KnowledgeQuery(
        query_text="你看见威尔逊把病人送去哪里？",
        asker_id="player",
        addressee_id="npc_anna",
    )
    unknown_query = KnowledgeQuery(
        query_text="今天彩票号码是什么？",
        asker_id="player",
        addressee_id="npc_anna",
    )

    automatic = policy.decide(state, automatic_query, resolver.retrieve(state, automatic_query))
    concealed = policy.decide(state, concealed_query, resolver.retrieve(state, concealed_query))
    unknown = policy.decide(state, unknown_query, resolver.retrieve(state, unknown_query))

    assert automatic.mode == "automatic"
    assert [item.fact_id for item in automatic.approved_evidence] == ["f_injection_marks"]
    assert concealed.mode == "check"
    assert concealed.skill in state.player.skills
    assert [item.fact_id for item in concealed.approved_evidence] == ["f_anna_saw_transfer"]
    assert unknown.mode == "unknown"
    assert unknown.approved_evidence == []


def test_free_text_question_discloses_fact_through_kernel_and_records_full_trace():
    scenario, state = _haunting_state()
    text = "孩子们后来怎么样了？"
    engine = GameEngine(scenario, PlannerLLM(_question_output(
        text,
        target_id="npc_knott",
        target_name="史蒂文·诺特",
    )))

    resolved, resolution = engine.play(state, text=text)

    assert resolution.accepted is True
    assert resolution.check and resolution.check.outcome == CheckOutcome.AUTOMATIC
    assert "f_macario_children_status" in resolved.player_known_fact_ids
    disclosed = next(event for event in resolved.event_log if event.type == "fact_disclosed")
    assert disclosed.actor == "npc_knott"
    assert disclosed.payload["fact_id"] == "f_macario_children_status"
    assert disclosed.payload["source_entity_id"] == "npc_knott"
    assert resolution.outcome_envelope is not None
    assert resolution.outcome_envelope["disclosed_facts"][0]["fact_id"] == "f_macario_children_status"

    trace = resolved.turn_traces[-1]
    assert trace["status"] == "resolved"
    assert trace["input"]["source"] == "free_text"
    assert trace["knowledge_query"]["query_text"] == text
    assert trace["evidence"][0]["fact_id"] == "f_macario_children_status"
    assert trace["disclosure"]["mode"] == "automatic"
    assert any(event["type"] == "fact_disclosed" for event in trace["kernel_events"])
    assert trace["state_diff"]["learned_fact_ids"] == ["f_macario_children_status"]


def test_unknown_npc_answer_does_not_fabricate_or_reveal_a_fact():
    scenario, state = _haunting_state()
    text = "他们在房子里住了多久？"
    engine = GameEngine(scenario, PlannerLLM(_question_output(
        text,
        target_id="npc_knott",
        target_name="史蒂文·诺特",
    )))
    known_before = set(state.player_known_fact_ids)

    resolved, resolution = engine.play(state, text=text)

    assert resolved.player_known_fact_ids == known_before
    assert not any(event.type == "fact_disclosed" for event in resolved.event_log)
    assert "不知道" in resolution.narrative_seed
    assert resolution.outcome_envelope["disclosed_facts"] == []
    assert resolved.turn_traces[-1]["disclosure"]["mode"] == "unknown"


def test_concealed_npc_knowledge_requires_a_roll_and_only_success_reveals():
    scenario = load_scenario()
    text = "安娜，你看见威尔逊把病人送去哪里？"
    output = _question_output(text, target_id="npc_anna", target_name="安娜·科瓦奇")
    successful = None
    failed = None
    for seed in range(1, 80):
        state = create_initial_state(scenario, seed=seed)
        resolved, resolution = GameEngine(scenario, PlannerLLM(output)).play(state, text=text)
        assert resolution.check is not None and resolution.check.required is True
        if resolution.check.succeeded and successful is None:
            successful = resolved
        if not resolution.check.succeeded and failed is None:
            failed = resolved
        if successful is not None and failed is not None:
            break

    assert successful is not None and "f_anna_saw_transfer" in successful.player_known_fact_ids
    assert failed is not None and "f_anna_saw_transfer" not in failed.player_known_fact_ids
    assert any(event.type == "fact_disclosed" for event in successful.event_log)
    assert not any(event.type == "fact_disclosed" for event in failed.event_log)


def test_pending_social_check_keeps_one_trace_through_rule_choice():
    scenario = load_scenario()
    text = "安娜，你看见威尔逊把病人送去哪里？"
    output = _question_output(text, target_id="npc_anna", target_name="安娜·科瓦奇")
    offered = None
    engine = None
    for seed in range(1, 100):
        candidate_engine = GameEngine(scenario, PlannerLLM(output))
        state = create_initial_state(scenario, seed=seed)
        candidate, resolution = candidate_engine.play(
            state,
            text=text,
            interactive_rules=True,
        )
        if resolution.awaiting_rule_choice:
            offered = candidate
            engine = candidate_engine
            break
    assert offered is not None and engine is not None
    trace_id = offered.pending_check.turn_trace_id
    assert trace_id is not None
    assert offered.turn_traces[-1]["status"] == "pending_rule_choice"
    assert {event["type"] for event in offered.turn_traces[-1]["kernel_events"]} >= {
        "action_started",
        "rule_choice_offered",
    }

    resolved, resolution = engine.play(offered, rule_choice=RuleChoice.ACCEPT_FAILURE)

    assert resolution.turn_trace_id == trace_id
    assert len(resolved.turn_traces) == 1
    trace = resolved.turn_traces[0]
    assert trace["status"] == "resolved"
    assert trace["outcome"]["player_text"] == text
    assert {event["type"] for event in trace["kernel_events"]} >= {
        "action_started",
        "rule_choice_offered",
        "rule_choice_made",
        "action_resolved",
    }


def test_button_and_free_text_share_envelope_and_trace_contract(scenario, state):
    button_engine = GameEngine(scenario, PlannerLLM({"existing_action_id": "lobby_guestbook"}))
    button_state, _ = button_engine.play(state, action_id="lobby_guestbook")
    assert button_state.turn_traces[-1]["input"]["source"] == "option"
    assert button_state.turn_traces[-1]["input"]["intent_seed"]["action_id"] == "lobby_guestbook"

    free_engine = GameEngine(scenario, PlannerLLM({"existing_action_id": "lobby_guestbook"}))
    free_state, _ = free_engine.play(state, text="我仔细翻看访客簿")
    assert free_state.turn_traces[-1]["input"]["source"] == "free_text"
    assert free_state.turn_traces[-1]["input"]["intent_seed"] is None
    assert set(button_state.turn_traces[-1]) == set(free_state.turn_traces[-1])


def test_open_action_is_not_off_main_until_world_location_is_off_main():
    scenario, state = _haunting_state()
    director = Director(scenario, WorldKernel(scenario))
    action = ActionDefinition(
        id="open__talk",
        label="继续聊天",
        type=ActionType.TALK,
        target="npc_knott",
        success_text="谈话继续。",
    )
    director.observe_action(
        state,
        action,
        CheckOutcome.AUTOMATIC,
        progress_before=0,
        location_before="loc_cafe",
    )
    assert state.director.off_main_streak == 0

    state.entities["loc_cafe"].tags.add("off_main")
    director.observe_action(
        state,
        action,
        CheckOutcome.AUTOMATIC,
        progress_before=0,
        location_before="loc_cafe",
    )
    assert state.director.off_main_streak == 1


def test_director_hint_only_unlocks_a_checked_opportunity():
    scenario, state = _haunting_state()
    kernel = WorldKernel(scenario)
    director = Director(scenario, kernel)
    state.director.actions_without_progress = 3

    intervention = director.decide(state)

    assert intervention is not None and intervention.action == "surface_clue"
    assert state.flags["director_case_note_opportunity"] is True
    assert "f_house_violent_history" not in state.player_known_fact_ids
    action = next(item for item in kernel.available_actions(state) if item.id == "cafe_review_case_note")
    assert action.skill == "research"
    assert action.success_effects[0].params["fact_id"] == "f_house_violent_history"
    assert all(effect.op != "reveal_fact" for effect in intervention.effects)


def test_director_internal_justification_is_not_player_visible_and_once_survives_legacy_save():
    scenario, state = _haunting_state()
    kernel = WorldKernel(scenario)
    director = Director(scenario, kernel)
    internal_text = "科比特的意识覆盖整栋宅邸"
    state.director.experience.success_streak = 3

    first = director.decide(state)

    assert first is not None
    assert first.source_definition_id == "complication_corbitt_attention"
    assert first.player_visible_text is None
    assert internal_text in first.world_justification
    first.source_definition_id = None  # Simulate an intervention loaded from an older save.
    state.director.experience.success_streak = 3
    second = director.decide(state)
    assert second is None or internal_text not in second.world_justification

    state = create_initial_state(scenario, seed=19)
    state.director.experience.success_streak = 2
    engine = GameEngine(
        scenario,
        OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None)),
    )
    resolved, resolution = engine.play(state, action_id="wait_and_listen")
    assert resolution.director_intervention is not None
    assert internal_text in resolution.director_intervention.world_justification
    sequence = resolved.narrative_sequence
    assert sequence is not None
    assert all(internal_text not in beat.text for beat in sequence.beats)
    assert (sequence.outcome_envelope or {}).get("director_opportunity") is None


def test_grounding_validator_rejects_unapproved_fact_and_absent_character():
    scenario, state = _haunting_state()
    validator = GroundingValidator(scenario)
    outcome = OutcomeEnvelope(
        turn_id="turn_weather",
        action_id="open__weather",
        action_label="谈论天气",
        action_type=ActionType.TALK,
        player_text="今天天气真好。",
        accepted=True,
        canonical_seed="“是啊，难得放晴。”诺特说。",
        scene={"id": "loc_cafe", "name": "咖啡馆", "description": "晨光落在桌面。"},
        present_entities=[{"id": "npc_knott", "name": "史蒂文·诺特", "type": "NPC"}],
    )

    accepted, report = validator.validate(
        state,
        outcome,
        [
            "窗外的阳光落在咖啡杯边。",
            "数代住户遭遇疾病、事故、自杀与仓促搬离。",
            "加布里埃拉忽然走进来谈起宅邸。",
        ],
    )

    assert accepted == ["窗外的阳光落在咖啡杯边。"]
    assert report.accepted is False
    assert len(report.rejected_beats) == 2


def test_grounding_validator_allows_an_explicitly_disclosed_fact():
    scenario, state = _haunting_state()
    fact = state.facts["f_macario_children_status"]
    outcome = OutcomeEnvelope.model_validate(
        {
            "turn_id": "turn_children",
            "action_id": "open__children",
            "action_label": "询问孩子近况",
            "action_type": "TALK",
            "player_text": "孩子们后来怎么样了？",
            "accepted": True,
            "canonical_seed": "诺特回答了问题。",
            "disclosed_facts": [
                {
                    "fact_id": fact.id,
                    "source_entity_id": "npc_knott",
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "value": fact.value,
                    "newly_learned": True,
                }
            ],
            "scene": {"id": "loc_cafe", "name": "咖啡馆", "description": ""},
            "present_entities": [
                {"id": "npc_knott", "name": "史蒂文·诺特", "type": "NPC"}
            ],
        }
    )

    accepted, report = GroundingValidator(scenario).validate(
        state,
        outcome,
        [f"“{fact.value}。”诺特说。"],
    )

    assert accepted == [f"“{fact.value}。”诺特说。"]
    assert report.accepted is True
    assert report.approved_fact_ids == [fact.id]


def test_grounding_validator_rejects_narrator_details_not_in_disclosed_fact():
    scenario, state = _haunting_state()
    fact = state.facts["f_macario_children_status"]
    outcome = OutcomeEnvelope.model_validate(
        {
            "turn_id": "turn_children",
            "action_id": "open__children",
            "action_label": "询问孩子近况",
            "action_type": "TALK",
            "player_text": "孩子们后来怎么样了？",
            "accepted": True,
            "canonical_seed": "诺特回答了问题。",
            "disclosed_facts": [
                {
                    "fact_id": fact.id,
                    "source_entity_id": "npc_knott",
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "value": fact.value,
                    "newly_learned": True,
                }
            ],
            "scene": {"id": "loc_cafe", "name": "咖啡馆", "description": ""},
            "present_entities": [
                {"id": "npc_knott", "name": "史蒂文·诺特", "type": "NPC"}
            ],
        }
    )

    accepted, report = GroundingValidator(scenario).validate(
        state,
        outcome,
        [
            f"“{fact.value}。”诺特说。",
            "“马卡里奥太太的姐姐带着孩子搬去了城郊的公寓。”",
            "“我猜他们现在过得很好。”",
            "“至于那两个孩子……",
        ],
    )

    assert accepted == [f"“{fact.value}。”诺特说。"]
    assert report.accepted is False
    assert any("unsupported concrete claim" in reason for reason in report.reasons)
    assert any("dialogue outside" in reason for reason in report.reasons)
    assert any("unfinished dialogue" in reason for reason in report.reasons)


def test_grounding_validator_rejects_ominous_case_drift_in_weather_smalltalk():
    scenario, state = _haunting_state()
    outcome = OutcomeEnvelope(
        turn_id="turn_weather",
        action_id="open__weather",
        action_label="闲聊天气",
        action_type=ActionType.TALK,
        player_text="今天天气真好，你觉得呢？",
        accepted=True,
        canonical_seed="“是啊，难得放晴。”诺特说。",
        scene={"id": "loc_cafe", "name": "咖啡馆", "description": "晨光落在桌面。"},
        present_entities=[{"id": "npc_knott", "name": "史蒂文·诺特", "type": "NPC"}],
    )

    accepted, report = GroundingValidator(scenario).validate(
        state,
        outcome,
        ["“阳光越好，某些人就越容易消失。”诺特压低声音。"],
    )

    assert accepted == []
    assert report.accepted is False
    assert report.reasons == ["switches an unrelated turn back to the case"]


def test_grounding_validator_allows_current_scene_name_and_non_factual_people_count():
    scenario, state = _haunting_state()
    outcome = OutcomeEnvelope(
        turn_id="turn_weather",
        action_id="open__weather",
        action_label="闲聊天气",
        action_type=ActionType.TALK,
        player_text="今天天气真好，你觉得呢？",
        accepted=True,
        canonical_seed="“是啊，难得放晴。”诺特说。",
        scene={"id": "loc_cafe", "name": "诺特约见的咖啡馆", "description": "晨光落在桌面。"},
        present_entities=[{"id": "npc_knott", "name": "史蒂文·诺特", "type": "NPC"}],
    )

    accepted, report = GroundingValidator(scenario).validate(
        state,
        outcome,
        ["诺特约见的咖啡馆里，两人之间的热气缓缓散去。"],
    )

    assert accepted == ["诺特约见的咖啡馆里，两人之间的热气缓缓散去。"]
    assert report.accepted is True


def test_narrator_payload_contains_outcome_facts_not_all_player_knowledge():
    scenario, state = _haunting_state()
    state.player_known_fact_ids.add("f_house_violent_history")
    outcome = OutcomeEnvelope(
        turn_id="turn_weather",
        action_id="open__weather",
        action_label="谈论天气",
        action_type=ActionType.TALK,
        player_text="今天天气真好。",
        accepted=True,
        mechanical_result={"outcome": "AUTOMATIC"},
        canonical_seed="“是啊，难得放晴。”诺特说。",
        scene={"id": "loc_cafe", "name": "咖啡馆", "description": "晨光落在桌面。"},
        present_entities=[{"id": "npc_knott", "name": "史蒂文·诺特", "type": "NPC"}],
    )
    sequence = NarrativeSequence(
        id="sequence_weather",
        state_version=state.version,
        action_id=outcome.action_id,
        action_label=outcome.action_label,
        action_type=outcome.action_type,
        player_text=outcome.player_text,
        status="pending",
        beats=[NarrativeBeat(id="beat_weather", text=outcome.canonical_seed)],
        canonical_seed=outcome.canonical_seed,
        mechanical_result=outcome.mechanical_result,
        outcome_envelope=outcome.model_dump(mode="json"),
        created_at=state.world_time,
    )
    llm = PlannerLLM({"beats": ["“是啊，难得放晴。”诺特抬眼看向窗外。"]})
    narrator = GameEngine(scenario, llm).narrator

    narrator.expand_sequence(state, sequence)

    payload = llm.calls[-1]["user_payload"]
    assert payload["outcome_envelope"]["disclosed_facts"] == []
    assert payload["recent_visible_facts"] == []
    assert "f_house_violent_history" not in str(payload)
    assert "数代住户" not in str(payload)


def test_old_snapshot_without_v2_fields_loads_with_defaults(state):
    payload = state.model_dump(mode="json")
    payload.pop("turn_traces")
    migrated = WorldState.model_validate(payload)

    assert migrated.turn_traces == []
    assert migrated.narrative_sequence is None


def test_scenario_upgrade_hydrates_new_atomic_knowledge_without_overwriting_state():
    scenario, state = _haunting_state()
    state.facts.pop("f_macario_children_status")
    state.facts.pop("f_macario_parents_status")
    state.npc_knowledge = [
        item
        for item in state.npc_knowledge
        if item.fact_id not in {"f_macario_children_status", "f_macario_parents_status"}
    ]
    state.flags.pop("director_case_note_opportunity")
    state.flags["house_uneasy"] = True

    changed = upgrade_world_state(state, scenario)
    changed_again = upgrade_world_state(state, scenario)

    assert changed is True
    assert changed_again is False
    assert "f_macario_children_status" in state.facts
    assert any(
        item.knower_id == "npc_knott" and item.fact_id == "f_macario_children_status"
        for item in state.npc_knowledge
    )
    assert state.flags["director_case_note_opportunity"] is False
    assert state.flags["house_uneasy"] is True


def test_disclosure_replay_preserves_simulation(tmp_path):
    scenario, state = _haunting_state()
    text = "孩子们后来怎么样了？"
    engine = GameEngine(scenario, PlannerLLM(_question_output(
        text,
        target_id="npc_knott",
        target_name="史蒂文·诺特",
    )))
    resolved, _ = engine.play(state, text=text)
    repository = SQLiteRepository(tmp_path / "disclosure-replay.db")
    repository.save(resolved)

    verified, expected, actual = verify_replay(repository.export(resolved.session_id), scenario)

    assert verified, (expected, actual)


def test_narrator_generation_never_mutates_structured_facts():
    scenario, state = _haunting_state()
    before = {fact_id: fact.model_dump(mode="json") for fact_id, fact in state.facts.items()}
    outcome = OutcomeEnvelope(
        turn_id="turn_test",
        action_id="open__test",
        action_label="停顿",
        action_type=ActionType.OTHER,
        accepted=True,
        canonical_seed="你停顿了一会儿。",
        scene={"id": "loc_cafe", "name": "咖啡馆", "description": ""},
        present_entities=[{"id": "npc_knott", "name": "史蒂文·诺特", "type": "NPC"}],
    )
    sequence = NarrativeSequence(
        id="sequence_test",
        state_version=state.version,
        action_id="open__test",
        action_label="停顿",
        action_type=ActionType.OTHER,
        status="pending",
        beats=[NarrativeBeat(id="beat_1", text="你停顿了一会儿。")],
        canonical_seed="你停顿了一会儿。",
        outcome_envelope=outcome.model_dump(mode="json"),
        created_at=state.world_time,
    )
    narrator = GameEngine(scenario, PlannerLLM({"beats": ["杯中的涟漪慢慢平复。"]})).narrator

    narrator.expand_sequence(state, sequence)

    after = {fact_id: fact.model_dump(mode="json") for fact_id, fact in state.facts.items()}
    assert after == before
