from __future__ import annotations

from copy import deepcopy

from living_tabletop.agent_runtime import (
    ContextAssembler,
    DialogueAgent,
    DisclosurePolicy,
    DialogueTurnOutput,
    GroundingValidator,
    KnowledgeQuery,
    KnowledgeResolver,
    OutcomeEnvelope,
    PlanValidator,
    PlanValidationError,
    PlayerIntentEnvelope,
    SoftFactValidator,
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
    Fact,
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
        if kwargs.get("schema_name") == "DialogueTurnOutput":
            evidence = kwargs["user_payload"].get("disclosable_evidence") or []
            questions = kwargs["user_payload"].get("question_parts") or []
            if evidence:
                first = evidence[0]
                return LLMResult(
                    data={
                        "beats": [f"“{first['value']}”对方直接回答。"],
                        "used_fact_ids": [first["fact_id"]],
                        "proposed_facts": [],
                        "answered_query_parts": questions,
                        "unresolved_query_parts": [],
                    },
                    latency_ms=3,
                    input_tokens=20,
                    output_tokens=20,
                )
            return LLMResult(
                data={
                    "beats": ["“这件事我确实不知道。”对方坦率地回答。"],
                    "used_fact_ids": [],
                    "proposed_facts": [],
                    "answered_query_parts": [],
                    "unresolved_query_parts": questions,
                },
                latency_ms=3,
                input_tokens=20,
                output_tokens=20,
            )
        return LLMResult(data=deepcopy(self.output), latency_ms=3, input_tokens=20, output_tokens=20)


class DialogueRoutingLLM:
    enabled = True

    def __init__(self, planner_output: dict, dialogue_output: dict):
        self.planner_output = planner_output
        self.dialogue_output = dialogue_output
        self.calls: list[dict] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        output = (
            self.dialogue_output
            if kwargs.get("schema_name") == "DialogueTurnOutput"
            else self.planner_output
        )
        return LLMResult(data=deepcopy(output), latency_ms=3, input_tokens=20, output_tokens=20)


class RepairingDialogueRoutingLLM(DialogueRoutingLLM):
    def __init__(self, planner_output: dict, dialogue_outputs: list[dict]):
        super().__init__(planner_output, dialogue_outputs[-1])
        self.dialogue_outputs = dialogue_outputs
        self.dialogue_attempt = 0

    def complete_json(self, **kwargs):
        if kwargs.get("schema_name") != "DialogueTurnOutput":
            return super().complete_json(**kwargs)
        self.calls.append(kwargs)
        output = self.dialogue_outputs[
            min(self.dialogue_attempt, len(self.dialogue_outputs) - 1)
        ]
        self.dialogue_attempt += 1
        return LLMResult(
            data=deepcopy(output),
            latency_ms=3,
            input_tokens=20,
            output_tokens=20,
        )


class FailingDialogueLLM(DialogueRoutingLLM):
    def complete_json(self, **kwargs):
        if kwargs.get("schema_name") == "DialogueTurnOutput":
            raise LLMUnavailable("dialogue provider unavailable")
        return super().complete_json(**kwargs)


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


def test_unsafe_move_metadata_is_downgraded_without_losing_the_turn():
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

    resolved, resolution = engine.play(state, text=text)

    assert len(llm.calls) == 1
    assert resolution.accepted is True
    assert resolved.entities["player"].location == state.entities["player"].location
    assert resolved.turn_traces[-1]["composition"] is not None
    assert resolved.agent_calls[-1].role == "turn_composer"
    assert resolved.agent_calls[-1].validation == "fallback"


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


def test_dialogue_context_preserves_player_role_and_raw_negotiation_transcript():
    scenario, state = _haunting_state()
    state.entities["player"].name = "林默"
    offline = OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    state, introduced = GameEngine(scenario, offline).play(
        state,
        action_id="cafe_question_knott",
    )
    assert introduced.accepted is True

    text = "说实话，钱太少了，这种有点危险的活要加钱"
    planner_output = {
        "existing_action_id": None,
        "confidence": 0.99,
        "open_plan": {
            "label": "玩家向诺特要求提高这次委托的报酬",
            "action_type": "TALK",
            "goal": text,
            "target_name": "史蒂文·诺特",
            "target_entity_id": "npc_knott",
            "addressee_id": "npc_knott",
            "duration_minutes": 0,
            "resolution": "automatic",
            "risk": "safe",
            "speech_act": "request",
            "referents": [],
            "knowledge_query": None,
        },
    }
    dialogue_output = {
        "beats": ["诺特沉默片刻。‘林默，我明白你的顾虑。报酬的事，我们可以再谈。’"],
        "used_fact_ids": [],
        "proposed_facts": [],
        "answered_query_parts": [],
        "unresolved_query_parts": [],
    }
    llm = DialogueRoutingLLM(planner_output, dialogue_output)

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    dialogue_call = next(
        call for call in llm.calls if call.get("schema_name") == "DialogueTurnOutput"
    )
    payload = dialogue_call["user_payload"]
    assert "player_input" not in payload
    assert "speaker" not in payload
    assert payload["current_turn"] == {
        "role": "player",
        "speaker_id": "player",
        "speaker_name": resolved.entities["player"].name,
        "addressee_id": "npc_knott",
        "addressee_name": "史蒂文·诺特",
        "utterance_verbatim": text,
        "semantic_reading": "玩家向诺特要求提高这次委托的报酬。",
        "conversation_focus": "紧接玩家当前发言作出回应：玩家向诺特要求提高这次委托的报酬。",
        "speech_act_hint": "request",
    }
    assert payload["npc_responder"]["id"] == "npc_knott"
    assert payload["question_parts"] == []
    assert "loc_library" not in {
        item["id"] for item in payload["generatable_world_entities"]
    }
    assert "玩家原话改写成 NPC 自己的第一人称立场" in dialogue_call["system"]
    assert "必须在第一段台词中回应诉求本身" in dialogue_call["system"]

    player_memory = next(item for item in resolved.visible_history if item.text == text)
    npc_memory = next(
        item for item in resolved.visible_history if item.text == dialogue_output["beats"][0]
    )
    assert (player_memory.source, player_memory.speaker_id, player_memory.addressee_id) == (
        "player",
        "player",
        "npc_knott",
    )
    assert (npc_memory.speaker_id, npc_memory.addressee_id) == ("npc_knott", "player")

    next_envelope = PlayerIntentEnvelope(
        id="intent_follow_up",
        source="free_text",
        text="那你愿意加多少？",
        actor_id="player",
        scene_id="loc_cafe",
    )
    next_context = ContextAssembler().assemble(
        resolved,
        next_envelope,
        WorldKernel(scenario).available_actions(resolved),
        semantic_hint="玩家继续与诺特商谈报酬",
    )
    transcript = {
        item["text"]: item for item in next_context.recent_visible_history
    }
    assert transcript[text]["role"] == "player"
    assert transcript[text]["speaker_id"] == "player"
    assert transcript[dialogue_output["beats"][0]]["role"] == "npc"
    assert transcript[dialogue_output["beats"][0]]["speaker_id"] == "npc_knott"


def test_dialogue_repairs_a_response_that_only_replays_the_previous_npc_turn():
    scenario, state = _haunting_state()
    opening_text = "说实话，钱有点少了，得加钱。"
    opening_plan = {
        "existing_action_id": None,
        "confidence": 0.99,
        "open_plan": {
            "label": "玩家向诺特要求提高报酬",
            "action_type": "TALK",
            "goal": opening_text,
            "target_name": "史蒂文·诺特",
            "target_entity_id": "npc_knott",
            "addressee_id": "npc_knott",
            "duration_minutes": 0,
            "resolution": "automatic",
            "risk": "safe",
            "speech_act": "request",
            "referents": [],
            "knowledge_query": None,
        },
    }
    opening_beat = "诺特皱起眉头。‘二十美元已经是我眼下能拿出的全部了。’"
    opening_output = {
        "beats": [opening_beat],
        "used_fact_ids": [],
        "proposed_facts": [],
        "answered_query_parts": [],
        "unresolved_query_parts": [],
    }
    state, introduced = GameEngine(
        scenario,
        DialogueRoutingLLM(opening_plan, opening_output),
    ).play(state, text=opening_text)
    assert introduced.accepted is True
    previous_npc_beat = next(
        item.text
        for item in state.visible_history
        if item.speaker_id == "npc_knott" and item.text == opening_beat
    )

    text = "那加一点？"
    planner_output = {
        "existing_action_id": None,
        "confidence": 0.99,
        "open_plan": {
            "label": "玩家继续向诺特商量增加报酬",
            "action_type": "TALK",
            "goal": text,
            "target_name": "史蒂文·诺特",
            "target_entity_id": "npc_knott",
            "addressee_id": "npc_knott",
            "duration_minutes": 0,
            "resolution": "automatic",
            "risk": "safe",
            "speech_act": "request",
            "referents": [],
            "knowledge_query": None,
        },
    }
    repeated_output = {
        "beats": [previous_npc_beat],
        "used_fact_ids": [],
        "proposed_facts": [],
        "answered_query_parts": [],
        "unresolved_query_parts": [],
    }
    repaired_beat = "诺特摇了摇头。‘我最多只能再加五美元，这是最后的余地。’"
    repaired_output = {**repeated_output, "beats": [repaired_beat]}
    llm = RepairingDialogueRoutingLLM(
        planner_output,
        [repeated_output, repaired_output],
    )

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    dialogue_calls = [
        call for call in llm.calls if call.get("schema_name") == "DialogueTurnOutput"
    ]
    assert resolution.accepted is True
    assert resolution.narrative_seed == repaired_beat
    assert len(dialogue_calls) == 2
    assert dialogue_calls[1]["user_payload"]["_harness_repair"]["validation_errors"].startswith(
        "dialogue only repeats"
    )
    assert any(item.text == repaired_beat for item in resolved.visible_history)


def test_dialogue_detects_a_previous_turn_replayed_with_extra_stage_directions():
    previous = [
        "诺特紧张地摩挲着衣角。",
        "‘二十块已经是我能拿出的全部了。’",
        "‘求你了，就按这个数吧。’",
    ]
    replayed = [
        "诺特紧张地摩挲着衣角，指节因用力而发白。",
        "‘二十块已经是我能拿出的全部了。’他的声音微微发颤。",
        "‘求你了，就按这个数吧。’他把钥匙推回桌面中央。",
    ]

    assert DialogueAgent._substantially_replays_turn(replayed, previous) is True
    assert DialogueAgent._substantially_replays_turn(
        ["‘我再加五块，但这是最后的条件。’"],
        previous,
    ) is False


def test_dialogue_output_closes_one_trailing_direct_speech_quote():
    output = DialogueTurnOutput.model_validate(
        {
            "beats": ["‘二十五块是极限。别让我为难……"],
            "used_fact_ids": [],
            "proposed_facts": [],
            "answered_query_parts": [],
            "unresolved_query_parts": [],
        }
    )

    assert output.beats == ["‘二十五块是极限。别让我为难……’"]


def test_ordinary_open_dialogue_request_is_not_double_resolved_by_a_skill_check():
    scenario, state = _haunting_state()
    state.player.skills["charm"] = 0
    text = "那加一点？"
    planner_output = {
        "existing_action_id": None,
        "confidence": 0.99,
        "open_plan": {
            "label": "玩家向诺特要求增加报酬",
            "action_type": "TALK",
            "goal": text,
            "target_name": "史蒂文·诺特",
            "target_entity_id": "npc_knott",
            "addressee_id": "npc_knott",
            "duration_minutes": 0,
            "resolution": "check",
            "skill": "charm",
            "difficulty": "hard",
            "risk": "safe",
            "speech_act": "request",
            "referents": [],
            "knowledge_query": None,
        },
    }
    dialogue_output = {
        "beats": ["诺特斟酌着回答：‘如果条件允许，我会认真考虑。’"],
        "used_fact_ids": [],
        "proposed_facts": [],
        "answered_query_parts": [],
        "unresolved_query_parts": [],
    }

    resolved, resolution = GameEngine(
        scenario,
        DialogueRoutingLLM(planner_output, dialogue_output),
    ).play(state, text=text)

    assert resolution.check is not None
    assert resolution.check.outcome == CheckOutcome.AUTOMATIC
    assert resolved.narrative_sequence is not None
    assert resolved.narrative_sequence.status == "ready"
    assert resolved.narrative_sequence.beats[0].text == dialogue_output["beats"][0]


def test_failed_open_deception_check_keeps_async_failure_performance_enabled():
    scenario, state = _haunting_state()
    state.player.skills["charm"] = 0
    text = "我说市政厅已经授权我查看所有材料。"
    planner_output = {
        "existing_action_id": None,
        "confidence": 0.99,
        "open_plan": {
            "label": "玩家试图欺骗诺特以获取配合",
            "action_type": "DECEIVE",
            "goal": text,
            "target_name": "史蒂文·诺特",
            "target_entity_id": "npc_knott",
            "addressee_id": "npc_knott",
            "duration_minutes": 0,
            "resolution": "check",
            "skill": "charm",
            "difficulty": "hard",
            "risk": "safe",
            "speech_act": "deception",
            "referents": [],
            "knowledge_query": None,
        },
    }
    dialogue_output = {
        "beats": ["诺特审视着你。‘把授权书拿给我看。’"],
        "used_fact_ids": [],
        "proposed_facts": [],
        "answered_query_parts": [],
        "unresolved_query_parts": [],
    }

    resolved, resolution = GameEngine(
        scenario,
        DialogueRoutingLLM(planner_output, dialogue_output),
    ).play(state, text=text)

    assert resolution.check is not None
    assert resolution.check.succeeded is False
    assert resolved.narrative_sequence is not None
    assert resolved.narrative_sequence.status == "pending"
    assert resolved.narrative_sequence.beats[0].text == resolution.narrative_seed


def test_dialogue_agent_can_create_persist_and_retrieve_missing_soft_location_fact(tmp_path):
    scenario, state = _haunting_state()
    engine = GameEngine(scenario, OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None)))
    state, introduced = engine.play(state, action_id="cafe_question_knott")
    assert introduced.accepted is True

    text = "他们现在都在疗养院？疗养院在哪儿？"
    planner_output = {
        "existing_action_id": None,
        "confidence": 0.98,
        "open_plan": {
            "label": "询问疗养院位置",
            "action_type": "TALK",
            "goal": text,
            "target_name": "史蒂文·诺特",
            "target_entity_id": "npc_knott",
            "addressee_id": "npc_knott",
            "duration_minutes": 2,
            "resolution": "automatic",
            "risk": "safe",
            "speech_act": "question",
            "referents": [
                {"mention": "疗养院", "entity_id": "loc_sanitarium", "confidence": 1.0}
            ],
            "knowledge_query": {
                "query_text": text,
                "asker_id": "player",
                "addressee_id": "npc_knott",
                "subject_entity_ids": ["npc_gabriela", "loc_sanitarium"],
                "atoms": [
                    {
                        "id": "parents_status",
                        "query_text": "他们现在都在疗养院吗",
                        "subject_entity_ids": ["npc_gabriela"],
                        "predicate_hints": ["parents_status"],
                        "relation_types": ["status"],
                    },
                    {
                        "id": "sanitarium_address",
                        "query_text": "罗克斯伯里疗养院在哪儿",
                        "subject_entity_ids": ["loc_sanitarium"],
                        "predicate_hints": ["address"],
                        "relation_types": ["location"],
                    },
                ],
                "max_results": 3,
            },
        },
    }
    dialogue_output = {
        "beats": [
            "“是的，马卡里奥夫妇目前都在那里。”诺特点头确认。",
            "“疗养院在罗克斯伯里区华盛顿街附近，从这里乘电车过去并不远。”",
        ],
        "used_fact_ids": ["f_macario_parents_status"],
        "proposed_facts": [
            {
                "subject_entity_id": "loc_sanitarium",
                "predicate": "address",
                "value": "罗克斯伯里区华盛顿街附近",
                "confidence": 0.9,
            }
        ],
        "answered_query_parts": ["他们现在都在疗养院吗", "罗克斯伯里疗养院在哪儿"],
        "unresolved_query_parts": [],
    }
    llm = DialogueRoutingLLM(planner_output, dialogue_output)
    engine = GameEngine(scenario, llm)

    resolved, resolution = engine.play(state, text=text)

    generated = next(
        fact
        for fact in resolved.facts.values()
        if fact.subject == "loc_sanitarium" and fact.predicate == "address"
    )
    assert generated.value == "罗克斯伯里区华盛顿街附近"
    assert generated.canon == "soft_canon"
    assert generated.immutable is False
    assert generated.id in resolved.player_known_fact_ids
    assert any(
        item.knower_id == "npc_knott" and item.fact_id == generated.id
        for item in resolved.npc_knowledge
    )
    assert resolution.narrative_seed == "\n\n".join(dialogue_output["beats"])
    assert resolved.narrative_sequence is not None
    assert resolved.narrative_sequence.status == "ready"
    assert [beat.text for beat in resolved.narrative_sequence.beats] == dialogue_output["beats"]
    assert resolved.turn_traces[-1]["dialogue"] == dialogue_output
    assert any(event.type == "fact_created" and event.target == generated.id for event in resolved.event_log)

    follow_up = KnowledgeQuery(
        query_text="罗克斯伯里疗养院的地址在哪里？",
        asker_id="player",
        addressee_id="npc_knott",
        subject_entity_ids=["loc_sanitarium"],
        predicate_hints=["address"],
        atoms=[
            {
                "id": "address",
                "query_text": "罗克斯伯里疗养院的地址在哪里？",
                "subject_entity_ids": ["loc_sanitarium"],
                "predicate_hints": ["address"],
                "relation_types": ["location"],
            }
        ],
    )
    retrieved = KnowledgeResolver().retrieve(resolved, follow_up)
    assert retrieved and retrieved[0].fact_id == generated.id

    repository = SQLiteRepository(tmp_path / "dialogue-soft-fact-replay.db")
    repository.save(resolved)
    verified, expected, actual = verify_replay(repository.export(resolved.session_id), scenario)
    assert verified, (expected, actual)


def test_soft_fact_validator_rejects_visible_conflict_with_established_world_detail():
    _scenario, state = _haunting_state()
    state.facts["f_existing_address"] = Fact(
        id="f_existing_address",
        subject="loc_sanitarium",
        predicate="address",
        value="罗克斯伯里区旧大道十号",
        visibility="PLAYER",
        created_at=state.world_time,
        source="scenario",
        immutable=True,
        canon="hard_canon",
    )
    state.player_known_fact_ids.add("f_existing_address")
    output = DialogueTurnOutput.model_validate(
        {
            "beats": ["“疗养院在罗克斯伯里区新街二号。”诺特回答。"],
            "used_fact_ids": [],
            "proposed_facts": [
                {
                    "subject_entity_id": "loc_sanitarium",
                    "predicate": "address",
                    "value": "罗克斯伯里区新街二号",
                    "confidence": 0.9,
                }
            ],
            "answered_query_parts": ["疗养院在哪里"],
            "unresolved_query_parts": [],
        }
    )

    with pytest.raises(ValueError, match="contradicts established canon"):
        SoftFactValidator().validate(
            state,
            output,
            speaker_id="npc_knott",
            allowed_entity_ids={"npc_knott", "loc_sanitarium"},
            allowed_fact_ids=set(),
        )


def test_soft_fact_metadata_mismatch_is_dropped_without_erasing_dialogue():
    _scenario, state = _haunting_state()
    output = DialogueTurnOutput.model_validate(
        {
            "beats": ["“可以，我和你一起去看看。”诺特点了点头。"],
            "used_fact_ids": [],
            "proposed_facts": [
                {
                    "subject_entity_id": "npc_knott",
                    "predicate": "access_notes",
                    "value": "诺特同意陪同调查员前往宅邸。",
                    "confidence": 0.9,
                }
            ],
            "answered_query_parts": ["你能和我一起去看看吗"],
            "unresolved_query_parts": [],
        }
    )

    SoftFactValidator().validate(
        state,
        output,
        speaker_id="npc_knott",
        allowed_entity_ids={"npc_knott"},
        allowed_fact_ids=set(),
    )

    assert output.beats == ["“可以，我和你一起去看看。”诺特点了点头。"]
    assert output.proposed_facts == []


def test_dialogue_output_repairs_qwen_duplicate_ascii_quote_separator():
    output = DialogueTurnOutput.model_validate(
        {
            "beats": [
                '"他们都在那里。"诺特攥紧钥匙，", "至于地址……"他停顿片刻，'
                '"就在市中心偏北。"'
            ]
        }
    )

    assert output.beats == [
        "“他们都在那里。”诺特攥紧钥匙，“至于地址……”他停顿片刻，“就在市中心偏北。”"
    ]


def test_dialogue_failure_does_not_commit_a_partial_turn():
    scenario, state = _haunting_state()
    offline = OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    state, introduced = GameEngine(scenario, offline).play(
        state,
        action_id="cafe_question_knott",
    )
    assert introduced.accepted is True
    snapshot = state.model_dump(mode="json")
    text = "诺特先生，今天的天气怎么样？"
    planner_output = {
        "existing_action_id": None,
        "confidence": 0.99,
        "open_plan": {
            "label": "谈论天气",
            "action_type": "TALK",
            "goal": text,
            "target_name": "史蒂文·诺特",
            "target_entity_id": "npc_knott",
            "addressee_id": "npc_knott",
            "duration_minutes": 2,
            "resolution": "automatic",
            "risk": "safe",
            "speech_act": "smalltalk",
            "referents": [],
            "knowledge_query": None,
        },
    }
    llm = FailingDialogueLLM(planner_output, {})

    with pytest.raises(LLMUnavailable, match="dialogue provider unavailable"):
        GameEngine(scenario, llm).play(state, text=text)

    assert state.model_dump(mode="json") == snapshot


def test_physical_action_with_addressed_question_also_gets_npc_reply():
    scenario, state = _haunting_state()
    offline = OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    state, introduced = GameEngine(scenario, offline).play(
        state,
        action_id="cafe_question_knott",
    )
    assert introduced.accepted is True
    text = "我先收起钥匙，说，你还有什么要告诉我的吗"
    planner_output = {
        "existing_action_id": None,
        "confidence": 0.95,
        "open_plan": {
            "label": "收起钥匙",
            "action_type": "TAKE",
            "goal": text,
            "target_name": "科比特宅邸钥匙",
            "target_entity_id": "item_house_key",
            "addressee_id": "npc_knott",
            "duration_minutes": 0,
            "resolution": "automatic",
            "risk": "safe",
            "speech_act": "statement",
            "referents": [
                {"mention": "钥匙", "entity_id": "item_house_key", "confidence": 1.0},
                {"mention": "你", "entity_id": "npc_knott", "confidence": 1.0},
            ],
            "knowledge_query": None,
        },
    }
    dialogue_output = {
        "beats": ["“还有一件事：那栋房子夜里会传出撞击声。”诺特压低声音回答。"],
        "used_fact_ids": [],
        "proposed_facts": [],
        "answered_query_parts": ["你还有什么要告诉我的吗"],
        "unresolved_query_parts": [],
    }
    llm = DialogueRoutingLLM(planner_output, dialogue_output)

    resolved, resolution = GameEngine(scenario, llm).play(state, text=text)

    assert resolution.accepted is True
    assert resolution.outcome_envelope["action_type"] == ActionType.TAKE
    assert resolved.turn_traces[-1]["planner_output"]["open_plan"]["speech_act"] == "question"
    assert resolved.turn_traces[-1]["knowledge_query"] is not None
    assert resolved.turn_traces[-1]["knowledge_query"]["query_text"] == "你还有什么要告诉我的吗"
    assert resolved.turn_traces[-1]["knowledge_query"]["subject_entity_ids"] == []
    assert resolved.turn_traces[-1]["dialogue"] == dialogue_output
    assert resolved.narrative_sequence is not None
    assert resolved.narrative_sequence.status == "ready"
    assert [beat.text for beat in resolved.narrative_sequence.beats] == [
        "你完成了收起钥匙。",
        dialogue_output["beats"][0],
    ]
    assert [call.get("schema_name") for call in llm.calls] == [
        "TurnCompositionOutput",
        "DialogueTurnOutput",
    ]


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
