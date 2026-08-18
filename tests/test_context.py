from __future__ import annotations

from datetime import datetime

from living_tabletop.context import (
    recent_visible_context,
    remember_visible_beats,
    substantially_repeats,
)
from living_tabletop.models import AgentCallRecord, NarrativeBeat, NarrativeSequence


def test_recent_context_recovers_generated_beats_from_legacy_save(state):
    state.agent_calls.append(
        AgentCallRecord(
            id="agent_call_legacy",
            role="narrator",
            input_state_version=7,
            output_digest="legacy",
            structured_output={
                "beats": ["地板上散落着一份关于马卡里奥一家的简报。"]
            },
            validation="accepted",
            latency_ms=20,
            result="success",
            created_at=datetime.fromisoformat("1927-12-17T10:30:00"),
        )
    )

    context = recent_visible_context(state)

    assert context[-1]["text"] == "地板上散落着一份关于马卡里奥一家的简报。"
    assert context[-1]["kind"] == "soft_canon"


def test_remembered_generated_dialogue_is_a_claim(state):
    sequence = NarrativeSequence(
        id="narrative_dialogue",
        state_version=state.version,
        action_type="TALK",
        beats=[],
        created_at=state.world_time,
    )
    generated = NarrativeBeat(
        id="narrative_dialogue_beat_01",
        text="“我从没进过那栋房子。”安娜说。",
        source="generated",
    )

    remember_visible_beats(state, sequence, [generated])

    assert state.visible_history[-1].kind == "dialogue_claim"
    assert state.visible_history[-1].sequence_id == sequence.id


def test_keeper_dialogue_seed_is_not_promoted_to_hard_canon(state):
    sequence = NarrativeSequence(
        id="narrative_keeper_dialogue",
        state_version=state.version,
        action_type="TALK",
        beats=[],
        created_at=state.world_time,
    )
    beat = NarrativeBeat(
        id="keeper_beat",
        text="“今天天气确实不错。”诺特点点头。",
        source="keeper",
    )

    remember_visible_beats(state, sequence, [beat])

    assert state.visible_history[-1].kind == "dialogue_claim"
    assert state.visible_history[-1].source == "keeper"


def test_recent_context_can_exclude_the_sequence_being_expanded(state):
    sequence = NarrativeSequence(
        id="narrative_current",
        state_version=state.version,
        beats=[NarrativeBeat(id="current", text="当前轮文本。")],
        created_at=state.world_time,
    )
    remember_visible_beats(state, sequence)

    assert recent_visible_context(state, exclude_sequence_id=sequence.id) == []


def test_legacy_open_plan_text_is_downgraded_from_authored_canon(state):
    text = "诺特避开你的目光，把话题重新转回那栋房子。"
    sequence = NarrativeSequence(
        id="narrative_legacy_open",
        state_version=state.version,
        action_type="TALK",
        beats=[NarrativeBeat(id="legacy_open", text=text, source="authored")],
        created_at=state.world_time,
    )
    remember_visible_beats(state, sequence)
    state.agent_calls.append(
        AgentCallRecord(
            id="agent_call_keeper_legacy",
            role="keeper",
            input_state_version=state.version,
            output_digest="legacy-keeper",
            structured_output={
                "open_plan": {"success_text": text, "failure_text": "没有发生什么。"}
            },
            validation="accepted",
            latency_ms=5,
            created_at=state.world_time,
        )
    )

    context = recent_visible_context(state)

    assert context[-1]["source"] == "keeper"
    assert context[-1]["kind"] == "dialogue_claim"


def test_substantial_repeat_filter_handles_exact_and_near_duplicate_text():
    earlier = "诺特的手指在杯沿上收紧，指节泛白。他不再看你，视线落在窗外流动的街道上。"

    assert substantially_repeats(earlier, [earlier])
    assert substantially_repeats(
        "诺特的手指仍在杯沿上收紧，指节泛白。他没有看你，视线落向窗外流动的街道。",
        [earlier],
    )
    assert not substantially_repeats("诺特望向晴朗的窗外，终于回应了你对天气的寒暄。", [earlier])


def test_recent_context_has_a_fixed_entry_and_character_budget(state):
    sequence = NarrativeSequence(
        id="narrative_budget",
        state_version=state.version,
        beats=[],
        created_at=state.world_time,
    )
    beats = [
        NarrativeBeat(id=f"beat_{index}", text=chr(65 + index) * 20, source="generated")
        for index in range(20)
    ]
    remember_visible_beats(state, sequence, beats)

    context = recent_visible_context(state, max_entries=5, max_characters=100)

    assert len(context) == 5
    assert sum(len(item["text"]) for item in context) <= 100
