from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .context import context_relevance_score, recent_visible_context, substantially_repeats
from .harness import StructuredHarness
from .llm import OpenAICompatibleLLM, record_agent_call
from .models import (
    ActionDefinition,
    ActionResolution,
    ActionType,
    NarrativeSequence,
    ScenarioDefinition,
    SessionStatus,
    WorldState,
)


class NarrativeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(min_length=1, max_length=5000)

    @field_validator("narrative")
    @classmethod
    def strip_narrative(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("narrative cannot be blank")
        return value


class NarrativeBeatsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beats: list[str] = Field(min_length=1, max_length=5)

    @field_validator("beats")
    @classmethod
    def strip_beats(cls, value: list[str]) -> list[str]:
        beats = [item.strip() for item in value if item.strip()]
        if not beats:
            raise ValueError("beats cannot be empty")
        return beats


class Narrator:
    def __init__(self, llm: OpenAICompatibleLLM, scenario: ScenarioDefinition):
        self.llm = llm
        self.scenario = scenario

    def narrate(
        self,
        state: WorldState,
        action: ActionDefinition,
        resolution: ActionResolution,
    ) -> str:
        event_texts = [
            str(event.payload.get("text"))
            for event in resolution.visible_events
            if event.payload.get("text")
        ]
        if resolution.interrupted:
            base = " ".join([*event_texts, action.interrupted_text]).strip()
        elif resolution.check and resolution.check.succeeded:
            base = action.success_text
        else:
            base = action.failure_text
        if event_texts and not resolution.interrupted:
            base = " ".join([*event_texts, base])

        if not self.llm.enabled or state.status != SessionStatus.ACTIVE:
            return state.last_narrative if state.status != SessionStatus.ACTIVE and state.last_narrative else base

        location_id = state.entities[state.player.entity_id].location
        location = state.entities[location_id] if location_id else None
        visible_facts = [
            {
                "subject": state.facts[fact_id].subject,
                "predicate": state.facts[fact_id].predicate,
                "value": state.facts[fact_id].value,
            }
            for fact_id in sorted(state.player_known_fact_ids)
            if fact_id in state.facts
        ]
        payload: dict[str, Any] = {
            "location": location.name if location else "未知地点",
            "location_description": location.attributes.get("description", "") if location else "",
            "resolved_action": action.label,
            "mechanical_result": resolution.check.model_dump(mode="json") if resolution.check else None,
            "canonical_narrative_seed": base,
            "visible_facts": visible_facts,
            "visible_events": event_texts,
            "tone": "1927 年调查恐怖，克制、具象、不夸张",
            "required_output": {"narrative": "2-4 short Chinese paragraphs"},
        }
        system = (
            "你是 Living Tabletop 的 Narrator。只表达输入中已经发生且玩家可见的事情。"
            "不得创建 NPC、物品、线索或结果，不得透露隐藏事实，不得改变时间、位置、数值或骰点。"
            "保持第二人称、简洁、有感官细节。只输出 JSON 对象。"
        )
        try:
            outcome = StructuredHarness(self.llm).run(
                NarrativeOutput,
                system=system,
                user_payload=payload,
            )
            record_agent_call(
                state,
                role="narrator",
                result=outcome.llm_result,
                validation="accepted",
            )
            return outcome.value.narrative
        except Exception:
            record_agent_call(state, role="narrator", result=None, validation="fallback", error=True)
            return base

    def expand_sequence(
        self,
        state: WorldState,
        sequence: NarrativeSequence,
    ) -> list[str]:
        """Generate optional presentation-only beats without blocking world resolution."""
        if not self.llm.enabled or state.status != SessionStatus.ACTIVE:
            return []

        location_id = state.entities[state.player.entity_id].location
        location = state.entities[location_id] if location_id else None
        present_entities = [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type.value,
                "role": entity.attributes.get("role"),
            }
            for entity in state.entities.values()
            if entity.active
            and entity.location == location_id
            and entity.id != state.player.entity_id
        ]
        known_fact_ids = sorted(state.player_known_fact_ids)[-8:]
        current_topic = sequence.player_text or sequence.canonical_seed
        prior_visible_history = recent_visible_context(
            state,
            exclude_sequence_id=sequence.id,
            query=sequence.player_text or sequence.canonical_seed,
            immediate_entries=0,
        )
        dedupe_history = recent_visible_context(
            state,
            exclude_sequence_id=sequence.id,
        )
        recent_visible_facts = [
            {
                "subject": state.facts[fact_id].subject,
                "predicate": state.facts[fact_id].predicate,
                "value": state.facts[fact_id].value,
            }
            for fact_id in known_fact_ids
            if fact_id in state.facts
            and context_relevance_score(
                current_topic,
                (
                    f"{state.facts[fact_id].subject} "
                    f"{state.facts[fact_id].predicate} "
                    f"{state.facts[fact_id].value}"
                ),
            )
            > 0
        ]
        light_dialogue = (
            sequence.action_type == ActionType.TALK
            and (sequence.mechanical_result or {}).get("outcome") == "AUTOMATIC"
            and not prior_visible_history
            and not recent_visible_facts
        )
        payload: dict[str, Any] = {
            "location": location.name if location else "未知地点",
            "location_description": (
                ""
                if light_dialogue
                else location.attributes.get("description", "") if location else ""
            ),
            "resolved_action": sequence.action_label,
            "player_text": sequence.player_text,
            "action_type": sequence.action_type.value if sequence.action_type else None,
            "continues_previous": sequence.continues_previous,
            "mechanical_result": sequence.mechanical_result,
            "canonical_narrative_seed": sequence.canonical_seed,
            "authored_beats": [beat.text for beat in sequence.beats],
            "present_entities": present_entities,
            "recent_visible_history": prior_visible_history,
            "recent_visible_facts": recent_visible_facts,
            "required_output": {
                "beats": (
                    "2 brief Chinese dialogue beats, each 40-140 Chinese characters; end the exchange on the current topic"
                    if light_dialogue
                    else
                    "3-5 additional Chinese dialogue beats, each 55-200 Chinese characters"
                    if sequence.action_type in {ActionType.TALK, ActionType.DECEIVE}
                    else "2-4 additional Chinese paragraphs, each 55-220 Chinese characters"
                ),
            },
        }
        dialogue_instruction = (
            "若 action_type 是 TALK 或 DECEIVE，必须以现场直接对话演出：玩家台词使用第一人称，"
            "重要 NPC 的回应必须放在引号内直接说出；不要用‘你询问了/对方回答说/某人承认’概述谈话。"
            "只有不重要的过场人物可以被间接转述。允许在台词之间加入短促的停顿、表情与动作，"
            "但不得替玩家作出输入之外的新承诺或决定。"
            if sequence.action_type in {ActionType.TALK, ActionType.DECEIVE}
            else ""
        )
        continuation_instruction = (
            "这是上一段演出的续段。不要重述玩家已经说过的话、不要重新描写行动开端，"
            "直接从 NPC 的回应或最终结果开始。"
            if sequence.continues_previous
            else ""
        )
        light_dialogue_instruction = (
            "这是不涉及检定、事实查询或案件资料的轻量对话。只演出当前话题的简短来回，最多两段；"
            "结束时停留在当前话题，不得借隐喻、预兆、反问或转折重新引向案件、委托和线索。"
            if light_dialogue
            else ""
        )
        system = (
            "你是 Living Tabletop 的异步 Narrator。补充已经结算完成的场景表现。"
            "当前轮的 player_text、resolved_action、mechanical_result、canonical_narrative_seed 和 authored_beats 拥有最高优先级；"
            "必须只延展当前轮的具体话题与结果。recent_visible_history 仅用于避免矛盾，绝不能用旧话题替换当前话题。"
            "recent_visible_facts 已按当前话题检索；不得主动把未出现在当前轮输入中的案件、委托或线索带入闲聊。"
            "只能描述输入中已经发生且玩家可见的事情；不得新增线索、人物、物品、结果或世界状态。"
            "recent_visible_history 是此前已经对玩家演出的内容：hard_canon 与 soft_canon 都不得被后文否认，"
            "dialogue_claim 只代表角色说过，不保证台词内容属实。不得让 present_entities 以外的人物在场行动或说话。"
            "不要复述 authored_beats 或 recent_visible_history 中已经出现的段落，不要给玩家规定下一步行动。保持第二人称、克制、具象。"
            f"{dialogue_instruction}"
            f"{continuation_instruction}"
            f"{light_dialogue_instruction}"
            "只输出 JSON 对象，格式为 {\"beats\":[\"段落一\",\"段落二\"]}。"
        )
        try:
            outcome = StructuredHarness(self.llm).run(
                NarrativeBeatsOutput,
                system=system,
                user_payload=payload,
                max_output_tokens=900,
            )
            limit = (
                2
                if light_dialogue
                else 5
                if sequence.action_type in {ActionType.TALK, ActionType.DECEIVE}
                else 4
            )
            earlier_texts = [
                *[beat.text for beat in sequence.beats],
                *[str(item["text"]) for item in dedupe_history],
            ]
            beats: list[str] = []
            for beat in outcome.value.beats:
                if substantially_repeats(beat, [*earlier_texts, *beats]):
                    continue
                beats.append(beat)
                if len(beats) >= limit:
                    break
            record_agent_call(
                state,
                role="narrator",
                result=outcome.llm_result,
                validation="accepted",
            )
            return beats
        except Exception:
            record_agent_call(state, role="narrator", result=None, validation="fallback", error=True)
            return []
