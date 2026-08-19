from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_runtime.contracts import OutcomeEnvelope
from .agent_runtime.outcome import GroundingValidator
from .context import recent_visible_context, substantially_repeats
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
        self.grounding_validator = GroundingValidator(scenario)

    @staticmethod
    def _light_dialogue_fallback(outcome: OutcomeEnvelope) -> str:
        npc = next(
            (
                str(entity.get("name"))
                for entity in outcome.present_entities
                if entity.get("type") == "NPC" and entity.get("name")
            ),
            "对方",
        )
        topic = f"{outcome.action_label}\n{outcome.player_text or ''}"
        if any(token in topic for token in ("天气", "气候", "下雨", "雨天", "晴天", "冷", "热", "刮风")):
            return f"“是啊，天气总会悄悄影响人的心情。”{npc}顺着这个话题回应了一句。"
        return f"“嗯，我明白你的意思。”{npc}顺着当前的话题回应了一句。"

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
        outcome_payload = resolution.outcome_envelope or {
            "canonical_seed": base,
            "disclosed_facts": [],
            "visible_events": event_texts,
            "scene": {
                "id": location.id if location else None,
                "name": location.name if location else "未知地点",
                "description": location.attributes.get("description", "") if location else "",
            },
        }
        payload: dict[str, Any] = {
            "location": location.name if location else "未知地点",
            "location_description": location.attributes.get("description", "") if location else "",
            "resolved_action": action.label,
            "mechanical_result": resolution.check.model_dump(mode="json") if resolution.check else None,
            "canonical_narrative_seed": base,
            "outcome_envelope": outcome_payload,
            "visible_facts": outcome_payload.get("disclosed_facts", []),
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
            narrative = outcome.value.narrative
            validation = "accepted"
            if resolution.outcome_envelope is not None:
                envelope = OutcomeEnvelope.model_validate(resolution.outcome_envelope)
                approved, grounding = self.grounding_validator.validate(
                    state,
                    envelope,
                    [narrative],
                )
                if not approved:
                    narrative = base
                    validation = "rejected"
            record_agent_call(
                state,
                role="narrator",
                result=outcome.llm_result,
                validation=validation,
            )
            return narrative
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
        if sequence.outcome_envelope is not None:
            outcome_envelope = OutcomeEnvelope.model_validate(sequence.outcome_envelope)
        else:
            present = [
                {"id": entity.id, "name": entity.name, "type": entity.type.value}
                for entity in state.entities.values()
                if entity.active
                and entity.location == location_id
                and entity.id != state.player.entity_id
            ]
            outcome_envelope = OutcomeEnvelope(
                turn_id=sequence.turn_trace_id or sequence.id,
                action_id=sequence.action_id or "legacy_action",
                action_label=sequence.action_label or "继续当前场景",
                action_type=sequence.action_type or ActionType.OTHER,
                player_text=sequence.player_text,
                accepted=True,
                mechanical_result=sequence.mechanical_result,
                canonical_seed=sequence.canonical_seed,
                scene={
                    "id": location.id if location else None,
                    "name": location.name if location else "未知地点",
                    "description": location.attributes.get("description", "") if location else "",
                },
                present_entities=present,
            )
        present_entities = outcome_envelope.present_entities
        dedupe_history = recent_visible_context(
            state,
            exclude_sequence_id=sequence.id,
        )
        recent_visible_facts = [
            item.model_dump(mode="json") for item in outcome_envelope.disclosed_facts
        ]
        light_dialogue = (
            sequence.action_type == ActionType.TALK
            and (sequence.mechanical_result or {}).get("outcome") == "AUTOMATIC"
            and not outcome_envelope.disclosed_facts
        )
        payload: dict[str, Any] = {
            "outcome_envelope": outcome_envelope.model_dump(mode="json"),
            "location": outcome_envelope.scene.get("name", "未知地点"),
            "location_description": (
                ""
                if light_dialogue
                else outcome_envelope.scene.get("description", "")
            ),
            "resolved_action": sequence.action_label,
            "player_text": sequence.player_text,
            "action_type": sequence.action_type.value if sequence.action_type else None,
            "continues_previous": sequence.continues_previous,
            "mechanical_result": sequence.mechanical_result,
            "canonical_narrative_seed": sequence.canonical_seed,
            "authored_beats": outcome_envelope.canonical_beats,
            "present_entities": present_entities,
            "recent_visible_history": [],
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
            "outcome_envelope 拥有最高优先级并且是唯一的事实边界；只能延展其中的当前动作、机械结果、disclosed_facts、可见事件、场景和在场人物。"
            "不得使用此前轮次的已知事实补写当前轮，不得猜测 NPC 知识，不得把案件、委托或线索带入无关闲聊。"
            "只能描述已经发生且玩家可见的事情；不得新增线索、人物、物品、因果结论、结果或世界状态。"
            "若 answer_coverage.unanswered_parts 非空，不得暗示这些问题已有答案；允许明确保留未知，"
            "也允许省略不能回答当前问题的材料。"
            "若 disclosed_facts 非空，不得替 NPC 增加 outcome_envelope 中没有的台词；只可原样复述已给出的事实台词，并用不引入新物品或关系的神态、停顿和环境感受扩写。"
            "不得让 present_entities 以外的人物在场行动或说话。不要复述 authored_beats，不要给玩家规定下一步行动。"
            "保持第二人称、克制、具象。"
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
            beats, grounding = self.grounding_validator.validate(
                state,
                outcome_envelope,
                beats,
            )
            if light_dialogue and len(beats) < 2:
                fallback = self._light_dialogue_fallback(outcome_envelope)
                fallback_beats, _fallback_grounding = self.grounding_validator.validate(
                    state,
                    outcome_envelope,
                    [fallback],
                )
                if fallback_beats and not substantially_repeats(
                    fallback,
                    [*earlier_texts, *beats],
                ):
                    # A rejected first beat often contained the actual NPC reply while a
                    # surviving second beat only contained its aftermath. Put the safe
                    # reply back before that aftermath so the exchange remains coherent.
                    beats = (
                        [fallback, *beats]
                        if grounding.rejected_beats
                        else [*beats, fallback]
                    )[:limit]
            sequence.grounding_report = grounding.model_dump(mode="json")
            record_agent_call(
                state,
                role="narrator",
                result=outcome.llm_result,
                validation="accepted" if grounding.accepted else "rejected",
            )
            return beats
        except Exception:
            record_agent_call(state, role="narrator", result=None, validation="fallback", error=True)
            return (
                [self._light_dialogue_fallback(outcome_envelope)]
                if light_dialogue
                else []
            )
