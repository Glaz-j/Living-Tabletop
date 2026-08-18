from __future__ import annotations

import re
from typing import Iterable

from ..models import ActionDefinition, ActionResolution, EventRecord, ScenarioDefinition, WorldState
from .contracts import DisclosedFact, GroundingReport, OutcomeEnvelope


def _normalized(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _claim_normalized(value: object) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()]+", "", str(value)).lower()


_CONCRETE_FACT_MARKERS = (
    # Relations and identity claims are world facts, not harmless stage dressing.
    "姐姐",
    "妹妹",
    "哥哥",
    "弟弟",
    "父亲",
    "母亲",
    "爸爸",
    "妈妈",
    "妻子",
    "丈夫",
    "女儿",
    "儿子",
    "叔叔",
    "阿姨",
    "姨妈",
    "舅舅",
    "亲戚",
    "亲人",
    "表亲",
    "堂亲",
    "监护人",
    # Concrete destinations and status transitions frequently invented by narrators.
    "公寓",
    "医院",
    "疗养院",
    "学校",
    "教堂",
    "墓地",
    "警局",
    "城郊",
    "乡下",
    "外地",
    "出生",
    "死亡",
    "去世",
    "自杀",
    "失踪",
    "结婚",
    "离婚",
    "怀孕",
    "收养",
    "搬去",
    "搬到",
    "送往",
    "住进",
    "埋葬",
    "被捕",
    "定罪",
    "唯一",
)


def _unsupported_concrete_claim(text: str, allowed_text: str) -> str | None:
    """Return the first concrete claim marker absent from the outcome boundary."""
    normalized = _normalized(text)
    allowed = _normalized(allowed_text)
    for marker in _CONCRETE_FACT_MARKERS:
        if marker in normalized and marker not in allowed:
            return marker
    for match in re.finditer(
        r"(?:\d+|[一二两三四五六七八九十百]+)(?:年|个月|月|周|星期|天|日|小时|分钟)",
        normalized,
    ):
        if match.group(0) not in allowed:
            return match.group(0)
    return None


def _has_unbalanced_dialogue(text: str) -> bool:
    return (
        text.count("“") != text.count("”")
        or text.count("‘") != text.count("’")
        or text.count('"') % 2 == 1
        or text.count("'") % 2 == 1
    )


def _unsupported_disclosure_dialogue(text: str, allowed_text: str) -> str | None:
    """Fact-disclosure dialogue may quote only text already inside the outcome."""
    allowed = _claim_normalized(allowed_text)
    patterns = (r"“([^”]+)”", r"‘([^’]+)’", r'"([^"]+)"', r"'([^']+)'")
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            quote = _claim_normalized(match.group(1))
            if len(quote) >= 4 and quote not in allowed:
                return match.group(1)
    return None


class OutcomeBuilder:
    def build(
        self,
        state: WorldState,
        action: ActionDefinition,
        resolution: ActionResolution,
        events: Iterable[EventRecord],
        *,
        turn_id: str,
    ) -> OutcomeEnvelope:
        event_list = list(events)
        player_text = next(
            (
                str(event.payload.get("player_text")).strip()
                for event in event_list
                if event.type == "action_started" and event.payload.get("player_text")
            ),
            None,
        )
        learned_by_id: dict[str, tuple[str | None, bool]] = {}
        for event in event_list:
            if event.type not in {"fact_disclosed", "player_learned_fact"}:
                continue
            fact_id = str(event.payload.get("fact_id") or event.target or "")
            if not fact_id:
                continue
            source_id = event.payload.get("source_entity_id") or event.actor
            newly_learned = bool(event.payload.get("newly_learned", True))
            previous = learned_by_id.get(fact_id)
            learned_by_id[fact_id] = (source_id or (previous[0] if previous else None), newly_learned)

        disclosed = []
        for fact_id, (source_id, newly_learned) in learned_by_id.items():
            fact = state.facts.get(fact_id)
            if fact is None:
                continue
            disclosed.append(
                DisclosedFact(
                    fact_id=fact.id,
                    source_entity_id=source_id,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    value=fact.value,
                    newly_learned=newly_learned,
                )
            )

        player_location = state.entities[state.player.entity_id].location
        location = state.entities.get(player_location or "")
        present = [
            {"id": entity.id, "name": entity.name, "type": entity.type.value}
            for entity in state.entities.values()
            if entity.active
            and entity.location == player_location
            and entity.id != state.player.entity_id
        ]
        visible_event_texts = [
            str(event.payload["text"])
            for event in event_list
            if event.visible_to_player and event.payload.get("text")
        ]
        intervention = resolution.director_intervention
        director_opportunity = None
        if intervention is not None:
            director_opportunity = {
                "kind": intervention.action,
                "text": intervention.world_justification,
                "affected_entities": intervention.affected_entities,
            }
        succeeded = resolution.check.succeeded if resolution.check else resolution.accepted
        canonical_beats = action.success_beats if succeeded else action.failure_beats
        return OutcomeEnvelope(
            turn_id=turn_id,
            action_id=action.id,
            action_label=action.label,
            action_type=action.type,
            player_text=player_text,
            accepted=resolution.accepted,
            mechanical_result=resolution.check.model_dump(mode="json") if resolution.check else None,
            canonical_seed=resolution.narrative_seed,
            canonical_beats=canonical_beats or [resolution.narrative_seed],
            disclosed_facts=disclosed,
            visible_events=visible_event_texts,
            scene={
                "id": location.id if location else None,
                "name": location.name if location else "未知地点",
                "description": location.attributes.get("description", "") if location else "",
            },
            present_entities=present,
            director_opportunity=director_opportunity,
        )


class GroundingValidator:
    """Conservative post-generation guard; rejection never mutates world state."""

    def __init__(self, scenario: ScenarioDefinition):
        self.scenario = scenario

    def validate(
        self,
        state: WorldState,
        outcome: OutcomeEnvelope,
        beats: Iterable[str],
    ) -> tuple[list[str], GroundingReport]:
        approved_ids = {item.fact_id for item in outcome.disclosed_facts}
        allowed_entities = {
            state.player.entity_id,
            *(item["id"] for item in outcome.present_entities),
            *(item.source_entity_id for item in outcome.disclosed_facts if item.source_entity_id),
        }
        scene_id = outcome.scene.get("id")
        if isinstance(scene_id, str) and scene_id:
            allowed_entities.add(scene_id)
        forbidden_values = [
            _normalized(fact.value)
            for fact in state.facts.values()
            if fact.id not in approved_ids
            and isinstance(fact.value, (str, int, float))
            and len(_normalized(fact.value)) >= 6
        ]
        forbidden_names = [
            (entity.id, _normalized(entity.name))
            for entity in state.entities.values()
            if entity.id not in allowed_entities and len(_normalized(entity.name)) >= 2
        ]

        accepted: list[str] = []
        rejected: list[str] = []
        reasons: list[str] = []
        current_topic = outcome.player_text or outcome.canonical_seed
        case_terms = (
            "案件",
            "委托",
            "线索",
            "宅邸",
            "房子",
            "调查",
            "地下室",
            "仪式",
            "诅咒",
            "死亡",
            "凶杀",
            "失踪",
            "消失",
            "阴影",
            "秘密",
        )
        topic_is_case_related = any(term in current_topic for term in case_terms)
        allowed_text = "\n".join(
            [
                outcome.action_label,
                outcome.player_text or "",
                outcome.canonical_seed,
                *outcome.canonical_beats,
                *outcome.visible_events,
                *(str(item.value) for item in outcome.disclosed_facts),
                str(outcome.scene.get("description", "")),
                *(str(item.get("name", "")) for item in outcome.present_entities),
            ]
        )
        for beat in beats:
            normalized = _normalized(beat)
            reason = None
            if _has_unbalanced_dialogue(beat):
                reason = "contains unfinished dialogue"
            elif any(value in normalized for value in forbidden_values):
                reason = "contains an unapproved fact value"
            else:
                entity_hit = next(
                    (entity_id for entity_id, name in forbidden_names if name in normalized),
                    None,
                )
                if entity_hit is not None:
                    reason = f"mentions an entity outside the outcome scene: {entity_hit}"
            if reason is None:
                unsupported_marker = _unsupported_concrete_claim(beat, allowed_text)
                if unsupported_marker is not None:
                    reason = f"adds an unsupported concrete claim: {unsupported_marker}"
            if reason is None and outcome.disclosed_facts:
                unsupported_quote = _unsupported_disclosure_dialogue(beat, allowed_text)
                if unsupported_quote is not None:
                    reason = "adds dialogue outside the disclosed outcome"
            if (
                reason is None
                and not outcome.disclosed_facts
                and not topic_is_case_related
                and any(term in beat for term in case_terms)
            ):
                reason = "switches an unrelated turn back to the case"
            if reason is None:
                accepted.append(beat)
            else:
                rejected.append(beat)
                reasons.append(reason)

        return accepted, GroundingReport(
            accepted=not rejected,
            rejected_beats=rejected,
            reasons=reasons,
            approved_fact_ids=sorted(approved_ids),
        )
