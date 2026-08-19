from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable

from ..context import context_relevance_score
from ..harness import HarnessValidationError, StructuredHarness
from ..llm import LLMUnavailable, OpenAICompatibleLLM, record_agent_call
from ..models import EntityType, Fact, OpenActionPlan, WorldState
from .contracts import (
    AssembledTurnContext,
    DialogueTurnOutput,
    DisclosureDecision,
    EvidenceCandidate,
    KnowledgeQuery,
    PlannedOpenAction,
    PlayerIntentEnvelope,
    SoftFactProposal,
)


logger = logging.getLogger(__name__)


class DialogueValidationError(ValueError):
    pass


class SoftFactValidator:
    """Allows creation by default for low-stakes details, while locking hard canon."""

    _LOCATION_PREDICATES = {
        "address",
        "district",
        "route_description",
        "travel_time",
        "opening_hours",
        "access_notes",
        "contact_details",
        "local_reputation",
        "appearance",
    }
    _NPC_PREDICATES = {
        "contact_details",
        "local_reputation",
        "appearance",
        "mannerism",
        "habit",
        "preference",
        "minor_background",
    }
    _NON_PLAYER_TYPES = {
        EntityType.LOCATION,
        EntityType.NPC,
        EntityType.ITEM,
        EntityType.OBJECT,
    }

    @staticmethod
    def _normalized(value: object) -> str:
        return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()]+", "", str(value)).lower()

    def validate(
        self,
        state: WorldState,
        output: DialogueTurnOutput,
        *,
        speaker_id: str,
        allowed_entity_ids: set[str],
        allowed_fact_ids: set[str],
    ) -> None:
        if speaker_id not in state.entities or state.entities[speaker_id].type != EntityType.NPC:
            raise DialogueValidationError("dialogue speaker must be a present NPC")
        player_location = state.entities[state.player.entity_id].location
        if state.entities[speaker_id].location != player_location:
            raise DialogueValidationError("dialogue speaker is not present in the current scene")
        if not output.beats or any(not beat.strip() for beat in output.beats):
            raise DialogueValidationError("dialogue must contain complete non-empty beats")
        if len(set(output.used_fact_ids)) != len(output.used_fact_ids):
            raise DialogueValidationError("used_fact_ids contains duplicates")
        invalid_used = set(output.used_fact_ids) - allowed_fact_ids
        if invalid_used:
            raise DialogueValidationError(
                f"dialogue cites facts the speaker was not allowed to disclose: {sorted(invalid_used)}"
            )

        reply_text = "\n".join(output.beats)
        normalized_reply = self._normalized(reply_text)
        for fact in state.facts.values():
            if fact.id in state.player_known_fact_ids or fact.id in allowed_fact_ids:
                continue
            if fact.canon != "hard_canon":
                continue
            hidden_value = self._normalized(fact.value)
            if len(hidden_value) >= 6 and hidden_value in normalized_reply:
                raise DialogueValidationError(f"dialogue exposes hidden hard canon: {fact.id}")
        for entity in state.entities.values():
            if entity.id in allowed_entity_ids or not entity.active:
                continue
            hidden_name = self._normalized(entity.name)
            if len(hidden_name) >= 2 and hidden_name in normalized_reply:
                raise DialogueValidationError(
                    f"dialogue mentions an entity outside accessible world context: {entity.id}"
                )

        proposal_keys: set[tuple[str, str]] = set()
        for proposal in output.proposed_facts:
            entity = state.entities.get(proposal.subject_entity_id)
            if proposal.subject_entity_id not in allowed_entity_ids or entity is None:
                raise DialogueValidationError(
                    f"soft fact subject is not an established accessible entity: {proposal.subject_entity_id}"
                )
            if entity.type not in self._NON_PLAYER_TYPES:
                raise DialogueValidationError("soft facts cannot modify the player or hostile creatures")
            if proposal.predicate in self._LOCATION_PREDICATES and entity.type != EntityType.LOCATION:
                if proposal.predicate not in self._NPC_PREDICATES:
                    raise DialogueValidationError(
                        f"{proposal.predicate} may only be generated for a location"
                    )
            if proposal.predicate in {"mannerism", "habit", "preference", "minor_background"}:
                if entity.type != EntityType.NPC:
                    raise DialogueValidationError(
                        f"{proposal.predicate} may only be generated for an NPC"
                    )
            key = (proposal.subject_entity_id, proposal.predicate)
            if key in proposal_keys:
                raise DialogueValidationError(f"duplicate soft fact slot: {key}")
            proposal_keys.add(key)
            attribute_value = entity.attributes.get(proposal.predicate)
            if attribute_value is not None and self._normalized(attribute_value) != self._normalized(proposal.value):
                raise DialogueValidationError(
                    f"soft fact conflicts with an established entity attribute: {key}"
                )
            for fact in state.facts.values():
                if fact.subject != proposal.subject_entity_id or fact.predicate != proposal.predicate:
                    continue
                if fact.id not in state.player_known_fact_ids and fact.canon == "hard_canon":
                    raise DialogueValidationError(f"soft fact attempts to expose hidden hard canon: {fact.id}")
                if self._normalized(fact.value) != self._normalized(proposal.value):
                    raise DialogueValidationError(f"soft fact conflicts with established canon: {fact.id}")

    def materialize(
        self,
        state: WorldState,
        proposals: Iterable[SoftFactProposal],
        *,
        speaker_id: str,
    ) -> tuple[list[Fact], list[str]]:
        created: list[Fact] = []
        reused_ids: list[str] = []
        for proposal in proposals:
            existing = next(
                (
                    fact
                    for fact in state.facts.values()
                    if fact.subject == proposal.subject_entity_id
                    and fact.predicate == proposal.predicate
                    and self._normalized(fact.value) == self._normalized(proposal.value)
                ),
                None,
            )
            if existing is not None:
                reused_ids.append(existing.id)
                continue
            digest = hashlib.sha1(
                (
                    f"{proposal.subject_entity_id}|{proposal.predicate}|"
                    f"{self._normalized(proposal.value)}"
                ).encode("utf-8")
            ).hexdigest()[:16]
            created.append(
                Fact(
                    id=f"f_generated_{digest}",
                    subject=proposal.subject_entity_id,
                    predicate=proposal.predicate,
                    value=proposal.value.strip(),
                    visibility="PLAYER",
                    created_at=state.world_time,
                    source=f"dialogue:{speaker_id}",
                    immutable=False,
                    canon="soft_canon",
                )
            )
        return created, reused_ids


class DialogueAgent:
    """LLM-first dialogue author; the validator governs only consistency and authority."""

    def __init__(self, llm: OpenAICompatibleLLM):
        self.llm = llm
        self.fact_validator = SoftFactValidator()

    @staticmethod
    def _safe_world_entities(
        state: WorldState,
        context: AssembledTurnContext,
        envelope: PlayerIntentEnvelope,
        speaker_id: str,
    ) -> list[dict[str, object]]:
        ids = {speaker_id, envelope.actor_id}
        ids.update(str(item["id"]) for item in context.present_entities if item.get("id"))
        ids.update(str(item["id"]) for item in context.inventory if item.get("id"))
        ids.update(str(item["subject"]) for item in context.player_known_facts if item.get("subject"))
        ids.update(
            str(item["target"])
            for item in context.available_actions
            if item.get("target") in state.entities
        )
        compact_input = re.sub(r"\s+", "", envelope.text)
        for entity in state.entities.values():
            if entity.active and entity.name and entity.name in compact_input:
                ids.add(entity.id)
        return [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type.value,
                "tags": sorted(entity.tags - {"hidden"}),
            }
            for entity_id in sorted(ids)
            if (entity := state.entities.get(entity_id)) is not None
            and entity.active
            and entity.type in SoftFactValidator._NON_PLAYER_TYPES
        ]

    @staticmethod
    def _fallback(
        plan: OpenActionPlan,
        disclosure: DisclosureDecision | None,
    ) -> DialogueTurnOutput:
        used = [item.fact_id for item in disclosure.approved_evidence] if disclosure else []
        unresolved = list(disclosure.unanswered_questions) if disclosure else []
        return DialogueTurnOutput(
            beats=[plan.success_text],
            used_fact_ids=used,
            proposed_facts=[],
            answered_query_parts=list(disclosure.answered_atom_ids) if disclosure else [],
            unresolved_query_parts=unresolved,
        )

    def compose(
        self,
        state: WorldState,
        *,
        envelope: PlayerIntentEnvelope,
        planned: PlannedOpenAction,
        plan: OpenActionPlan,
        context: AssembledTurnContext,
        query: KnowledgeQuery | None,
        evidence: list[EvidenceCandidate],
        disclosure: DisclosureDecision | None,
    ) -> DialogueTurnOutput:
        speaker_id = planned.addressee_id or planned.target_entity_id or plan.target_entity_id
        if not speaker_id:
            return self._fallback(plan, disclosure)
        speaker = state.entities.get(speaker_id)
        if speaker is None or speaker.type != EntityType.NPC:
            return self._fallback(plan, disclosure)

        safe_entities = self._safe_world_entities(state, context, envelope, speaker_id)
        allowed_entity_ids = {str(item["id"]) for item in safe_entities}
        allowed_evidence = list(disclosure.approved_evidence) if disclosure else []
        allowed_fact_ids = {item.fact_id for item in allowed_evidence}
        speaker_knowledge = {
            item.fact_id
            for item in state.npc_knowledge
            if item.knower_id == speaker_id and not item.concealed
        }
        continuity_facts = [
            item
            for item in context.player_known_facts
            if item.get("id") in speaker_knowledge
            and context_relevance_score(envelope.text, str(item.get("value", ""))) > 0
        ]
        allowed_fact_ids.update(str(item["id"]) for item in continuity_facts)
        raw_question_parts = [
            part.strip(" ，,。；;")
            for part in re.split(r"[？?]+", envelope.text)
            if part.strip(" ，,。；;")
        ]
        query_parts = (
            raw_question_parts
            if len(raw_question_parts) > 1
            else [atom.query_text for atom in query.atoms]
            if query and query.atoms
            else [query.query_text]
            if query
            else raw_question_parts
        )
        payload = {
            "player_input": envelope.text,
            "speech_act": planned.speech_act,
            "speaker": {
                "id": speaker.id,
                "name": speaker.name,
                "role": speaker.attributes.get("role"),
                "mood": speaker.attributes.get("mood"),
            },
            "scene": context.scene,
            "present_entities": context.present_entities,
            "recent_visible_history": context.recent_visible_history,
            "question_parts": query_parts,
            "retrieval_diagnostics": [
                {
                    "fact_id": item.fact_id,
                    "subject": item.subject,
                    "predicate": item.predicate,
                    "matched_atom_ids": item.matched_atom_ids,
                    "relation_types": item.relation_types,
                    "score": item.score,
                    "concealed": item.concealed,
                }
                for item in evidence
            ],
            "disclosable_evidence": [
                *[item.model_dump(mode="json") for item in allowed_evidence],
                *continuity_facts,
            ],
            "previous_coverage_assessment": (
                disclosure.model_dump(mode="json") if disclosure else None
            ),
            "player_known_facts": context.player_known_facts,
            "generatable_world_entities": safe_entities,
            "allowed_soft_predicates": list(SoftFactProposal.model_fields["predicate"].annotation.__args__),
            "output_rules": {
                "beats": "1-4 complete Chinese performance beats with the NPC speaking directly in quotation marks",
                "used_fact_ids": "only ids from disclosable_evidence that are actually stated",
                "proposed_facts": "new low-stakes details needed for a complete answer; use only listed entities and predicates",
                "coverage": "cover every question part, or list the exact unresolved part explicitly",
            },
        }
        system = (
            "你是 Living Tabletop 的 Dialogue Turn Agent，是当前对话内容的主要作者。"
            "先完整回应玩家这一轮真正问的每个部分，再考虑气氛；不要把相关但不回答问题的事实冒充答案。"
            "retrieval_diagnostics 只是无事实值的检索诊断，不是白名单，也不保证能回答问题；只有 disclosable_evidence 可以作为既有事实引用。"
            "若回答缺少地址、城区、路线、路程、营业时间、联络方式、普通名声、外观、习惯或无关谜底的小背景，"
            "可以为 generatable_world_entities 中已经存在的实体提出 proposed_facts，并在台词中清楚表达同一信息；允许自然的同义改写。"
            "不得提出凶手、谜底、隐藏线索、超自然真相、身份、生死、亲属关系、所有权、核心历史、伤害、数值或位置移动；"
            "这些内容缺失时应自然承认不知道。不得创建新实体，不得替玩家决定行动或移动玩家。"
            "NPC 的关键回复必须放在引号内直接说出，并用第一人称说话，不能用自己的姓名称呼自己；可以加入简短动作和环境描写。"
            "recent_visible_history 只用于承接关系和指代，不要复述与当前问题无关的旧台词、旧建议、主线提示或神秘预兆；当前问题回答完整后自然收束。"
            "普通闲聊允许自由、自然地发挥，不需要把纯气氛写入 proposed_facts。"
            "只输出符合 JSON Schema 的对象。"
        )

        try:
            outcome = StructuredHarness(self.llm).run(
                DialogueTurnOutput,
                system=system,
                user_payload=payload,
                max_output_tokens=1100,
                temperature=0.45,
                post_validate=lambda value: self.fact_validator.validate(
                    state,
                    value,
                    speaker_id=speaker_id,
                    allowed_entity_ids=allowed_entity_ids,
                    allowed_fact_ids=allowed_fact_ids,
                ),
            )
            record_agent_call(
                state,
                role="dialogue",
                result=outcome.llm_result,
                validation="accepted",
            )
            return outcome.value
        except HarnessValidationError as exc:
            record_agent_call(
                state,
                role="dialogue",
                result=exc.result,
                validation="rejected",
                error=True,
            )
            logger.warning("Dialogue output failed validation after repair: %s", exc)
            raise LLMUnavailable(
                "LLM returned an invalid dialogue turn",
                public_message=(
                    "模型服务已经响应，但完整对话在自动修复后仍未通过世界一致性检查。"
                    "行动未提交，游戏状态没有改变；请重试或切换模型。"
                ),
            ) from exc
        except LLMUnavailable:
            record_agent_call(state, role="dialogue", result=None, validation="rejected", error=True)
            raise
        except Exception as exc:
            record_agent_call(state, role="dialogue", result=None, validation="rejected", error=True)
            logger.exception("Dialogue composition failed: %s", exc)
            raise LLMUnavailable(
                "LLM could not compose the dialogue turn",
                public_message=(
                    "完整对话生成模块处理失败，行动未提交，游戏状态没有改变。请稍后重试。"
                ),
            ) from exc

    def apply(
        self,
        state: WorldState,
        plan: OpenActionPlan,
        output: DialogueTurnOutput,
        *,
        speaker_id: str,
    ) -> OpenActionPlan:
        generated, reused_ids = self.fact_validator.materialize(
            state,
            output.proposed_facts,
            speaker_id=speaker_id,
        )
        fact_ids = list(dict.fromkeys([*output.used_fact_ids, *reused_ids, *(fact.id for fact in generated)]))
        beats = [beat.strip() for beat in output.beats if beat.strip()]
        return plan.model_copy(
            update={
                "success_text": "\n\n".join(beats),
                "success_beats": beats,
                "generated_facts": generated,
                "approved_fact_ids": fact_ids,
                "knowledge_source_id": speaker_id,
                "answered_query_parts": list(output.answered_query_parts),
                "unanswered_query_parts": list(output.unresolved_query_parts),
            }
        )
