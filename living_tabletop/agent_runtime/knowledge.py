from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..context import context_relevance_score
from ..models import OpenActionPlan, WorldState
from .contracts import DisclosureDecision, EvidenceCandidate, KnowledgeQuery


class KnowledgeRetriever(ABC):
    @abstractmethod
    def retrieve(self, state: WorldState, query: KnowledgeQuery) -> list[EvidenceCandidate]:
        raise NotImplementedError


class KnowledgeResolver(KnowledgeRetriever):
    """Structured NPC knowledge retrieval with lexical ranking as a bounded fallback."""

    @staticmethod
    def _entity_text(state: WorldState, entity_id: str) -> str:
        entity = state.entities.get(entity_id)
        return f"{entity_id} {entity.name}" if entity else entity_id

    @staticmethod
    def _relation_kinds(text: str) -> set[str]:
        value = text.lower()
        groups = {
            "duration": ("duration", "period", "how_long", "多久", "多长时间", "时长", "住了多"),
            "time": ("when", "date", "time", "什么时候", "何时", "几点", "日期", "时间"),
            "location": (
                "where", "location", "place", "destination", "哪里", "哪儿", "何处", "地点", "去向"
            ),
            "status": (
                "status", "state", "happen", "outcome", "fate", "condition",
                "怎么样", "怎么了", "现状", "遭遇", "情况", "如何"
            ),
            "identity": ("who", "identity", "是谁", "什么人", "哪位"),
            "cause": ("why", "cause", "reason", "为什么", "为何", "原因"),
            "quantity": ("how_many", "count", "多少", "几个", "几名"),
        }
        return {
            kind
            for kind, markers in groups.items()
            if any(marker in value for marker in markers)
        }

    @classmethod
    def _fact_relation_kinds(cls, predicate: str, value: object) -> set[str]:
        text = f"{predicate} {value}".lower()
        kinds = cls._relation_kinds(predicate)
        if any(marker in text for marker in ("送往", "通往", "位于", "前往", "抵达", "location")):
            kinds.add("location")
        if any(marker in predicate.lower() for marker in ("status", "state", "condition", "fate")):
            kinds.add("status")
        if any(marker in predicate.lower() for marker in ("duration", "period", "length_of_stay")):
            kinds.add("duration")
        if any(marker in predicate.lower() for marker in ("time", "date", "at")):
            kinds.add("time")
        if any(marker in predicate.lower() for marker in ("count", "quantity", "number")):
            kinds.add("quantity")
        return kinds

    def retrieve(self, state: WorldState, query: KnowledgeQuery) -> list[EvidenceCandidate]:
        candidates: list[EvidenceCandidate] = []
        explicit = set(query.explicit_fact_ids)
        subjects = set(query.subject_entity_ids)
        requested_relations = self._relation_kinds(
            " ".join([query.query_text, *query.predicate_hints])
        )
        focus_groups = {
            "children": ("孩子", "子女", "儿子", "女儿", "children", "child"),
            "parents": ("夫妇", "夫妻", "父母", "丈夫", "妻子", "parents", "couple"),
        }
        query_lower = query.query_text.lower()
        requested_focus = {
            kind
            for kind, markers in focus_groups.items()
            if any(marker in query_lower for marker in markers)
        }
        for entry in state.npc_knowledge:
            if entry.knower_id != query.addressee_id:
                continue
            fact = state.facts.get(entry.fact_id)
            if fact is None:
                continue
            value = entry.belief_value if entry.belief_value is not None else fact.value
            fact_relations = self._fact_relation_kinds(fact.predicate, value)
            if requested_relations and not (requested_relations & fact_relations):
                continue
            searchable = " ".join(
                [
                    self._entity_text(state, fact.subject),
                    fact.predicate,
                    str(value),
                    entry.source,
                ]
            )
            searchable_lower = searchable.lower()
            if requested_focus and not any(
                any(marker in searchable_lower for marker in focus_groups[kind])
                for kind in requested_focus
            ):
                continue
            score = context_relevance_score(query.query_text, searchable)
            if requested_focus:
                score += 40
            if fact.id in explicit:
                score += 100
            if fact.subject in subjects:
                score += 24
            for hint in query.predicate_hints:
                score += context_relevance_score(hint, searchable)
            if score <= 0:
                continue
            candidates.append(
                EvidenceCandidate(
                    fact_id=fact.id,
                    source_entity_id=entry.knower_id,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    value=value,
                    confidence=entry.confidence,
                    concealed=entry.concealed,
                    score=score,
                    knowledge_source=entry.source,
                )
            )
        candidates.sort(key=lambda item: (item.score, item.confidence, item.fact_id), reverse=True)
        return candidates[: query.max_results]


class DisclosurePolicy:
    def decide(
        self,
        state: WorldState,
        query: KnowledgeQuery,
        candidates: Iterable[EvidenceCandidate],
    ) -> DisclosureDecision:
        ranked = list(candidates)
        source = state.entities.get(query.addressee_id)
        source_name = source.name if source else "对方"
        if not ranked:
            return DisclosureDecision(
                mode="unknown",
                reason="the addressed NPC has no matching structured knowledge",
                source_entity_id=query.addressee_id,
                canonical_success=f"“这件事我不知道，至少没有能确定的消息。”{source_name}说。",
                canonical_failure=f"{source_name}没有提供可确认的信息。",
            )

        top_score = ranked[0].score
        selected = [item for item in ranked if item.score >= max(1, top_score - 8)][:2]
        concealed = any(item.concealed for item in selected)
        values = "；".join(str(item.value).rstrip("。") for item in selected)
        success = f"“就我所知，{values}。”{source_name}直接回答。"
        failure = f"{source_name}听懂了问题，却没有披露这部分信息。"
        if concealed:
            skills = state.player.skills
            skill = max(
                (name for name in ("persuasion", "charm", "psychology", "deception") if name in skills),
                key=lambda name: skills[name],
                default=None,
            )
            if skill is None:
                return DisclosureDecision(
                    mode="refuse",
                    reason="the NPC conceals the fact and the player has no applicable social skill",
                    source_entity_id=query.addressee_id,
                    approved_evidence=[],
                    canonical_success=failure,
                    canonical_failure=failure,
                )
            return DisclosureDecision(
                mode="check",
                reason="matching knowledge is deliberately concealed",
                source_entity_id=query.addressee_id,
                approved_evidence=selected,
                skill=skill,
                difficulty="regular",
                canonical_success=success,
                canonical_failure=failure,
            )
        return DisclosureDecision(
            mode="automatic",
            reason="the NPC knows the matching fact and does not conceal it",
            source_entity_id=query.addressee_id,
            approved_evidence=selected,
            canonical_success=success,
            canonical_failure=failure,
        )

    @staticmethod
    def apply(plan: OpenActionPlan, decision: DisclosureDecision) -> OpenActionPlan:
        plan = plan.model_copy(deep=True)
        plan.disclosure_mode = decision.mode
        plan.knowledge_source_id = decision.source_entity_id
        plan.approved_fact_ids = [item.fact_id for item in decision.approved_evidence]
        plan.success_text = decision.canonical_success
        plan.failure_text = decision.canonical_failure
        if decision.mode == "check":
            plan.resolution = "check"
            plan.skill = decision.skill
            plan.difficulty = decision.difficulty
            plan.risk = "uncertain"
        elif decision.mode in {"automatic", "unknown", "refuse"}:
            plan.resolution = "automatic"
            plan.skill = None
            plan.difficulty = "regular"
        return plan
