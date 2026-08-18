from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from ..context import context_relevance_score
from ..models import OpenActionPlan, WorldState
from .contracts import DisclosureDecision, EvidenceCandidate, KnowledgeQuery, KnowledgeQueryAtom


RetrievalStrategy = Literal["legacy", "bm25", "typed_hybrid_v1", "typed_hybrid_v2"]


_RELATION_MARKERS: dict[str, tuple[str, ...]] = {
    "duration": ("duration", "period", "how_long", "tenancy_duration", "lease_period", "多久", "多长时间", "时长", "住了多", "待了多", "呆了多"),
    "time": ("when", "date", "time", "什么时候", "何时", "几点", "日期", "时间", "哪年"),
    "location": ("where", "location", "place", "destination", "current_residents", "哪里", "哪儿", "何处", "地点", "去向", "去了哪", "住在哪里", "送去"),
    "status": ("status", "state", "outcome", "fate", "condition", "怎么样", "怎么了", "现状", "情况", "如何", "后来"),
    "identity": ("who", "identity", "是谁", "什么人", "哪位"),
    "cause": ("why", "cause", "reason", "为什么", "为何", "原因"),
    "quantity": ("how_many", "count", "number", "多少", "几个", "几名"),
    "history": ("history", "previous", "former", "过去", "历史", "以前", "曾经", "前任", "旧住户"),
    "historical_pattern": ("previous_incidents", "历任", "数代", "多任", "前任们", "其他前任", "其他住户", "以前的住户", "反复", "类似情况"),
    "experience": ("experience", "incident", "event", "haunting", "经历", "遭遇", "发生", "出事", "异象", "异常事件", "看见", "飞起来", "自行飞", "自己飞"),
    "family": ("children", "parents", "child", "孩子", "子女", "儿子", "女儿", "夫妇", "夫妻", "父母"),
    "weakness": ("weakness", "counter", "弱点", "害怕", "怎么对付", "如何反制"),
    "ownership": ("owner", "purchase", "bought", "产权", "主人", "买下", "拥有"),
    "burial": ("burial", "grave", "埋葬", "埋在", "葬在", "遗体", "墓地"),
}

_PREDICATE_RELATIONS: dict[str, set[str]] = {
    "last_tenants": {"history", "experience", "status"},
    "children_status": {"family", "status", "location"},
    "parents_status": {"family", "status", "location"},
    "history": {"history", "historical_pattern", "experience"},
    "location": {"location"},
    "haunting": {"experience"},
    "manifestation": {"experience"},
    "weakness": {"weakness"},
    "purchased_house": {"ownership", "time", "history"},
    "burial_request": {"burial", "location"},
    "burial_record": {"burial", "location"},
    "police_raid": {"history", "experience", "time"},
    "commission": {"status"},
}

_SPECIFIC_RELATIONS = {
    "duration", "time", "location", "identity", "cause", "quantity",
    "historical_pattern", "weakness", "ownership", "burial",
}

_ATOM_SPLIT = re.compile(r"(?:[？?；;]+|(?:，|,)?(?:以及|还有|并且|同时|另外|再问|而且))")

_TERM_STOPWORDS = {
    "什么", "怎么", "哪里", "哪儿", "如何", "这个", "那个", "他们", "她们",
    "自己", "真的", "后来", "现在", "请问", "一下", "了吗", "的吗",
}

_FOCUS_MARKERS: dict[str, tuple[str, ...]] = {
    "children": ("children", "child", "孩子", "子女", "儿子", "女儿"),
    "parents": ("parents", "couple", "夫妇", "夫妻", "父母", "丈夫", "妻子"),
}


def _normalized(text: object) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()]+", "", str(text)).lower()


def _terms(text: object) -> list[str]:
    value = _normalized(text)
    terms = re.findall(r"[a-z0-9_]{2,}", value)
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", value))
    terms.extend(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    terms.extend(chinese[index : index + 3] for index in range(max(0, len(chinese) - 2)))
    if 0 < len(chinese) <= 2:
        terms.append(chinese)
    return [term for term in terms if term not in _TERM_STOPWORDS]


def relation_kinds(text: object) -> set[str]:
    value = str(text).lower()
    kinds = {
        kind
        for kind, markers in _RELATION_MARKERS.items()
        if any(marker in value for marker in markers)
    }
    # Broad conversational words must not weaken a more precise question.
    if kinds & _SPECIFIC_RELATIONS:
        kinds.discard("status")
    if "historical_pattern" in kinds:
        kinds.add("history")
    return kinds


def _focus_kinds(text: object) -> set[str]:
    value = str(text).lower()
    return {
        kind
        for kind, markers in _FOCUS_MARKERS.items()
        if any(marker in value for marker in markers)
    }


def fact_relation_kinds(predicate: str, value: object) -> set[str]:
    predicate_value = predicate.lower()
    kinds = set(_PREDICATE_RELATIONS.get(predicate_value, set()))
    kinds.update(relation_kinds(predicate_value))
    if any(marker in predicate_value for marker in ("status", "state", "condition", "fate")):
        kinds.add("status")
    if any(marker in predicate_value for marker in ("duration", "period", "length_of_stay")):
        kinds.add("duration")
    if any(marker in predicate_value for marker in ("time", "date", "_at")):
        kinds.add("time")
    if any(marker in predicate_value for marker in ("count", "quantity", "number")):
        kinds.add("quantity")
    if any(marker in predicate_value for marker in ("history", "previous", "former")):
        kinds.add("history")
    if any(marker in predicate_value for marker in ("location", "destination", "where")):
        kinds.add("location")
    if any(marker in predicate_value for marker in ("weakness", "counter")):
        kinds.add("weakness")
    value_text = str(value).lower()
    if any(marker in value_text for marker in ("送往", "通往", "位于", "前往", "抵达", "搬到")):
        kinds.add("location")
    return kinds


def query_atoms(query: KnowledgeQuery) -> list[KnowledgeQueryAtom]:
    """Return explicit planner atoms, or a conservative deterministic fallback."""

    if query.atoms:
        normalized: list[KnowledgeQueryAtom] = []
        for index, atom in enumerate(query.atoms, start=1):
            planner_relations = {
                relation for relation in atom.relation_types if relation in _RELATION_MARKERS
            }
            relations = planner_relations | relation_kinds(
                " ".join([atom.query_text, *atom.predicate_hints])
            )
            normalized.append(
                atom.model_copy(
                    update={"id": atom.id or f"atom_{index}", "relation_types": sorted(relations)}
                )
            )
        return normalized

    parts = [part.strip(" ，,。") for part in _ATOM_SPLIT.split(query.query_text) if part.strip(" ，,。")]
    if not parts:
        parts = [query.query_text]
    hint_relations = {hint: relation_kinds(hint) for hint in query.predicate_hints}
    atoms: list[KnowledgeQueryAtom] = []
    for index, part in enumerate(parts, start=1):
        part_relations = relation_kinds(part)
        matching_hints = [
            hint
            for hint, relations in hint_relations.items()
            if not part_relations or not relations or part_relations & relations
        ]
        relations = part_relations | relation_kinds(" ".join(matching_hints))
        atoms.append(
            KnowledgeQueryAtom(
                id=f"atom_{index}",
                query_text=part,
                subject_entity_ids=list(query.subject_entity_ids),
                predicate_hints=matching_hints,
                relation_types=sorted(relations),
            )
        )
    return atoms


@dataclass(frozen=True)
class _KnowledgeDocument:
    fact_id: str
    source_entity_id: str
    subject: str
    predicate: str
    value: object
    confidence: float
    concealed: bool
    knowledge_source: str
    text: str
    terms: tuple[str, ...]
    relations: frozenset[str]
    focus: frozenset[str]


@dataclass(frozen=True)
class _KnowledgeIndex:
    documents: tuple[_KnowledgeDocument, ...]
    document_counts: dict[str, Counter[str]]
    document_frequency: Counter[str]
    average_length: float


class KnowledgeRetriever(ABC):
    @abstractmethod
    def retrieve(self, state: WorldState, query: KnowledgeQuery) -> list[EvidenceCandidate]:
        raise NotImplementedError


class KnowledgeResolver(KnowledgeRetriever):
    """Low-resource typed hybrid retrieval over the canonical NPC knowledge graph."""

    def __init__(self, strategy: RetrievalStrategy = "typed_hybrid_v2"):
        self.strategy = strategy
        self._index_cache: dict[tuple[object, ...], _KnowledgeIndex] = {}

    @staticmethod
    def _entity_text(state: WorldState, entity_id: str) -> str:
        entity = state.entities.get(entity_id)
        return f"{entity_id} {entity.name}" if entity else entity_id

    def _documents(self, state: WorldState, addressee_id: str) -> list[_KnowledgeDocument]:
        documents: list[_KnowledgeDocument] = []
        for entry in state.npc_knowledge:
            if entry.knower_id != addressee_id:
                continue
            fact = state.facts.get(entry.fact_id)
            if fact is None:
                continue
            value = entry.belief_value if entry.belief_value is not None else fact.value
            text = " ".join([
                self._entity_text(state, fact.subject), fact.predicate, str(value), entry.source,
            ])
            documents.append(
                _KnowledgeDocument(
                    fact_id=fact.id,
                    source_entity_id=entry.knower_id,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    value=value,
                    confidence=entry.confidence,
                    concealed=entry.concealed,
                    knowledge_source=entry.source,
                    text=text,
                    terms=tuple(_terms(text)),
                    relations=frozenset(fact_relation_kinds(fact.predicate, value)),
                    focus=frozenset(_focus_kinds(f"{fact.predicate} {value}")),
                )
            )
        return documents

    @staticmethod
    def _build_index(documents: Sequence[_KnowledgeDocument]) -> _KnowledgeIndex:
        counts = {item.fact_id: Counter(item.terms) for item in documents}
        frequency: Counter[str] = Counter()
        for item in documents:
            frequency.update(counts[item.fact_id].keys())
        return _KnowledgeIndex(
            documents=tuple(documents),
            document_counts=counts,
            document_frequency=frequency,
            average_length=(
                sum(len(item.terms) for item in documents) / len(documents)
                if documents
                else 0.0
            ),
        )

    def _knowledge_signature(self, state: WorldState, addressee_id: str) -> tuple[object, ...]:
        entries: list[tuple[object, ...]] = []
        for entry in state.npc_knowledge:
            if entry.knower_id != addressee_id:
                continue
            fact = state.facts.get(entry.fact_id)
            if fact is None:
                continue
            value = entry.belief_value if entry.belief_value is not None else fact.value
            subject = state.entities.get(fact.subject)
            entries.append(
                (
                    entry.fact_id,
                    fact.subject,
                    subject.name if subject else "",
                    fact.predicate,
                    str(value),
                    entry.source,
                    entry.confidence,
                    entry.concealed,
                )
            )
        return (state.scenario_id, addressee_id, *entries)

    def _cached_index(self, state: WorldState, addressee_id: str) -> _KnowledgeIndex:
        signature = self._knowledge_signature(state, addressee_id)
        cached = self._index_cache.get(signature)
        if cached is not None:
            return cached
        index = self._build_index(self._documents(state, addressee_id))
        if len(self._index_cache) >= 32:
            self._index_cache.pop(next(iter(self._index_cache)))
        self._index_cache[signature] = index
        return index

    def cache_size_bytes(self) -> int:
        """Deterministic payload estimate for benchmark reporting."""

        total = 0
        for index in self._index_cache.values():
            for document in index.documents:
                total += len(document.text.encode("utf-8"))
                total += sum(len(term.encode("utf-8")) for term in document.terms)
                total += len(document.relations) * 16 + len(document.focus) * 16
            total += sum(len(counts) * 24 for counts in index.document_counts.values())
            total += len(index.document_frequency) * 24
        return total

    @staticmethod
    def _bm25(
        query_terms: list[str],
        documents: Sequence[_KnowledgeDocument],
        index: _KnowledgeIndex | None = None,
    ) -> dict[str, float]:
        if not query_terms or not documents:
            return {}
        query_counts = Counter(query_terms)
        prepared = index or KnowledgeResolver._build_index(documents)
        document_counts = prepared.document_counts
        average_length = prepared.average_length
        scores: dict[str, float] = {}
        for document in documents:
            counts = document_counts[document.fact_id]
            length = max(1, len(document.terms))
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = prepared.document_frequency.get(term, 0)
                inverse_frequency = math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / max(1.0, average_length))
                score += query_frequency * inverse_frequency * (frequency * 2.2 / denominator)
            if score > 0:
                scores[document.fact_id] = score
        return scores

    @staticmethod
    def _candidate(document: _KnowledgeDocument, **updates: object) -> EvidenceCandidate:
        return EvidenceCandidate(
            fact_id=document.fact_id,
            source_entity_id=document.source_entity_id,
            subject=document.subject,
            predicate=document.predicate,
            value=document.value,
            confidence=document.confidence,
            concealed=document.concealed,
            knowledge_source=document.knowledge_source,
            **updates,
        )

    def _retrieve_legacy(
        self, state: WorldState, query: KnowledgeQuery, documents: list[_KnowledgeDocument]
    ) -> list[EvidenceCandidate]:
        explicit = set(query.explicit_fact_ids)
        subjects = set(query.subject_entity_ids)
        requested_relations = relation_kinds(" ".join([query.query_text, *query.predicate_hints]))
        candidates: list[EvidenceCandidate] = []
        for document in documents:
            if requested_relations and not (requested_relations & set(document.relations)):
                continue
            score = context_relevance_score(query.query_text, document.text)
            if document.fact_id in explicit:
                score += 100
            if document.subject in subjects:
                score += 24
            for hint in query.predicate_hints:
                score += context_relevance_score(hint, document.text)
            if score <= 0:
                continue
            candidates.append(self._candidate(
                document,
                score=float(score),
                relation_types=sorted(document.relations),
                component_scores={"legacy": float(score)},
            ))
        candidates.sort(key=lambda item: (item.score, item.confidence, item.fact_id), reverse=True)
        return candidates[: query.max_results]

    def _retrieve_bm25(
        self, query: KnowledgeQuery, documents: Sequence[_KnowledgeDocument]
    ) -> list[EvidenceCandidate]:
        scores = self._bm25(_terms(" ".join([query.query_text, *query.predicate_hints])), documents)
        candidates = [
            self._candidate(
                document,
                score=score,
                relation_types=sorted(document.relations),
                component_scores={"bm25": score},
            )
            for document in documents
            if (score := scores.get(document.fact_id, 0.0)) > 0
        ]
        candidates.sort(key=lambda item: (item.score, item.confidence, item.fact_id), reverse=True)
        return candidates[: query.max_results]

    def _retrieve_typed(
        self,
        query: KnowledgeQuery,
        documents: Sequence[_KnowledgeDocument],
        *,
        atomic: bool,
        index: _KnowledgeIndex | None = None,
    ) -> list[EvidenceCandidate]:
        atoms = query_atoms(query) if atomic else [KnowledgeQueryAtom(
            id="query",
            query_text=query.query_text,
            subject_entity_ids=query.subject_entity_ids,
            predicate_hints=query.predicate_hints,
            relation_types=sorted(relation_kinds(" ".join([query.query_text, *query.predicate_hints]))),
        )]
        explicit = set(query.explicit_fact_ids)
        accumulated: dict[str, dict[str, object]] = {}
        for atom in atoms:
            requested_relations = set(atom.relation_types)
            atom_text = " ".join([atom.query_text, *atom.predicate_hints])
            requested_focus = _focus_kinds(atom_text)
            strict_subject_ids = {
                entity_id
                for entity_id in atom.subject_entity_ids
                if entity_id != query.addressee_id
            }
            required_relations: set[str] = set()
            if atomic:
                if "duration" in requested_relations:
                    required_relations.add("duration")
                else:
                    required_relations.update(requested_relations & {
                        "time", "location", "identity", "quantity", "historical_pattern",
                        "weakness", "ownership", "burial",
                    })
            bm25_scores = self._bm25(_terms(atom_text), documents, index)
            bm25_rank = {
                fact_id: rank
                for rank, (fact_id, _score) in enumerate(
                    sorted(bm25_scores.items(), key=lambda item: (item[1], item[0]), reverse=True), start=1
                )
            }
            structured_scores: dict[str, float] = {}
            for document in documents:
                fact_relations = set(document.relations)
                exact_predicate_hint = document.predicate.lower() in {
                    hint.lower() for hint in atom.predicate_hints
                }
                if requested_focus and not (requested_focus & set(document.focus)):
                    continue
                if atomic and strict_subject_ids and document.subject not in strict_subject_ids:
                    continue
                if (
                    not exact_predicate_hint
                    and requested_relations
                    and not (requested_relations & fact_relations)
                ):
                    continue
                if (
                    not exact_predicate_hint
                    and required_relations
                    and not required_relations.issubset(fact_relations)
                ):
                    continue
                score = 0.0
                if document.fact_id in explicit:
                    score += 1000.0
                if document.subject in atom.subject_entity_ids:
                    score += 72.0
                if requested_relations:
                    score += 48.0 + 6.0 * len(requested_relations & fact_relations)
                for hint in atom.predicate_hints:
                    if hint.lower() == document.predicate.lower():
                        score += 90.0
                    else:
                        score += min(20.0, context_relevance_score(hint, document.predicate))
                structured_scores[document.fact_id] = score

            structured_rank = {
                fact_id: rank
                for rank, (fact_id, _score) in enumerate(
                    sorted(structured_scores.items(), key=lambda item: (item[1], item[0]), reverse=True), start=1
                )
            }
            for document in documents:
                if document.fact_id not in structured_scores:
                    continue
                bm25 = bm25_scores.get(document.fact_id, 0.0)
                structured = structured_scores[document.fact_id]
                if (
                    document.fact_id not in explicit
                    and not requested_relations
                    and not atom.predicate_hints
                    and bm25 <= 0
                ):
                    continue
                if document.fact_id not in explicit and bm25 <= 0 and structured <= 0:
                    continue
                rrf = 0.0
                if document.fact_id in bm25_rank:
                    rrf += 1.0 / (60 + bm25_rank[document.fact_id])
                if document.fact_id in structured_rank:
                    rrf += 1.0 / (60 + structured_rank[document.fact_id])
                final_score = structured + bm25 * 12.0 + rrf * 120.0 + document.confidence * 2.0
                current = accumulated.setdefault(document.fact_id, {
                    "document": document, "score": 0.0, "atoms": [],
                    "bm25": 0.0, "structured": 0.0, "rrf": 0.0,
                })
                current["score"] = max(float(current["score"]), final_score)
                current["bm25"] = max(float(current["bm25"]), bm25)
                current["structured"] = max(float(current["structured"]), structured)
                current["rrf"] = max(float(current["rrf"]), rrf)
                atom_ids = current["atoms"]
                assert isinstance(atom_ids, list)
                atom_ids.append(atom.id)

        candidates: list[EvidenceCandidate] = []
        for raw in accumulated.values():
            document = raw["document"]
            assert isinstance(document, _KnowledgeDocument)
            candidates.append(self._candidate(
                document,
                score=float(raw["score"]),
                matched_atom_ids=list(dict.fromkeys(raw["atoms"])),
                relation_types=sorted(document.relations),
                component_scores={
                    "bm25": float(raw["bm25"]),
                    "structured": float(raw["structured"]),
                    "rrf": float(raw["rrf"]),
                },
            ))
        candidates.sort(key=lambda item: (item.score, item.confidence, item.fact_id), reverse=True)
        return candidates[: query.max_results]

    def retrieve(self, state: WorldState, query: KnowledgeQuery) -> list[EvidenceCandidate]:
        if self.strategy == "typed_hybrid_v2":
            index = self._cached_index(state, query.addressee_id)
            return self._retrieve_typed(query, index.documents, atomic=True, index=index)
        documents = self._documents(state, query.addressee_id)
        if self.strategy == "legacy":
            return self._retrieve_legacy(state, query, documents)
        if self.strategy == "bm25":
            return self._retrieve_bm25(query, documents)
        if self.strategy == "typed_hybrid_v1":
            return self._retrieve_typed(query, documents, atomic=False)
        raise ValueError(f"Unsupported retrieval strategy: {self.strategy}")


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
        atoms = query_atoms(query)
        selected: list[EvidenceCandidate] = []
        answered_atom_ids: list[str] = []
        for atom in atoms:
            match = next((item for item in ranked if atom.id in item.matched_atom_ids), None)
            if match is None and len(atoms) == 1:
                # Backward-compatible retrievers do not annotate atom coverage.
                match = next(iter(ranked), None)
            if match is not None:
                answered_atom_ids.append(atom.id)
                if match.fact_id not in {item.fact_id for item in selected}:
                    selected.append(match)
        selected = selected[: query.max_results]
        unanswered = [atom for atom in atoms if atom.id not in answered_atom_ids]

        if not selected:
            return DisclosureDecision(
                mode="unknown",
                reason="the addressed NPC has no evidence that supports the requested relation",
                source_entity_id=query.addressee_id,
                canonical_success=f"“这件事我不知道，至少没有能确定的消息。”{source_name}说。",
                canonical_failure=f"{source_name}没有提供可确认的信息。",
                unanswered_atom_ids=[atom.id for atom in atoms],
                unanswered_questions=[atom.query_text for atom in atoms],
            )

        concealed = any(item.concealed for item in selected)
        values = "；".join(str(item.value).rstrip("。") for item in selected)
        uncertainty = "至于问题中的其余部分，我没有能确定的消息。" if unanswered else ""
        joined = "；".join(part for part in (values, uncertainty) if part)
        success = f"“就我所知，{joined}。”{source_name}直接回答。"
        failure = f"{source_name}听懂了问题，却没有披露这部分信息。"
        decision_fields = {
            "source_entity_id": query.addressee_id,
            "approved_evidence": selected,
            "canonical_success": success,
            "canonical_failure": failure,
            "answered_atom_ids": answered_atom_ids,
            "unanswered_atom_ids": [atom.id for atom in unanswered],
            "unanswered_questions": [atom.query_text for atom in unanswered],
        }
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
                    **{**decision_fields, "approved_evidence": []},
                )
            return DisclosureDecision(
                mode="check",
                reason="matching knowledge is deliberately concealed",
                skill=skill,
                difficulty="regular",
                **decision_fields,
            )
        return DisclosureDecision(
            mode="automatic",
            reason=(
                "the NPC has evidence for only part of the compound question"
                if unanswered
                else "the NPC knows facts that support every query atom"
            ),
            **decision_fields,
        )

    @staticmethod
    def apply(plan: OpenActionPlan, decision: DisclosureDecision) -> OpenActionPlan:
        plan = plan.model_copy(deep=True)
        plan.disclosure_mode = decision.mode
        plan.knowledge_source_id = decision.source_entity_id
        plan.approved_fact_ids = [item.fact_id for item in decision.approved_evidence]
        plan.answered_query_parts = list(decision.answered_atom_ids)
        plan.unanswered_query_parts = list(decision.unanswered_questions)
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
