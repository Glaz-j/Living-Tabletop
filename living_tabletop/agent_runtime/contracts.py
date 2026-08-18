from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import ActionType, OpenActionPlan


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class IntentSeed(RuntimeModel):
    action_id: str
    label: str
    action_type: ActionType
    target_entity_id: str | None = None
    hard_constraints: dict[str, Any] = Field(default_factory=dict)


class PlayerIntentEnvelope(RuntimeModel):
    id: str
    source: Literal["option", "free_text"]
    text: str = Field(min_length=1, max_length=1000)
    actor_id: str
    scene_id: str | None = None
    intent_seed: IntentSeed | None = None


class ResolvedReferent(RuntimeModel):
    mention: str
    entity_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class KnowledgeQueryAtom(RuntimeModel):
    """One independently answerable part of a player's factual question."""

    id: str = Field(min_length=1, max_length=80)
    query_text: str = Field(min_length=1, max_length=500)
    subject_entity_ids: list[str] = Field(default_factory=list)
    predicate_hints: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)


class KnowledgeQuery(RuntimeModel):
    query_text: str = Field(min_length=1, max_length=1000)
    asker_id: str
    addressee_id: str
    subject_entity_ids: list[str] = Field(default_factory=list)
    predicate_hints: list[str] = Field(default_factory=list)
    explicit_fact_ids: list[str] = Field(default_factory=list)
    atoms: list[KnowledgeQueryAtom] = Field(default_factory=list, max_length=6)
    max_results: int = Field(default=3, ge=1, le=8)


class PlannedOpenAction(RuntimeModel):
    """Planner-owned intent fields. It deliberately has no effects or prose result."""

    label: str = Field(min_length=1, max_length=120)
    action_type: ActionType = ActionType.OTHER
    goal: str = Field(min_length=1, max_length=1000)
    target_name: str | None = Field(default=None, max_length=120)
    target_entity_id: str | None = Field(default=None, max_length=120)
    destination_name: str | None = Field(default=None, max_length=120)
    destination_entity_id: str | None = Field(default=None, max_length=120)
    destination_description: str | None = Field(default=None, max_length=600)
    duration_minutes: int = Field(default=5, ge=0, le=1440)
    resolution: Literal["automatic", "check", "impossible"] = "automatic"
    skill: str | None = Field(default=None, max_length=80)
    difficulty: Literal["regular", "hard", "extreme"] = "regular"
    risk: Literal["safe", "uncertain", "dangerous"] = "safe"
    rest_until_hour: int | None = Field(default=None, ge=0, le=23)
    rest_day_offset: int = Field(default=0, ge=0, le=7)
    speech_act: Literal[
        "none", "question", "statement", "request", "smalltalk", "deception", "threat"
    ] = "none"
    addressee_id: str | None = None
    referents: list[ResolvedReferent] = Field(default_factory=list)
    knowledge_query: KnowledgeQuery | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_authoritative_prose(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            value.pop("success_text", None)
            value.pop("failure_text", None)
            value.pop("success_effects", None)
            value.pop("failure_effects", None)
        return value


class TurnPlannerDecision(RuntimeModel):
    existing_action_id: str | None = None
    open_plan: PlannedOpenAction | None = None
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def exactly_one_choice(self) -> "TurnPlannerDecision":
        if (self.existing_action_id is None) == (self.open_plan is None):
            raise ValueError("planner must select one existing action or one open plan")
        return self


class AssembledTurnContext(RuntimeModel):
    world_time: str
    scene: dict[str, Any]
    present_entities: list[dict[str, Any]] = Field(default_factory=list)
    player_known_facts: list[dict[str, Any]] = Field(default_factory=list)
    recent_visible_history: list[dict[str, Any]] = Field(default_factory=list)
    inventory: list[dict[str, Any]] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    available_actions: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceCandidate(RuntimeModel):
    fact_id: str
    source_entity_id: str
    subject: str
    predicate: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    concealed: bool = False
    score: float = Field(default=0, ge=0)
    knowledge_source: str
    matched_atom_ids: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    component_scores: dict[str, float] = Field(default_factory=dict)


class DisclosureDecision(RuntimeModel):
    mode: Literal["automatic", "check", "refuse", "unknown"]
    reason: str
    source_entity_id: str | None = None
    approved_evidence: list[EvidenceCandidate] = Field(default_factory=list)
    skill: str | None = None
    difficulty: Literal["regular", "hard", "extreme"] = "regular"
    canonical_success: str
    canonical_failure: str
    answered_atom_ids: list[str] = Field(default_factory=list)
    unanswered_atom_ids: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)


class ValidatedActionPlan(RuntimeModel):
    envelope: PlayerIntentEnvelope
    existing_action_id: str | None = None
    open_plan: OpenActionPlan | None = None
    planner_output: dict[str, Any] | None = None
    knowledge_query: KnowledgeQuery | None = None
    evidence: list[EvidenceCandidate] = Field(default_factory=list)
    disclosure: DisclosureDecision | None = None

    @model_validator(mode="after")
    def exactly_one_action(self) -> "ValidatedActionPlan":
        if (self.existing_action_id is None) == (self.open_plan is None):
            raise ValueError("validated plan must contain one action")
        return self


class DisclosedFact(RuntimeModel):
    fact_id: str
    source_entity_id: str | None = None
    subject: str
    predicate: str
    value: Any
    newly_learned: bool = True


class OutcomeEnvelope(RuntimeModel):
    turn_id: str
    action_id: str
    action_label: str
    action_type: ActionType
    player_text: str | None = None
    accepted: bool
    mechanical_result: dict[str, Any] | None = None
    canonical_seed: str
    canonical_beats: list[str] = Field(default_factory=list)
    disclosed_facts: list[DisclosedFact] = Field(default_factory=list)
    visible_events: list[str] = Field(default_factory=list)
    scene: dict[str, Any] = Field(default_factory=dict)
    present_entities: list[dict[str, Any]] = Field(default_factory=list)
    director_opportunity: dict[str, Any] | None = None
    knowledge_query_text: str | None = None
    answer_coverage: dict[str, Any] = Field(default_factory=dict)


class GroundingReport(RuntimeModel):
    accepted: bool
    rejected_beats: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    approved_fact_ids: list[str] = Field(default_factory=list)


class TurnTrace(RuntimeModel):
    id: str
    state_version_before: int = Field(ge=1)
    status: Literal["planned", "pending_rule_choice", "resolved", "rejected"] = "planned"
    input: dict[str, Any]
    context_summary: dict[str, Any] = Field(default_factory=dict)
    planner_output: dict[str, Any] | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    knowledge_query: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    disclosure: dict[str, Any] | None = None
    action_id: str | None = None
    kernel_events: list[dict[str, Any]] = Field(default_factory=list)
    state_diff: dict[str, Any] = Field(default_factory=dict)
    outcome: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
