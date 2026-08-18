from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EntityType(StrEnum):
    PLAYER = "PLAYER"
    NPC = "NPC"
    CREATURE = "CREATURE"
    ITEM = "ITEM"
    LOCATION = "LOCATION"
    FACTION = "FACTION"
    OBJECT = "OBJECT"


class ActionType(StrEnum):
    MOVE = "MOVE"
    SEARCH = "SEARCH"
    EXAMINE = "EXAMINE"
    TALK = "TALK"
    DECEIVE = "DECEIVE"
    TAKE = "TAKE"
    USE = "USE"
    FORCE = "FORCE"
    WAIT = "WAIT"
    REST = "REST"
    RESCUE = "RESCUE"
    DISRUPT = "DISRUPT"
    CONFRONT = "CONFRONT"
    ESCAPE = "ESCAPE"
    OTHER = "OTHER"


class CheckOutcome(StrEnum):
    CRITICAL = "CRITICAL"
    EXTREME = "EXTREME"
    HARD = "HARD"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    FUMBLE = "FUMBLE"
    AUTOMATIC = "AUTOMATIC"
    INTERRUPTED = "INTERRUPTED"


class RuleChoice(StrEnum):
    ACCEPT_FAILURE = "accept_failure"
    SPEND_LUCK = "spend_luck"
    PUSH_ROLL = "push_roll"


class PacingPhase(StrEnum):
    EXPLORE = "EXPLORE"
    BUILD = "BUILD"
    PRESSURE = "PRESSURE"
    PEAK = "PEAK"
    RELIEF = "RELIEF"


class SessionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    WON = "WON"
    LOST = "LOST"
    ESCAPED = "ESCAPED"


class Entity(DomainModel):
    id: str
    type: EntityType
    name: str
    location: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    tags: set[str] = Field(default_factory=set)
    active: bool = True


class Fact(DomainModel):
    id: str
    subject: str
    predicate: str
    value: Any
    visibility: Literal["PUBLIC", "PLAYER", "HIDDEN"] = "HIDDEN"
    created_at: datetime
    source: str
    immutable: bool = False
    canon: Literal["hard_canon", "soft_canon"] = "soft_canon"


class Relationship(DomainModel):
    subject: str
    relation: str
    object: str
    value: float = Field(ge=-1.0, le=1.0)


class KnowledgeEntry(DomainModel):
    knower_id: str
    fact_id: str
    belief_value: Any | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str
    learned_at: datetime
    concealed: bool = False


class MemoryEntry(DomainModel):
    id: str
    npc_id: str
    text: str
    occurred_at: datetime
    source_event_id: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class Condition(DomainModel):
    subject: str
    predicate: str
    operator: Literal[
        "eq",
        "ne",
        "gte",
        "lte",
        "contains",
        "not_contains",
        "known",
        "not_known",
    ] = "eq"
    value: Any = None


class Effect(DomainModel):
    op: Literal[
        "reveal_fact",
        "set_fact",
        "move_entity",
        "modify_player",
        "add_inventory",
        "remove_inventory",
        "add_npc_knowledge",
        "add_memory",
        "schedule_event",
        "cancel_event",
        "advance_threat",
        "set_entity_active",
        "set_status",
        "mark_flag",
        "sanity_check",
        "damage_player",
        "create_entity",
    ]
    params: dict[str, Any] = Field(default_factory=dict)
    conditions: list[Condition] = Field(default_factory=list)


class ScheduledEvent(DomainModel):
    id: str
    time: datetime
    type: str
    actor: str | None = None
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    conditions: list[Condition] = Field(default_factory=list)
    priority: int = 100
    cancelable: bool = True
    interrupt_action: bool = False
    interrupt_locations: list[str] = Field(default_factory=list)
    canceled: bool = False


class EventRecord(DomainModel):
    id: str
    seq: int
    time: datetime
    type: str
    actor: str | None = None
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    visible_to_player: bool = False


class ThreatThreshold(DomainModel):
    at: int = Field(ge=0, le=100)
    label: str
    effects: list[Effect] = Field(default_factory=list)
    narrative: str = ""


class ThreatClock(DomainModel):
    id: str
    name: str
    progress: int = Field(default=0, ge=0, le=100)
    thresholds: list[ThreatThreshold] = Field(default_factory=list)
    crossed: set[int] = Field(default_factory=set)


class ClueDefinition(DomainModel):
    id: str
    fact_id: str
    title: str
    description: str
    critical: bool = False
    sources: list[str] = Field(default_factory=list)


class ActionDefinition(DomainModel):
    id: str
    label: str
    type: ActionType
    location: str | None = None
    target: str | None = None
    duration_minutes: int = Field(default=1, ge=0, le=2880)
    skill: str | None = None
    difficulty: Literal["regular", "hard", "extreme"] = "regular"
    bonus_dice: int = Field(default=0, ge=-2, le=2)
    modifiers: list["RollModifierDefinition"] = Field(default_factory=list)
    opposed: "OpposedCheckDefinition | None" = None
    sanity_check: "SanityCheckDefinition | None" = None
    pushable: bool = True
    push_failure_effects: list[Effect] = Field(default_factory=list)
    push_failure_text: str | None = None
    aliases: list[str] = Field(default_factory=list)
    dialogue_text: str | None = Field(default=None, max_length=500)
    requirements: list[Condition] = Field(default_factory=list)
    forbidden: list[Condition] = Field(default_factory=list)
    success_effects: list[Effect] = Field(default_factory=list)
    failure_effects: list[Effect] = Field(default_factory=list)
    always_effects: list[Effect] = Field(default_factory=list)
    success_text: str
    failure_text: str = "这次尝试没有产生预期的结果。"
    success_beats: list[str] = Field(default_factory=list)
    failure_beats: list[str] = Field(default_factory=list)
    interrupted_text: str = "突发事件打断了你的行动。"
    once: bool = False
    complete_on_attempt: bool = False
    suggest: bool = True
    risk: Literal["safe", "uncertain", "dangerous"] = "safe"
    category: Literal["investigate", "social", "risk", "move", "other"] = "other"


class ComplicationDefinition(DomainModel):
    id: str
    label: str
    requirements: list[Condition] = Field(default_factory=list)
    effects: list[Effect]
    world_justification: str
    player_visible_text: str | None = None
    affected_entities: list[str] = Field(default_factory=list)
    once: bool = False


class RespiteDefinition(DomainModel):
    id: str
    label: str
    requirements: list[Condition] = Field(default_factory=list)
    effects: list[Effect]
    world_justification: str
    player_visible_text: str | None = None
    affected_entities: list[str] = Field(default_factory=list)


class DirectorHintDefinition(DomainModel):
    id: str
    label: str
    requirements: list[Condition] = Field(default_factory=list)
    effects: list[Effect]
    world_justification: str
    player_visible_text: str | None = None
    affected_entities: list[str] = Field(default_factory=list)


class DirectorConfig(DomainModel):
    primary_threat_id: str | None = None
    progress_flags: list[str] = Field(default_factory=list)
    hint_opportunities: list[DirectorHintDefinition] = Field(default_factory=list)


class ScenarioSource(DomainModel):
    title: str
    publisher: str
    authors: list[str] = Field(default_factory=list)
    url: str
    rights_note: str
    adaptation_note: str


class ScenarioPresentation(DomainModel):
    case_label: str = "LIVING TABLETOP"
    headline: str = "今晚，世界不会等你。"
    description: str = "一场由状态、时间和选择共同推进的调查。"
    start_button: str = "开始调查"
    session_note: str = "CoC 7版基础规则 · 单人"
    free_action_placeholder: str = "例如：仔细检查眼前的房间……"


class EndingDefinition(DomainModel):
    id: str
    title: str
    status: SessionStatus
    requirements: list[Condition]
    narrative: str
    priority: int = 100


class RollModifierDefinition(DomainModel):
    dice: int = Field(ge=-2, le=2)
    label: str
    conditions: list[Condition] = Field(default_factory=list)


class OpposedCheckDefinition(DomainModel):
    entity_id: str | None = None
    skill: str
    value: int | None = Field(default=None, ge=0, le=100)
    label: str = "对手"
    response: Literal["fight_back", "dodge", "oppose"] = "oppose"
    bonus_dice: int = Field(default=0, ge=-2, le=2)
    tie_favors: Literal["initiator", "opponent"] = "initiator"
    both_fail_no_winner: bool = True


class SanityCheckDefinition(DomainModel):
    success_loss: str = "0"
    failure_loss: str
    reason: str
    on: Literal["always", "success", "failure"] = "always"
    once: bool = True


class PlayerState(DomainModel):
    entity_id: str = "player"
    name: str = "调查员"
    hp: int = 10
    max_hp: int = 10
    stress: int = 0
    max_stress: int = 10
    characteristics: dict[str, int] = Field(
        default_factory=lambda: {
            "str": 50,
            "con": 50,
            "siz": 50,
            "dex": 50,
            "app": 50,
            "int": 60,
            "pow": 60,
            "edu": 60,
        }
    )
    sanity: int = 60
    max_sanity: int = 99
    starting_sanity: int = 60
    daily_sanity_loss: int = 0
    luck: int = 50
    max_luck: int = 99
    magic_points: int = 12
    move_rate: int = 8
    damage_bonus: str = "0"
    build: int = 0
    major_wound: bool = False
    unconscious: bool = False
    dying: bool = False
    temporary_insanity_until: datetime | None = None
    indefinite_insanity: bool = False
    bout_of_madness: str | None = None
    checked_skills: set[str] = Field(default_factory=set)
    skills: dict[str, int] = Field(default_factory=dict)
    inventory: list[str] = Field(default_factory=list)

    @field_validator("hp", "stress", "sanity", "luck", "magic_points", "daily_sanity_loss")
    @classmethod
    def non_negative(cls, value: int) -> int:
        return max(0, value)


class ExperienceState(DomainModel):
    tension: int = Field(default=15, ge=0, le=100)
    danger: int = Field(default=5, ge=0, le=100)
    progress: int = Field(default=0, ge=0, le=100)
    mystery: int = Field(default=90, ge=0, le=100)
    resource_pressure: int = Field(default=0, ge=0, le=100)
    success_streak: int = Field(default=0, ge=0)
    failure_streak: int = Field(default=0, ge=0)
    agency: int = Field(default=80, ge=0, le=100)
    frustration: int = Field(default=0, ge=0, le=100)
    relief_need: int = Field(default=0, ge=0, le=100)
    novelty: int = Field(default=70, ge=0, le=100)
    time_pressure: int = Field(default=10, ge=0, le=100)


class DirectorIntervention(DomainModel):
    id: str
    action: Literal[
        "advance_threat",
        "surface_clue",
        "offer_respite",
        "increase_pressure",
        "guide_affordances",
    ]
    reason: str
    world_justification: str
    source_definition_id: str | None = None
    player_visible_text: str | None = None
    affected_entities: list[str] = Field(default_factory=list)
    expected_experience_effect: str
    effects: list[Effect] = Field(default_factory=list)
    applied_at: datetime
    valid: bool = True


class DirectorState(DomainModel):
    experience: ExperienceState = Field(default_factory=ExperienceState)
    phase: PacingPhase = PacingPhase.EXPLORE
    meaningful_actions: int = 0
    actions_since_evaluation: int = 0
    actions_without_progress: int = 0
    scene_history: list[str] = Field(default_factory=list)
    recent_action_types: list[ActionType] = Field(default_factory=list)
    interventions: list[DirectorIntervention] = Field(default_factory=list)
    affordance_bias: str | None = None
    off_main_streak: int = Field(default=0, ge=0)
    last_open_goal: str | None = None


class RollRecord(DomainModel):
    id: str
    dice: str = "d100"
    result: int = Field(ge=1, le=100)
    skill: str
    target: int = Field(ge=0, le=100)
    outcome: CheckOutcome
    reason: str
    timestamp: datetime
    difficulty: Literal["regular", "hard", "extreme"] = "regular"
    candidates: list[int] = Field(default_factory=list)
    bonus_dice: int = Field(default=0, ge=-2, le=2)
    pushed: bool = False
    luck_spent: int = 0


class OpposedRollResult(DomainModel):
    entity_id: str | None = None
    label: str
    skill: str
    value: int = Field(ge=0, le=100)
    roll: int = Field(ge=1, le=100)
    outcome: CheckOutcome
    candidates: list[int] = Field(default_factory=list)
    bonus_dice: int = Field(default=0, ge=-2, le=2)


class SanityCheckResult(DomainModel):
    id: str
    reason: str
    roll: int = Field(ge=1, le=100)
    target: int = Field(ge=0, le=99)
    succeeded: bool
    loss: int = Field(ge=0)
    sanity_before: int = Field(ge=0)
    sanity_after: int = Field(ge=0)
    temporary_insanity_triggered: bool = False
    indefinite_insanity_triggered: bool = False
    bout: str | None = None
    timestamp: datetime


class DamageResult(DomainModel):
    id: str
    source: str
    amount: int = Field(ge=0)
    hp_before: int = Field(ge=0)
    hp_after: int = Field(ge=0)
    major_wound_triggered: bool = False
    con_roll: int | None = Field(default=None, ge=1, le=100)
    unconscious: bool = False
    dying: bool = False
    instant_death: bool = False
    timestamp: datetime


class PendingCheck(DomainModel):
    action_id: str
    check: "CheckResult"
    luck_cost: int | None = Field(default=None, ge=1)
    can_spend_luck: bool = False
    can_push: bool = False
    progress_before: int = 0
    location_before: str | None = None
    dynamic_action: "ActionDefinition | None" = None
    turn_trace_id: str | None = None


class AgentCallRecord(DomainModel):
    id: str
    role: Literal["action_interpreter", "keeper", "turn_planner", "director", "narrator"]
    input_state_version: int
    output_digest: str
    structured_output: dict[str, Any] | None = None
    validation: Literal["accepted", "rejected", "fallback"]
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    result: Literal["success", "error"] = "success"
    created_at: datetime


class NarrativeBeat(DomainModel):
    id: str
    text: str = Field(min_length=1, max_length=1600)
    source: Literal["authored", "event", "scene", "director", "keeper", "generated", "system"] = "authored"
    skippable: bool = True


class NarrativeSequence(DomainModel):
    id: str
    state_version: int = Field(ge=1)
    action_id: str | None = None
    action_label: str | None = Field(default=None, max_length=160)
    action_type: ActionType | None = None
    player_text: str | None = Field(default=None, max_length=1000)
    continues_previous: bool = False
    status: Literal["pending", "ready", "fallback"] = "ready"
    beats: list[NarrativeBeat] = Field(default_factory=list)
    canonical_seed: str = Field(default="", max_length=4000)
    mechanical_result: dict[str, Any] | None = None
    outcome_envelope: dict[str, Any] | None = None
    grounding_report: dict[str, Any] | None = None
    turn_trace_id: str | None = None
    created_at: datetime


class PlayerVisibleMemory(DomainModel):
    """A compact record of what the player has actually seen or heard."""

    id: str
    state_version: int = Field(ge=1)
    world_time: datetime
    location_id: str | None = None
    sequence_id: str | None = None
    kind: Literal["hard_canon", "soft_canon", "dialogue_claim"]
    source: Literal["authored", "event", "scene", "director", "keeper", "generated", "system"]
    action_type: ActionType | None = None
    text: str = Field(min_length=1, max_length=1600)


class WorldState(DomainModel):
    session_id: str
    scenario_id: str
    world_time: datetime
    version: int = 1
    status: SessionStatus = SessionStatus.ACTIVE
    ending_id: str | None = None
    entities: dict[str, Entity]
    facts: dict[str, Fact]
    relationships: list[Relationship] = Field(default_factory=list)
    player: PlayerState
    player_known_fact_ids: set[str] = Field(default_factory=set)
    npc_knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    npc_memories: list[MemoryEntry] = Field(default_factory=list)
    event_queue: list[ScheduledEvent] = Field(default_factory=list)
    event_log: list[EventRecord] = Field(default_factory=list)
    threats: dict[str, ThreatClock] = Field(default_factory=dict)
    discovered_clue_ids: set[str] = Field(default_factory=set)
    completed_actions: set[str] = Field(default_factory=set)
    flags: dict[str, Any] = Field(default_factory=dict)
    rolls: list[RollRecord] = Field(default_factory=list)
    sanity_checks: list[SanityCheckResult] = Field(default_factory=list)
    damage_log: list[DamageResult] = Field(default_factory=list)
    pending_check: PendingCheck | None = None
    director: DirectorState = Field(default_factory=DirectorState)
    rng_seed: int = 1927
    rng_draws: int = 0
    rng_sides: list[int] = Field(default_factory=list)
    last_narrative: str = ""
    narrative_sequence: NarrativeSequence | None = None
    visible_history: list[PlayerVisibleMemory] = Field(default_factory=list)
    agent_calls: list[AgentCallRecord] = Field(default_factory=list)
    turn_traces: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioDefinition(DomainModel):
    id: str
    title: str
    subtitle: str
    start_time: datetime
    opening_narrative: str
    start_location: str
    entities: list[Entity]
    facts: list[Fact]
    relationships: list[Relationship] = Field(default_factory=list)
    player: PlayerState
    initial_player_known_fact_ids: list[str] = Field(default_factory=list)
    npc_knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    events: list[ScheduledEvent] = Field(default_factory=list)
    threats: list[ThreatClock] = Field(default_factory=list)
    clues: list[ClueDefinition]
    actions: list[ActionDefinition]
    complications: list[ComplicationDefinition] = Field(default_factory=list)
    respites: list[RespiteDefinition] = Field(default_factory=list)
    endings: list[EndingDefinition]
    location_graph: dict[str, dict[str, int]]
    minimum_evidence: int = 5
    source: ScenarioSource | None = None
    presentation: ScenarioPresentation = Field(default_factory=ScenarioPresentation)
    initial_flags: dict[str, Any] = Field(default_factory=dict)
    movement_requirements: dict[str, list[Condition]] = Field(default_factory=dict)
    director_config: DirectorConfig = Field(default_factory=DirectorConfig)


class ActionIntent(DomainModel):
    action_id: str | None = None
    action_type: ActionType | None = None
    target: str | None = None
    content: str | None = None
    goal: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification: str | None = None
    turn_trace_id: str | None = None
    source: Literal["button", "player_text", "llm", "open", "clarification"] = "player_text"


class OpenActionPlan(DomainModel):
    label: str = Field(min_length=1, max_length=120)
    action_type: ActionType = ActionType.OTHER
    goal: str = Field(min_length=1, max_length=500)
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
    success_text: str = Field(min_length=1, max_length=1200)
    failure_text: str = Field(default="这次尝试没有产生预期的结果。", min_length=1, max_length=1200)
    approved_fact_ids: list[str] = Field(default_factory=list)
    knowledge_source_id: str | None = None
    disclosure_mode: Literal["automatic", "check", "refuse", "unknown"] | None = None
    knowledge_query_text: str | None = Field(default=None, max_length=1000)
    answered_query_parts: list[str] = Field(default_factory=list)
    unanswered_query_parts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_resolution(self) -> "OpenActionPlan":
        if self.resolution != "check":
            object.__setattr__(self, "skill", None)
        return self


class KeeperDecision(DomainModel):
    existing_action_id: str | None = None
    open_plan: OpenActionPlan | None = None
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def exactly_one_choice(self) -> "KeeperDecision":
        if (self.existing_action_id is None) == (self.open_plan is None):
            raise ValueError("Keeper decision must select one existing action or one open plan")
        return self


class CheckResult(DomainModel):
    required: bool
    skill: str | None = None
    target: int | None = None
    roll: int | None = None
    outcome: CheckOutcome
    succeeded: bool
    difficulty: Literal["regular", "hard", "extreme"] = "regular"
    base_value: int | None = None
    candidates: list[int] = Field(default_factory=list)
    bonus_dice: int = Field(default=0, ge=-2, le=2)
    modifier_labels: list[str] = Field(default_factory=list)
    pushed: bool = False
    luck_spent: int = 0
    roll_id: str | None = None
    opponent: OpposedRollResult | None = None


class ActionResolution(DomainModel):
    action_id: str | None = None
    accepted: bool
    needs_clarification: bool = False
    clarification: str | None = None
    check: CheckResult | None = None
    sanity_check: SanityCheckResult | None = None
    awaiting_rule_choice: bool = False
    rule_choices: list[RuleChoice] = Field(default_factory=list)
    luck_cost: int | None = None
    interrupted: bool = False
    interrupting_event_id: str | None = None
    narrative_seed: str = ""
    visible_events: list[EventRecord] = Field(default_factory=list)
    director_intervention: DirectorIntervention | None = None
    disclosed_fact_ids: list[str] = Field(default_factory=list)
    knowledge_source_id: str | None = None
    outcome_envelope: dict[str, Any] | None = None
    turn_trace_id: str | None = None
    continues_previous_narrative: bool = False
    state_version: int


class LLMResult(DomainModel):
    data: dict[str, Any]
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
