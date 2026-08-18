from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


class KernelModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntityKind(StrEnum):
    ACTOR = "actor"
    OBJECT = "object"


class PlacementKind(StrEnum):
    AT_LOCATION = "at_location"
    HELD_BY = "held_by"
    INSIDE_CONTAINER = "inside_container"
    EQUIPPED_BY = "equipped_by"


class CommandKind(StrEnum):
    MOVE = "move"
    WAIT = "wait"
    INSPECT = "inspect"
    INTERACT = "interact"
    SET_STATE = "set_state"


class Placement(KernelModel):
    kind: PlacementKind
    target_id: str


class LocationDefinition(KernelModel):
    id: str
    name: str
    parent_id: str | None = None
    x: float = 0
    y: float = 0
    initially_known: bool = False
    description: str = ""
    tags: set[str] = Field(default_factory=set)


class ConnectionDefinition(KernelModel):
    id: str
    name: str
    from_location: str
    to_location: str
    bidirectional: bool = True
    travel_minutes: int = Field(default=1, ge=0, le=1440)
    initial_state: dict[str, Any] = Field(
        default_factory=lambda: {"open": True, "locked": False, "discovered": True}
    )


class EntityDefinition(KernelModel):
    id: str
    kind: EntityKind
    name: str
    initial_placement: Placement
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    tags: set[str] = Field(default_factory=set)
    asset_id: str | None = None


class PropositionDefinition(KernelModel):
    id: str
    subject: str
    predicate: str
    value: Any
    canon: Literal["hard_canon", "soft_canon"] = "hard_canon"
    source: str = "definition"
    discoverable_via: list[str] = Field(default_factory=list)


class InitialKnowledge(KernelModel):
    observer_id: str
    proposition_id: str
    stance: Literal["known", "believed", "disbelieved"] = "known"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "background"


class StateVariableDefinition(KernelModel):
    id: str
    value_type: Literal["bool", "int", "float", "str"]
    initial_value: bool | int | float | str
    mutable_by: set[Literal["system", "script"]] = Field(
        default_factory=lambda: {"system", "script"}
    )

    @model_validator(mode="after")
    def value_matches_type(self) -> "StateVariableDefinition":
        expected = {"bool": bool, "int": int, "float": (int, float), "str": str}[self.value_type]
        if self.value_type == "int" and isinstance(self.initial_value, bool):
            raise ValueError(f"state variable {self.id} must be an int")
        if not isinstance(self.initial_value, expected):
            raise ValueError(f"state variable {self.id} has the wrong initial value type")
        return self


class TriggerEventDefinition(KernelModel):
    type: str
    actor_id: str | None = None
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    visibility: Literal["public", "participants", "dev"] = "public"


class ScheduledTriggerDefinition(KernelModel):
    id: str
    scheduled_time: datetime
    priority: int = 100
    interrupt: bool = False
    events: list[TriggerEventDefinition]


class WorldDefinition(KernelModel):
    schema_version: int = 1
    id: str
    version: str
    title: str
    start_time: datetime
    locations: list[LocationDefinition]
    connections: list[ConnectionDefinition]
    entities: list[EntityDefinition]
    propositions: list[PropositionDefinition] = Field(default_factory=list)
    initial_knowledge: list[InitialKnowledge] = Field(default_factory=list)
    state_variables: list[StateVariableDefinition] = Field(default_factory=list)
    scheduled_triggers: list[ScheduledTriggerDefinition] = Field(default_factory=list)

    @property
    def content_digest(self) -> str:
        payload = _canonical_value(self)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def referential_integrity(self) -> "WorldDefinition":
        groups = {
            "location": [item.id for item in self.locations],
            "connection": [item.id for item in self.connections],
            "entity": [item.id for item in self.entities],
            "proposition": [item.id for item in self.propositions],
            "state variable": [item.id for item in self.state_variables],
            "scheduled trigger": [item.id for item in self.scheduled_triggers],
        }
        errors: list[str] = []
        for label, identifiers in groups.items():
            duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
            if duplicates:
                errors.append(f"duplicate {label} ids: {duplicates}")

        location_ids = set(groups["location"])
        entity_ids = set(groups["entity"])
        proposition_ids = set(groups["proposition"])
        for location in self.locations:
            if location.parent_id and location.parent_id not in location_ids:
                errors.append(f"location {location.id} has unknown parent {location.parent_id}")
            if location.parent_id == location.id:
                errors.append(f"location {location.id} cannot contain itself")
        for connection in self.connections:
            if connection.from_location not in location_ids or connection.to_location not in location_ids:
                errors.append(f"connection {connection.id} has an unknown endpoint")
        for entity in self.entities:
            placement = entity.initial_placement
            valid_targets = location_ids if placement.kind == PlacementKind.AT_LOCATION else entity_ids
            if placement.target_id not in valid_targets:
                errors.append(f"entity {entity.id} has invalid placement target {placement.target_id}")
            if placement.target_id == entity.id:
                errors.append(f"entity {entity.id} cannot contain or hold itself")
        for record in self.initial_knowledge:
            if record.observer_id not in entity_ids:
                errors.append(f"knowledge has unknown observer {record.observer_id}")
            if record.proposition_id not in proposition_ids:
                errors.append(f"knowledge has unknown proposition {record.proposition_id}")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class EntityRuntime(KernelModel):
    entity_id: str
    placement: Placement
    active: bool = True
    attributes: dict[str, Any] = Field(default_factory=dict)


class ConnectionRuntime(KernelModel):
    connection_id: str
    state: dict[str, Any] = Field(default_factory=dict)


class EpistemicRecord(KernelModel):
    id: str
    observer_id: str
    subject: str
    predicate: str
    claimed_value: Any
    stance: Literal["known", "believed", "disbelieved"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    learned_at: datetime
    truth_proposition_id: str | None = None


class ScheduledTrigger(KernelModel):
    id: str
    scheduled_time: datetime
    priority: int
    interrupt: bool
    events: list[TriggerEventDefinition]


class DomainEvent(KernelModel):
    schema_version: int = 1
    event_id: str
    session_id: str
    sequence: int = Field(ge=1)
    transaction_version: int = Field(ge=1)
    world_time: datetime
    type: str
    actor_id: str | None = None
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    command_id: str
    correlation_id: str
    causation_id: str | None = None
    world_definition_version: str
    visibility: Literal["public", "participants", "dev"] = "public"


class WorldRuntime(KernelModel):
    schema_version: int = 1
    session_id: str
    world_definition_id: str
    world_definition_version: str
    world_definition_digest: str
    version: int = 0
    event_sequence: int = 0
    clock: datetime
    entities: dict[str, EntityRuntime]
    connections: dict[str, ConnectionRuntime]
    state_variables: dict[str, bool | int | float | str] = Field(default_factory=dict)
    knowledge: list[EpistemicRecord] = Field(default_factory=list)
    observed_locations: dict[str, datetime] = Field(default_factory=dict)
    observed_entities: dict[str, datetime] = Field(default_factory=dict)
    scheduled_triggers: list[ScheduledTrigger] = Field(default_factory=list)
    event_log: list[DomainEvent] = Field(default_factory=list)


class CommandEnvelope(KernelModel):
    schema_version: int = 1
    command_id: str
    session_id: str
    issuer_id: str
    actor_id: str | None
    kind: CommandKind
    payload: dict[str, Any] = Field(default_factory=dict)
    expected_state_version: int = Field(ge=0)
    idempotency_key: str
    correlation_id: str | None = None


class CommandReceipt(KernelModel):
    command_id: str
    idempotency_key: str
    accepted: bool
    outcome: Literal["succeeded", "failed", "interrupted", "rejected", "duplicate"]
    reason: str | None = None
    state_version: int
    event_ids: list[str] = Field(default_factory=list)


class EventDraft(KernelModel):
    world_time: datetime
    type: str
    actor_id: str | None = None
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    visibility: Literal["public", "participants", "dev"] = "public"
    causation_id: str | None = None
