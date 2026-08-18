from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

from .models import (
    CommandEnvelope,
    CommandKind,
    CommandReceipt,
    ConnectionDefinition,
    ConnectionRuntime,
    DomainEvent,
    EntityKind,
    EntityRuntime,
    EpistemicRecord,
    EventDraft,
    Placement,
    PlacementKind,
    ScheduledTrigger,
    WorldDefinition,
    WorldRuntime,
)


class CommandRejected(ValueError):
    """A malformed, unauthorized, or stale protocol command.

    In-world failure is deliberately represented by accepted domain events and
    never raised as this exception.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Decision:
    drafts: list[EventDraft]
    outcome: str
    reason: str | None = None


class VisualWorldKernel:
    def __init__(self, definition: WorldDefinition):
        self.definition = definition
        self.locations = {item.id: item for item in definition.locations}
        self.connection_definitions = {item.id: item for item in definition.connections}
        self.entity_definitions = {item.id: item for item in definition.entities}
        self.propositions = {item.id: item for item in definition.propositions}
        self.state_variable_definitions = {item.id: item for item in definition.state_variables}

    def initial_state(self, session_id: str) -> WorldRuntime:
        entities = {
            item.id: EntityRuntime(
                entity_id=item.id,
                placement=item.initial_placement.model_copy(deep=True),
                attributes=deepcopy(item.attributes),
            )
            for item in self.definition.entities
        }
        connections = {
            item.id: ConnectionRuntime(
                connection_id=item.id,
                state=deepcopy(item.initial_state),
            )
            for item in self.definition.connections
        }
        knowledge: list[EpistemicRecord] = []
        for index, item in enumerate(self.definition.initial_knowledge, start=1):
            proposition = self.propositions[item.proposition_id]
            knowledge.append(
                EpistemicRecord(
                    id=f"initial-knowledge:{index:04d}",
                    observer_id=item.observer_id,
                    subject=proposition.subject,
                    predicate=proposition.predicate,
                    claimed_value=deepcopy(proposition.value),
                    stance=item.stance,
                    confidence=item.confidence,
                    source=item.source,
                    learned_at=self.definition.start_time,
                    truth_proposition_id=proposition.id,
                )
            )

        observed_locations: dict[str, datetime] = {}
        observed_entities: dict[str, datetime] = {}
        actors = [item for item in self.definition.entities if item.kind == EntityKind.ACTOR]
        for actor in actors:
            for location in self.definition.locations:
                if location.initially_known:
                    observed_locations[self._observation_key(actor.id, location.id)] = self.definition.start_time
            actor_location = self._root_location_from_entities(entities, actor.id)
            if actor_location:
                observed_locations[self._observation_key(actor.id, actor_location)] = self.definition.start_time
            observed_entities[self._observation_key(actor.id, actor.id)] = self.definition.start_time

        state = WorldRuntime(
            session_id=session_id,
            world_definition_id=self.definition.id,
            world_definition_version=self.definition.version,
            world_definition_digest=self.definition.content_digest,
            clock=self.definition.start_time,
            entities=entities,
            connections=connections,
            state_variables={item.id: item.initial_value for item in self.definition.state_variables},
            knowledge=knowledge,
            observed_locations=observed_locations,
            observed_entities=observed_entities,
            scheduled_triggers=[
                ScheduledTrigger(
                    id=item.id,
                    scheduled_time=item.scheduled_time,
                    priority=item.priority,
                    interrupt=item.interrupt,
                    events=[event.model_copy(deep=True) for event in item.events],
                )
                for item in self.definition.scheduled_triggers
            ],
        )
        self.assert_invariants(state)
        return state

    @staticmethod
    def _observation_key(observer_id: str, object_id: str) -> str:
        return f"{observer_id}|{object_id}"

    @staticmethod
    def _root_location_from_entities(
        entities: dict[str, EntityRuntime], entity_id: str
    ) -> str | None:
        seen: set[str] = set()
        current_id = entity_id
        while current_id in entities:
            if current_id in seen:
                return None
            seen.add(current_id)
            placement = entities[current_id].placement
            if placement.kind == PlacementKind.AT_LOCATION:
                return placement.target_id
            current_id = placement.target_id
        return None

    def root_location(self, state: WorldRuntime, entity_id: str) -> str | None:
        return self._root_location_from_entities(state.entities, entity_id)

    def _actor(self, state: WorldRuntime, command: CommandEnvelope) -> EntityRuntime:
        if not command.actor_id or command.actor_id not in state.entities:
            raise CommandRejected("unknown_actor", "Command must reference an existing actor")
        definition = self.entity_definitions[command.actor_id]
        if definition.kind != EntityKind.ACTOR:
            raise CommandRejected("not_an_actor", "The command actor is not an actor")
        if command.issuer_id not in {command.actor_id, "system", "script"}:
            raise CommandRejected("unauthorized", "An issuer cannot control this actor")
        return state.entities[command.actor_id]

    def _draft(
        self,
        state: WorldRuntime,
        event_type: str,
        *,
        when: datetime | None = None,
        actor_id: str | None = None,
        target_id: str | None = None,
        payload: dict[str, Any] | None = None,
        visibility: str = "public",
        causation_id: str | None = None,
    ) -> EventDraft:
        return EventDraft(
            world_time=when or state.clock,
            type=event_type,
            actor_id=actor_id,
            target_id=target_id,
            payload=payload or {},
            visibility=visibility,
            causation_id=causation_id,
        )

    def _connection_between(
        self, state: WorldRuntime, origin: str, destination: str
    ) -> tuple[ConnectionDefinition, ConnectionRuntime] | None:
        for definition in self.definition.connections:
            forward = definition.from_location == origin and definition.to_location == destination
            reverse = (
                definition.bidirectional
                and definition.to_location == origin
                and definition.from_location == destination
            )
            if forward or reverse:
                return definition, state.connections[definition.id]
        return None

    def _advance_drafts(
        self,
        state: WorldRuntime,
        *,
        start: datetime,
        target: datetime,
        actor_id: str | None,
    ) -> tuple[list[EventDraft], datetime, str | None]:
        drafts: list[EventDraft] = []
        cursor = start
        due = sorted(
            (
                trigger
                for trigger in state.scheduled_triggers
                if cursor <= trigger.scheduled_time <= target
            ),
            key=lambda trigger: (trigger.scheduled_time, trigger.priority, trigger.id),
        )
        for trigger in due:
            if trigger.scheduled_time > cursor:
                drafts.append(
                    self._draft(
                        state,
                        "TimeAdvanced",
                        when=trigger.scheduled_time,
                        actor_id=actor_id,
                        payload={"from": cursor.isoformat(), "to": trigger.scheduled_time.isoformat()},
                    )
                )
                cursor = trigger.scheduled_time
            drafts.append(
                self._draft(
                    state,
                    "ScheduledTriggerConsumed",
                    when=cursor,
                    target_id=trigger.id,
                    payload={"trigger_id": trigger.id},
                    visibility="dev",
                )
            )
            for event in trigger.events:
                drafts.append(
                    self._draft(
                        state,
                        event.type,
                        when=cursor,
                        actor_id=event.actor_id,
                        target_id=event.target_id,
                        payload=deepcopy(event.payload),
                        visibility=event.visibility,
                        causation_id=trigger.id,
                    )
                )
            if trigger.interrupt:
                drafts.append(
                    self._draft(
                        state,
                        "ActionInterrupted",
                        when=cursor,
                        actor_id=actor_id,
                        target_id=trigger.id,
                        payload={"trigger_id": trigger.id},
                        visibility="participants",
                    )
                )
                return drafts, cursor, trigger.id
        if target > cursor:
            drafts.append(
                self._draft(
                    state,
                    "TimeAdvanced",
                    when=target,
                    actor_id=actor_id,
                    payload={"from": cursor.isoformat(), "to": target.isoformat()},
                )
            )
            cursor = target
        return drafts, cursor, None

    def _move(self, state: WorldRuntime, command: CommandEnvelope) -> Decision:
        self._actor(state, command)
        destination = str(command.payload.get("destination_id", ""))
        if destination not in self.locations:
            raise CommandRejected("unknown_destination", "Move requires an existing destination")
        origin = self.root_location(state, command.actor_id or "")
        if origin is None:
            raise CommandRejected("actor_not_placed", "The actor has no world location")
        attempted = self._draft(
            state,
            "MovementAttempted",
            actor_id=command.actor_id,
            target_id=destination,
            payload={"origin": origin, "destination": destination},
            visibility="participants",
        )
        connection_pair = self._connection_between(state, origin, destination)
        failure_reason: str | None = None
        travel_minutes = 1
        connection_id: str | None = None
        if connection_pair is None:
            failure_reason = "no_connection"
        else:
            connection_definition, connection_state = connection_pair
            connection_id = connection_definition.id
            travel_minutes = connection_definition.travel_minutes
            if not bool(connection_state.state.get("discovered", True)):
                failure_reason = "connection_unknown"
            elif bool(connection_state.state.get("locked", False)):
                failure_reason = "connection_locked"
            elif not bool(connection_state.state.get("open", True)):
                failure_reason = "connection_closed"

        advance, final_time, interrupted_by = self._advance_drafts(
            state,
            start=state.clock,
            target=state.clock + timedelta(minutes=travel_minutes if not failure_reason else 1),
            actor_id=command.actor_id,
        )
        drafts = [attempted, *advance]
        if interrupted_by:
            return Decision(drafts, "interrupted", f"interrupted_by:{interrupted_by}")
        if failure_reason:
            drafts.append(
                self._draft(
                    state,
                    "MovementAttemptFailed",
                    when=final_time,
                    actor_id=command.actor_id,
                    target_id=destination,
                    payload={"origin": origin, "destination": destination, "reason": failure_reason},
                    visibility="participants",
                )
            )
            return Decision(drafts, "failed", failure_reason)
        drafts.extend(
            [
                self._draft(
                    state,
                    "EntityMoved",
                    when=final_time,
                    actor_id=command.actor_id,
                    target_id=destination,
                    payload={
                        "entity_id": command.actor_id,
                        "origin": origin,
                        "destination": destination,
                        "connection_id": connection_id,
                    },
                    visibility="public",
                ),
                self._draft(
                    state,
                    "LocationObserved",
                    when=final_time,
                    actor_id=command.actor_id,
                    target_id=destination,
                    payload={"observer_id": command.actor_id, "location_id": destination},
                    visibility="participants",
                ),
            ]
        )
        return Decision(drafts, "succeeded")

    def _wait(self, state: WorldRuntime, command: CommandEnvelope) -> Decision:
        self._actor(state, command)
        try:
            minutes = int(command.payload.get("minutes", 5))
        except (TypeError, ValueError) as exc:
            raise CommandRejected("invalid_duration", "Wait duration must be an integer") from exc
        if not 1 <= minutes <= 1440:
            raise CommandRejected("invalid_duration", "Wait duration must be between 1 and 1440 minutes")
        drafts = [
            self._draft(
                state,
                "WaitAttempted",
                actor_id=command.actor_id,
                payload={"minutes": minutes},
                visibility="participants",
            )
        ]
        advance, final_time, interrupted_by = self._advance_drafts(
            state,
            start=state.clock,
            target=state.clock + timedelta(minutes=minutes),
            actor_id=command.actor_id,
        )
        drafts.extend(advance)
        if interrupted_by:
            return Decision(drafts, "interrupted", f"interrupted_by:{interrupted_by}")
        drafts.append(
            self._draft(
                state,
                "WaitCompleted",
                when=final_time,
                actor_id=command.actor_id,
                payload={"minutes": minutes},
                visibility="participants",
            )
        )
        return Decision(drafts, "succeeded")

    def _target_is_reachable(self, state: WorldRuntime, actor_id: str, target_id: str) -> bool:
        actor_location = self.root_location(state, actor_id)
        if target_id in state.entities:
            return self.root_location(state, target_id) == actor_location
        if target_id in self.locations:
            return target_id == actor_location
        if target_id in state.connections:
            definition = self.connection_definitions[target_id]
            return actor_location in {definition.from_location, definition.to_location}
        return False

    def _inspection(self, state: WorldRuntime, command: CommandEnvelope) -> Decision:
        self._actor(state, command)
        target_id = str(command.payload.get("target_id", ""))
        if target_id not in {*state.entities, *state.connections, *self.locations}:
            raise CommandRejected("unknown_target", "Inspect requires an existing target")
        attempted = self._draft(
            state,
            "InspectionAttempted",
            actor_id=command.actor_id,
            target_id=target_id,
            visibility="participants",
        )
        advance, final_time, interrupted_by = self._advance_drafts(
            state,
            start=state.clock,
            target=state.clock + timedelta(minutes=5),
            actor_id=command.actor_id,
        )
        drafts = [attempted, *advance]
        if interrupted_by:
            return Decision(drafts, "interrupted", f"interrupted_by:{interrupted_by}")
        if not self._target_is_reachable(state, command.actor_id or "", target_id):
            drafts.append(
                self._draft(
                    state,
                    "InspectionAttemptFailed",
                    when=final_time,
                    actor_id=command.actor_id,
                    target_id=target_id,
                    payload={"reason": "target_not_reachable"},
                    visibility="participants",
                )
            )
            return Decision(drafts, "failed", "target_not_reachable")

        drafts.append(
            self._draft(
                state,
                "EntityInspected",
                when=final_time,
                actor_id=command.actor_id,
                target_id=target_id,
                visibility="participants",
            )
        )
        discovery_key = f"inspect:{target_id}"
        known_truth_ids = {
            item.truth_proposition_id
            for item in state.knowledge
            if item.observer_id == command.actor_id and item.truth_proposition_id
        }
        for proposition in self.definition.propositions:
            if discovery_key not in proposition.discoverable_via or proposition.id in known_truth_ids:
                continue
            drafts.append(
                self._draft(
                    state,
                    "KnowledgeLearned",
                    when=final_time,
                    actor_id=command.actor_id,
                    target_id=proposition.id,
                    payload={
                        "observer_id": command.actor_id,
                        "subject": proposition.subject,
                        "predicate": proposition.predicate,
                        "claimed_value": deepcopy(proposition.value),
                        "stance": "known",
                        "confidence": 1.0,
                        "source": discovery_key,
                        "truth_proposition_id": proposition.id,
                    },
                    visibility="participants",
                )
            )
        if target_id in self.entity_definitions:
            reveal_connection = self.entity_definitions[target_id].attributes.get("reveals_connection_id")
            if reveal_connection in state.connections and not state.connections[reveal_connection].state.get(
                "discovered", True
            ):
                drafts.append(
                    self._draft(
                        state,
                        "ConnectionDiscovered",
                        when=final_time,
                        actor_id=command.actor_id,
                        target_id=str(reveal_connection),
                        payload={"connection_id": reveal_connection, "observer_id": command.actor_id},
                        visibility="participants",
                    )
                )
        return Decision(drafts, "succeeded")

    def _interaction(self, state: WorldRuntime, command: CommandEnvelope) -> Decision:
        self._actor(state, command)
        target_id = str(command.payload.get("target_id", ""))
        verb = str(command.payload.get("verb", "interact")).lower()
        if target_id not in {*state.entities, *state.connections}:
            raise CommandRejected("unknown_target", "Interact requires an existing target")
        attempted = self._draft(
            state,
            "InteractionAttempted",
            actor_id=command.actor_id,
            target_id=target_id,
            payload={"verb": verb},
            visibility="participants",
        )
        advance, final_time, interrupted_by = self._advance_drafts(
            state,
            start=state.clock,
            target=state.clock + timedelta(minutes=1),
            actor_id=command.actor_id,
        )
        drafts = [attempted, *advance]
        if interrupted_by:
            return Decision(drafts, "interrupted", f"interrupted_by:{interrupted_by}")
        if not self._target_is_reachable(state, command.actor_id or "", target_id):
            drafts.append(
                self._draft(
                    state,
                    "InteractionAttemptFailed",
                    when=final_time,
                    actor_id=command.actor_id,
                    target_id=target_id,
                    payload={"verb": verb, "reason": "target_not_reachable"},
                    visibility="participants",
                )
            )
            return Decision(drafts, "failed", "target_not_reachable")

        failure: str | None = None
        if verb == "take" and target_id in state.entities:
            target_definition = self.entity_definitions[target_id]
            if target_definition.kind != EntityKind.OBJECT:
                failure = "target_not_portable"
            else:
                drafts.append(
                    self._draft(
                        state,
                        "EntityPlaced",
                        when=final_time,
                        actor_id=command.actor_id,
                        target_id=target_id,
                        payload={
                            "entity_id": target_id,
                            "placement": {"kind": "held_by", "target_id": command.actor_id},
                        },
                        visibility="public",
                    )
                )
        elif verb == "drop" and target_id in state.entities:
            placement = state.entities[target_id].placement
            if placement.kind != PlacementKind.HELD_BY or placement.target_id != command.actor_id:
                failure = "item_not_held"
            else:
                location = self.root_location(state, command.actor_id or "")
                drafts.append(
                    self._draft(
                        state,
                        "EntityPlaced",
                        when=final_time,
                        actor_id=command.actor_id,
                        target_id=target_id,
                        payload={
                            "entity_id": target_id,
                            "placement": {"kind": "at_location", "target_id": location},
                        },
                        visibility="public",
                    )
                )
        elif verb == "unlock" and target_id in state.connections:
            runtime = state.connections[target_id]
            required_key = runtime.state.get("key_id")
            holds_key = bool(
                required_key in state.entities
                and state.entities[str(required_key)].placement.kind == PlacementKind.HELD_BY
                and state.entities[str(required_key)].placement.target_id == command.actor_id
            )
            if not runtime.state.get("locked", False):
                failure = "already_unlocked"
            elif required_key and not holds_key:
                failure = "missing_key"
            else:
                drafts.append(
                    self._draft(
                        state,
                        "ConnectionStateChanged",
                        when=final_time,
                        actor_id=command.actor_id,
                        target_id=target_id,
                        payload={"connection_id": target_id, "key": "locked", "value": False},
                        visibility="public",
                    )
                )
        elif verb == "open" and target_id in state.connections:
            runtime = state.connections[target_id]
            if runtime.state.get("locked", False):
                failure = "connection_locked"
            elif runtime.state.get("open", True):
                failure = "already_open"
            else:
                drafts.append(
                    self._draft(
                        state,
                        "ConnectionStateChanged",
                        when=final_time,
                        actor_id=command.actor_id,
                        target_id=target_id,
                        payload={"connection_id": target_id, "key": "open", "value": True},
                        visibility="public",
                    )
                )
        if failure:
            drafts.append(
                self._draft(
                    state,
                    "InteractionAttemptFailed",
                    when=final_time,
                    actor_id=command.actor_id,
                    target_id=target_id,
                    payload={"verb": verb, "reason": failure},
                    visibility="participants",
                )
            )
            return Decision(drafts, "failed", failure)
        drafts.append(
            self._draft(
                state,
                "InteractionCompleted",
                when=final_time,
                actor_id=command.actor_id,
                target_id=target_id,
                payload={"verb": verb},
                visibility="participants",
            )
        )
        return Decision(drafts, "succeeded")

    def _set_state(self, state: WorldRuntime, command: CommandEnvelope) -> Decision:
        variable_id = str(command.payload.get("variable_id", ""))
        if variable_id not in self.state_variable_definitions:
            raise CommandRejected("unknown_state_variable", "Unknown typed state variable")
        definition = self.state_variable_definitions[variable_id]
        if command.issuer_id not in definition.mutable_by:
            raise CommandRejected("unauthorized", "Only an authorized system role may set this variable")
        value = command.payload.get("value")
        expected = {"bool": bool, "int": int, "float": (int, float), "str": str}[definition.value_type]
        if definition.value_type == "int" and isinstance(value, bool):
            raise CommandRejected("invalid_state_value", "State variable has the wrong value type")
        if not isinstance(value, expected):
            raise CommandRejected("invalid_state_value", "State variable has the wrong value type")
        draft = self._draft(
            state,
            "StateVariableChanged",
            actor_id=command.actor_id,
            target_id=variable_id,
            payload={
                "variable_id": variable_id,
                "old_value": state.state_variables[variable_id],
                "value": value,
            },
            visibility="dev",
        )
        return Decision([draft], "succeeded")

    def decide(self, state: WorldRuntime, command: CommandEnvelope) -> Decision:
        if command.session_id != state.session_id:
            raise CommandRejected("wrong_session", "Command session does not match runtime")
        if command.expected_state_version != state.version:
            raise CommandRejected(
                "stale_version",
                f"Expected world version {command.expected_state_version}, current version is {state.version}",
            )
        existing = [event for event in state.event_log if event.command_id == command.command_id]
        if existing:
            return Decision([], "duplicate", "command_already_applied")
        dispatch = {
            CommandKind.MOVE: self._move,
            CommandKind.WAIT: self._wait,
            CommandKind.INSPECT: self._inspection,
            CommandKind.INTERACT: self._interaction,
            CommandKind.SET_STATE: self._set_state,
        }
        return dispatch[command.kind](state, command)

    def materialize_events(
        self, state: WorldRuntime, command: CommandEnvelope, drafts: Iterable[EventDraft]
    ) -> list[DomainEvent]:
        transaction_version = state.version + 1
        correlation_id = command.correlation_id or command.command_id
        events: list[DomainEvent] = []
        for offset, draft in enumerate(drafts, start=1):
            sequence = state.event_sequence + offset
            events.append(
                DomainEvent(
                    event_id=f"{state.session_id}:event:{sequence:08d}",
                    session_id=state.session_id,
                    sequence=sequence,
                    transaction_version=transaction_version,
                    world_time=draft.world_time,
                    type=draft.type,
                    actor_id=draft.actor_id,
                    target_id=draft.target_id,
                    payload=deepcopy(draft.payload),
                    command_id=command.command_id,
                    correlation_id=correlation_id,
                    causation_id=draft.causation_id or command.command_id,
                    world_definition_version=state.world_definition_version,
                    visibility=draft.visibility,
                )
            )
        return events

    def reduce_event(self, state: WorldRuntime, event: DomainEvent) -> WorldRuntime:
        if event.session_id != state.session_id:
            raise ValueError("Cannot reduce an event from another session")
        if event.sequence != state.event_sequence + 1:
            raise ValueError("Event sequence is not contiguous")
        if event.world_definition_version != state.world_definition_version:
            raise ValueError("Event uses a different world definition version")
        result = state.model_copy(deep=True)
        payload = event.payload
        if event.type == "TimeAdvanced":
            result.clock = datetime.fromisoformat(str(payload["to"]))
        elif event.type == "EntityMoved":
            result.entities[str(payload["entity_id"])].placement = Placement(
                kind=PlacementKind.AT_LOCATION,
                target_id=str(payload["destination"]),
            )
        elif event.type == "EntityPlaced":
            result.entities[str(payload["entity_id"])].placement = Placement.model_validate(
                payload["placement"]
            )
        elif event.type == "ConnectionStateChanged":
            result.connections[str(payload["connection_id"])].state[str(payload["key"])] = deepcopy(
                payload["value"]
            )
        elif event.type == "ConnectionDiscovered":
            result.connections[str(payload["connection_id"])].state["discovered"] = True
        elif event.type == "StateVariableChanged":
            result.state_variables[str(payload["variable_id"])] = deepcopy(payload["value"])
        elif event.type == "KnowledgeLearned":
            result.knowledge.append(
                EpistemicRecord(
                    id=f"knowledge:{event.event_id}",
                    observer_id=str(payload["observer_id"]),
                    subject=str(payload["subject"]),
                    predicate=str(payload["predicate"]),
                    claimed_value=deepcopy(payload["claimed_value"]),
                    stance=str(payload["stance"]),
                    confidence=float(payload["confidence"]),
                    source=str(payload["source"]),
                    learned_at=event.world_time,
                    truth_proposition_id=payload.get("truth_proposition_id"),
                )
            )
        elif event.type == "LocationObserved":
            key = self._observation_key(str(payload["observer_id"]), str(payload["location_id"]))
            result.observed_locations[key] = event.world_time
        elif event.type == "EntityObserved":
            key = self._observation_key(str(payload["observer_id"]), str(payload["entity_id"]))
            result.observed_entities[key] = event.world_time
        elif event.type == "ScheduledTriggerConsumed":
            trigger_id = str(payload["trigger_id"])
            result.scheduled_triggers = [item for item in result.scheduled_triggers if item.id != trigger_id]
        result.version = max(result.version, event.transaction_version)
        result.event_sequence = event.sequence
        result.event_log.append(event)
        self.assert_invariants(result)
        return result

    def reduce_all(self, state: WorldRuntime, events: Iterable[DomainEvent]) -> WorldRuntime:
        result = state
        for event in events:
            result = self.reduce_event(result, event)
        return result

    def process(
        self, state: WorldRuntime, command: CommandEnvelope
    ) -> tuple[WorldRuntime, CommandReceipt]:
        decision = self.decide(state, command)
        if decision.outcome == "duplicate":
            return state, CommandReceipt(
                command_id=command.command_id,
                idempotency_key=command.idempotency_key,
                accepted=True,
                outcome="duplicate",
                reason=decision.reason,
                state_version=state.version,
            )
        events = self.materialize_events(state, command, decision.drafts)
        next_state = self.reduce_all(state, events)
        receipt = CommandReceipt(
            command_id=command.command_id,
            idempotency_key=command.idempotency_key,
            accepted=True,
            outcome=decision.outcome,
            reason=decision.reason,
            state_version=next_state.version,
            event_ids=[event.event_id for event in events],
        )
        return next_state, receipt

    def assert_invariants(self, state: WorldRuntime) -> None:
        if state.world_definition_digest != self.definition.content_digest:
            raise ValueError("Runtime is pinned to a different definition digest")
        if state.clock < self.definition.start_time:
            raise ValueError("World clock cannot precede definition start time")
        if state.event_log:
            sequences = [event.sequence for event in state.event_log]
            if sequences != list(range(1, len(sequences) + 1)):
                raise ValueError("Event log sequences must be contiguous")
            if state.event_sequence != sequences[-1]:
                raise ValueError("Runtime event sequence disagrees with event log")
        elif state.event_sequence != 0:
            raise ValueError("Empty event log must have sequence zero")
        for entity_id in state.entities:
            placement = state.entities[entity_id].placement
            if placement.kind == PlacementKind.AT_LOCATION:
                if placement.target_id not in self.locations:
                    raise ValueError(f"Entity {entity_id} has an unknown location")
            elif placement.target_id not in state.entities:
                raise ValueError(f"Entity {entity_id} has an unknown placement owner")
            if self.root_location(state, entity_id) is None:
                raise ValueError(f"Entity {entity_id} has a placement cycle")
        trigger_keys = [(item.scheduled_time, item.priority, item.id) for item in state.scheduled_triggers]
        if len({item.id for item in state.scheduled_triggers}) != len(state.scheduled_triggers):
            raise ValueError("Scheduled trigger ids must remain unique")
        if any(item.scheduled_time < self.definition.start_time for item in state.scheduled_triggers):
            raise ValueError("Scheduled trigger cannot precede world start")
        if state.event_log and state.version != state.event_log[-1].transaction_version:
            raise ValueError("Runtime version must equal the latest transaction version")
