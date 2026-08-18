from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable

from .models import (
    Condition,
    Effect,
    Entity,
    EntityType,
    EventRecord,
    KnowledgeEntry,
    MemoryEntry,
    ScenarioDefinition,
    ScheduledEvent,
    SessionStatus,
    WorldState,
)
from .rules import RuleEngine


class KernelValidationError(ValueError):
    """Raised when a proposed mutation violates canonical world rules."""


class WorldKernel:
    def __init__(self, scenario: ScenarioDefinition):
        self.scenario = scenario
        self._clue_by_fact = {clue.fact_id: clue.id for clue in scenario.clues}
        self.rules = RuleEngine()

    def append_event(
        self,
        state: WorldState,
        event_type: str,
        *,
        actor: str | None = None,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
        visible: bool = False,
    ) -> EventRecord:
        record = EventRecord(
            id=f"event_{len(state.event_log) + 1:05d}",
            seq=len(state.event_log) + 1,
            time=state.world_time,
            type=event_type,
            actor=actor,
            target=target,
            payload=payload or {},
            visible_to_player=visible,
        )
        state.event_log.append(record)
        return record

    def _value_for(self, state: WorldState, condition: Condition) -> Any:
        subject = condition.subject
        predicate = condition.predicate

        if subject == "player":
            if predicate == "location":
                return state.entities[state.player.entity_id].location
            if predicate == "inventory":
                return state.player.inventory
            if predicate == "clue_count":
                return len(state.discovered_clue_ids)
            if predicate == "known_fact_count":
                return len(state.player_known_fact_ids)
            if hasattr(state.player, predicate):
                return getattr(state.player, predicate)
        elif subject == "world":
            if predicate == "time":
                return state.world_time
            if predicate == "status":
                return state.status.value
            return state.flags.get(predicate)
        elif subject.startswith("fact:"):
            fact_id = subject.split(":", 1)[1]
            fact = state.facts.get(fact_id)
            return fact.value if fact else None
        elif subject.startswith("flag:"):
            return state.flags.get(subject.split(":", 1)[1])
        elif subject.startswith("threat:"):
            clock = state.threats.get(subject.split(":", 1)[1])
            return clock.progress if clock else None
        elif subject.startswith("action:"):
            return subject.split(":", 1)[1] in state.completed_actions
        elif subject.startswith("entity:"):
            entity = state.entities.get(subject.split(":", 1)[1])
            if entity is None:
                return None
            if predicate in {"location", "active", "name", "type"}:
                value = getattr(entity, predicate)
                return value.value if hasattr(value, "value") else value
            return entity.attributes.get(predicate)
        return None

    def condition_met(self, state: WorldState, condition: Condition) -> bool:
        if condition.operator == "known":
            return str(condition.value) in state.player_known_fact_ids
        if condition.operator == "not_known":
            return str(condition.value) not in state.player_known_fact_ids

        actual = self._value_for(state, condition)
        expected = condition.value
        if condition.operator == "eq":
            return actual == expected
        if condition.operator == "ne":
            return actual != expected
        if condition.operator == "gte":
            return actual is not None and actual >= expected
        if condition.operator == "lte":
            return actual is not None and actual <= expected
        if condition.operator == "contains":
            return actual is not None and expected in actual
        if condition.operator == "not_contains":
            return actual is None or expected not in actual
        return False

    def all_conditions_met(self, state: WorldState, conditions: Iterable[Condition]) -> bool:
        return all(self.condition_met(state, condition) for condition in conditions)

    def action_is_available(self, state: WorldState, action_id: str) -> tuple[bool, str | None]:
        action = next((item for item in self.scenario.actions if item.id == action_id), None)
        if action is None:
            return False, "未知行动。"
        if state.status != SessionStatus.ACTIVE:
            return False, "本局游戏已经结束。"
        if state.pending_check is not None:
            return False, "请先处理刚才的失败检定。"
        if state.player.unconscious:
            return False, "调查员已经失去意识，无法继续行动。"
        player_location = state.entities[state.player.entity_id].location
        if action.location is not None and action.location != player_location:
            return False, "你当前不在可以执行该行动的位置。"
        if action.once and action.id in state.completed_actions:
            return False, "这个行动已经完成。"
        if not self.all_conditions_met(state, action.requirements):
            return False, "当前条件不满足。"
        if any(self.condition_met(state, condition) for condition in action.forbidden):
            return False, "世界状态阻止了这个行动。"
        return True, None

    def available_actions(self, state: WorldState) -> list:
        result = []
        for action in self.scenario.actions:
            available, _ = self.action_is_available(state, action.id)
            if available:
                result.append(action)
        return result

    def apply_effect(self, state: WorldState, effect: Effect, *, source: str) -> None:
        if effect.conditions and not self.all_conditions_met(state, effect.conditions):
            return
        params = effect.params
        op = effect.op

        if op == "create_entity":
            entity = Entity.model_validate(params["entity"])
            if entity.id in state.entities:
                raise KernelValidationError(f"Cannot replace existing entity: {entity.id}")
            if entity.type not in {EntityType.LOCATION, EntityType.NPC, EntityType.ITEM, EntityType.OBJECT}:
                raise KernelValidationError(f"Dynamic entity type is not allowed: {entity.type}")
            dynamic_count = sum("dynamic" in existing.tags for existing in state.entities.values())
            if dynamic_count >= 200:
                raise KernelValidationError("Dynamic entity limit reached")
            if entity.location is not None and entity.location not in state.entities:
                raise KernelValidationError(f"Dynamic entity has an unknown location: {entity.location}")
            entity.tags.update({"dynamic", "soft_canon"})
            state.entities[entity.id] = entity
            self.append_event(
                state,
                "entity_created",
                target=entity.id,
                payload={
                    "name": entity.name,
                    "entity_type": entity.type.value,
                    "location": entity.location,
                    "source": source,
                },
                visible=entity.location == state.entities[state.player.entity_id].location,
            )
            return

        if op == "reveal_fact":
            fact_id = str(params["fact_id"])
            if fact_id not in state.facts:
                raise KernelValidationError(f"Cannot reveal missing fact: {fact_id}")
            source_entity_id = params.get("source_entity_id")
            if source_entity_id is not None and str(source_entity_id) not in state.entities:
                raise KernelValidationError(f"Cannot disclose from missing entity: {source_entity_id}")
            newly_known = fact_id not in state.player_known_fact_ids
            state.player_known_fact_ids.add(fact_id)
            clue_id = self._clue_by_fact.get(fact_id)
            if clue_id:
                state.discovered_clue_ids.add(clue_id)
            if params.get("disclosure"):
                self.append_event(
                    state,
                    "fact_disclosed",
                    actor=str(source_entity_id) if source_entity_id is not None else None,
                    target=fact_id,
                    payload={
                        "fact_id": fact_id,
                        "source_entity_id": source_entity_id,
                        "source": source,
                        "newly_learned": newly_known,
                    },
                    visible=True,
                )
            if newly_known:
                self.append_event(
                    state,
                    "player_learned_fact",
                    target=fact_id,
                    payload={
                        "fact_id": fact_id,
                        "source": source,
                        "source_entity_id": source_entity_id,
                        "clue_id": clue_id,
                        "newly_learned": True,
                    },
                    visible=True,
                )
            return

        if op == "set_fact":
            fact_id = str(params["fact_id"])
            if fact_id not in state.facts:
                raise KernelValidationError(f"Cannot mutate missing fact: {fact_id}")
            fact = state.facts[fact_id]
            new_value = params.get("value")
            if fact.immutable and fact.value != new_value:
                raise KernelValidationError(f"Immutable fact rejected: {fact_id}")
            old_value = fact.value
            fact.value = new_value
            self.append_event(
                state,
                "fact_changed",
                target=fact_id,
                payload={"old": old_value, "new": new_value, "source": source},
                visible=fact_id in state.player_known_fact_ids,
            )
            return

        if op == "move_entity":
            entity_id = str(params["entity_id"])
            destination = str(params["destination"])
            entity = state.entities.get(entity_id)
            destination_entity = state.entities.get(destination)
            if entity is None or destination_entity is None or destination_entity.type.value != "LOCATION":
                raise KernelValidationError("Invalid entity movement")
            origin = entity.location
            entity.location = destination
            player_location = state.entities[state.player.entity_id].location
            visible = entity_id == state.player.entity_id or origin == player_location or destination == player_location
            self.append_event(
                state,
                "entity_moved",
                actor=entity_id,
                target=destination,
                payload={"origin": origin, "destination": destination, "source": source},
                visible=visible,
            )
            return

        if op == "modify_player":
            field = str(params["field"])
            if field not in {"hp", "stress", "sanity", "luck", "magic_points"}:
                raise KernelValidationError(f"Illegal player field mutation: {field}")
            current = int(getattr(state.player, field))
            value = int(params["set"]) if "set" in params else current + int(params.get("delta", 0))
            maximum = {
                "hp": state.player.max_hp,
                "stress": state.player.max_stress,
                "sanity": state.player.max_sanity,
                "luck": state.player.max_luck,
                "magic_points": max(0, int(state.player.characteristics.get("pow", 0)) // 5),
            }[field]
            value = max(0, min(maximum, value))
            if field == "hp" and value < current:
                damage = self.rules.apply_damage(state, current - value, source=source)
                value = state.player.hp
                self.append_event(
                    state,
                    "player_damaged",
                    target="hp",
                    payload=damage.model_dump(mode="json"),
                    visible=True,
                )
            else:
                setattr(state.player, field, value)
                if field == "hp" and value > current:
                    state.player.unconscious = False
                    state.player.dying = False
                    if value >= (state.player.max_hp + 1) // 2:
                        state.player.major_wound = False
                self.rules.refresh_conditions(state)
            self.append_event(
                state,
                "player_state_changed",
                target=field,
                payload={"old": current, "new": value, "source": source},
                visible=True,
            )
            return

        if op == "damage_player":
            try:
                amount = self.rules.roll_expression(state, str(params["amount"]))
            except (KeyError, ValueError) as exc:
                raise KernelValidationError("Invalid damage expression") from exc
            damage = self.rules.apply_damage(state, amount, source=source)
            self.append_event(
                state,
                "player_damaged",
                target="hp",
                payload=damage.model_dump(mode="json"),
                visible=True,
            )
            return

        if op == "sanity_check":
            try:
                result = self.rules.sanity_check(
                    state,
                    success_loss=str(params.get("success_loss", "0")),
                    failure_loss=str(params["failure_loss"]),
                    reason=str(params.get("reason", "遭遇骇人景象")),
                )
            except (KeyError, ValueError) as exc:
                raise KernelValidationError("Invalid sanity check") from exc
            self.append_event(
                state,
                "sanity_check_resolved",
                target=state.player.entity_id,
                payload=result.model_dump(mode="json"),
                visible=True,
            )
            return

        if op == "add_inventory":
            item_id = str(params["item_id"])
            if item_id not in state.entities:
                raise KernelValidationError(f"Unknown inventory item: {item_id}")
            if item_id not in state.player.inventory:
                state.player.inventory.append(item_id)
                state.entities[item_id].location = state.player.entity_id
                self.append_event(state, "item_acquired", target=item_id, payload={"source": source}, visible=True)
            return

        if op == "remove_inventory":
            item_id = str(params["item_id"])
            if item_id in state.player.inventory:
                state.player.inventory.remove(item_id)
                self.append_event(state, "item_removed", target=item_id, payload={"source": source}, visible=True)
            return

        if op == "add_npc_knowledge":
            npc_id = str(params["npc_id"])
            fact_id = str(params["fact_id"])
            if npc_id not in state.entities or fact_id not in state.facts:
                raise KernelValidationError("Invalid NPC knowledge reference")
            if not any(item.knower_id == npc_id and item.fact_id == fact_id for item in state.npc_knowledge):
                state.npc_knowledge.append(
                    KnowledgeEntry(
                        knower_id=npc_id,
                        fact_id=fact_id,
                        belief_value=params.get("belief_value"),
                        confidence=float(params.get("confidence", 1.0)),
                        source=str(params.get("source", source)),
                        learned_at=state.world_time,
                        concealed=bool(params.get("concealed", False)),
                    )
                )
            return

        if op == "add_memory":
            npc_id = str(params["npc_id"])
            if npc_id not in state.entities:
                raise KernelValidationError(f"Unknown NPC: {npc_id}")
            memory = MemoryEntry(
                id=f"memory_{len(state.npc_memories) + 1:05d}",
                npc_id=npc_id,
                text=str(params["text"]),
                occurred_at=state.world_time,
                source_event_id=params.get("source_event_id"),
                importance=float(params.get("importance", 0.5)),
            )
            state.npc_memories.append(memory)
            return

        if op == "schedule_event":
            event = ScheduledEvent.model_validate(params["event"])
            if any(existing.id == event.id and not existing.canceled for existing in state.event_queue):
                raise KernelValidationError(f"Duplicate scheduled event: {event.id}")
            if event.time < state.world_time:
                raise KernelValidationError("Cannot schedule an event in the past")
            state.event_queue.append(event)
            self._sort_queue(state)
            self.append_event(state, "event_scheduled", target=event.id, payload={"source": source}, visible=False)
            return

        if op == "cancel_event":
            event_id = str(params["event_id"])
            event = next((item for item in state.event_queue if item.id == event_id), None)
            if event is None and bool(params.get("if_present", False)):
                return
            if event is None or not event.cancelable:
                raise KernelValidationError(f"Event cannot be canceled: {event_id}")
            event.canceled = True
            self.append_event(state, "event_canceled", target=event_id, payload={"source": source}, visible=False)
            return

        if op == "advance_threat":
            self.advance_threat(state, str(params["threat_id"]), int(params.get("amount", 10)), source=source)
            return

        if op == "set_entity_active":
            entity_id = str(params["entity_id"])
            if entity_id not in state.entities:
                raise KernelValidationError(f"Unknown entity: {entity_id}")
            state.entities[entity_id].active = bool(params["active"])
            self.append_event(
                state,
                "entity_activity_changed",
                target=entity_id,
                payload={"active": bool(params["active"]), "source": source},
                visible=state.entities[entity_id].location == state.entities[state.player.entity_id].location,
            )
            return

        if op == "set_status":
            status = SessionStatus(str(params["status"]))
            if state.status != SessionStatus.ACTIVE:
                raise KernelValidationError("Cannot replace a terminal game status")
            state.status = status
            state.ending_id = params.get("ending_id")
            return

        if op == "mark_flag":
            key = str(params["key"])
            state.flags[key] = params.get("value", True)
            self.append_event(
                state,
                "flag_changed",
                target=key,
                payload={"value": state.flags[key], "source": source},
                visible=bool(params.get("visible", False)),
            )
            return

        raise KernelValidationError(f"Unsupported effect: {op}")

    def apply_effects(self, state: WorldState, effects: Iterable[Effect], *, source: str) -> None:
        for effect in effects:
            self.apply_effect(state, effect, source=source)

    def advance_threat(self, state: WorldState, threat_id: str, amount: int, *, source: str) -> None:
        clock = state.threats.get(threat_id)
        if clock is None:
            raise KernelValidationError(f"Unknown threat: {threat_id}")
        old = clock.progress
        clock.progress = max(0, min(100, old + amount))
        self.append_event(
            state,
            "threat_advanced",
            target=threat_id,
            payload={"old": old, "new": clock.progress, "amount": amount, "source": source},
            visible=False,
        )
        for threshold in sorted(clock.thresholds, key=lambda item: item.at):
            if old < threshold.at <= clock.progress and threshold.at not in clock.crossed:
                clock.crossed.add(threshold.at)
                self.apply_effects(state, threshold.effects, source=f"threat:{threat_id}:{threshold.at}")
                self.append_event(
                    state,
                    "threat_threshold_crossed",
                    target=threat_id,
                    payload={"at": threshold.at, "label": threshold.label, "text": threshold.narrative},
                    visible=bool(threshold.narrative),
                )

    @staticmethod
    def _sort_queue(state: WorldState) -> None:
        state.event_queue.sort(key=lambda event: (event.time, event.priority, event.id))

    def execute_scheduled_event(self, state: WorldState, event: ScheduledEvent) -> EventRecord | None:
        if event.canceled or not self.all_conditions_met(state, event.conditions):
            self.append_event(
                state,
                "scheduled_event_skipped",
                target=event.id,
                payload={"canceled": event.canceled},
                visible=False,
            )
            return None

        raw_effects = event.payload.get("effects", [])
        effects = [Effect.model_validate(raw) for raw in raw_effects]
        self.apply_effects(state, effects, source=f"scheduled_event:{event.id}")
        player_location = state.entities[state.player.entity_id].location
        visible_locations = event.payload.get("visible_locations", [])
        visible = bool(event.payload.get("always_visible", False) or player_location in visible_locations)
        return self.append_event(
            state,
            event.type,
            actor=event.actor,
            target=event.target,
            payload={
                "scheduled_event_id": event.id,
                "text": event.payload.get("text", ""),
            },
            visible=visible,
        )

    def advance_time(
        self,
        state: WorldState,
        minutes: int,
        *,
        action_interruptible: bool = True,
    ) -> tuple[list[EventRecord], ScheduledEvent | None]:
        if minutes < 0:
            raise KernelValidationError("Time cannot move backwards")
        target_time = state.world_time + timedelta(minutes=minutes)
        visible_events: list[EventRecord] = []
        interrupting_event: ScheduledEvent | None = None
        self._sort_queue(state)

        while state.event_queue and state.event_queue[0].time <= target_time:
            event = state.event_queue.pop(0)
            state.world_time = event.time
            record = self.execute_scheduled_event(state, event)
            if record is not None and record.visible_to_player:
                visible_events.append(record)
            player_location = state.entities[state.player.entity_id].location
            location_matches = not event.interrupt_locations or player_location in event.interrupt_locations
            if record is not None and event.interrupt_action and action_interruptible and location_matches:
                interrupting_event = event
                break

        if interrupting_event is None:
            state.world_time = target_time
        self.rules.refresh_conditions(state)
        return visible_events, interrupting_event

    def evaluate_ending(self, state: WorldState) -> str | None:
        if state.status != SessionStatus.ACTIVE:
            if state.ending_id and not any(
                event.type == "game_ended" and event.target == state.ending_id
                for event in state.event_log
            ):
                ending = next(
                    (item for item in self.scenario.endings if item.id == state.ending_id),
                    None,
                )
                self.append_event(
                    state,
                    "game_ended",
                    target=state.ending_id,
                    payload={"title": ending.title if ending else state.ending_id},
                    visible=True,
                )
            return state.ending_id
        for ending in sorted(self.scenario.endings, key=lambda item: item.priority):
            if self.all_conditions_met(state, ending.requirements):
                state.status = ending.status
                state.ending_id = ending.id
                state.last_narrative = ending.narrative
                self.append_event(
                    state,
                    "game_ended",
                    target=ending.id,
                    payload={"title": ending.title},
                    visible=True,
                )
                return ending.id
        return None
