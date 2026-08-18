from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from .kernel import WorldKernel
from .models import (
    ActionDefinition,
    ActionType,
    Condition,
    Effect,
    EntityType,
    ScenarioDefinition,
    WorldState,
)


DEFAULT_SCENARIO_ID = "st_mary_hospital_v0"


class ScenarioIntegrityError(ValueError):
    pass


def validate_scenario_integrity(scenario: ScenarioDefinition) -> None:
    errors: list[str] = []
    dice_expression = re.compile(r"^(?:\d+|\d+d\d+(?:[+-]\d+)?)$", re.IGNORECASE)

    def unique_ids(label: str, values: list[str]) -> set[str]:
        duplicates = sorted(item for item, count in Counter(values).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate {label} ids: {duplicates}")
        return set(values)

    entity_ids = unique_ids("entity", [item.id for item in scenario.entities])
    fact_ids = unique_ids("fact", [item.id for item in scenario.facts])
    clue_ids = unique_ids("clue", [item.id for item in scenario.clues])
    action_ids = unique_ids("action", [item.id for item in scenario.actions])
    event_ids = unique_ids("event", [item.id for item in scenario.events])
    threat_ids = unique_ids("threat", [item.id for item in scenario.threats])
    ending_ids = unique_ids("ending", [item.id for item in scenario.endings])
    location_ids = {
        item.id for item in scenario.entities if item.type == EntityType.LOCATION
    }

    if scenario.start_location not in location_ids:
        errors.append(f"unknown start_location: {scenario.start_location}")
    if scenario.player.entity_id not in entity_ids:
        errors.append(f"unknown player entity: {scenario.player.entity_id}")
    for item_id in scenario.player.inventory:
        if item_id not in entity_ids:
            errors.append(f"unknown initial inventory item: {item_id}")
    for fact_id in scenario.initial_player_known_fact_ids:
        if fact_id not in fact_ids:
            errors.append(f"unknown initial known fact: {fact_id}")
    for clue in scenario.clues:
        if clue.fact_id not in fact_ids:
            errors.append(f"clue {clue.id} references unknown fact {clue.fact_id}")

    def check_condition(condition: Condition, owner: str) -> None:
        subject = condition.subject
        if subject.startswith("fact:") and subject.split(":", 1)[1] not in fact_ids:
            errors.append(f"{owner} references unknown fact condition {subject}")
        elif subject.startswith("entity:") and subject.split(":", 1)[1] not in entity_ids:
            errors.append(f"{owner} references unknown entity condition {subject}")
        elif subject.startswith("threat:") and subject.split(":", 1)[1] not in threat_ids:
            errors.append(f"{owner} references unknown threat condition {subject}")
        elif subject.startswith("action:") and subject.split(":", 1)[1] not in action_ids:
            errors.append(f"{owner} references unknown action condition {subject}")
        if condition.operator in {"known", "not_known"} and str(condition.value) not in fact_ids:
            errors.append(f"{owner} references unknown knowledge fact {condition.value}")

    def check_effect(effect: Effect, owner: str) -> None:
        for condition in effect.conditions:
            check_condition(condition, f"{owner} conditional effect")
        params = effect.params
        if effect.op in {"reveal_fact", "set_fact"}:
            fact_id = str(params.get("fact_id"))
            if fact_id not in fact_ids:
                errors.append(f"{owner} effect references unknown fact {fact_id}")
        elif effect.op == "move_entity":
            entity_id = str(params.get("entity_id"))
            destination = str(params.get("destination"))
            if entity_id not in entity_ids:
                errors.append(f"{owner} moves unknown entity {entity_id}")
            if destination not in location_ids:
                errors.append(f"{owner} moves to unknown location {destination}")
        elif effect.op in {"add_inventory", "remove_inventory"}:
            item_id = str(params.get("item_id"))
            if item_id not in entity_ids:
                errors.append(f"{owner} references unknown inventory item {item_id}")
        elif effect.op == "add_npc_knowledge":
            if str(params.get("npc_id")) not in entity_ids:
                errors.append(f"{owner} references unknown NPC {params.get('npc_id')}")
            if str(params.get("fact_id")) not in fact_ids:
                errors.append(f"{owner} references unknown NPC fact {params.get('fact_id')}")
        elif effect.op in {"add_memory", "set_entity_active"}:
            entity_id = str(params.get("npc_id") or params.get("entity_id"))
            if entity_id not in entity_ids:
                errors.append(f"{owner} references unknown entity {entity_id}")
        elif effect.op == "advance_threat":
            threat_id = str(params.get("threat_id"))
            if threat_id not in threat_ids:
                errors.append(f"{owner} advances unknown threat {threat_id}")
        elif effect.op == "cancel_event" and not params.get("if_present", False):
            event_id = str(params.get("event_id"))
            if event_id not in event_ids:
                errors.append(f"{owner} cancels unknown event {event_id}")
        elif effect.op == "set_status":
            ending_id = params.get("ending_id")
            if ending_id is not None and str(ending_id) not in ending_ids:
                errors.append(f"{owner} sets unknown ending {ending_id}")
        elif effect.op == "damage_player":
            amount = str(params.get("amount", ""))
            if not dice_expression.fullmatch(amount.replace(" ", "")):
                errors.append(f"{owner} has invalid damage expression {amount}")
        elif effect.op == "sanity_check":
            for key in ("success_loss", "failure_loss"):
                expression = str(params.get(key, ""))
                if not dice_expression.fullmatch(expression.replace(" ", "")):
                    errors.append(f"{owner} has invalid sanity expression {key}={expression}")

    for origin, destinations in scenario.location_graph.items():
        if origin not in location_ids:
            errors.append(f"location graph has unknown origin {origin}")
        for destination, duration in destinations.items():
            if destination not in location_ids:
                errors.append(f"location graph has unknown destination {destination}")
            if duration < 0:
                errors.append(f"location graph has negative duration {origin}->{destination}")
    graph_edges = {
        f"{origin}->{destination}"
        for origin, destinations in scenario.location_graph.items()
        for destination in destinations
    }
    for edge, conditions in scenario.movement_requirements.items():
        if edge not in graph_edges:
            errors.append(f"movement requirements reference unknown edge {edge}")
        for condition in conditions:
            check_condition(condition, f"movement edge {edge}")

    for action in scenario.actions:
        if action.location is not None and action.location not in location_ids:
            errors.append(f"action {action.id} has unknown location {action.location}")
        for condition in [*action.requirements, *action.forbidden]:
            check_condition(condition, f"action {action.id}")
        for modifier in action.modifiers:
            for condition in modifier.conditions:
                check_condition(condition, f"action {action.id} modifier {modifier.label}")
        if action.opposed and action.opposed.entity_id and action.opposed.entity_id not in entity_ids:
            errors.append(f"action {action.id} opposes unknown entity {action.opposed.entity_id}")
        if action.sanity_check:
            for key, expression in (
                ("success_loss", action.sanity_check.success_loss),
                ("failure_loss", action.sanity_check.failure_loss),
            ):
                if not dice_expression.fullmatch(expression.replace(" ", "")):
                    errors.append(f"action {action.id} has invalid sanity expression {key}={expression}")
        for effect in [
            *action.success_effects,
            *action.failure_effects,
            *action.always_effects,
            *action.push_failure_effects,
        ]:
            check_effect(effect, f"action {action.id}")
    for event in scenario.events:
        for condition in event.conditions:
            check_condition(condition, f"event {event.id}")
        for raw_effect in event.payload.get("effects", []):
            check_effect(Effect.model_validate(raw_effect), f"event {event.id}")
    for threat in scenario.threats:
        for threshold in threat.thresholds:
            for effect in threshold.effects:
                check_effect(effect, f"threat {threat.id}@{threshold.at}")
    for definition in [*scenario.complications, *scenario.respites]:
        for condition in definition.requirements:
            check_condition(condition, f"director option {definition.id}")
        for effect in definition.effects:
            check_effect(effect, f"director option {definition.id}")
    for hint in scenario.director_config.hint_opportunities:
        for condition in hint.requirements:
            check_condition(condition, f"director hint {hint.id}")
        for effect in hint.effects:
            check_effect(effect, f"director hint {hint.id}")
    for ending in scenario.endings:
        for condition in ending.requirements:
            check_condition(condition, f"ending {ending.id}")
    primary_threat = scenario.director_config.primary_threat_id
    if primary_threat is not None and primary_threat not in threat_ids:
        errors.append(f"director_config references unknown primary threat {primary_threat}")

    if errors:
        raise ScenarioIntegrityError("; ".join(errors))


def _add_graph_movement_actions(scenario: ScenarioDefinition) -> None:
    locations = {entity.id: entity for entity in scenario.entities if entity.type == EntityType.LOCATION}
    existing = {action.id for action in scenario.actions}
    for origin, destinations in scenario.location_graph.items():
        for destination, duration in destinations.items():
            action_id = f"move__{origin}__{destination}"
            if action_id in existing:
                continue
            destination_name = locations[destination].name
            edge_key = f"{origin}->{destination}"
            requirements = deepcopy(scenario.movement_requirements.get(edge_key, []))
            scenario.actions.append(
                ActionDefinition(
                    id=action_id,
                    label=f"前往{destination_name}",
                    type=ActionType.MOVE,
                    location=origin,
                    target=destination,
                    duration_minutes=duration,
                    aliases=[f"去{destination_name}", f"前往{destination_name}", f"走到{destination_name}", destination_name],
                    requirements=requirements,
                    success_effects=[
                        Effect(
                            op="move_entity",
                            params={"entity_id": "player", "destination": destination},
                        )
                    ],
                    success_text=f"你离开原处，抵达{destination_name}。",
                    category="move",
                    risk="safe",
                )
            )
            existing.add(action_id)


def _parse_scenario(raw: str) -> ScenarioDefinition:
    scenario = ScenarioDefinition.model_validate(json.loads(raw))
    _add_graph_movement_actions(scenario)
    validate_scenario_integrity(scenario)
    return scenario


def load_scenarios(directory: str | Path | None = None) -> dict[str, ScenarioDefinition]:
    if directory is None:
        resources = sorted(
            (
                item
                for item in files("living_tabletop").joinpath("scenarios").iterdir()
                if item.name.endswith(".json")
            ),
            key=lambda item: item.name,
        )
        scenarios = [_parse_scenario(item.read_text(encoding="utf-8")) for item in resources]
    else:
        paths = sorted(Path(directory).glob("*.json"))
        scenarios = [_parse_scenario(path.read_text(encoding="utf-8")) for path in paths]
    result = {scenario.id: scenario for scenario in scenarios}
    if len(result) != len(scenarios):
        raise ValueError("Scenario ids must be unique")
    if not result:
        raise ValueError("No scenario definitions found")
    return result


def load_scenario(
    path: str | Path | None = None,
    *,
    scenario_id: str | None = None,
) -> ScenarioDefinition:
    if path is not None:
        return _parse_scenario(Path(path).read_text(encoding="utf-8"))
    scenarios = load_scenarios()
    selected_id = scenario_id or DEFAULT_SCENARIO_ID
    try:
        return scenarios[selected_id]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario: {selected_id}") from exc


def create_initial_state(
    scenario: ScenarioDefinition,
    *,
    player_name: str = "调查员",
    seed: int = 1927,
    session_id: str | None = None,
) -> WorldState:
    entities = {entity.id: deepcopy(entity) for entity in scenario.entities}
    entities[scenario.player.entity_id].location = scenario.start_location
    player = deepcopy(scenario.player)
    player.name = player_name.strip() or "调查员"
    entities[player.entity_id].name = player.name
    for item_id in player.inventory:
        if item_id in entities:
            entities[item_id].location = player.entity_id

    state = WorldState(
        session_id=session_id or uuid.uuid4().hex,
        scenario_id=scenario.id,
        world_time=scenario.start_time,
        entities=entities,
        facts={fact.id: deepcopy(fact) for fact in scenario.facts},
        relationships=deepcopy(scenario.relationships),
        player=player,
        player_known_fact_ids=set(scenario.initial_player_known_fact_ids),
        npc_knowledge=deepcopy(scenario.npc_knowledge),
        event_queue=deepcopy(scenario.events),
        threats={threat.id: deepcopy(threat) for threat in scenario.threats},
        rng_seed=seed,
        last_narrative=scenario.opening_narrative,
        flags=deepcopy(scenario.initial_flags),
    )
    kernel = WorldKernel(scenario)
    kernel._sort_queue(state)
    kernel.append_event(
        state,
        "session_started",
        actor=player.entity_id,
        payload={"scenario_id": scenario.id, "seed": seed},
        visible=True,
    )
    return state
