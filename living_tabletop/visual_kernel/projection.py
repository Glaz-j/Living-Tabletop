from __future__ import annotations

from typing import Any

from .kernel import VisualWorldKernel
from .models import EntityKind, PlacementKind, WorldRuntime


def _event_is_visible(event: Any, viewer_id: str) -> bool:
    if event.visibility == "public":
        return True
    if event.visibility == "participants":
        return viewer_id in {event.actor_id, event.target_id} or event.actor_id == viewer_id
    return False


def _event_view(event: Any) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "world_time": event.world_time.isoformat(),
        "type": event.type,
        "actor_id": event.actor_id,
        "target_id": event.target_id,
        "payload": event.payload,
    }


def player_projection(
    kernel: VisualWorldKernel, state: WorldRuntime, viewer_id: str
) -> dict[str, Any]:
    if viewer_id not in state.entities:
        raise KeyError(f"Unknown viewer: {viewer_id}")
    viewer_location = kernel.root_location(state, viewer_id)
    known_locations = {
        key.split("|", 1)[1]
        for key in state.observed_locations
        if key.startswith(f"{viewer_id}|")
    }

    # A discovered route makes its immediate destination a known map affordance,
    # without claiming the actor has personally observed the room.
    changed = True
    while changed:
        changed = False
        for connection in kernel.definition.connections:
            runtime = state.connections[connection.id]
            if not runtime.state.get("discovered", True):
                continue
            if connection.from_location in known_locations and connection.to_location not in known_locations:
                known_locations.add(connection.to_location)
                changed = True
            if (
                connection.bidirectional
                and connection.to_location in known_locations
                and connection.from_location not in known_locations
            ):
                known_locations.add(connection.from_location)
                changed = True

    locations = []
    for location in kernel.definition.locations:
        if location.id not in known_locations:
            continue
        observed = kernel._observation_key(viewer_id, location.id) in state.observed_locations
        locations.append(
            {
                "id": location.id,
                "name": location.name if observed else "未知区域",
                "parent_id": location.parent_id,
                "x": location.x,
                "y": location.y,
                "description": location.description if observed else "一条已经发现、但尚未踏足的路径。",
                "observed": observed,
                "current": location.id == viewer_location,
            }
        )

    connections = []
    for connection in kernel.definition.connections:
        runtime = state.connections[connection.id]
        if not runtime.state.get("discovered", True):
            continue
        if connection.from_location not in known_locations or connection.to_location not in known_locations:
            continue
        connections.append(
            {
                "id": connection.id,
                "name": connection.name,
                "from_location": connection.from_location,
                "to_location": connection.to_location,
                "bidirectional": connection.bidirectional,
                "travel_minutes": connection.travel_minutes,
                "state": {
                    "open": bool(runtime.state.get("open", True)),
                    "locked": bool(runtime.state.get("locked", False)),
                    "discovered": True,
                },
            }
        )

    visible_entities = []
    inventory = []
    for entity_id, runtime in state.entities.items():
        definition = kernel.entity_definitions[entity_id]
        held = runtime.placement.kind in {PlacementKind.HELD_BY, PlacementKind.EQUIPPED_BY}
        if held and runtime.placement.target_id == viewer_id:
            inventory.append(
                {
                    "id": entity_id,
                    "name": definition.name,
                    "kind": definition.kind,
                    "description": definition.description,
                }
            )
            continue
        if entity_id == viewer_id or kernel.root_location(state, entity_id) != viewer_location:
            continue
        visible_entities.append(
            {
                "id": entity_id,
                "name": definition.name,
                "kind": definition.kind,
                "description": definition.description,
                "asset_id": definition.asset_id,
                "tags": sorted(definition.tags),
            }
        )

    knowledge = [
        {
            "id": item.id,
            "subject": item.subject,
            "predicate": item.predicate,
            "claimed_value": item.claimed_value,
            "stance": item.stance,
            "confidence": item.confidence,
            "source": item.source,
            "learned_at": item.learned_at.isoformat(),
        }
        for item in state.knowledge
        if item.observer_id == viewer_id
    ]

    affordances: list[dict[str, Any]] = []
    for connection in kernel.definition.connections:
        runtime = state.connections[connection.id]
        if not runtime.state.get("discovered", True):
            continue
        destination: str | None = None
        if connection.from_location == viewer_location:
            destination = connection.to_location
        elif connection.bidirectional and connection.to_location == viewer_location:
            destination = connection.from_location
        if destination:
            destination_name = next(
                (item["name"] for item in locations if item["id"] == destination), "未知区域"
            )
            if runtime.state.get("locked", False):
                affordances.append(
                    {
                        "label": f"尝试解锁{connection.name}",
                        "kind": "interact",
                        "payload": {"target_id": connection.id, "verb": "unlock"},
                    }
                )
            elif not runtime.state.get("open", True):
                affordances.append(
                    {
                        "label": f"打开{connection.name}",
                        "kind": "interact",
                        "payload": {"target_id": connection.id, "verb": "open"},
                    }
                )
            else:
                affordances.append(
                    {
                        "label": f"前往{destination_name}",
                        "kind": "move",
                        "payload": {"destination_id": destination},
                    }
                )
    for entity in visible_entities:
        affordances.append(
            {
                "label": f"查看{entity['name']}",
                "kind": "inspect",
                "payload": {"target_id": entity["id"]},
            }
        )
        if entity["kind"] == EntityKind.OBJECT and "portable" in entity["tags"]:
            affordances.append(
                {
                    "label": f"拿起{entity['name']}",
                    "kind": "interact",
                    "payload": {"target_id": entity["id"], "verb": "take"},
                }
            )
        elif entity["kind"] == EntityKind.ACTOR:
            affordances.append(
                {
                    "label": f"与{entity['name']}交谈",
                    "kind": "interact",
                    "payload": {"target_id": entity["id"], "verb": "talk"},
                }
            )
    affordances.append({"label": "等待五分钟", "kind": "wait", "payload": {"minutes": 5}})

    events = [_event_view(event) for event in state.event_log if _event_is_visible(event, viewer_id)]
    viewer_definition = kernel.entity_definitions[viewer_id]
    return {
        "projection_type": "player",
        "projection_version": 1,
        "world_version": state.version,
        "viewer_id": viewer_id,
        "world": {
            "id": kernel.definition.id,
            "definition_version": kernel.definition.version,
            "title": kernel.definition.title,
            "time": state.clock.isoformat(),
        },
        "observer": {
            "id": viewer_id,
            "name": viewer_definition.name,
            "location_id": viewer_location,
            "inventory": inventory,
        },
        "map": {"locations": locations, "connections": connections},
        "visible_entities": visible_entities,
        "knowledge": knowledge,
        "affordances": affordances,
        "recent_events": events[-16:],
    }


def dev_projection(kernel: VisualWorldKernel, state: WorldRuntime) -> dict[str, Any]:
    return {
        "projection_type": "dev",
        "projection_version": 1,
        "world_version": state.version,
        "world_definition_digest": state.world_definition_digest,
        "definition": kernel.definition.model_dump(mode="json"),
        "runtime": state.model_dump(mode="json"),
        "map": {
            "locations": [item.model_dump(mode="json") for item in kernel.definition.locations],
            "connections": [
                {
                    **item.model_dump(mode="json"),
                    "runtime_state": state.connections[item.id].state,
                }
                for item in kernel.definition.connections
            ],
        },
        "recent_events": [_event_view(event) for event in state.event_log[-50:]],
    }
