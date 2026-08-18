from __future__ import annotations

from hashlib import sha256
from typing import Any

from .kernel import WorldKernel
from .models import ActionDefinition, ActionType, Entity, EntityType, ScenarioDefinition, WorldState


class SceneProjectionAdapter:
    """Project the legacy scenario runtime into a local RPGMaker-like scene.

    This adapter is deliberately read-only.  Scene clicks carry an existing
    authored action id back into the same GameEngine path used by tavern-style
    buttons, so the visual layer never becomes a second rules engine.
    """

    ACTOR_SLOTS = (
        (31, 43),
        (69, 39),
        (22, 66),
        (78, 66),
        (50, 29),
        (39, 62),
        (62, 58),
    )
    OBJECT_SLOTS = (
        (20, 28),
        (78, 27),
        (15, 53),
        (85, 52),
        (31, 75),
        (69, 76),
        (47, 47),
    )
    HOTSPOT_SLOTS = (
        (25, 23),
        (74, 23),
        (17, 48),
        (83, 48),
        (29, 70),
        (71, 70),
    )
    EXIT_SLOTS = (
        (50, 5, "north"),
        (95, 48, "east"),
        (50, 95, "south"),
        (5, 48, "west"),
        (82, 7, "north"),
        (94, 76, "east"),
        (18, 94, "south"),
        (6, 22, "west"),
    )

    def __init__(self, scenario: ScenarioDefinition, kernel: WorldKernel):
        self.scenario = scenario
        self.kernel = kernel

    @staticmethod
    def _stable_offset(identifier: str, span: float = 2.5) -> tuple[float, float]:
        digest = sha256(identifier.encode("utf-8")).digest()
        x = ((digest[0] / 255) - 0.5) * span
        y = ((digest[1] / 255) - 0.5) * span
        return round(x, 2), round(y, 2)

    @classmethod
    def _placed(cls, slot: tuple[int, int], identifier: str) -> dict[str, float]:
        offset_x, offset_y = cls._stable_offset(identifier)
        return {"x": round(slot[0] + offset_x, 2), "y": round(slot[1] + offset_y, 2)}

    @staticmethod
    def _archetype(location: Entity) -> str:
        tags = location.tags
        lowered = f"{location.id} {location.name}".lower()
        if tags & {"outside", "exit"}:
            return "exterior"
        if "chapel" in tags or "教堂" in location.name:
            return "chapel"
        if tags & {"lair", "ritual"} or any(
            word in lowered for word in ("basement", "crypt", "cellar", "地下", "地窖", "墓室")
        ):
            return "underground"
        if "hospital" in tags:
            return "hospital"
        if tags & {"house", "residence"}:
            return "residence"
        if tags & {"research", "city"}:
            return "civic"
        return "interior"

    @staticmethod
    def _action_view(action: ActionDefinition) -> dict[str, Any]:
        return {
            "action_id": action.id,
            "label": action.label,
            "type": action.type.value,
            "category": action.category,
            "risk": action.risk,
            "duration_minutes": action.duration_minutes,
        }

    @staticmethod
    def _first_action(
        actions: list[ActionDefinition],
        *,
        target_id: str,
        kinds: set[ActionType],
    ) -> ActionDefinition | None:
        return next(
            (
                action
                for action in actions
                if action.target == target_id and action.type in kinds
            ),
            None,
        )

    def project(self, state: WorldState) -> dict[str, Any]:
        player = state.entities[state.player.entity_id]
        location_id = player.location
        if not location_id or location_id not in state.entities:
            return {
                "projection_version": 1,
                "location_id": None,
                "archetype": "void",
                "player": None,
                "actors": [],
                "objects": [],
                "hotspots": [],
                "exits": [],
            }
        location = state.entities[location_id]
        available = sorted(
            self.kernel.available_actions(state),
            key=lambda action: (action.category, action.id),
        )
        available_ids = {action.id for action in available}

        actors = sorted(
            (
                entity
                for entity in state.entities.values()
                if entity.id != state.player.entity_id
                and entity.type in {EntityType.NPC, EntityType.CREATURE}
                and entity.active
                and entity.location == location_id
            ),
            key=lambda entity: entity.id,
        )
        actor_views: list[dict[str, Any]] = []
        for index, entity in enumerate(actors):
            slot = self.ACTOR_SLOTS[index % len(self.ACTOR_SLOTS)]
            dialogue = self._first_action(
                available,
                target_id=entity.id,
                kinds={ActionType.TALK, ActionType.DECEIVE},
            )
            actor_views.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "role": entity.attributes.get("role"),
                    "kind": "creature" if entity.type == EntityType.CREATURE else "npc",
                    "position": self._placed(slot, entity.id),
                    "interaction": self._action_view(dialogue) if dialogue else None,
                }
            )

        objects = sorted(
            (
                entity
                for entity in state.entities.values()
                if entity.type in {EntityType.ITEM, EntityType.OBJECT}
                and entity.active
                and entity.location == location_id
            ),
            key=lambda entity: entity.id,
        )
        object_views: list[dict[str, Any]] = []
        attached_action_ids: set[str] = set()
        object_action_kinds = {
            ActionType.SEARCH,
            ActionType.EXAMINE,
            ActionType.TAKE,
            ActionType.USE,
            ActionType.FORCE,
            ActionType.RESCUE,
            ActionType.DISRUPT,
            ActionType.CONFRONT,
            ActionType.OTHER,
        }
        for index, entity in enumerate(objects):
            slot = self.OBJECT_SLOTS[index % len(self.OBJECT_SLOTS)]
            interaction = self._first_action(
                available,
                target_id=entity.id,
                kinds=object_action_kinds,
            )
            if interaction:
                attached_action_ids.add(interaction.id)
            object_views.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "kind": "item" if entity.type == EntityType.ITEM else "object",
                    "tags": sorted(entity.tags),
                    "position": self._placed(slot, entity.id),
                    "interaction": self._action_view(interaction) if interaction else None,
                }
            )

        exits: list[dict[str, Any]] = []
        move_actions = [
            action
            for action in self.scenario.actions
            if action.type == ActionType.MOVE and action.location == location_id
        ]
        for index, action in enumerate(sorted(move_actions, key=lambda item: item.target or item.id)):
            if not action.target or action.target not in state.entities:
                continue
            x, y, edge = self.EXIT_SLOTS[index % len(self.EXIT_SLOTS)]
            destination = state.entities[action.target]
            exits.append(
                {
                    "id": f"exit:{action.id}",
                    "destination_id": destination.id,
                    "label": destination.name,
                    "position": {"x": x, "y": y},
                    "edge": edge,
                    "available": action.id in available_ids,
                    "interaction": self._action_view(action) if action.id in available_ids else None,
                }
            )

        excluded_types = {ActionType.MOVE, ActionType.TALK, ActionType.DECEIVE, ActionType.WAIT, ActionType.REST}
        hotspots = [
            action
            for action in available
            if action.type not in excluded_types
            and action.id not in attached_action_ids
            and (action.location in {None, location_id})
        ]
        hotspot_views = []
        for index, action in enumerate(hotspots[: len(self.HOTSPOT_SLOTS)]):
            slot = self.HOTSPOT_SLOTS[index]
            hotspot_views.append(
                {
                    "id": f"hotspot:{action.id}",
                    "name": action.label,
                    "position": self._placed(slot, action.id),
                    "interaction": self._action_view(action),
                }
            )

        return {
            "projection_version": 1,
            "location_id": location_id,
            "archetype": self._archetype(location),
            "danger": "danger" in location.tags,
            "description": location.attributes.get("description", ""),
            "grid": {"columns": 16, "rows": 9},
            "player": {
                "id": player.id,
                "name": player.name,
                "position": {"x": 50, "y": 76},
                "facing": "north",
            },
            "actors": actor_views,
            "objects": object_views,
            "hotspots": hotspot_views,
            "exits": exits,
        }

    def developer_world_projection(self, state: WorldState) -> dict[str, Any]:
        locations = [
            {
                "id": entity.id,
                "name": entity.name,
                "tags": sorted(entity.tags),
                "player_here": state.entities[state.player.entity_id].location == entity.id,
            }
            for entity in state.entities.values()
            if entity.type == EntityType.LOCATION
        ]
        connections = [
            {
                "from": origin,
                "to": destination,
                "minutes": minutes,
            }
            for origin, destinations in self.scenario.location_graph.items()
            for destination, minutes in destinations.items()
        ]
        return {
            "projection_version": 1,
            "locations": locations,
            "connections": connections,
            "entities": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "location": entity.location,
                    "active": entity.active,
                }
                for entity in state.entities.values()
                if entity.type != EntityType.LOCATION
            ],
        }
