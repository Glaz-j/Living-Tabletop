from __future__ import annotations

from ..context import context_relevance_score, recent_visible_context
from ..models import ActionDefinition, EntityType, WorldState
from .contracts import AssembledTurnContext, PlayerIntentEnvelope


class ContextAssembler:
    """Builds the bounded, player-safe context consumed by TurnPlanner."""

    def assemble(
        self,
        state: WorldState,
        envelope: PlayerIntentEnvelope,
        available_actions: list[ActionDefinition],
        *,
        semantic_hint: str | None = None,
    ) -> AssembledTurnContext:
        retrieval_query = "\n".join(
            item for item in (envelope.text, (semantic_hint or "").strip()) if item
        )
        player_entity = state.entities[state.player.entity_id]
        location = state.entities.get(player_entity.location or "")
        present = [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type.value,
                "role": entity.attributes.get("role"),
            }
            for entity in state.entities.values()
            if entity.active
            and entity.location == player_entity.location
            and entity.id != state.player.entity_id
        ]

        known_facts = []
        for fact_id in sorted(state.player_known_fact_ids):
            fact = state.facts.get(fact_id)
            if fact is None:
                continue
            serialized = {
                "id": fact.id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "value": fact.value,
                "canon": fact.canon,
            }
            score = context_relevance_score(
                retrieval_query,
                f"{fact.subject} {fact.predicate} {fact.value}",
            )
            known_facts.append((score, serialized))
        known_facts.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)

        # A conversational model needs the NPC's actual knowledge before it can
        # answer naturally.  Supplying it here avoids a second foreground
        # Planner -> Retriever -> Dialogue round trip.  Concealed entries remain
        # absent, while relevance only changes ordering, never authority.
        npc_knowledge = []
        present_ids = {item["id"] for item in present}
        for entry in state.npc_knowledge:
            if entry.knower_id not in present_ids or entry.concealed:
                continue
            fact = state.facts.get(entry.fact_id)
            if fact is None:
                continue
            value = entry.belief_value if entry.belief_value is not None else fact.value
            serialized = {
                "knower_id": entry.knower_id,
                "fact_id": fact.id,
                "subject_entity_id": fact.subject,
                "predicate": fact.predicate,
                "value": value,
                "canon": fact.canon,
                "confidence": entry.confidence,
            }
            score = context_relevance_score(
                retrieval_query,
                f"{entry.knower_id} {fact.subject} {fact.predicate} {value}",
            )
            npc_knowledge.append((score, serialized))
        npc_knowledge.sort(
            key=lambda item: (item[0], item[1]["confidence"], item[1]["fact_id"]),
            reverse=True,
        )
        relevant_npc_knowledge = [item for item in npc_knowledge if item[0] > 0][:16]

        referenced_ids = set(present_ids)
        referenced_ids.update(item[1]["subject"] for item in known_facts[:24])
        referenced_ids.update(
            item[1]["subject_entity_id"] for item in relevant_npc_knowledge
        )
        accessible_fact_values = [str(item[1]["value"]) for item in known_facts[:24]]
        accessible_fact_values.extend(
            str(item[1]["value"]) for item in relevant_npc_knowledge
        )
        for entity in state.entities.values():
            if entity.active and entity.name and any(
                entity.name in value for value in accessible_fact_values
            ):
                referenced_ids.add(entity.id)
        # If the player explicitly names an established entity, it is valid
        # context even when no prior structured fact points at it yet.  This is
        # what lets an NPC invent a harmless route/address for a known place.
        for entity in state.entities.values():
            if entity.active and entity.name and entity.name in envelope.text:
                referenced_ids.add(entity.id)
        referenced_entities = [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type.value,
                "role": entity.attributes.get("role"),
                "known_attributes": {
                    key: value
                    for key, value in entity.attributes.items()
                    if key in {"role", "mood", "description"}
                },
            }
            for entity in state.entities.values()
            if entity.id in referenced_ids and entity.type != EntityType.PLAYER
        ]

        return AssembledTurnContext(
            world_time=state.world_time.isoformat(),
            scene={
                "id": location.id if location else None,
                "name": location.name if location else "未知地点",
                "description": location.attributes.get("description", "") if location else "",
            },
            present_entities=present,
            player_known_facts=[item for _score, item in known_facts[:24]],
            recent_visible_history=recent_visible_context(
                state,
                query=retrieval_query,
                max_entries=24,
                max_characters=8000,
                immediate_entries=10,
            ),
            inventory=[
                {"id": item_id, "name": state.entities[item_id].name}
                for item_id in state.player.inventory
                if item_id in state.entities
            ],
            capabilities={
                "skills": state.player.skills,
                "characteristics": state.player.characteristics,
            },
            available_actions=[
                {
                    "id": action.id,
                    "label": action.label,
                    "type": action.type.value,
                    "target": action.target,
                    "aliases": action.aliases,
                    "risk": action.risk,
                    "category": action.category,
                    "requires_explicit_intent": action.risk == "dangerous" or not action.suggest,
                }
                for action in available_actions
            ],
            hard_state={
                "player_id": state.player.entity_id,
                "location_id": player_entity.location,
                "hp": state.player.hp,
                "max_hp": state.player.max_hp,
                "san": state.player.sanity,
                "luck": state.player.luck,
                "session_status": state.status.value,
            },
            referenced_entities=referenced_entities,
            present_npc_knowledge=[item for _score, item in relevant_npc_knowledge],
        )

    @staticmethod
    def trace_summary(context: AssembledTurnContext) -> dict[str, object]:
        return {
            "scene_id": context.scene.get("id"),
            "present_entity_ids": [item["id"] for item in context.present_entities],
            "known_fact_ids": [item["id"] for item in context.player_known_facts],
            "visible_history_entries": len(context.recent_visible_history),
            "available_action_ids": [item["id"] for item in context.available_actions],
            "referenced_entity_ids": [item["id"] for item in context.referenced_entities],
            "npc_knowledge_fact_ids": [
                item["fact_id"] for item in context.present_npc_knowledge
            ],
        }
