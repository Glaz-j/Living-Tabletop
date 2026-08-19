from __future__ import annotations

from ..context import context_relevance_score, recent_visible_context
from ..models import ActionDefinition, WorldState
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
                    "dialogue_text": action.dialogue_text,
                    "risk": action.risk,
                    "category": action.category,
                    "requires_explicit_intent": action.risk == "dangerous" or not action.suggest,
                }
                for action in available_actions
            ],
        )

    @staticmethod
    def trace_summary(context: AssembledTurnContext) -> dict[str, object]:
        return {
            "scene_id": context.scene.get("id"),
            "present_entity_ids": [item["id"] for item in context.present_entities],
            "known_fact_ids": [item["id"] for item in context.player_known_facts],
            "visible_history_entries": len(context.recent_visible_history),
            "available_action_ids": [item["id"] for item in context.available_actions],
        }
