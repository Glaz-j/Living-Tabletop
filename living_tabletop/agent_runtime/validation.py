from __future__ import annotations

from copy import deepcopy

from ..models import ActionDefinition, ActionType, OpenActionPlan, WorldState
from .contracts import PlayerIntentEnvelope, TurnPlannerDecision


class PlanValidationError(ValueError):
    pass


class PlanValidator:
    """Rejects invented authority while preserving the player's literal intent."""

    def validate(
        self,
        state: WorldState,
        envelope: PlayerIntentEnvelope,
        decision: TurnPlannerDecision,
        available_actions: list[ActionDefinition],
    ) -> TurnPlannerDecision:
        available_by_id = {action.id: action for action in available_actions}
        if decision.existing_action_id is not None:
            if decision.existing_action_id not in available_by_id:
                raise PlanValidationError(
                    f"existing action is not available: {decision.existing_action_id}"
                )
            return decision

        plan = deepcopy(decision.open_plan)
        if plan is None:
            raise PlanValidationError("planner omitted open plan")
        if plan.goal.strip() != envelope.text.strip():
            raise PlanValidationError("open plan goal must exactly preserve player text")

        allowed_skills = {*state.player.skills, *state.player.characteristics, "luck"}
        if plan.skill not in allowed_skills:
            plan.skill = None
            if plan.resolution == "check":
                plan.resolution = "automatic"

        player_location = state.entities[state.player.entity_id].location
        present_ids = {
            entity.id
            for entity in state.entities.values()
            if entity.active and entity.location == player_location
        }
        if plan.action_type in {ActionType.TALK, ActionType.DECEIVE}:
            addressee_id = plan.addressee_id or plan.target_entity_id
            if addressee_id is None:
                present_npcs = [
                    entity.id
                    for entity in state.entities.values()
                    if entity.active
                    and entity.location == player_location
                    and entity.type.value == "NPC"
                ]
                if len(present_npcs) == 1:
                    addressee_id = present_npcs[0]
            if addressee_id is not None and addressee_id not in present_ids:
                raise PlanValidationError("dialogue addressee is not present")
            plan.addressee_id = addressee_id
            plan.target_entity_id = addressee_id
            plan.referents = [
                referent.model_copy(
                    update={
                        "entity_id": (
                            referent.entity_id
                            if referent.entity_id in state.entities
                            else None
                        )
                    }
                )
                for referent in plan.referents
            ]
            if plan.knowledge_query is not None:
                if addressee_id is None:
                    raise PlanValidationError("knowledge query has no present addressee")
                plan.knowledge_query.addressee_id = addressee_id
                plan.knowledge_query.asker_id = envelope.actor_id
                plan.knowledge_query.subject_entity_ids = [
                    entity_id
                    for entity_id in plan.knowledge_query.subject_entity_ids
                    if entity_id in state.entities
                ]
                # Fact identifiers are resolver-owned authority, never Planner authority.
                plan.knowledge_query.explicit_fact_ids = []

        for destination_id in [plan.destination_entity_id]:
            if destination_id is None:
                continue
            entity = state.entities.get(destination_id)
            if entity is None or entity.type.value != "LOCATION":
                plan.destination_entity_id = None

        return TurnPlannerDecision(open_plan=plan, confidence=decision.confidence)

    @staticmethod
    def materialize(decision: TurnPlannerDecision) -> OpenActionPlan:
        plan = decision.open_plan
        if plan is None:
            raise PlanValidationError("cannot materialize an existing action")

        target = plan.target_name or plan.destination_name or "眼前目标"
        if plan.resolution == "impossible":
            success = "世界状态没有因为宣称而改变。"
            failure = f"你尝试了“{plan.label}”，但现实条件不允许这个结果发生。"
        elif plan.action_type in {ActionType.TALK, ActionType.DECEIVE}:
            success = f"你把这句话直接说给{target}听。"
            failure = f"{target}没有对这句话作出有效回应。"
        elif plan.action_type in {ActionType.SEARCH, ActionType.EXAMINE}:
            success = f"你完成了这次{plan.label}；没有未经验证的新事实被写入世界。"
            failure = f"你尝试了{plan.label}，但没有取得可靠结果。"
        elif plan.action_type in {ActionType.MOVE, ActionType.ESCAPE}:
            success = f"你按计划动身前往{target}。"
            failure = f"你没能抵达{target}。"
        elif plan.action_type == ActionType.REST:
            success = "你按自己的安排离开当前事务并休息。"
            failure = "突发状况使这次休息未能完成。"
        else:
            success = f"你完成了{plan.label}。"
            failure = f"你尝试了{plan.label}，但没有达到目标。"

        difficulty = plan.difficulty if plan.resolution == "check" else "regular"
        return OpenActionPlan(
            label=plan.label,
            action_type=plan.action_type,
            goal=plan.goal,
            target_name=plan.target_name,
            target_entity_id=plan.target_entity_id,
            destination_name=plan.destination_name,
            destination_entity_id=plan.destination_entity_id,
            destination_description=plan.destination_description,
            duration_minutes=plan.duration_minutes,
            resolution=plan.resolution,
            skill=plan.skill,
            difficulty=difficulty,
            risk=plan.risk,
            rest_until_hour=plan.rest_until_hour,
            rest_day_offset=plan.rest_day_offset,
            success_text=success,
            failure_text=failure,
        )
