from __future__ import annotations

from collections import Counter

from .kernel import WorldKernel
from .models import (
    ActionDefinition,
    ActionType,
    CheckOutcome,
    DirectorIntervention,
    Effect,
    PacingPhase,
    ScenarioDefinition,
    WorldState,
)


def _clamp(value: float) -> int:
    return max(0, min(100, round(value)))


class Director:
    """Telemetry plus a bounded, explainable policy.

    An LLM advisor may rank already-legal candidates later, but this class owns the
    invariant-preserving fallback and is always available offline.
    """

    def __init__(self, scenario: ScenarioDefinition, kernel: WorldKernel):
        self.scenario = scenario
        self.kernel = kernel

    @staticmethod
    def _definition_was_used(state: WorldState, definition_id: str, justification: str) -> bool:
        """Recognize both new interventions and legacy saves without definition IDs."""

        return any(
            previous.source_definition_id == definition_id
            or (
                previous.source_definition_id is None
                and previous.world_justification == justification
            )
            for previous in state.director.interventions
        )

    def observe_action(
        self,
        state: WorldState,
        action: ActionDefinition,
        outcome: CheckOutcome,
        *,
        progress_before: int,
        location_before: str | None,
    ) -> None:
        director = state.director
        exp = director.experience
        success = outcome not in {CheckOutcome.FAILURE, CheckOutcome.FUMBLE, CheckOutcome.INTERRUPTED}
        if success:
            exp.success_streak += 1
            exp.failure_streak = 0
            exp.frustration = _clamp(exp.frustration - 8)
        else:
            exp.failure_streak += 1
            exp.success_streak = 0
            exp.frustration = _clamp(exp.frustration + (14 if outcome == CheckOutcome.FUMBLE else 9))

        director.meaningful_actions += 1
        director.actions_since_evaluation += 1
        director.recent_action_types.append(action.type)
        director.recent_action_types = director.recent_action_types[-10:]
        current_location = state.entities.get(state.entities[state.player.entity_id].location or "")
        is_off_main = current_location is not None and "off_main" in current_location.tags
        if action.id.startswith("open__"):
            director.last_open_goal = action.label
        if is_off_main:
            director.off_main_streak += 1
        else:
            director.off_main_streak = 0
        if len(state.discovered_clue_ids) > progress_before:
            director.actions_without_progress = 0
        elif action.type not in {ActionType.MOVE, ActionType.WAIT}:
            director.actions_without_progress += 1

        location = state.entities[state.player.entity_id].location
        if location and location != location_before:
            director.scene_history.append(location)
            director.scene_history = director.scene_history[-8:]

        self.recompute(state)

    def recompute(self, state: WorldState) -> None:
        exp = state.director.experience
        clue_ratio = len(state.discovered_clue_ids) / max(1, len(self.scenario.clues))
        max_threat = max((clock.progress for clock in state.threats.values()), default=0)
        hp_loss = state.player.max_hp - state.player.hp
        sanity_loss = max(0, state.player.starting_sanity - state.player.sanity)
        hostile_here = any(
            entity.active
            and "hostile" in entity.tags
            and entity.location == state.entities[state.player.entity_id].location
            for entity in state.entities.values()
        )

        progress_bonus = 15 if any(
            state.flags.get(flag_name) for flag_name in self.scenario.director_config.progress_flags
        ) else 0
        exp.progress = _clamp(clue_ratio * 85 + progress_bonus)
        exp.mystery = _clamp(100 - clue_ratio * 80)
        exp.danger = _clamp(
            hp_loss * 9
            + sanity_loss * 3
            + state.player.stress * 4
            + max_threat * 0.35
            + (18 if hostile_here else 0)
        )
        exp.time_pressure = _clamp(max_threat)
        exp.resource_pressure = _clamp(hp_loss * 10 + sanity_loss * 4 + state.player.stress * 5)
        recent_threats = sum(1 for event in state.event_log[-8:] if "threat" in event.type or "ritual" in event.type)
        exp.tension = _clamp(max_threat * 0.55 + exp.danger * 0.25 + recent_threats * 6)
        exp.relief_need = _clamp(exp.danger * 0.55 + exp.frustration * 0.45 + exp.failure_streak * 8)
        available_count = len(self.kernel.available_actions(state))
        # Suggested buttons are conveniences; open KP adjudication is always available.
        exp.agency = max(85, _clamp(55 + available_count * 6 - (15 if available_count <= 1 else 0)))

        recent = state.director.recent_action_types
        if recent:
            most_common = Counter(recent).most_common(1)[0][1]
            exp.novelty = _clamp(100 - (most_common / len(recent)) * 70)

        if exp.relief_need >= 70:
            state.director.phase = PacingPhase.RELIEF
        elif exp.tension >= 75 or max_threat >= 80:
            state.director.phase = PacingPhase.PEAK
        elif exp.tension >= 50 or max_threat >= 50:
            state.director.phase = PacingPhase.PRESSURE
        elif state.director.meaningful_actions >= 2:
            state.director.phase = PacingPhase.BUILD
        else:
            state.director.phase = PacingPhase.EXPLORE

    def should_evaluate(self, state: WorldState, *, major_event: bool = False, scene_changed: bool = False) -> bool:
        exp = state.director.experience
        return any(
            (
                major_event,
                scene_changed,
                state.director.actions_since_evaluation >= 3,
                exp.success_streak >= 3,
                exp.failure_streak >= 2,
                state.director.actions_without_progress >= 3,
                state.player.hp <= 3,
                state.player.stress >= 8,
                state.player.sanity <= max(10, state.player.starting_sanity // 2),
            )
        )

    def decide(self, state: WorldState) -> DirectorIntervention | None:
        self.recompute(state)
        exp = state.director.experience
        intervention: DirectorIntervention | None = None

        if exp.relief_need >= 65 or exp.failure_streak >= 2:
            respite = next(
                (
                    item
                    for item in self.scenario.respites
                    if self.kernel.all_conditions_met(state, item.requirements)
                ),
                None,
            )
            if respite is not None:
                intervention = DirectorIntervention(
                    id=f"intervention_{len(state.director.interventions) + 1:04d}",
                    action="offer_respite",
                    reason=f"连续失败 {exp.failure_streak} 次，喘息需求为 {exp.relief_need}。",
                    world_justification=respite.world_justification,
                    source_definition_id=respite.id,
                    player_visible_text=respite.player_visible_text,
                    affected_entities=respite.affected_entities or [state.player.entity_id],
                    expected_experience_effect="降低挫败和资源压力，但不取消任何既有失败。",
                    effects=respite.effects,
                    applied_at=state.world_time,
                )

        if intervention is None and (
            state.director.actions_without_progress >= 3 or state.director.off_main_streak >= 3
        ):
            hint = next(
                (
                    item
                    for item in self.scenario.director_config.hint_opportunities
                    if self.kernel.all_conditions_met(state, item.requirements)
                ),
                None,
            )
            if hint is not None:
                intervention = DirectorIntervention(
                    id=f"intervention_{len(state.director.interventions) + 1:04d}",
                    action="surface_clue",
                    reason=(
                        f"玩家连续 {state.director.off_main_streak} 次选择开放路线；"
                        "用世界内事件重新呈现未解决线索，但不阻止当前路线。"
                        if state.director.off_main_streak >= 3
                        else f"玩家已有 {state.director.actions_without_progress} 个有效行动未获得调查进展。"
                    ),
                    world_justification=hint.world_justification,
                    source_definition_id=hint.id,
                    player_visible_text=hint.player_visible_text,
                    affected_entities=hint.affected_entities,
                    expected_experience_effect="增加一个可选择的线索发现机会，不直接授予事实。",
                    effects=hint.effects,
                    applied_at=state.world_time,
                )

        if intervention is None and exp.success_streak >= 3 and exp.tension < 70:
            complication = next(
                (
                    item
                    for item in self.scenario.complications
                    if self.kernel.all_conditions_met(state, item.requirements)
                    and (
                        not item.once
                        or not self._definition_was_used(state, item.id, item.world_justification)
                    )
                ),
                None,
            )
            if complication is not None:
                intervention = DirectorIntervention(
                    id=f"intervention_{len(state.director.interventions) + 1:04d}",
                    action="increase_pressure",
                    reason=f"玩家连续成功 {exp.success_streak} 次，当前张力仅为 {exp.tension}。",
                    world_justification=complication.world_justification,
                    source_definition_id=complication.id,
                    player_visible_text=complication.player_visible_text,
                    affected_entities=complication.affected_entities,
                    expected_experience_effect="推进尚未完成的敌对准备，提升时间压力，不改变玩家成功结果。",
                    effects=complication.effects,
                    applied_at=state.world_time,
                )

        primary_threat_id = self.scenario.director_config.primary_threat_id
        if (
            intervention is None
            and primary_threat_id in state.threats
            and exp.tension < 25
            and exp.progress >= 20
        ):
            threat = state.threats[primary_threat_id]
            intervention = DirectorIntervention(
                id=f"intervention_{len(state.director.interventions) + 1:04d}",
                action="advance_threat",
                reason=f"调查进展为 {exp.progress}，张力仅为 {exp.tension}。",
                world_justification=f"{threat.name}会在玩家调查其他地点时继续发展。",
                affected_entities=[primary_threat_id],
                expected_experience_effect="轻微提高时间压力。",
                effects=[Effect(op="advance_threat", params={"threat_id": primary_threat_id, "amount": 5})],
                applied_at=state.world_time,
            )

        if intervention is not None:
            try:
                self.kernel.apply_effects(state, intervention.effects, source=f"director:{intervention.id}")
                self.kernel.append_event(
                    state,
                    "director_intervention",
                    target=intervention.action,
                    payload={
                        "id": intervention.id,
                        "reason": intervention.reason,
                        "world_justification": intervention.world_justification,
                        "expected_effect": intervention.expected_experience_effect,
                    },
                    visible=False,
                )
            except Exception:
                intervention.valid = False
                raise
            state.director.interventions.append(intervention)
            if intervention.action in {"increase_pressure", "advance_threat"}:
                state.director.experience.success_streak = 0
            elif intervention.action == "offer_respite":
                state.director.experience.failure_streak = 0
            elif intervention.action == "surface_clue":
                state.director.actions_without_progress = 0
                state.director.off_main_streak = 0

        state.director.actions_since_evaluation = 0
        return intervention

    def rank_suggested_actions(self, state: WorldState, actions: list[ActionDefinition]) -> list[ActionDefinition]:
        bias = state.director.affordance_bias
        category_order = {
            "investigate": 0,
            "social": 1,
            "risk": 2,
            "move": 3,
            "other": 4,
        }
        if bias in category_order:
            category_order[bias] = -1
        return sorted(
            actions,
            key=lambda action: (
                category_order.get(action.category, 9),
                0 if action.suggest else 1,
                action.duration_minutes,
                action.id,
            ),
        )
