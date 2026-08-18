from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib
import heapq
import re
from typing import Any

from .agents import Narrator
from .context import remember_visible_beats
from .director import Director
from .keeper import Keeper
from .kernel import KernelValidationError, WorldKernel
from .llm import OpenAICompatibleLLM
from .models import (
    ActionDefinition,
    ActionIntent,
    ActionResolution,
    ActionType,
    CheckOutcome,
    CheckResult,
    Effect,
    EntityType,
    NarrativeBeat,
    NarrativeSequence,
    OpenActionPlan,
    PendingCheck,
    RuleChoice,
    ScenarioDefinition,
    SessionStatus,
    WorldState,
)
from .rules import RuleEngine
from .scene_projection import SceneProjectionAdapter


class GameEngine:
    def __init__(self, scenario: ScenarioDefinition, llm: OpenAICompatibleLLM | None = None):
        self.scenario = scenario
        self.kernel = WorldKernel(scenario)
        self.rules = RuleEngine()
        self.llm = llm or OpenAICompatibleLLM()
        self.keeper = Keeper(self.llm, scenario)
        self.narrator = Narrator(self.llm, scenario)
        self.director = Director(scenario, self.kernel)
        self.scene_projection = SceneProjectionAdapter(scenario, self.kernel)
        self._actions = {action.id: action for action in scenario.actions}
        self._clues = {clue.id: clue for clue in scenario.clues}

    def interpret(self, state: WorldState, *, action_id: str | None = None, text: str | None = None) -> ActionIntent:
        available = self.kernel.available_actions(state)
        if action_id:
            action = next((item for item in available if item.id == action_id), None)
            if action is None:
                return ActionIntent(
                    action_id=None,
                    confidence=0.0,
                    clarification="这个行动当前不可执行。请根据眼前情境选择其他行动。",
                    source="clarification",
                )
            return ActionIntent(
                action_id=action.id,
                action_type=action.type,
                target=action.target,
                confidence=1.0,
                source="button",
            )
        if text and text.strip():
            player_text = text.strip()
            return ActionIntent(
                action_id=None,
                content=player_text,
                goal=player_text,
                confidence=1.0,
                source="player_text",
            )
        return ActionIntent(
            action_id=None,
            confidence=0.0,
            clarification="请输入你想采取的行动。",
            source="clarification",
        )

    def play(
        self,
        state: WorldState,
        *,
        action_id: str | None = None,
        text: str | None = None,
        utterance: str | None = None,
        rule_choice: RuleChoice | str | None = None,
        interactive_rules: bool = False,
        recorded_open_plan: OpenActionPlan | dict[str, Any] | None = None,
    ) -> tuple[WorldState, ActionResolution]:
        working = deepcopy(state)
        if working.pending_check is not None:
            if rule_choice is None:
                return working, ActionResolution(
                    accepted=False,
                    needs_clarification=True,
                    clarification="请先决定：接受失败、消耗幸运，或孤注一掷。",
                    narrative_seed=working.last_narrative,
                    state_version=working.version,
                )
            try:
                choice = RuleChoice(rule_choice)
            except ValueError:
                return working, ActionResolution(
                    accepted=False,
                    needs_clarification=True,
                    clarification="未知的规则选择。",
                    state_version=working.version,
                )
            return self._resolve_pending(working, choice)
        if rule_choice is not None:
            return working, ActionResolution(
                accepted=False,
                needs_clarification=True,
                clarification="当前没有等待处理的检定。",
                state_version=working.version,
            )

        intent = self.interpret(working, action_id=action_id, text=text)
        open_plan: OpenActionPlan | None = None
        if recorded_open_plan is not None:
            open_plan = OpenActionPlan.model_validate(recorded_open_plan)
            intent = ActionIntent(
                action_id=None,
                action_type=open_plan.action_type,
                target=open_plan.target_entity_id,
                content=text or open_plan.goal,
                goal=open_plan.goal,
                confidence=1.0,
                source="open",
            )
        elif intent.action_id is None and intent.source == "player_text":
            decision = self.keeper.adjudicate(
                working,
                intent,
                self.kernel.available_actions(working),
            )
            if decision.existing_action_id is not None:
                action = self._actions[decision.existing_action_id]
                intent = ActionIntent(
                    action_id=action.id,
                    action_type=action.type,
                    target=action.target,
                    content=text,
                    goal=intent.goal,
                    confidence=decision.confidence,
                    source="llm",
                )
            else:
                open_plan = decision.open_plan
                intent = ActionIntent(
                    action_id=None,
                    action_type=open_plan.action_type,
                    target=open_plan.target_entity_id,
                    content=text,
                    goal=open_plan.goal,
                    confidence=decision.confidence,
                    source="llm",
                )

        if open_plan is not None:
            return self._execute_open_plan(
                working,
                intent,
                open_plan,
                text=text,
                interactive_rules=interactive_rules,
            )

        if intent.action_id is None:
            resolution = ActionResolution(
                accepted=False,
                needs_clarification=True,
                clarification=intent.clarification,
                narrative_seed=intent.clarification or "请明确你的行动。",
                state_version=working.version,
            )
            return working, resolution

        action = self._actions[intent.action_id]
        presentation_text = text
        if utterance and action.type in {ActionType.TALK, ActionType.DECEIVE}:
            presentation_text = utterance.strip()
            action = action.model_copy(update={"dialogue_text": presentation_text})
        return self._execute_action(
            working,
            intent,
            action,
            text=presentation_text,
            interactive_rules=interactive_rules,
            validate_predefined=True,
        )

    def _execute_action(
        self,
        working: WorldState,
        intent: ActionIntent,
        action: ActionDefinition,
        *,
        text: str | None,
        interactive_rules: bool,
        validate_predefined: bool,
        open_plan: OpenActionPlan | None = None,
    ) -> tuple[WorldState, ActionResolution]:
        if validate_predefined:
            available, reason = self.kernel.action_is_available(working, action.id)
            if not available:
                return working, ActionResolution(
                    action_id=action.id,
                    accepted=False,
                    needs_clarification=True,
                    clarification=reason,
                    narrative_seed=reason or "行动被世界规则拒绝。",
                    state_version=working.version,
                )

        if open_plan is not None:
            progress_before, location_before, log_start = self._begin_open_action(
                working,
                intent,
                action,
                open_plan,
                text,
                interactive_rules=interactive_rules,
            )
        else:
            progress_before = len(working.discovered_clue_ids)
            location_before = working.entities[working.player.entity_id].location
            log_start = len(working.event_log)
            self.kernel.append_event(
                working,
                "action_started",
                actor=working.player.entity_id,
                target=action.target,
                payload={
                    "action_id": action.id,
                    "intent_source": intent.source,
                    "player_text": text,
                    "interactive_rules": interactive_rules,
                    "open_plan": None,
                },
                visible=False,
            )

        visible_events, interrupting_event = self.kernel.advance_time(
            working,
            action.duration_minutes,
            action_interruptible=True,
        )

        if interrupting_event is not None:
            return self._complete_interrupted(
                working,
                action,
                interrupting_event.id,
                visible_events,
                progress_before,
                location_before,
                log_start,
            )

        check = self._roll_action_check(working, action)
        if interactive_rules and not check.succeeded:
            luck_cost = self.rules.luck_cost(check)
            can_spend_luck = luck_cost is not None and luck_cost <= working.player.luck
            can_push = (
                action.pushable
                and action.type != ActionType.CONFRONT
                and action.opposed is None
                and check.outcome != CheckOutcome.FUMBLE
            )
            if can_spend_luck or can_push:
                choices = [RuleChoice.ACCEPT_FAILURE]
                if can_spend_luck:
                    choices.append(RuleChoice.SPEND_LUCK)
                if can_push:
                    choices.append(RuleChoice.PUSH_ROLL)
                working.pending_check = PendingCheck(
                    action_id=action.id,
                    check=check,
                    luck_cost=luck_cost,
                    can_spend_luck=can_spend_luck,
                    can_push=can_push,
                    progress_before=progress_before,
                    location_before=location_before,
                    dynamic_action=(
                        action
                        if action.id not in self._actions
                        or action.dialogue_text != self._actions[action.id].dialogue_text
                        else None
                    ),
                )
                self.kernel.append_event(
                    working,
                    "rule_choice_offered",
                    actor=working.player.entity_id,
                    target=action.id,
                    payload={
                        "choices": [item.value for item in choices],
                        "luck_cost": luck_cost,
                    },
                    visible=True,
                )
                if action.type in {ActionType.TALK, ActionType.DECEIVE}:
                    working.last_narrative = (
                        "你的话已经说出口。对方的神情有了一瞬变化，"
                        "但真正的回答仍悬在短暂的沉默里。"
                    )
                else:
                    working.last_narrative = (
                        "行动已经推进到结果即将显现的一刻，但最终后果尚未落定。"
                    )
                working.version += 1
                resolution = ActionResolution(
                    action_id=action.id,
                    accepted=True,
                    check=check,
                    awaiting_rule_choice=True,
                    rule_choices=choices,
                    luck_cost=luck_cost,
                    narrative_seed=working.last_narrative,
                    visible_events=visible_events,
                    state_version=working.version,
                )
                working.narrative_sequence = self._build_narrative_sequence(
                    working,
                    action,
                    resolution,
                    location_before=location_before,
                    allow_generation=False,
                )
                remember_visible_beats(working, working.narrative_sequence)
                return working, resolution

        return self._complete_checked(
            working,
            action,
            check,
            visible_events,
            progress_before,
            location_before,
            log_start,
        )

    def _execute_open_plan(
        self,
        state: WorldState,
        intent: ActionIntent,
        plan: OpenActionPlan,
        *,
        text: str | None,
        interactive_rules: bool,
    ) -> tuple[WorldState, ActionResolution]:
        action = self._action_from_open_plan(state, plan)
        if plan.resolution == "impossible":
            return self._execute_impossible_plan(state, intent, action, plan, text=text)
        if plan.action_type in {ActionType.MOVE, ActionType.ESCAPE, ActionType.REST}:
            return self._execute_travel_or_rest(
                state,
                intent,
                action,
                plan,
                text=text,
                interactive_rules=interactive_rules,
            )
        return self._execute_action(
            state,
            intent,
            action,
            text=text,
            interactive_rules=interactive_rules,
            validate_predefined=False,
            open_plan=plan,
        )

    def _action_from_open_plan(self, state: WorldState, plan: OpenActionPlan) -> ActionDefinition:
        digest = hashlib.sha1(f"{state.version}:{plan.goal}".encode("utf-8")).hexdigest()[:12]
        category = {
            ActionType.MOVE: "move",
            ActionType.ESCAPE: "move",
            ActionType.SEARCH: "investigate",
            ActionType.EXAMINE: "investigate",
            ActionType.TALK: "social",
            ActionType.DECEIVE: "social",
            ActionType.CONFRONT: "risk",
            ActionType.FORCE: "risk",
            ActionType.DISRUPT: "risk",
        }.get(plan.action_type, "other")
        target = plan.target_entity_id or plan.destination_entity_id
        return ActionDefinition(
            id=f"open__{digest}",
            label=plan.label,
            type=plan.action_type,
            location=None,
            target=target,
            duration_minutes=plan.duration_minutes,
            skill=plan.skill if plan.resolution == "check" else None,
            difficulty=plan.difficulty,
            pushable=plan.resolution == "check",
            success_text=plan.success_text,
            failure_text=plan.failure_text,
            suggest=False,
            risk=plan.risk,
            category=category,
        )

    def _begin_open_action(
        self,
        state: WorldState,
        intent: ActionIntent,
        action: ActionDefinition,
        plan: OpenActionPlan,
        text: str | None,
        *,
        interactive_rules: bool = False,
    ) -> tuple[int, str | None, int]:
        progress_before = len(state.discovered_clue_ids)
        location_before = state.entities[state.player.entity_id].location
        log_start = len(state.event_log)
        state.flags["open_action_count"] = int(state.flags.get("open_action_count", 0)) + 1
        self.kernel.append_event(
            state,
            "action_started",
            actor=state.player.entity_id,
            target=action.target,
            payload={
                "action_id": action.id,
                "intent_source": intent.source,
                "player_text": text,
                "interactive_rules": interactive_rules,
                "open_plan": plan.model_dump(mode="json"),
            },
            visible=False,
        )
        self.kernel.append_event(
            state,
            "open_plan_committed",
            actor=state.player.entity_id,
            target=action.target,
            payload={
                "label": plan.label,
                "goal": plan.goal,
                "action_type": plan.action_type.value,
                "destination": plan.destination_name or plan.destination_entity_id,
            },
            visible=True,
        )
        return progress_before, location_before, log_start

    def _execute_impossible_plan(
        self,
        state: WorldState,
        intent: ActionIntent,
        action: ActionDefinition,
        plan: OpenActionPlan,
        *,
        text: str | None,
    ) -> tuple[WorldState, ActionResolution]:
        progress_before, location_before, log_start = self._begin_open_action(
            state, intent, action, plan, text
        )
        visible_events, interrupting_event = self.kernel.advance_time(
            state, plan.duration_minutes, action_interruptible=True
        )
        if interrupting_event is not None:
            return self._complete_interrupted(
                state,
                action,
                interrupting_event.id,
                visible_events,
                progress_before,
                location_before,
                log_start,
            )
        check = CheckResult(
            required=False,
            outcome=CheckOutcome.FAILURE,
            succeeded=False,
            difficulty=plan.difficulty,
        )
        return self._complete_checked(
            state,
            action,
            check,
            visible_events,
            progress_before,
            location_before,
            log_start,
        )

    def _execute_travel_or_rest(
        self,
        state: WorldState,
        intent: ActionIntent,
        action: ActionDefinition,
        plan: OpenActionPlan,
        *,
        text: str | None,
        interactive_rules: bool,
    ) -> tuple[WorldState, ActionResolution]:
        if plan.resolution == "check":
            destination_id = self._resolve_destination(state, plan)
            current_id = state.entities[state.player.entity_id].location
            route = (
                self._shortest_route(state, current_id, destination_id)
                if destination_id
                else self._route_to_open_world(state, current_id)
                if plan.destination_name
                else []
            )
            if current_id != destination_id and route is None:
                action.failure_text = self._blocked_travel_text(plan)
                return self._execute_impossible_plan(state, intent, action, plan, text=text)
            action.success_effects = self._destination_effects(state, plan, destination_id)
            return self._execute_action(
                state,
                intent,
                action,
                text=text,
                interactive_rules=interactive_rules,
                validate_predefined=False,
                open_plan=plan,
            )

        progress_before, location_before, log_start = self._begin_open_action(
            state, intent, action, plan, text
        )
        visible_events: list = []
        destination_id = self._resolve_destination(state, plan)
        current_id = state.entities[state.player.entity_id].location
        route = self._shortest_route(state, current_id, destination_id) if destination_id else []
        if destination_id is not None and current_id != destination_id and not route:
            route = None
        if destination_id is None and plan.destination_name:
            route = self._route_to_open_world(state, current_id)

        if route is None:
            action.failure_text = self._blocked_travel_text(plan)
            visible_events, interrupting_event = self.kernel.advance_time(
                state,
                min(5, max(1, plan.duration_minutes)),
                action_interruptible=True,
            )
            if interrupting_event is not None:
                return self._complete_interrupted(
                    state,
                    action,
                    interrupting_event.id,
                    visible_events,
                    progress_before,
                    location_before,
                    log_start,
                )
            check = CheckResult(
                required=False,
                outcome=CheckOutcome.FAILURE,
                succeeded=False,
                difficulty=plan.difficulty,
            )
            return self._complete_checked(
                state,
                action,
                check,
                visible_events,
                progress_before,
                location_before,
                log_start,
            )

        traveled_minutes = 0
        for next_location, minutes in route:
            events, interrupting_event = self.kernel.advance_time(
                state, minutes, action_interruptible=True
            )
            visible_events.extend(events)
            if interrupting_event is not None:
                return self._complete_interrupted(
                    state,
                    action,
                    interrupting_event.id,
                    visible_events,
                    progress_before,
                    location_before,
                    log_start,
                )
            self.kernel.apply_effect(
                state,
                Effect(op="move_entity", params={"entity_id": state.player.entity_id, "destination": next_location}),
                source=f"open_plan:{action.id}:route",
            )
            traveled_minutes += minutes

        travel_budget = (
            0
            if plan.action_type == ActionType.REST
            and not (plan.destination_name or plan.destination_entity_id)
            else plan.duration_minutes
        )
        remaining = max(0, travel_budget - traveled_minutes)
        if remaining:
            events, interrupting_event = self.kernel.advance_time(
                state, remaining, action_interruptible=True
            )
            visible_events.extend(events)
            if interrupting_event is not None:
                return self._complete_interrupted(
                    state,
                    action,
                    interrupting_event.id,
                    visible_events,
                    progress_before,
                    location_before,
                    log_start,
                )

        if plan.destination_name or plan.destination_entity_id:
            destination_id = destination_id or self._create_dynamic_location(state, plan, action.id)
            if state.entities[state.player.entity_id].location != destination_id:
                self.kernel.apply_effect(
                    state,
                    Effect(op="move_entity", params={"entity_id": state.player.entity_id, "destination": destination_id}),
                    source=f"open_plan:{action.id}:destination",
                )
            action.target = destination_id

        if plan.action_type == ActionType.REST:
            rest_minutes = self._rest_minutes(state, plan)
            if rest_minutes:
                events, interrupting_event = self.kernel.advance_time(
                    state, rest_minutes, action_interruptible=True
                )
                visible_events.extend(events)
                if interrupting_event is not None:
                    return self._complete_interrupted(
                        state,
                        action,
                        interrupting_event.id,
                        visible_events,
                        progress_before,
                        location_before,
                        log_start,
                    )
            self.kernel.apply_effects(
                state,
                [
                    Effect(op="modify_player", params={"field": "hp", "delta": 1}),
                    Effect(op="modify_player", params={"field": "stress", "delta": -2}),
                    Effect(
                        op="modify_player",
                        params={
                            "field": "magic_points",
                            "set": max(0, int(state.player.characteristics.get("pow", 0)) // 5),
                        },
                    ),
                ],
                source=f"open_plan:{action.id}:rest",
            )
            self.kernel.append_event(
                state,
                "rest_completed",
                actor=state.player.entity_id,
                payload={"minutes": rest_minutes, "goal": plan.goal},
                visible=True,
            )

        check = CheckResult(
            required=False,
            outcome=CheckOutcome.AUTOMATIC,
            succeeded=True,
            difficulty=plan.difficulty,
        )
        return self._complete_checked(
            state,
            action,
            check,
            visible_events,
            progress_before,
            location_before,
            log_start,
        )

    def _resolve_destination(self, state: WorldState, plan: OpenActionPlan) -> str | None:
        if plan.destination_entity_id:
            entity = state.entities.get(plan.destination_entity_id)
            if entity is not None and entity.type == EntityType.LOCATION:
                return entity.id
        if not plan.destination_name:
            return None
        wanted = plan.destination_name.strip().lower()
        exact = [
            entity.id
            for entity in state.entities.values()
            if entity.type == EntityType.LOCATION and entity.name.strip().lower() == wanted
        ]
        if exact:
            return exact[0]
        partial = [
            entity.id
            for entity in state.entities.values()
            if entity.type == EntityType.LOCATION
            and (wanted in entity.name.lower() or entity.name.lower() in wanted)
        ]
        return partial[0] if len(partial) == 1 else None

    def _destination_effects(
        self,
        state: WorldState,
        plan: OpenActionPlan,
        destination_id: str | None,
    ) -> list[Effect]:
        effects: list[Effect] = []
        resolved_id = destination_id
        if resolved_id is None and plan.destination_name:
            resolved_id = self._dynamic_location_id(plan.destination_name)
            if resolved_id not in state.entities:
                effects.append(
                    Effect(
                        op="create_entity",
                        params={"entity": self._dynamic_location_payload(plan, resolved_id)},
                    )
                )
        if resolved_id:
            effects.append(
                Effect(op="move_entity", params={"entity_id": state.player.entity_id, "destination": resolved_id})
            )
        return effects

    @staticmethod
    def _dynamic_location_id(name: str) -> str:
        digest = hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:12]
        return f"dynamic_location_{digest}"

    def _dynamic_location_payload(
        self,
        plan: OpenActionPlan,
        location_id: str,
    ) -> dict[str, Any]:
        name = plan.destination_name or "新的地点"
        return {
            "id": location_id,
            "type": EntityType.LOCATION.value,
            "name": name,
            "location": None,
            "attributes": {
                "description": plan.destination_description
                or f"这是玩家选择前往的地点：{name}。",
                "origin": "open_action",
            },
            "tags": ["off_main", "safe" if plan.risk == "safe" else "uncertain"],
            "active": True,
        }

    def _create_dynamic_location(
        self,
        state: WorldState,
        plan: OpenActionPlan,
        action_id: str,
    ) -> str:
        location_id = self._dynamic_location_id(plan.destination_name or plan.goal)
        if location_id not in state.entities:
            self.kernel.apply_effect(
                state,
                Effect(
                    op="create_entity",
                    params={"entity": self._dynamic_location_payload(plan, location_id)},
                ),
                source=f"open_plan:{action_id}:create_location",
            )
        return location_id

    def _shortest_route(
        self,
        state: WorldState,
        origin: str | None,
        destination: str | None,
    ) -> list[tuple[str, int]]:
        if not origin or not destination or origin == destination:
            return []
        queue: list[tuple[int, str, list[tuple[str, int]]]] = [(0, origin, [])]
        best: dict[str, int] = {origin: 0}
        while queue:
            cost, current, path = heapq.heappop(queue)
            if current == destination:
                return path
            if cost != best.get(current):
                continue
            for next_location, minutes in self.scenario.location_graph.get(current, {}).items():
                edge = f"{current}->{next_location}"
                requirements = self.scenario.movement_requirements.get(edge, [])
                if not self.kernel.all_conditions_met(state, requirements):
                    continue
                next_cost = cost + minutes
                if next_cost < best.get(next_location, 10**9):
                    best[next_location] = next_cost
                    heapq.heappush(
                        queue,
                        (next_cost, next_location, [*path, (next_location, minutes)]),
                    )
        return []

    def _route_to_open_world(
        self,
        state: WorldState,
        origin: str | None,
    ) -> list[tuple[str, int]] | None:
        if not origin or origin not in state.entities:
            return None
        current = state.entities[origin]
        if current.tags & {"outside", "exit", "city", "safe", "off_main"}:
            return []
        candidates = [
            entity.id
            for entity in state.entities.values()
            if entity.type == EntityType.LOCATION
            and entity.tags & {"outside", "exit", "city", "safe"}
        ]
        routes = [self._shortest_route(state, origin, candidate) for candidate in candidates]
        routes = [route for route in routes if route]
        return min(routes, key=lambda route: sum(minutes for _, minutes in route), default=None)

    @staticmethod
    def _blocked_travel_text(plan: OpenActionPlan) -> str:
        destination = plan.destination_name or "那里"
        return f"你尝试前往{destination}，但眼前的路线或阻碍暂时让你无法抵达。"

    @staticmethod
    def _rest_minutes(state: WorldState, plan: OpenActionPlan) -> int:
        if plan.rest_until_hour is None:
            return 0 if plan.destination_name else plan.duration_minutes
        target = state.world_time.replace(
            hour=plan.rest_until_hour,
            minute=0,
            second=0,
            microsecond=0,
        ) + timedelta(days=plan.rest_day_offset)
        if target <= state.world_time:
            target += timedelta(days=1)
        return max(0, round((target - state.world_time).total_seconds() / 60))

    def _roll_action_check(
        self,
        state: WorldState,
        action: ActionDefinition,
        *,
        pushed: bool = False,
    ) -> CheckResult:
        bonus_dice = action.bonus_dice
        labels: list[str] = []
        for modifier in action.modifiers:
            if self.kernel.all_conditions_met(state, modifier.conditions):
                bonus_dice += modifier.dice
                labels.append(modifier.label)
        if state.player.major_wound:
            bonus_dice -= 1
            labels.append("重伤")
        if state.player.temporary_insanity_until is not None or state.player.indefinite_insanity:
            bonus_dice -= 1
            labels.append("疯狂发作")
        bonus_dice = max(-2, min(2, bonus_dice))
        check = self.rules.check(
            state,
            action.skill,
            action.difficulty,
            action.label,
            bonus_dice=bonus_dice,
            modifier_labels=labels,
            pushed=pushed,
        )
        if action.opposed is not None and check.required:
            check = self.rules.opposed_check(state, check, action.opposed)
        return check

    def _resolve_pending(
        self,
        state: WorldState,
        choice: RuleChoice,
    ) -> tuple[WorldState, ActionResolution]:
        pending = state.pending_check
        assert pending is not None
        action = pending.dynamic_action or self._actions[pending.action_id]
        check = pending.check

        def record_choice() -> None:
            self.kernel.append_event(
                state,
                "rule_choice_made",
                actor=state.player.entity_id,
                target=action.id,
                payload={"choice": choice.value},
                visible=True,
            )

        if choice == RuleChoice.SPEND_LUCK:
            if not pending.can_spend_luck or pending.luck_cost is None:
                return state, ActionResolution(
                    action_id=action.id,
                    accepted=False,
                    needs_clarification=True,
                    clarification="这次检定不能用幸运改写。",
                    state_version=state.version,
                )
            try:
                check = self.rules.spend_luck(state, check, pending.luck_cost)
            except ValueError:
                return state, ActionResolution(
                    action_id=action.id,
                    accepted=False,
                    needs_clarification=True,
                    clarification="当前幸运不足，无法改写结果。",
                    state_version=state.version,
                )
            record_choice()
            state.pending_check = None
            return self._complete_checked(
                state,
                action,
                check,
                [],
                pending.progress_before,
                pending.location_before,
                len(state.event_log) - 1,
                continues_previous_narrative=True,
            )

        if choice == RuleChoice.PUSH_ROLL:
            if not pending.can_push:
                return state, ActionResolution(
                    action_id=action.id,
                    accepted=False,
                    needs_clarification=True,
                    clarification="这次检定不能孤注一掷。",
                    state_version=state.version,
                )
            record_choice()
            visible_events, interrupting_event = self.kernel.advance_time(
                state,
                action.duration_minutes,
                action_interruptible=True,
            )
            state.pending_check = None
            if interrupting_event is not None:
                return self._complete_interrupted(
                    state,
                    action,
                    interrupting_event.id,
                    visible_events,
                    pending.progress_before,
                    pending.location_before,
                    len(state.event_log),
                    continues_previous_narrative=True,
                )
            check = self._roll_action_check(state, action, pushed=True)
            return self._complete_checked(
                state,
                action,
                check,
                visible_events,
                pending.progress_before,
                pending.location_before,
                len(state.event_log) - 1,
                pushed_failure=not check.succeeded,
                continues_previous_narrative=True,
            )

        record_choice()
        state.pending_check = None
        return self._complete_checked(
            state,
            action,
            check,
            [],
            pending.progress_before,
            pending.location_before,
            len(state.event_log) - 1,
            continues_previous_narrative=True,
        )

    def _complete_checked(
        self,
        state: WorldState,
        action: ActionDefinition,
        check: CheckResult,
        visible_events: list,
        progress_before: int,
        location_before: str | None,
        log_start: int,
        *,
        pushed_failure: bool = False,
        continues_previous_narrative: bool = False,
    ) -> tuple[WorldState, ActionResolution]:
        effects = action.success_effects if check.succeeded else action.failure_effects
        sanity_start = len(state.sanity_checks)
        try:
            self.kernel.apply_effects(state, effects, source=f"action:{action.id}")
            if pushed_failure:
                push_effects = action.push_failure_effects or self._default_push_failure_effects(action)
                self.kernel.apply_effects(state, push_effects, source=f"action:{action.id}:pushed_failure")
                self.kernel.append_event(
                    state,
                    "pushed_roll_failed",
                    target=action.id,
                    payload={"risk": action.risk},
                    visible=True,
                )
            self.kernel.apply_effects(state, action.always_effects, source=f"action:{action.id}:always")
            self._apply_action_sanity(state, action, check.succeeded)
        except KernelValidationError:
            return state, ActionResolution(
                action_id=action.id,
                accepted=False,
                narrative_seed="世界规则拒绝了这个行动产生的非法副作用。",
                state_version=state.version,
            )

        if check.succeeded or action.complete_on_attempt:
            state.completed_actions.add(action.id)
        self.kernel.append_event(
            state,
            "action_resolved",
            actor=state.player.entity_id,
            target=action.target,
            payload={
                "action_id": action.id,
                "outcome": check.outcome.value,
                "succeeded": check.succeeded,
                "roll": check.roll,
                "target": check.target,
                "bonus_dice": check.bonus_dice,
                "pushed": check.pushed,
                "luck_spent": check.luck_spent,
            },
            visible=True,
        )
        failure_text = action.push_failure_text if pushed_failure and action.push_failure_text else action.failure_text
        resolution = ActionResolution(
            action_id=action.id,
            accepted=True,
            check=check,
            sanity_check=state.sanity_checks[-1] if len(state.sanity_checks) > sanity_start else None,
            narrative_seed=action.success_text if check.succeeded else failure_text,
            visible_events=visible_events,
            continues_previous_narrative=continues_previous_narrative,
            state_version=state.version + 1,
        )
        observed_outcome = check.outcome if check.succeeded else (
            CheckOutcome.FUMBLE if check.outcome == CheckOutcome.FUMBLE else CheckOutcome.FAILURE
        )
        self.director.observe_action(
            state,
            action,
            observed_outcome,
            progress_before=progress_before,
            location_before=location_before,
        )
        return self._finish_action(state, action, resolution, location_before, log_start)

    def _complete_interrupted(
        self,
        state: WorldState,
        action: ActionDefinition,
        interrupting_event_id: str,
        visible_events: list,
        progress_before: int,
        location_before: str | None,
        log_start: int,
        *,
        continues_previous_narrative: bool = False,
    ) -> tuple[WorldState, ActionResolution]:
        check = CheckResult(required=False, outcome=CheckOutcome.INTERRUPTED, succeeded=False)
        self.kernel.append_event(
            state,
            "action_interrupted",
            actor=state.player.entity_id,
            target=action.id,
            payload={"scheduled_event_id": interrupting_event_id},
            visible=True,
        )
        resolution = ActionResolution(
            action_id=action.id,
            accepted=True,
            check=check,
            interrupted=True,
            interrupting_event_id=interrupting_event_id,
            narrative_seed=action.interrupted_text,
            visible_events=visible_events,
            continues_previous_narrative=continues_previous_narrative,
            state_version=state.version + 1,
        )
        self.director.observe_action(
            state,
            action,
            CheckOutcome.INTERRUPTED,
            progress_before=progress_before,
            location_before=location_before,
        )
        return self._finish_action(state, action, resolution, location_before, log_start)

    def _apply_action_sanity(self, state: WorldState, action: ActionDefinition, succeeded: bool) -> None:
        definition = action.sanity_check
        if definition is None:
            return
        if definition.on == "success" and not succeeded:
            return
        if definition.on == "failure" and succeeded:
            return
        flag = f"sanity_seen:{action.id}"
        if definition.once and state.flags.get(flag):
            return
        self.kernel.apply_effect(
            state,
            Effect(
                op="sanity_check",
                params={
                    "success_loss": definition.success_loss,
                    "failure_loss": definition.failure_loss,
                    "reason": definition.reason,
                },
            ),
            source=f"action:{action.id}:sanity",
        )
        if definition.once:
            state.flags[flag] = True

    def _default_push_failure_effects(self, action: ActionDefinition) -> list[Effect]:
        threat_id = self.scenario.director_config.primary_threat_id
        if threat_id:
            amount = 10 if action.risk == "dangerous" else 5
            return [Effect(op="advance_threat", params={"threat_id": threat_id, "amount": amount})]
        return [Effect(op="modify_player", params={"field": "stress", "delta": 1})]

    def _finish_action(
        self,
        state: WorldState,
        action: ActionDefinition,
        resolution: ActionResolution,
        location_before: str | None,
        log_start: int,
    ) -> tuple[WorldState, ActionResolution]:
        location_after = state.entities[state.player.entity_id].location
        new_events = state.event_log[log_start:]
        major_event = any(
            event.type in {
                "threat_threshold_crossed",
                "hospital_power_failed",
                "ritual_begins",
            }
            for event in new_events
        )
        if self.director.should_evaluate(
            state,
            major_event=major_event,
            scene_changed=location_after != location_before,
        ):
            resolution.director_intervention = self.director.decide(state)

        self.kernel.evaluate_ending(state)
        if state.status != SessionStatus.ACTIVE:
            ending = next((item for item in self.scenario.endings if item.id == state.ending_id), None)
            if ending is not None:
                state.last_narrative = ending.narrative
                resolution.narrative_seed = ending.narrative
        else:
            state.last_narrative = resolution.narrative_seed

        state.version += 1
        resolution.state_version = state.version
        state.narrative_sequence = self._build_narrative_sequence(
            state,
            action,
            resolution,
            location_before=location_before,
        )
        remember_visible_beats(state, state.narrative_sequence)
        return state, resolution

    @staticmethod
    def _split_narrative(text: str, *, limit: int = 4) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        if len(paragraphs) == 1 and len(paragraphs[0]) > 90:
            sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])", paragraphs[0]) if part.strip()]
            paragraphs = []
            current = ""
            for sentence in sentences:
                if current and len(current) + len(sentence) > 95:
                    paragraphs.append(current)
                    current = sentence
                else:
                    current += sentence
            if current:
                paragraphs.append(current)
        if len(paragraphs) <= limit:
            return paragraphs
        return [*paragraphs[: limit - 1], "\n\n".join(paragraphs[limit - 1 :])]

    def prepare_opening_sequence(self, state: WorldState) -> None:
        if state.narrative_sequence is not None:
            return
        sequence_id = f"narrative_{state.version:06d}_opening"
        beats = [
            NarrativeBeat(
                id=f"{sequence_id}_beat_{index:02d}",
                text=text,
                source="authored",
                skippable=index > 1,
            )
            for index, text in enumerate(self._split_narrative(state.last_narrative), start=1)
        ]
        state.narrative_sequence = NarrativeSequence(
            id=sequence_id,
            state_version=state.version,
            status="ready",
            beats=beats,
            canonical_seed=state.last_narrative,
            created_at=state.world_time,
        )
        remember_visible_beats(state, state.narrative_sequence)

    def _build_narrative_sequence(
        self,
        state: WorldState,
        action: ActionDefinition,
        resolution: ActionResolution,
        *,
        location_before: str | None,
        allow_generation: bool = True,
    ) -> NarrativeSequence:
        sequence_id = f"narrative_{state.version:06d}_{len(state.event_log):05d}"
        raw_beats: list[tuple[str, str, bool]] = []

        if (
            not resolution.continues_previous_narrative
            and action.type in {ActionType.TALK, ActionType.DECEIVE}
            and action.dialogue_text
        ):
            raw_beats.append((action.dialogue_text.strip(), "authored", False))

        for event in resolution.visible_events:
            text = str(event.payload.get("text", "")).strip()
            if text:
                raw_beats.append((text, "event", False))

        performance_beats: list[str] = []
        if not resolution.interrupted and not resolution.awaiting_rule_choice:
            succeeded = resolution.check.succeeded if resolution.check else resolution.accepted
            performance_beats = action.success_beats if succeeded else action.failure_beats
        narrative_parts = performance_beats or self._split_narrative(resolution.narrative_seed)
        narrative_source = (
            "keeper"
            if action.id.startswith("open__") and state.status == SessionStatus.ACTIVE
            else "authored"
        )
        for text in narrative_parts:
            raw_beats.append((text, narrative_source, False))

        location_after = state.entities[state.player.entity_id].location
        if location_after and location_after != location_before:
            location = state.entities.get(location_after)
            if location is not None:
                description = str(location.attributes.get("description", "")).strip()
                if description:
                    raw_beats.append((description, "scene", True))
                arrival_count = sum(
                    1
                    for event in state.event_log
                    if event.type == "entity_moved"
                    and event.actor == state.player.entity_id
                    and event.target == location_after
                )
                entry_beats = location.attributes.get("entry_beats", []) if arrival_count <= 1 else []
                if isinstance(entry_beats, list):
                    raw_beats.extend(
                        (str(text).strip(), "scene", True)
                        for text in entry_beats
                        if str(text).strip()
                    )

        intervention = resolution.director_intervention
        if intervention is not None and intervention.world_justification.strip():
            raw_beats.append((intervention.world_justification.strip(), "director", True))

        seen: set[str] = set()
        beats: list[NarrativeBeat] = []
        for text, source, skippable in raw_beats:
            normalized = re.sub(r"\s+", "", text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            beats.append(
                NarrativeBeat(
                    id=f"{sequence_id}_beat_{len(beats) + 1:02d}",
                    text=text,
                    source=source,
                    skippable=skippable,
                )
            )

        if not beats:
            beats.append(
                NarrativeBeat(
                    id=f"{sequence_id}_beat_01",
                    text=state.last_narrative or "世界短暂地安静下来。",
                    source="system",
                    skippable=False,
                )
            )

        player_text = next(
            (
                str(event.payload.get("player_text")).strip()
                for event in reversed(state.event_log)
                if event.type == "action_started"
                and event.payload.get("action_id") == action.id
                and event.payload.get("player_text")
            ),
            None,
        )
        pending = allow_generation and self.llm.enabled and state.status == SessionStatus.ACTIVE
        return NarrativeSequence(
            id=sequence_id,
            state_version=state.version,
            action_id=action.id,
            action_label=action.label,
            action_type=action.type,
            player_text=player_text,
            continues_previous=resolution.continues_previous_narrative,
            status="pending" if pending else "ready",
            beats=beats,
            canonical_seed=resolution.narrative_seed,
            mechanical_result=resolution.check.model_dump(mode="json") if resolution.check else None,
            created_at=state.world_time,
        )

    def _suggested(
        self,
        state: WorldState,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[ActionDefinition]:
        if state.status != SessionStatus.ACTIVE:
            return []
        excluded = exclude_ids or set()
        candidates = [
            action
            for action in self.kernel.available_actions(state)
            if action.suggest and action.id not in excluded
        ]
        ranked = self.director.rank_suggested_actions(state, candidates)
        selected: list[ActionDefinition] = []
        used_categories: set[str] = set()
        for action in ranked:
            if action.category not in used_categories:
                selected.append(action)
                used_categories.add(action.category)
            if len(selected) == 3:
                return selected
        for action in ranked:
            if action not in selected:
                selected.append(action)
            if len(selected) == 3:
                break
        return selected

    def _dialogue_options(
        self,
        state: WorldState,
        present_npcs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if state.status != SessionStatus.ACTIVE or not present_npcs:
            return []
        npc_names = {npc["id"]: npc["name"] for npc in present_npcs}
        actions = [
            action
            for action in self.kernel.available_actions(state)
            if action.suggest
            and action.type in {ActionType.TALK, ActionType.DECEIVE}
            and action.target in npc_names
        ]
        options = [
            {
                "id": f"dialogue__{action.id}",
                "action_id": action.id,
                "text": action.dialogue_text
                or f"“{npc_names[action.target]}，关于这件事，我想听你亲口说明。”",
                "target_name": npc_names[action.target],
                "duration_minutes": action.duration_minutes,
            }
            for action in actions
        ]
        if actions and len(options) < 3:
            anchor = actions[0]
            name = npc_names[anchor.target]
            variants = [
                f"“{name}，先别急。请把你亲眼看到的事情告诉我。”",
                f"“{name}，我愿意听。请从最早让你觉得不对劲的地方说起。”",
            ]
            for index, text in enumerate(variants, start=1):
                if len(options) >= 3:
                    break
                options.append(
                    {
                        "id": f"dialogue__{anchor.id}__variant_{index}",
                        "action_id": anchor.id,
                        "text": text,
                        "target_name": name,
                        "duration_minutes": anchor.duration_minutes,
                    }
                )
        return options[:3]

    def public_view(self, state: WorldState, resolution: ActionResolution | None = None) -> dict[str, Any]:
        player_entity = state.entities[state.player.entity_id]
        location = state.entities[player_entity.location] if player_entity.location else None
        present_npcs = [
            {"id": entity.id, "name": entity.name, "role": entity.attributes.get("role")}
            for entity in state.entities.values()
            if entity.id != state.player.entity_id
            and entity.type.value == "NPC"
            and entity.active
            and entity.location == player_entity.location
        ]
        discovered = [
            {
                "id": clue_id,
                "title": self._clues[clue_id].title,
                "description": self._clues[clue_id].description,
            }
            for clue_id in sorted(state.discovered_clue_ids)
            if clue_id in self._clues
        ]
        check = resolution.check.model_dump(mode="json") if resolution and resolution.check else None
        conditions: list[str] = []
        if state.player.major_wound:
            conditions.append("重伤")
        if state.player.unconscious:
            conditions.append("昏迷")
        if state.player.dying:
            conditions.append("濒死")
        if state.player.temporary_insanity_until is not None:
            conditions.append("临时疯狂")
        if state.player.indefinite_insanity:
            conditions.append("不定期疯狂")
        pending = state.pending_check
        rule_prompt = None
        if pending is not None:
            choices = [RuleChoice.ACCEPT_FAILURE]
            if pending.can_spend_luck:
                choices.append(RuleChoice.SPEND_LUCK)
            if pending.can_push:
                choices.append(RuleChoice.PUSH_ROLL)
            rule_prompt = {
                "action_id": pending.action_id,
                "check": pending.check.model_dump(mode="json"),
                "choices": [item.value for item in choices],
                "luck_cost": pending.luck_cost,
            }
        sequence = state.narrative_sequence
        if sequence is None or sequence.state_version != state.version:
            fallback_id = f"narrative_{state.version:06d}_legacy"
            public_sequence = {
                "id": fallback_id,
                "state_version": state.version,
                "status": "ready",
                "beats": [
                    {
                        "id": f"{fallback_id}_beat_01",
                        "text": state.last_narrative,
                        "source": "authored",
                        "skippable": False,
                    }
                ],
            }
        else:
            public_sequence = {
                "id": sequence.id,
                "state_version": sequence.state_version,
                "status": sequence.status,
                "beats": [beat.model_dump(mode="json") for beat in sequence.beats],
            }
        dialogue_options = self._dialogue_options(state, present_npcs)
        dialogue_action_ids = {
            option["action_id"] for option in dialogue_options if option.get("action_id")
        }
        return {
            "session_id": state.session_id,
            "scenario": {
                "id": self.scenario.id,
                "title": self.scenario.title,
                "subtitle": self.scenario.subtitle,
                "presentation": self.scenario.presentation.model_dump(mode="json"),
                "source": self.scenario.source.model_dump(mode="json") if self.scenario.source else None,
                "ruleset": "coc7_quickstart_subset_v1",
                "narrative_mode": "async_beats_v1",
            },
            "version": state.version,
            "status": state.status.value,
            "ending_id": state.ending_id,
            "world_time": state.world_time.isoformat(),
            "time_label": state.world_time.strftime("%H:%M"),
            "narrative": state.last_narrative,
            "narrative_sequence": public_sequence,
            "scene": {
                "id": location.id if location else None,
                "name": location.name if location else "未知地点",
                "description": location.attributes.get("description", "") if location else "",
                "present_npcs": present_npcs,
                "visual": self.scene_projection.project(state),
            },
            "player": {
                "name": state.player.name,
                "hp": state.player.hp,
                "max_hp": state.player.max_hp,
                "stress": state.player.stress,
                "max_stress": state.player.max_stress,
                "sanity": state.player.sanity,
                "max_sanity": state.player.max_sanity,
                "luck": state.player.luck,
                "max_luck": state.player.max_luck,
                "magic_points": state.player.magic_points,
                "characteristics": state.player.characteristics,
                "skills": state.player.skills,
                "conditions": conditions,
                "bout_of_madness": state.player.bout_of_madness,
                "inventory": [state.entities[item_id].name for item_id in state.player.inventory if item_id in state.entities],
            },
            "clues": discovered,
            "dialogue_options": dialogue_options,
            "suggested_actions": [
                {
                    "id": action.id,
                    "label": action.label,
                    "type": action.type.value,
                    "category": action.category,
                    "risk": action.risk,
                    "duration_minutes": action.duration_minutes,
                }
                for action in self._suggested(state, exclude_ids=dialogue_action_ids)
            ],
            "rule_prompt": rule_prompt,
            "last_resolution": {
                "accepted": resolution.accepted,
                "needs_clarification": resolution.needs_clarification,
                "clarification": resolution.clarification,
                "interrupted": resolution.interrupted,
                "check": check,
                "sanity_check": resolution.sanity_check.model_dump(mode="json") if resolution.sanity_check else None,
                "awaiting_rule_choice": resolution.awaiting_rule_choice,
                "rule_choices": [item.value for item in resolution.rule_choices],
                "luck_cost": resolution.luck_cost,
            }
            if resolution
            else None,
        }

    def developer_view(self, state: WorldState) -> dict[str, Any]:
        return {
            "session_id": state.session_id,
            "version": state.version,
            "world_time": state.world_time.isoformat(),
            "world_map": self.scene_projection.developer_world_projection(state),
            "director": state.director.model_dump(mode="json"),
            "threats": [clock.model_dump(mode="json") for clock in state.threats.values()],
            "event_queue": [event.model_dump(mode="json") for event in state.event_queue],
            "npc_locations": [
                {"id": entity.id, "name": entity.name, "location": entity.location, "active": entity.active}
                for entity in state.entities.values()
                if entity.type.value == "NPC"
            ],
            "known_facts": [
                state.facts[fact_id].model_dump(mode="json")
                for fact_id in sorted(state.player_known_fact_ids)
                if fact_id in state.facts
            ],
            "canonical_facts": [fact.model_dump(mode="json") for fact in state.facts.values()],
            "npc_knowledge": [entry.model_dump(mode="json") for entry in state.npc_knowledge],
            "event_log": [event.model_dump(mode="json") for event in state.event_log[-60:]],
            "rolls": [roll.model_dump(mode="json") for roll in state.rolls],
            "sanity_checks": [item.model_dump(mode="json") for item in state.sanity_checks],
            "damage_log": [item.model_dump(mode="json") for item in state.damage_log],
            "pending_check": state.pending_check.model_dump(mode="json") if state.pending_check else None,
            "narrative_sequence": (
                state.narrative_sequence.model_dump(mode="json")
                if state.narrative_sequence
                else None
            ),
            "visible_history": [
                entry.model_dump(mode="json") for entry in state.visible_history[-20:]
            ],
            "agent_calls": [call.model_dump(mode="json") for call in state.agent_calls],
        }
