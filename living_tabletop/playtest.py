from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Literal

from .engine import GameEngine
from .llm import LLMSettings, OpenAICompatibleLLM
from .models import ActionDefinition, ActionType, ScenarioDefinition, SessionStatus, WorldState
from .scenario import create_initial_state


class PlayStrategy(StrEnum):
    INVESTIGATOR = "investigator"
    SOCIAL = "social"
    BOLD = "bold"
    EXPLORER = "explorer"
    RANDOM = "random"


@dataclass(slots=True)
class PlaytestIssue:
    severity: str
    code: str
    message: str
    turn: int | None = None
    action_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScriptedStep:
    action_id: str
    until: Literal["success", "attempt"] = "success"


@dataclass(slots=True)
class TraceStep:
    turn: int
    action_id: str
    action_label: str
    location_before: str | None
    location_after: str | None
    world_time: str
    accepted: bool
    interrupted: bool
    outcome: str
    hp: int
    stress: int
    sanity: int
    luck: int
    major_wound: bool
    clue_count: int
    status: str
    ending_id: str | None
    director_action: str | None


@dataclass(slots=True)
class PlaytestRun:
    name: str
    scenario_id: str
    seed: int
    strategy: str
    terminal_required: bool
    steps: list[TraceStep] = field(default_factory=list)
    issues: list[PlaytestIssue] = field(default_factory=list)
    final_status: str = SessionStatus.ACTIVE.value
    ending_id: str | None = None
    discovered_clues: list[str] = field(default_factory=list)
    visited_locations: list[str] = field(default_factory=list)
    action_coverage: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    sanity_check_count: int = 0
    sanity_loss: int = 0
    damage_event_count: int = 0
    major_wound_seen: bool = False

    @property
    def errors(self) -> list[PlaytestIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["error_count"] = len(self.errors)
        return payload


@dataclass(slots=True)
class PlaytestReport:
    scenario_id: str
    runs: list[PlaytestRun]
    authored_action_count: int
    covered_authored_actions: list[str]
    uncovered_authored_actions: list[str]
    endings_seen: dict[str, int]
    statuses_seen: dict[str, int]
    roll_outcomes_seen: dict[str, int]
    director_actions_seen: dict[str, int]
    sanity_check_count: int
    sanity_loss: int
    damage_event_count: int
    major_wound_runs: int

    @property
    def error_count(self) -> int:
        return sum(len(run.errors) for run in self.runs)

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for run in self.runs
            for issue in run.issues
            if issue.severity == "warning"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_count": len(self.runs),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "authored_action_count": self.authored_action_count,
            "covered_authored_actions": self.covered_authored_actions,
            "uncovered_authored_actions": self.uncovered_authored_actions,
            "endings_seen": self.endings_seen,
            "statuses_seen": self.statuses_seen,
            "roll_outcomes_seen": self.roll_outcomes_seen,
            "director_actions_seen": self.director_actions_seen,
            "sanity_check_count": self.sanity_check_count,
            "sanity_loss": self.sanity_loss,
            "damage_event_count": self.damage_event_count,
            "major_wound_runs": self.major_wound_runs,
            "runs": [run.to_dict() for run in self.runs],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Playtest report: `{self.scenario_id}`",
            "",
            f"- Runs: {len(self.runs)}",
            f"- Errors: {self.error_count}",
            f"- Warnings: {self.warning_count}",
            f"- Authored action coverage: {len(self.covered_authored_actions)}/{self.authored_action_count}",
            f"- Endings: {self.endings_seen}",
            f"- Roll outcomes: {self.roll_outcomes_seen}",
            f"- Director interventions: {self.director_actions_seen}",
            f"- SAN checks / loss: {self.sanity_check_count} / {self.sanity_loss}",
            f"- Damage events / runs with major wounds: {self.damage_event_count} / {self.major_wound_runs}",
            "",
            "| Run | Strategy | Seed | Turns | Status | Ending | Issues |",
            "| --- | --- | ---: | ---: | --- | --- | ---: |",
        ]
        for run in self.runs:
            lines.append(
                f"| {run.name} | {run.strategy} | {run.seed} | {len(run.steps)} | "
                f"{run.final_status} | {run.ending_id or '-'} | {len(run.issues)} |"
            )
        if self.uncovered_authored_actions:
            lines.extend(
                [
                    "",
                    "## Uncovered authored actions",
                    "",
                    ", ".join(f"`{item}`" for item in self.uncovered_authored_actions),
                ]
            )
        all_issues = [(run.name, issue) for run in self.runs for issue in run.issues]
        if all_issues:
            lines.extend(["", "## Issues", ""])
            for run_name, issue in all_issues:
                where = f" turn {issue.turn}" if issue.turn is not None else ""
                lines.append(
                    f"- **{issue.severity.upper()}** `{issue.code}` in `{run_name}`{where}: {issue.message}"
                )
        return "\n".join(lines) + "\n"


def _offline_engine(scenario: ScenarioDefinition) -> GameEngine:
    return GameEngine(
        scenario,
        OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None)),
    )


class AutoPlayer:
    """A deterministic policy player that only selects currently legal affordances."""

    CATEGORY_WEIGHTS: dict[PlayStrategy, dict[str, int]] = {
        PlayStrategy.INVESTIGATOR: {"investigate": 55, "social": 42, "move": 22, "risk": 8, "other": 0},
        PlayStrategy.SOCIAL: {"social": 60, "investigate": 42, "move": 22, "risk": 5, "other": 0},
        PlayStrategy.BOLD: {"risk": 62, "investigate": 32, "move": 26, "social": 12, "other": 0},
        PlayStrategy.EXPLORER: {"move": 55, "investigate": 35, "social": 25, "risk": 18, "other": 0},
        PlayStrategy.RANDOM: {"move": 0, "investigate": 0, "social": 0, "risk": 0, "other": 0},
    }

    def __init__(self, strategy: PlayStrategy, seed: int):
        self.strategy = strategy
        self.random = random.Random(seed ^ 0x5E1F)
        self.action_counts: Counter[str] = Counter()
        self.location_visits: Counter[str] = Counter()

    @staticmethod
    def _distances_to_tag(scenario: ScenarioDefinition, tag: str) -> dict[str, int]:
        locations = {
            entity.id
            for entity in scenario.entities
            if entity.type.value == "LOCATION"
        }
        goals = {
            entity.id
            for entity in scenario.entities
            if entity.type.value == "LOCATION" and tag in entity.tags
        }
        reverse_edges: dict[str, set[str]] = {location_id: set() for location_id in locations}
        for action in scenario.actions:
            if (
                action.type == ActionType.MOVE
                and action.location in locations
                and action.target in locations
            ):
                reverse_edges[action.target].add(action.location)
        distances = {goal: 0 for goal in goals}
        frontier = list(goals)
        while frontier:
            destination = frontier.pop(0)
            for origin in reverse_edges[destination]:
                if origin in distances:
                    continue
                distances[origin] = distances[destination] + 1
                frontier.append(origin)
        return distances

    @staticmethod
    def _reveals_unknown_fact(action: ActionDefinition, state: WorldState) -> bool:
        return any(
            effect.op == "reveal_fact"
            and str(effect.params.get("fact_id")) not in state.player_known_fact_ids
            for effect in action.success_effects
        )

    @staticmethod
    def _advances_resolution(action: ActionDefinition, scenario: ScenarioDefinition) -> bool:
        progress_flags = set(scenario.director_config.progress_flags)
        ending_flags = {
            condition.subject.split(":", 1)[1]
            for ending in scenario.endings
            if ending.status == SessionStatus.WON
            for condition in ending.requirements
            if condition.subject.startswith("flag:") and condition.value is True
        }
        useful_flags = progress_flags | ending_flags
        return any(
            effect.op == "mark_flag"
            and str(effect.params.get("key")) in useful_flags
            and effect.params.get("value", True) is True
            for effect in action.success_effects
        )

    @staticmethod
    def _causes_losing_ending(action: ActionDefinition, scenario: ScenarioDefinition) -> bool:
        losing_flags = {
            condition.subject.split(":", 1)[1]
            for ending in scenario.endings
            if ending.status == SessionStatus.LOST
            for condition in ending.requirements
            if condition.subject.startswith("flag:") and condition.value is True
        }
        return any(
            effect.op == "mark_flag"
            and str(effect.params.get("key")) in losing_flags
            and effect.params.get("value", True) is True
            for effect in action.success_effects
        )

    def choose(
        self,
        state: WorldState,
        scenario: ScenarioDefinition,
        actions: list[ActionDefinition],
    ) -> ActionDefinition:
        if self.strategy == PlayStrategy.RANDOM:
            weights = [max(1, 10 - self.action_counts[action.id] * 3) for action in actions]
            return self.random.choices(actions, weights=weights, k=1)[0]

        location_id = state.entities[state.player.entity_id].location
        if location_id:
            self.location_visits[location_id] += 1
        clue_goal_met = len(state.discovered_clue_ids) >= scenario.minimum_evidence
        hp_ratio = state.player.hp / max(1, state.player.max_hp)
        objective_met = any(
            state.flags.get(flag_name)
            for flag_name in scenario.director_config.progress_flags
        )
        exit_distances = self._distances_to_tag(scenario, "exit")
        current_location = state.entities[state.player.entity_id].location
        current_exit_distance = exit_distances.get(current_location or "")
        scored: list[tuple[float, ActionDefinition]] = []
        for action in actions:
            score = float(self.CATEGORY_WEIGHTS[self.strategy].get(action.category, 0))
            score -= self.action_counts[action.id] * 24
            score += self.random.random() * 4
            if self._reveals_unknown_fact(action, state):
                score += 55 if not clue_goal_met else 24
            if self._advances_resolution(action, scenario):
                score += 80 if clue_goal_met else 28
            if self._causes_losing_ending(action, scenario):
                score -= 180
            if action.type in {ActionType.CONFRONT, ActionType.DISRUPT, ActionType.RESCUE}:
                score += 48 if clue_goal_met else (-18 if self.strategy != PlayStrategy.BOLD else 18)
            if action.type == ActionType.ESCAPE:
                score += 110 if hp_ratio <= 0.35 else (18 if objective_met else -90)
            if action.type == ActionType.MOVE and action.target in state.entities:
                target = state.entities[action.target]
                visits = self.location_visits[action.target]
                score += 38 if visits == 0 else -visits * 9
                if "exit" in target.tags:
                    score += 100 if hp_ratio <= 0.35 else (20 if objective_met else -120)
                target_exit_distance = exit_distances.get(action.target)
                if (
                    hp_ratio <= 0.35
                    and current_exit_distance is not None
                    and target_exit_distance is not None
                ):
                    improvement = current_exit_distance - target_exit_distance
                    score += improvement * 110
            if action.risk == "dangerous" and hp_ratio <= 0.35:
                score -= 80
            if action.type == ActionType.WAIT:
                score -= 70
            scored.append((score, action))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        chosen = scored[0][1]
        self.action_counts[chosen.id] += 1
        return chosen


class PlaytestLab:
    def __init__(self, scenario: ScenarioDefinition):
        self.scenario = scenario

    @staticmethod
    def _public_invariant_issues(
        engine: GameEngine,
        state: WorldState,
        turn: int,
        action_id: str,
    ) -> list[PlaytestIssue]:
        issues: list[PlaytestIssue] = []
        view = engine.public_view(state)
        public_clues = {item["id"] for item in view["clues"]}
        if public_clues != state.discovered_clue_ids:
            issues.append(
                PlaytestIssue(
                    "error",
                    "PUBLIC_CLUE_MISMATCH",
                    "Public clue projection does not match discovered clues.",
                    turn,
                    action_id,
                )
            )
        player_location = state.entities[state.player.entity_id].location
        expected_npcs = {
            entity.id
            for entity in state.entities.values()
            if entity.type.value == "NPC"
            and entity.active
            and entity.location == player_location
        }
        public_npcs = {item["id"] for item in view["scene"]["present_npcs"]}
        if public_npcs != expected_npcs:
            issues.append(
                PlaytestIssue(
                    "error",
                    "PUBLIC_NPC_MISMATCH",
                    "Public NPC projection includes an absent/hidden NPC or omits a present one.",
                    turn,
                    action_id,
                )
            )
        narrative = view["narrative"]
        for fact in state.facts.values():
            if fact.id in state.player_known_fact_ids or fact.visibility != "HIDDEN":
                continue
            value = fact.value
            if isinstance(value, str) and len(value) >= 12 and value in narrative:
                issues.append(
                    PlaytestIssue(
                        "error",
                        "HIDDEN_FACT_LEAK",
                        f"Narrative contains the full value of unseen fact {fact.id}.",
                        turn,
                        action_id,
                    )
                )
        expected_seq = list(range(1, len(state.event_log) + 1))
        if [event.seq for event in state.event_log] != expected_seq:
            issues.append(
                PlaytestIssue(
                    "error",
                    "EVENT_SEQUENCE_GAP",
                    "Event log sequence is not contiguous.",
                    turn,
                    action_id,
                )
            )
        return issues

    @staticmethod
    def _meaningful_signature(state: WorldState) -> tuple[Any, ...]:
        return (
            state.entities[state.player.entity_id].location,
            tuple(sorted(state.discovered_clue_ids)),
            tuple(state.player.inventory),
            state.player.hp,
            state.player.stress,
            state.player.sanity,
            state.player.luck,
            state.player.major_wound,
            tuple(sorted((key, repr(value)) for key, value in state.flags.items())),
            state.status.value,
            state.ending_id,
        )

    def _execute(
        self,
        *,
        name: str,
        seed: int,
        strategy_name: str,
        choose_action: Any,
        max_turns: int,
        terminal_required: bool,
    ) -> PlaytestRun:
        engine = _offline_engine(self.scenario)
        state = create_initial_state(self.scenario, player_name="自动调查员", seed=seed)
        engine.director.recompute(state)
        run = PlaytestRun(
            name=name,
            scenario_id=self.scenario.id,
            seed=seed,
            strategy=strategy_name,
            terminal_required=terminal_required,
        )
        visited_locations: set[str] = {state.entities[state.player.entity_id].location or ""}
        action_coverage: set[str] = set()
        repeated_without_change = 0
        previous_signature = self._meaningful_signature(state)

        for turn in range(1, max_turns + 1):
            if state.status != SessionStatus.ACTIVE:
                break
            available = engine.kernel.available_actions(state)
            if not available:
                run.issues.append(
                    PlaytestIssue(
                        "error",
                        "ACTIVE_WITHOUT_ACTIONS",
                        "The session is active but has no legal actions.",
                        turn,
                    )
                )
                break
            try:
                action = choose_action(state, available, turn)
            except LookupError as exc:
                run.issues.append(
                    PlaytestIssue(
                        "error",
                        "SCRIPT_ROUTE_INVALID",
                        str(exc),
                        turn,
                    )
                )
                break
            if action not in available:
                run.issues.append(
                    PlaytestIssue(
                        "error",
                        "POLICY_SELECTED_ILLEGAL_ACTION",
                        f"Policy selected unavailable action {action.id}.",
                        turn,
                        action.id,
                    )
                )
                break
            location_before = state.entities[state.player.entity_id].location
            time_before = state.world_time
            version_before = state.version
            state, resolution = engine.play(state, action_id=action.id)
            location_after = state.entities[state.player.entity_id].location
            action_coverage.add(action.id)
            if location_after:
                visited_locations.add(location_after)
            outcome = (
                "INTERRUPTED"
                if resolution.interrupted
                else resolution.check.outcome.value
                if resolution.check
                else "NONE"
            )
            run.steps.append(
                TraceStep(
                    turn=turn,
                    action_id=action.id,
                    action_label=action.label,
                    location_before=location_before,
                    location_after=location_after,
                    world_time=state.world_time.isoformat(),
                    accepted=resolution.accepted,
                    interrupted=resolution.interrupted,
                    outcome=outcome,
                    hp=state.player.hp,
                    stress=state.player.stress,
                    sanity=state.player.sanity,
                    luck=state.player.luck,
                    major_wound=state.player.major_wound,
                    clue_count=len(state.discovered_clue_ids),
                    status=state.status.value,
                    ending_id=state.ending_id,
                    director_action=(
                        resolution.director_intervention.action
                        if resolution.director_intervention
                        else None
                    ),
                )
            )
            if not resolution.accepted:
                run.issues.append(
                    PlaytestIssue(
                        "error",
                        "LEGAL_ACTION_REJECTED",
                        f"Kernel listed {action.id} as legal but engine rejected it.",
                        turn,
                        action.id,
                    )
                )
                break
            if state.version != version_before + 1:
                run.issues.append(
                    PlaytestIssue(
                        "error",
                        "VERSION_INCREMENT",
                        f"Accepted action changed version {version_before} -> {state.version}.",
                        turn,
                        action.id,
                    )
                )
            if state.world_time < time_before:
                run.issues.append(
                    PlaytestIssue(
                        "error",
                        "TIME_MOVED_BACKWARD",
                        "World time moved backward.",
                        turn,
                        action.id,
                    )
                )
            run.issues.extend(self._public_invariant_issues(engine, state, turn, action.id))
            signature = self._meaningful_signature(state)
            if signature == previous_signature:
                repeated_without_change += 1
            else:
                repeated_without_change = 0
            previous_signature = signature
            if repeated_without_change == 8:
                run.issues.append(
                    PlaytestIssue(
                        "warning",
                        "PROLONGED_STALL",
                        "Eight consecutive actions produced no meaningful state change.",
                        turn,
                        action.id,
                    )
                )

        if terminal_required and state.status == SessionStatus.ACTIVE:
            run.issues.append(
                PlaytestIssue(
                    "error",
                    "TERMINAL_NOT_REACHED",
                    f"Required terminal state was not reached within {max_turns} turns.",
                )
            )
        elif state.status == SessionStatus.ACTIVE:
            run.issues.append(
                PlaytestIssue(
                    "warning",
                    "TERMINAL_NOT_REACHED",
                    f"Exploratory run remained active after {max_turns} turns.",
                )
            )
        run.final_status = state.status.value
        run.ending_id = state.ending_id
        run.discovered_clues = sorted(state.discovered_clue_ids)
        run.visited_locations = sorted(visited_locations - {""})
        run.action_coverage = sorted(action_coverage)
        run.event_types = sorted({event.type for event in state.event_log})
        run.sanity_check_count = len(state.sanity_checks)
        run.sanity_loss = sum(item.loss for item in state.sanity_checks)
        run.damage_event_count = len(state.damage_log)
        run.major_wound_seen = any(item.major_wound_triggered for item in state.damage_log)
        return run

    def run_policy(
        self,
        strategy: PlayStrategy,
        *,
        seed: int,
        max_turns: int = 80,
        terminal_required: bool = False,
        name: str | None = None,
    ) -> PlaytestRun:
        player = AutoPlayer(strategy, seed)
        return self._execute(
            name=name or f"{strategy.value}-{seed}",
            seed=seed,
            strategy_name=strategy.value,
            choose_action=lambda state, actions, _turn: player.choose(state, self.scenario, actions),
            max_turns=max_turns,
            terminal_required=terminal_required,
        )

    def run_scripted(
        self,
        name: str,
        action_ids: Iterable[str | ScriptedStep],
        *,
        seed: int,
        attempts_per_action: int = 8,
        terminal_required: bool = True,
        expected_endings: set[str] | None = None,
    ) -> PlaytestRun:
        action_list = [
            item if isinstance(item, ScriptedStep) else ScriptedStep(item)
            for item in action_ids
        ]
        route_index = 0
        attempts = 0
        step_started = False

        def choose(state: WorldState, actions: list[ActionDefinition], _turn: int) -> ActionDefinition:
            nonlocal route_index, attempts, step_started
            available_by_id = {action.id: action for action in actions}
            if route_index < len(action_list) and step_started:
                current = action_list[route_index]
                if current.until == "attempt" or current.action_id in state.completed_actions:
                    route_index += 1
                    attempts = 0
                    step_started = False
                elif attempts >= attempts_per_action:
                    raise LookupError(
                        f"Scripted action did not succeed after {attempts_per_action} attempts: "
                        f"{current.action_id}"
                    )
            if route_index < len(action_list):
                current = action_list[route_index]
                if current.action_id not in available_by_id:
                    raise LookupError(f"Scripted action is unavailable: {current.action_id}")
                attempts += 1
                step_started = True
                return available_by_id[current.action_id]
            raise LookupError("Scripted route exhausted before a terminal state")

        max_turns = max(1, len(action_list) * attempts_per_action)
        run = self._execute(
            name=name,
            seed=seed,
            strategy_name="scripted",
            choose_action=choose,
            max_turns=max_turns,
            terminal_required=terminal_required,
        )
        if expected_endings and run.ending_id not in expected_endings:
            run.issues.append(
                PlaytestIssue(
                    "error",
                    "UNEXPECTED_ENDING",
                    f"Expected one of {sorted(expected_endings)}, got {run.ending_id or run.final_status}.",
                )
            )
        return run


def build_report(scenario: ScenarioDefinition, runs: list[PlaytestRun]) -> PlaytestReport:
    authored_actions = {action.id for action in scenario.actions if not action.id.startswith("move__")}
    covered = {action_id for run in runs for action_id in run.action_coverage}
    endings = Counter(run.ending_id or "ACTIVE" for run in runs)
    statuses = Counter(run.final_status for run in runs)
    outcomes = Counter(step.outcome for run in runs for step in run.steps)
    director_actions = Counter(
        step.director_action
        for run in runs
        for step in run.steps
        if step.director_action
    )
    return PlaytestReport(
        scenario_id=scenario.id,
        runs=runs,
        authored_action_count=len(authored_actions),
        covered_authored_actions=sorted(authored_actions & covered),
        uncovered_authored_actions=sorted(authored_actions - covered),
        endings_seen=dict(sorted(endings.items())),
        statuses_seen=dict(sorted(statuses.items())),
        roll_outcomes_seen=dict(sorted(outcomes.items())),
        director_actions_seen=dict(sorted(director_actions.items())),
        sanity_check_count=sum(run.sanity_check_count for run in runs),
        sanity_loss=sum(run.sanity_loss for run in runs),
        damage_event_count=sum(run.damage_event_count for run in runs),
        major_wound_runs=sum(1 for run in runs if run.major_wound_seen),
    )
