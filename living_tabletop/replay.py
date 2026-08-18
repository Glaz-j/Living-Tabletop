from __future__ import annotations

from typing import Any

from .engine import GameEngine
from .llm import LLMSettings, OpenAICompatibleLLM
from .models import ScenarioDefinition, WorldState
from .scenario import create_initial_state
from .storage import SQLiteRepository


class ReplayMismatch(AssertionError):
    pass


def replay_actions(export: dict[str, Any], scenario: ScenarioDefinition) -> WorldState:
    """Re-run recorded resolved actions without invoking an LLM.

    Narrator wording and agent observability records are projections, so verification
    compares the simulation snapshot rather than regenerated prose.
    """

    final_snapshot = export["snapshot"]
    state = create_initial_state(
        scenario,
        player_name=final_snapshot["player"]["name"],
        seed=int(export["rng_seed"]),
        session_id=str(export["session_id"]),
    )
    engine = GameEngine(
        scenario,
        OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None)),
    )
    engine.director.recompute(state)
    inputs = export.get("turn_inputs") or [
        {"kind": "action", **item, "interactive_rules": False}
        for item in export["action_inputs"]
    ]
    for item in inputs:
        if item.get("kind") == "rule_choice":
            choice = item.get("choice")
            if not choice:
                raise ReplayMismatch("Recorded rule choice is missing its value")
            state, resolution = engine.play(state, rule_choice=str(choice))
        else:
            action_id = item.get("action_id")
            open_plan = item.get("open_plan")
            if open_plan:
                state, resolution = engine.play(
                    state,
                    text=item.get("player_text") or open_plan.get("goal"),
                    interactive_rules=bool(item.get("interactive_rules", False)),
                    recorded_open_plan=open_plan,
                )
            elif action_id:
                state, resolution = engine.play(
                    state,
                    action_id=str(action_id),
                    interactive_rules=bool(item.get("interactive_rules", False)),
                )
            else:
                raise ReplayMismatch("Recorded action is missing its resolved action_id")
        if not resolution.accepted:
            raise ReplayMismatch(f"Replay input rejected: {item}")
    return state


def verify_replay(export: dict[str, Any], scenario: ScenarioDefinition) -> tuple[bool, str, str]:
    replayed = replay_actions(export, scenario)
    actual = SQLiteRepository.simulation_digest(replayed)
    expected = str(export["simulation_digest"])
    return actual == expected, expected, actual
