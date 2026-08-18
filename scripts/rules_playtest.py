from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from living_tabletop.engine import GameEngine
from living_tabletop.kernel import WorldKernel
from living_tabletop.llm import LLMSettings, OpenAICompatibleLLM
from living_tabletop.models import Effect, RuleChoice
from living_tabletop.scenario import create_initial_state, load_scenario


def _engine(scenario_id: str) -> tuple[Any, GameEngine]:
    scenario = load_scenario(scenario_id=scenario_id)
    engine = GameEngine(
        scenario,
        OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None)),
    )
    return scenario, engine


def _pending_case(*, pushed_should_succeed: bool | None = None):
    scenario, engine = _engine("st_mary_hospital_v0")
    for seed in range(1, 500):
        state = create_initial_state(scenario, seed=seed)
        state.player.luck = 99
        offered, first = engine.play(
            state,
            action_id="lobby_guestbook",
            interactive_rules=True,
        )
        if not first.awaiting_rule_choice:
            continue
        if pushed_should_succeed is None:
            return scenario, engine, offered, first
        pushed, second = engine.play(offered, rule_choice=RuleChoice.PUSH_ROLL)
        if second.check and second.check.succeeded is pushed_should_succeed:
            return scenario, engine, offered, first, pushed, second
    raise AssertionError("Could not find deterministic rule-choice seed")


def run_rules_matrix() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    _, engine, offered, first = _pending_case()
    accepted, resolution = engine.play(offered, rule_choice=RuleChoice.ACCEPT_FAILURE)
    assert accepted.pending_check is None and not resolution.check.succeeded
    cases.append({"case": "accept_failure", "passed": True, "outcome": resolution.check.outcome.value})

    _, engine, offered, first = _pending_case()
    luck_before = offered.player.luck
    luck_cost = first.luck_cost
    lucky, resolution = engine.play(offered, rule_choice=RuleChoice.SPEND_LUCK)
    assert resolution.check.succeeded and lucky.player.luck == luck_before - luck_cost
    cases.append({"case": "spend_luck", "passed": True, "luck_spent": luck_cost})

    _, _, offered, _, pushed, resolution = _pending_case(pushed_should_succeed=True)
    assert resolution.check.pushed and pushed.world_time > offered.world_time
    cases.append({"case": "push_success", "passed": True, "outcome": resolution.check.outcome.value})

    _, _, offered, _, pushed, resolution = _pending_case(pushed_should_succeed=False)
    assert resolution.check.pushed and any(event.type == "pushed_roll_failed" for event in pushed.event_log)
    cases.append({"case": "push_failure", "passed": True, "outcome": resolution.check.outcome.value})

    scenario, engine = _engine("the_haunting_corbitt_house_v1")
    state = create_initial_state(scenario, seed=9)
    state.entities["player"].location = "loc_hidden_lair"
    state.entities["creature_corbitt"].active = True
    state.player.inventory.extend(["item_magic_dagger", "item_trash_lid"])
    state.entities["item_magic_dagger"].location = "player"
    state.entities["item_trash_lid"].location = "player"
    _, combat = engine.play(state, action_id="lair_use_dagger", interactive_rules=True)
    assert combat.check.opponent and combat.check.bonus_dice == 1 and len(combat.check.candidates) == 2
    cases.append(
        {
            "case": "opposed_combat_bonus_die",
            "passed": True,
            "player_roll": combat.check.roll,
            "opponent_roll": combat.check.opponent.roll,
        }
    )

    state = create_initial_state(scenario, seed=19)
    state.entities["player"].location = "loc_hidden_lair"
    state.entities["creature_corbitt"].active = True
    state, horror = engine.play(state, action_id="lair_examine_corbitt")
    assert horror.sanity_check and len(state.sanity_checks) == 1
    cases.append(
        {
            "case": "sanity_check",
            "passed": True,
            "roll": horror.sanity_check.roll,
            "loss": horror.sanity_check.loss,
        }
    )

    state = create_initial_state(scenario, seed=31)
    state.player.hp = state.player.max_hp = 12
    WorldKernel(scenario).apply_effect(
        state,
        Effect(op="damage_player", params={"amount": "6"}),
        source="rules-playtest",
    )
    assert state.player.major_wound and state.damage_log[-1].major_wound_triggered
    cases.append(
        {
            "case": "major_wound",
            "passed": True,
            "hp": state.player.hp,
            "unconscious": state.player.unconscious,
        }
    )

    return {"passed": all(case["passed"] for case in cases), "case_count": len(cases), "cases": cases}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CoC rules playtest",
        "",
        f"- Cases: {report['case_count']}",
        f"- Passed: {report['passed']}",
        "",
        "| Case | Result | Details |",
        "| --- | --- | --- |",
    ]
    for case in report["cases"]:
        details = ", ".join(f"{key}={value}" for key, value in case.items() if key not in {"case", "passed"})
        lines.append(f"| {case['case']} | {'PASS' if case['passed'] else 'FAIL'} | {details} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise interactive CoC rules branches deterministically.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run_rules_matrix()
    markdown = _markdown(report)
    print(markdown)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "coc-rules-playtest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "coc-rules-playtest.md").write_text(markdown, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
