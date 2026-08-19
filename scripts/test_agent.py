from __future__ import annotations

import argparse
import json
from pathlib import Path

from living_tabletop.engine import GameEngine
from living_tabletop.llm import RoutedLLM
from living_tabletop.scenario import load_scenarios
from living_tabletop.test_agent import (
    HeuristicPlayerDriver,
    LLMPlayerDriver,
    PERSONAS,
    SyntheticComposerLLM,
    TestAgentRunner,
    TestAgentReport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run dynamic multi-persona free-text Test Agents against Living Tabletop V2."
    )
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--runs-per-persona", type=int, default=2)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument(
        "--game-backend",
        choices=("synthetic", "live"),
        default="synthetic",
        help="Synthetic is fast invariant fuzzing; live uses the configured local/remote model.",
    )
    parser.add_argument(
        "--player-backend",
        choices=("heuristic", "llm"),
        default="heuristic",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("test-agent-reports"))
    args = parser.parse_args()

    scenarios = load_scenarios()
    scenario_ids = list(scenarios) if args.scenario == "all" else [args.scenario]
    if any(item not in scenarios for item in scenario_ids):
        parser.error("unknown scenario id")

    exit_code = 0
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        runs = []
        shared_inputs: set[str] = set()
        for persona in PERSONAS:
            for run_index in range(args.runs_per_persona):
                seed = 1009 + run_index * 101 + PERSONAS.index(persona) * 17
                game_llm = SyntheticComposerLLM(seed) if args.game_backend == "synthetic" else RoutedLLM.from_env()
                engine = GameEngine(scenario, game_llm)
                if args.player_backend == "llm":
                    player_llm = RoutedLLM.from_env()
                    driver = LLMPlayerDriver(player_llm, seed)
                else:
                    driver = HeuristicPlayerDriver(seed)
                runs.append(
                    TestAgentRunner(scenario, engine, driver, shared_inputs).run(
                        persona,
                        seed=seed,
                        max_turns=args.turns,
                    )
                )
        report = TestAgentReport(scenario_id=scenario_id, runs=runs)
        print(report.to_markdown())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"test-agent-{scenario_id}"
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / f"{stem}.md").write_text(report.to_markdown(), encoding="utf-8")
        if report.issues:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
