from __future__ import annotations

import argparse
import json
from pathlib import Path

from living_tabletop.playtest import PlayStrategy, PlaytestLab, ScriptedStep, build_report
from living_tabletop.scenario import load_scenarios


HAUNTING_RESEARCH_ROUTE = [
    "cafe_question_knott",
    "move__loc_cafe__loc_globe",
    "globe_persuade_arty",
    "move__loc_globe__loc_cafe",
    "move__loc_cafe__loc_library",
    "library_research_owner",
    "library_research_lawsuit",
    "library_research_obituary",
    "move__loc_library__loc_cafe",
    "move__loc_cafe__loc_records",
    "records_trace_executor",
    "move__loc_records__loc_cafe",
    "move__loc_cafe__loc_police",
    "police_gain_access",
    "move__loc_police__loc_cafe",
    "move__loc_cafe__loc_neighborhood",
    "neighborhood_talk_dooley",
    "move__loc_neighborhood__loc_cafe",
    "move__loc_cafe__loc_sanitarium",
    "sanitarium_talk_gabriela",
    "sanitarium_listen_vittorio",
    "move__loc_sanitarium__loc_cafe",
    "move__loc_cafe__loc_chapel",
    "chapel_search_cellar",
    "chapel_take_liber",
    "move__loc_chapel__loc_house_exterior",
    "exterior_examine_house",
    "move__loc_house_exterior__loc_house_ground",
    "ground_search_storage",
    "ground_restore_power",
    "move__loc_house_ground__loc_house_upper",
    "upper_search_family_rooms",
    "move__loc_house_upper__loc_spare_bedroom",
    "spare_trigger_bed",
    "move__loc_spare_bedroom__loc_house_upper",
    "move__loc_house_upper__loc_house_ground",
    "ground_descend_basement",
    "basement_take_lid",
    "basement_search_knife",
    "basement_find_wall",
    "basement_break_wall",
    "lair_examine_corbitt",
    "lair_use_dagger",
    "lair_flee",
    "basement_return_ground",
    "move__loc_house_ground__loc_house_exterior",
    "move__loc_house_exterior__loc_cafe",
    "cafe_report_success",
]

HAUNTING_EVIDENCE_EXIT_ROUTE = [
    "cafe_question_knott",
    "move__loc_cafe__loc_globe",
    "globe_persuade_arty",
    "move__loc_globe__loc_cafe",
    "move__loc_cafe__loc_library",
    "library_research_owner",
    "library_research_lawsuit",
    "library_research_obituary",
    "move__loc_library__loc_cafe",
    "move__loc_cafe__loc_records",
    "records_trace_executor",
    "move__loc_records__loc_cafe",
    "move__loc_cafe__loc_street",
    "street_leave_case",
]

HAUNTING_DIRECT_DAGGER_ROUTE = [
    "move__loc_cafe__loc_house_exterior",
    "move__loc_house_exterior__loc_house_ground",
    "ground_descend_basement",
    "basement_take_lid",
    "basement_search_knife",
    "basement_find_wall",
    "basement_break_wall",
    "lair_use_dagger",
    "lair_flee",
    "basement_return_ground",
    "move__loc_house_ground__loc_house_exterior",
    "move__loc_house_exterior__loc_cafe",
    "cafe_report_success",
]

HAUNTING_DIRECT_FORCE_ROUTE = [
    "move__loc_cafe__loc_house_exterior",
    "move__loc_house_exterior__loc_house_ground",
    "ground_descend_basement",
    "basement_take_lid",
    "basement_find_wall",
    "basement_break_wall",
    "lair_attack_direct",
    "lair_finish_direct",
    "lair_flee",
    "basement_return_ground",
    "move__loc_house_ground__loc_house_exterior",
    "move__loc_house_exterior__loc_cafe",
    "cafe_report_success",
]

ST_MARY_COMPLETE_ROUTE = [
    "lobby_guestbook",
    ScriptedStep("lobby_talk_anna", until="attempt"),
    "move__loc_lobby__loc_office",
    "office_search_desk",
    "office_inspect_painting",
    "office_unlock_archives",
    "move__loc_office__loc_archives",
    ScriptedStep("archives_take_records", until="attempt"),
    ScriptedStep("archives_search_ashes", until="attempt"),
    "move__loc_archives__loc_office",
    "move__loc_office__loc_lobby",
    "lobby_unlock_basement",
    "move__loc_lobby__loc_basement",
    "basement_follow_trail",
    "basement_rescue_samuel",
    "move__loc_basement__loc_chapel",
    "chapel_talk_doyle",
    "chapel_inspect_altar",
    "chapel_enter_crypt",
    "crypt_disrupt_ritual",
    "crypt_return_chapel",
    "move__loc_chapel__loc_road",
    "road_escape",
]


def scenario_runs(scenario_id: str):
    scenario = load_scenarios()[scenario_id]
    lab = PlaytestLab(scenario)
    runs = []
    if scenario_id == "the_haunting_corbitt_house_v1":
        runs.extend(
            [
                lab.run_scripted(
                    "research-complete",
                    HAUNTING_RESEARCH_ROUTE,
                    seed=19,
                    expected_endings={"case_proven"},
                ),
                lab.run_scripted(
                    "evidence-withdrawal",
                    HAUNTING_EVIDENCE_EXIT_ROUTE,
                    seed=7,
                    expected_endings={"escaped_with_evidence"},
                ),
                lab.run_scripted(
                    "false-assurance",
                    ["cafe_false_report"],
                    seed=3,
                    expected_endings={"false_assurance"},
                ),
                lab.run_scripted(
                    "early-exit",
                    ["move__loc_cafe__loc_street", "street_leave_case"],
                    seed=4,
                    expected_endings={"escaped_alive"},
                ),
                lab.run_scripted(
                    "direct-dagger",
                    HAUNTING_DIRECT_DAGGER_ROUTE,
                    seed=2,
                    expected_endings={"case_survived"},
                ),
                lab.run_scripted(
                    "direct-force",
                    HAUNTING_DIRECT_FORCE_ROUTE,
                    seed=9,
                    expected_endings={"case_survived"},
                ),
            ]
        )
    elif scenario_id == "st_mary_hospital_v0":
        runs.extend(
            [
                lab.run_scripted(
                    "complete-rescue",
                    ST_MARY_COMPLETE_ROUTE,
                    seed=77,
                    expected_endings={"truth_rescue_and_silence"},
                ),
                lab.run_scripted(
                    "leave-immediately",
                    ["move__loc_lobby__loc_road", "road_escape"],
                    seed=3,
                    expected_endings={"escaped_alive"},
                ),
                lab.run_scripted(
                    "force-archive-and-leave",
                    [
                        "lobby_take_lantern",
                        "move__loc_lobby__loc_office",
                        "office_force_archives",
                        "move__loc_office__loc_lobby",
                        "move__loc_lobby__loc_road",
                        "road_escape",
                    ],
                    seed=1,
                    expected_endings={"escaped_alive"},
                ),
            ]
        )
    policy_seeds = {
        PlayStrategy.INVESTIGATOR: (7, 11),
        PlayStrategy.SOCIAL: (15, 37),
        PlayStrategy.BOLD: (6, 37),
        PlayStrategy.EXPLORER: (11, 12),
        PlayStrategy.RANDOM: (11, 37),
    }
    for strategy in PlayStrategy:
        for seed in policy_seeds[strategy]:
            runs.append(
                lab.run_policy(
                    strategy,
                    seed=seed,
                    max_turns=120,
                    terminal_required=False,
                )
            )
    return scenario, runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic multi-route Living Tabletop self-play.")
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario id or 'all' (default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for JSON and Markdown reports.",
    )
    args = parser.parse_args()
    scenarios = load_scenarios()
    scenario_ids = list(scenarios) if args.scenario == "all" else [args.scenario]
    unknown = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenarios]
    if unknown:
        parser.error(f"Unknown scenario id: {unknown[0]}")

    error_count = 0
    for scenario_id in scenario_ids:
        scenario, runs = scenario_runs(scenario_id)
        report = build_report(scenario, runs)
        print(report.to_markdown())
        error_count += report.error_count
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"playtest-{scenario_id}"
            (args.output_dir / f"{stem}.json").write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (args.output_dir / f"{stem}.md").write_text(report.to_markdown(), encoding="utf-8")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
