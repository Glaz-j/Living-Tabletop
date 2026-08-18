from __future__ import annotations

from living_tabletop.playtest import build_report
from living_tabletop.llm import LLMSettings
from living_tabletop.models import SessionStatus
from living_tabletop.replay import verify_replay
from living_tabletop.service import GameService
from scripts.self_play import HAUNTING_DIRECT_DAGGER_ROUTE, scenario_runs


def test_haunting_multi_route_self_play_has_full_authored_coverage():
    scenario, runs = scenario_runs("the_haunting_corbitt_house_v1")
    report = build_report(scenario, runs)
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.uncovered_authored_actions == []
    assert {"case_proven", "case_survived", "incapacitated", "escaped_with_evidence", "false_assurance"} <= set(
        report.endings_seen
    )
    assert {"FAILURE", "SUCCESS", "HARD", "EXTREME", "AUTOMATIC"} <= set(
        report.roll_outcomes_seen
    )


def test_st_mary_regression_self_play_has_full_authored_coverage():
    scenario, runs = scenario_runs("st_mary_hospital_v0")
    report = build_report(scenario, runs)
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.uncovered_authored_actions == []
    assert "truth_rescue_and_silence" in report.endings_seen


def test_haunting_service_playthrough_persists_and_replays(tmp_path):
    scenario_id = "the_haunting_corbitt_house_v1"
    service = GameService(
        db_path=tmp_path / "haunting-playthrough.db",
        llm_settings=LLMSettings(enabled=False, api_key=None),
    )
    view = service.create_session(
        scenario_id=scenario_id,
        player_name="回放调查员",
        seed=2,
    )
    session_id = view["session_id"]

    for action_id in HAUNTING_DIRECT_DAGGER_ROUTE:
        for _ in range(8):
            view = service.act(session_id, action_id=action_id)
            assert view["last_resolution"]["accepted"]
            state = service.repository.load(session_id)
            if state.status != SessionStatus.ACTIVE or action_id in state.completed_actions:
                break
        else:
            raise AssertionError(f"Action did not complete: {action_id}")
        if state.status != SessionStatus.ACTIVE:
            break

    assert state.status == SessionStatus.WON
    assert state.ending_id == "case_survived"
    replay = service.export_replay(session_id)
    verified, expected, actual = verify_replay(replay, service.scenarios[scenario_id])
    assert verified, (expected, actual)


def test_interactive_rule_choice_is_deterministic_in_replay(tmp_path):
    service = GameService(
        db_path=tmp_path / "interactive-replay.db",
        llm_settings=LLMSettings(enabled=False, api_key=None),
    )
    view = None
    for seed in range(1, 200):
        created = service.create_session(seed=seed)
        candidate = service.act(
            created["session_id"],
            action_id="lobby_guestbook",
            interactive_rules=True,
        )
        if candidate["rule_prompt"]:
            view = candidate
            break
    assert view is not None
    session_id = view["session_id"]
    service.act(session_id, rule_choice="accept_failure")

    replay = service.export_replay(session_id)
    assert any(item["kind"] == "rule_choice" for item in replay["turn_inputs"])
    verified, expected, actual = verify_replay(replay, service.scenarios["st_mary_hospital_v0"])
    assert verified, (expected, actual)
