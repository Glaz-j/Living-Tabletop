from __future__ import annotations

import tempfile
from pathlib import Path

from living_tabletop.llm import LLMSettings
from living_tabletop.models import SessionStatus
from living_tabletop.replay import verify_replay
from living_tabletop.service import GameService


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="living-tabletop-") as temp_dir:
        service = GameService(
            db_path=Path(temp_dir) / "playthrough.db",
            llm_settings=LLMSettings(enabled=False, api_key=None),
        )
        view = service.create_session(player_name="自动调查员", seed=19)
        session_id = view["session_id"]

        def act(action_id: str) -> dict:
            result = service.act(session_id, action_id=action_id)
            resolution = result["last_resolution"] or {}
            check = resolution.get("check") or {}
            outcome = "INTERRUPTED" if resolution.get("interrupted") else check.get("outcome", "AUTO")
            print(f"{result['time_label']}  {action_id:<38} {outcome:<12} clues={len(result['clues'])}")
            if not resolution.get("accepted"):
                raise RuntimeError(f"Action rejected: {action_id}: {resolution}")
            return result

        for _ in range(2):
            view = act("lobby_guestbook")
            if any(clue["id"] == "clue_missing_note" for clue in view["clues"]):
                break
        for action_id in [
            "lobby_talk_anna",
            "move__loc_lobby__loc_office",
            "office_search_desk",
            "office_inspect_painting",
            "office_unlock_archives",
            "move__loc_office__loc_archives",
            "archives_take_records",
            "archives_search_ashes",
            "move__loc_archives__loc_office",
            "move__loc_office__loc_lobby",
            "lobby_unlock_basement",
            "move__loc_lobby__loc_basement",
            "basement_follow_trail",
            "basement_rescue_samuel",
            "move__loc_basement__loc_chapel",
            "chapel_talk_doyle",
            "chapel_inspect_altar",
            "chapel_inspect_altar",
            "chapel_enter_crypt",
            "crypt_disrupt_ritual",
            "crypt_return_chapel",
            "move__loc_chapel__loc_road",
            "road_escape",
        ]:
            view = act(action_id)

        state = service.repository.load(session_id)
        reloaded_digest = service.repository.canonical_digest(state)
        replay = service.export_replay(session_id)
        assert state.status == SessionStatus.WON
        assert state.ending_id == "truth_rescue_and_silence"
        assert reloaded_digest == replay["canonical_digest"]
        assert state.flags["ritual_disrupted"] and state.flags["samuel_rescued"]
        assert len(state.discovered_clue_ids) >= service.scenario.minimum_evidence
        replay_ok, expected_simulation, replayed_simulation = verify_replay(replay, service.scenario)
        assert replay_ok, (expected_simulation, replayed_simulation)
        print("\nPASS: truth_rescue_and_silence")
        print(f"events={len(state.event_log)} rolls={len(state.rolls)} interventions={len(state.director.interventions)}")
        print(f"canonical_digest={reloaded_digest}")
        print(f"replay_verified={replay_ok} simulation_digest={expected_simulation}")


if __name__ == "__main__":
    run()
