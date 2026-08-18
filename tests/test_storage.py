from __future__ import annotations

from living_tabletop.kernel import WorldKernel
from living_tabletop.models import Effect
from living_tabletop.storage import SQLiteRepository
from living_tabletop.replay import verify_replay


def test_snapshot_and_event_log_round_trip(tmp_path, scenario, state):
    repository = SQLiteRepository(tmp_path / "game.db")
    digest_before = repository.save(state)
    loaded = repository.load(state.session_id)
    assert loaded == state
    assert repository.canonical_digest(loaded) == digest_before


def test_event_log_is_append_only_across_saves(tmp_path, scenario, state):
    repository = SQLiteRepository(tmp_path / "game.db")
    repository.save(state)
    kernel = WorldKernel(scenario)
    kernel.apply_effect(state, Effect(op="reveal_fact", params={"fact_id": "f_missing_note"}), source="test")
    repository.save(state)
    loaded = repository.load(state.session_id)
    assert [event.seq for event in loaded.event_log] == list(range(1, len(loaded.event_log) + 1))
    assert loaded.event_log[0].type == "session_started"


def test_replay_export_contains_inputs_and_no_credentials(tmp_path, scenario, engine, state):
    repository = SQLiteRepository(tmp_path / "game.db")
    played, _ = engine.play(state, action_id="lobby_guestbook")
    repository.save(played)
    replay = repository.export(played.session_id)
    assert replay["format"] == "living-tabletop-replay-v0"
    assert replay["action_inputs"][0]["action_id"] == "lobby_guestbook"
    assert "api_key" not in str(replay).lower()


def test_recorded_actions_rebuild_same_simulation(tmp_path, scenario, engine, state):
    repository = SQLiteRepository(tmp_path / "game.db")
    for action_id in ("lobby_guestbook", "lobby_talk_anna", "move__loc_lobby__loc_office"):
        state, resolution = engine.play(state, action_id=action_id)
        assert resolution.accepted
    repository.save(state)
    verified, expected, actual = verify_replay(repository.export(state.session_id), scenario)
    assert verified, (expected, actual)
