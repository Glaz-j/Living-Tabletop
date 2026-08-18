from __future__ import annotations

from living_tabletop.models import SessionStatus


def test_complete_truth_and_rescue_playthrough(engine, state):
    def act(action_id):
        nonlocal state
        state, resolution = engine.play(state, action_id=action_id)
        assert resolution.accepted, (action_id, resolution)
        return resolution

    for _ in range(2):
        resolution = act("lobby_guestbook")
        if "f_missing_note" in state.player_known_fact_ids:
            break
    act("lobby_talk_anna")  # intentionally interrupted by the 21:18 outage
    act("move__loc_lobby__loc_office")
    while "item_archive_key" not in state.player.inventory:
        act("office_search_desk")
    for _ in range(2):
        if "f_cult_photo" in state.player_known_fact_ids:
            break
        act("office_inspect_painting")
    act("office_unlock_archives")
    act("move__loc_office__loc_archives")
    act("archives_take_records")  # Wilson burns them during the search, interrupting it.
    act("archives_search_ashes")
    act("move__loc_archives__loc_office")
    act("move__loc_office__loc_lobby")
    act("lobby_unlock_basement")
    act("move__loc_lobby__loc_basement")
    act("basement_follow_trail")
    act("basement_rescue_samuel")
    act("move__loc_basement__loc_chapel")
    act("chapel_talk_doyle")
    act("chapel_inspect_altar")  # interrupted as the ritual begins
    act("chapel_inspect_altar")
    act("chapel_enter_crypt")
    act("crypt_disrupt_ritual")
    act("crypt_return_chapel")
    act("move__loc_chapel__loc_road")
    act("road_escape")

    assert state.status == SessionStatus.WON
    assert state.ending_id == "truth_rescue_and_silence"
    assert state.flags["ritual_disrupted"] is True
    assert state.flags["samuel_rescued"] is True
    assert len(state.discovered_clue_ids) >= engine.scenario.minimum_evidence
