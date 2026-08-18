from __future__ import annotations

from datetime import datetime

import pytest

from living_tabletop.kernel import KernelValidationError, WorldKernel
from living_tabletop.models import Effect


def test_locked_locations_are_not_available(scenario, state):
    kernel = WorldKernel(scenario)
    ids = {action.id for action in kernel.available_actions(state)}
    assert "move__loc_lobby__loc_basement" not in ids
    assert "lobby_unlock_basement" not in ids


def test_immutable_fact_mutation_fails_closed(scenario, state):
    kernel = WorldKernel(scenario)
    with pytest.raises(KernelValidationError, match="Immutable"):
        kernel.apply_effect(
            state,
            Effect(op="set_fact", params={"fact_id": "f_wren_leader", "value": "player"}),
            source="illegal-director",
        )
    assert state.facts["f_wren_leader"].value == "faction_choir"


def test_time_executes_events_in_order_and_interrupts(scenario, state):
    kernel = WorldKernel(scenario)
    visible, interrupt = kernel.advance_time(state, 30)
    assert interrupt is not None
    assert interrupt.id == "scheduled_power_failure"
    assert state.world_time == datetime.fromisoformat("1927-12-17T21:18:00")
    assert state.facts["f_power_on"].value is False
    assert [event.type for event in visible] == ["hospital_power_failed"]


def test_location_scoped_event_interrupts_only_at_affected_location(scenario, state):
    kernel = WorldKernel(scenario)
    kernel.advance_time(state, 30)  # global power interruption at 21:18
    kernel.advance_time(state, 20)  # ritual at 21:30, continue to 21:38
    state.entities["player"].location = "loc_archives"
    _, interrupt = kernel.advance_time(state, 10)
    assert interrupt and interrupt.id == "scheduled_wilson_burns_records"
    assert state.world_time == datetime.fromisoformat("1927-12-17T21:42:00")
    assert state.facts["f_records_intact"].value is False
    assert state.entities["npc_wilson"].location == "loc_chapel"


def test_revealing_fact_updates_player_knowledge_and_clue(scenario, state):
    kernel = WorldKernel(scenario)
    kernel.apply_effect(
        state,
        Effect(op="reveal_fact", params={"fact_id": "f_missing_note"}),
        source="test",
    )
    assert "f_missing_note" in state.player_known_fact_ids
    assert "clue_missing_note" in state.discovered_clue_ids
    assert state.event_log[-1].type == "player_learned_fact"


def test_npc_hidden_knowledge_is_not_player_knowledge(state):
    assert any(k.knower_id == "npc_wren" and k.fact_id == "f_wren_leader" for k in state.npc_knowledge)
    assert "f_wren_leader" not in state.player_known_fact_ids


def test_illegal_movement_is_rejected(scenario, state):
    kernel = WorldKernel(scenario)
    with pytest.raises(KernelValidationError, match="Invalid entity movement"):
        kernel.apply_effect(
            state,
            Effect(op="move_entity", params={"entity_id": "player", "destination": "nowhere"}),
            source="test",
        )


def test_dynamic_soft_canon_location_can_be_created_but_player_cannot_be_replaced(scenario, state):
    kernel = WorldKernel(scenario)
    kernel.apply_effect(
        state,
        Effect(
            op="create_entity",
            params={
                "entity": {
                    "id": "dynamic_location_station",
                    "type": "LOCATION",
                    "name": "临时车站",
                    "attributes": {"description": "由开放行动建立的地点。"},
                    "tags": ["off_main"],
                }
            },
        ),
        source="test",
    )
    assert {"dynamic", "soft_canon", "off_main"}.issubset(
        state.entities["dynamic_location_station"].tags
    )
    with pytest.raises(KernelValidationError):
        kernel.apply_effect(
            state,
            Effect(
                op="create_entity",
                params={
                    "entity": {
                        "id": "another_player",
                        "type": "PLAYER",
                        "name": "替代调查员",
                    }
                },
            ),
            source="test",
        )
