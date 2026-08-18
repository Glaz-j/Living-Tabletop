from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from living_tabletop.api import create_app
from living_tabletop.engine import GameEngine
from living_tabletop.kernel import WorldKernel
from living_tabletop.llm import LLMSettings, OpenAICompatibleLLM
from living_tabletop.models import Condition, Effect, SessionStatus
from living_tabletop.scenario import (
    ScenarioIntegrityError,
    create_initial_state,
    load_scenario,
    load_scenarios,
    validate_scenario_integrity,
)
from living_tabletop.service import GameService


HAUNTING_ID = "the_haunting_corbitt_house_v1"


def test_scenario_registry_loads_both_cases_and_source_notice():
    scenarios = load_scenarios()
    assert set(scenarios) == {"st_mary_hospital_v0", HAUNTING_ID}
    haunting = scenarios[HAUNTING_ID]
    assert haunting.title == "科比特宅邸"
    assert haunting.source is not None
    assert haunting.source.publisher == "Chaosium Inc."
    assert "不等于开放许可" in haunting.source.rights_note
    assert len(haunting.clues) == 16
    assert len(haunting.actions) >= 60


def test_integrity_validator_rejects_dangling_fact_effect():
    scenario = deepcopy(load_scenario(scenario_id=HAUNTING_ID))
    scenario.actions[0].success_effects[0].params["fact_id"] = "missing_fact"
    with pytest.raises(ScenarioIntegrityError, match="unknown fact"):
        validate_scenario_integrity(scenario)


def test_integrity_validator_rejects_dangling_opposed_entity_and_bad_dice():
    scenario = deepcopy(load_scenario(scenario_id=HAUNTING_ID))
    combat = next(action for action in scenario.actions if action.id == "lair_use_dagger")
    combat.opposed.entity_id = "missing_creature"
    combat.failure_effects[0].params["amount"] = "many-dice"
    with pytest.raises(ScenarioIntegrityError) as exc:
        validate_scenario_integrity(scenario)
    assert "opposes unknown entity" in str(exc.value)
    assert "invalid damage expression" in str(exc.value)


def test_api_catalog_creates_and_restores_selected_scenario(tmp_path):
    service = GameService(
        db_path=tmp_path / "multi.db",
        llm_settings=LLMSettings(enabled=False, api_key=None),
    )
    client = TestClient(create_app(service))
    catalog = client.get("/api/scenarios")
    assert catalog.status_code == 200
    assert {item["id"] for item in catalog.json()["scenarios"]} == {
        "st_mary_hospital_v0",
        HAUNTING_ID,
    }

    created = client.post(
        "/api/sessions",
        json={"scenario_id": HAUNTING_ID, "player_name": "测试调查员", "seed": 19},
    )
    assert created.status_code == 201
    view = created.json()
    assert view["scenario"]["id"] == HAUNTING_ID
    assert view["scene"]["id"] == "loc_cafe"
    assert view["scene"]["visual"]["location_id"] == "loc_cafe"
    assert view["scene"]["visual"]["archetype"] == "civic"
    assert any(actor["id"] == "npc_knott" for actor in view["scene"]["visual"]["actors"])

    acted = client.post(
        f"/api/sessions/{view['session_id']}/actions",
        json={"action_id": "cafe_question_knott"},
    )
    assert acted.status_code == 200
    assert acted.json()["scenario"]["id"] == HAUNTING_ID
    assert client.get(f"/api/sessions/{view['session_id']}").json()["version"] == 2


def test_api_rejects_unknown_scenario(tmp_path):
    service = GameService(
        db_path=tmp_path / "unknown.db",
        llm_settings=LLMSettings(enabled=False, api_key=None),
    )
    client = TestClient(create_app(service))
    response = client.post("/api/sessions", json={"scenario_id": "missing"})
    assert response.status_code == 404


def test_conditional_effect_makes_basement_shield_mechanical():
    scenario = load_scenario(scenario_id=HAUNTING_ID)
    state = create_initial_state(scenario)
    state.player.inventory.append("item_trash_lid")
    kernel = WorldKernel(scenario)
    kernel.apply_effect(
        state,
        Effect(
            op="modify_player",
            params={"field": "hp", "delta": -4},
            conditions=[
                Condition(
                    subject="player",
                    predicate="inventory",
                    operator="not_contains",
                    value="item_trash_lid",
                )
            ],
        ),
        source="test",
    )
    kernel.apply_effect(
        state,
        Effect(
            op="modify_player",
            params={"field": "hp", "delta": -2},
            conditions=[
                Condition(
                    subject="player",
                    predicate="inventory",
                    operator="contains",
                    value="item_trash_lid",
                )
            ],
        ),
        source="test",
    )
    assert state.player.hp == 10


def test_zero_hp_is_resolved_as_incapacitation_not_automatic_death():
    scenario = load_scenario(scenario_id=HAUNTING_ID)
    state = create_initial_state(scenario)
    state.player.hp = 1
    kernel = WorldKernel(scenario)
    kernel.apply_effect(
        state,
        Effect(op="modify_player", params={"field": "hp", "delta": -2}),
        source="test",
    )
    assert state.status == SessionStatus.ACTIVE
    assert kernel.evaluate_ending(state) == "incapacitated"
    assert state.status == SessionStatus.LOST
    assert state.ending_id == "incapacitated"
    assert "一线生机" in state.last_narrative
    assert state.event_log[-1].type == "game_ended"


def test_effect_triggered_terminal_uses_scenario_narrative_and_event():
    scenario = load_scenario()
    state = create_initial_state(scenario)
    state.world_time = datetime.fromisoformat("1927-12-17T22:59:00")
    state.event_queue = [
        deepcopy(next(event for event in scenario.events if event.id == "scheduled_ritual_complete"))
    ]
    engine = GameEngine(
        scenario,
        OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None)),
    )
    state, resolution = engine.play(state, action_id="wait_five_minutes")
    assert resolution.accepted
    assert state.status == SessionStatus.LOST
    assert state.ending_id == "ritual_complete"
    assert state.last_narrative == next(
        ending.narrative for ending in scenario.endings if ending.id == "ritual_complete"
    )
    assert len(
        [event for event in state.event_log if event.type == "game_ended" and event.target == "ritual_complete"]
    ) == 1
