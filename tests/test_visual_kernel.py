from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from living_tabletop.api import create_app
from living_tabletop.llm import LLMSettings
from living_tabletop.service import GameService
from living_tabletop.visual_kernel import CommandEnvelope, CommandRejected, VisualWorldKernel
from living_tabletop.visual_kernel.projection import dev_projection, player_projection
from living_tabletop.visual_kernel.service import VisualWorldService, load_demo_definition
from living_tabletop.visual_kernel.storage import DefinitionVersionConflict


def command(state, kind, payload, *, issuer="player", actor="player", key="test"):
    return CommandEnvelope(
        command_id=f"cmd-{key}",
        session_id=state.session_id,
        issuer_id=issuer,
        actor_id=actor,
        kind=kind,
        payload=payload,
        expected_state_version=state.version,
        idempotency_key=key,
    )


def test_definition_is_versioned_and_referentially_valid():
    definition = load_demo_definition()
    assert definition.version == "1.0.1"
    assert len(definition.content_digest) == 64
    round_trip = type(definition).model_validate_json(definition.model_dump_json())
    assert round_trip.content_digest == definition.content_digest
    assert {item.id for item in definition.locations} >= {"loc_street", "loc_basement"}


def test_in_world_movement_failure_is_an_accepted_attempt():
    kernel = VisualWorldKernel(load_demo_definition())
    state = kernel.initial_state("world-1")
    next_state, receipt = kernel.process(
        state,
        command(state, "move", {"destination_id": "loc_basement"}),
    )
    assert receipt.accepted is True
    assert receipt.outcome == "failed"
    assert receipt.reason == "no_connection"
    assert kernel.root_location(next_state, "player") == "loc_street"
    assert next_state.version == 1
    assert [event.type for event in next_state.event_log] == [
        "MovementAttempted",
        "TimeAdvanced",
        "MovementAttemptFailed",
    ]


def test_structurally_invalid_and_stale_commands_are_rejected():
    kernel = VisualWorldKernel(load_demo_definition())
    state = kernel.initial_state("world-2")
    with pytest.raises(CommandRejected, match="existing destination"):
        kernel.process(state, command(state, "move", {"destination_id": "missing"}))
    stale = command(state, "wait", {"minutes": 5})
    stale.expected_state_version = 99
    with pytest.raises(CommandRejected) as caught:
        kernel.process(state, stale)
    assert caught.value.code == "stale_version"


def test_director_has_no_set_state_backdoor():
    kernel = VisualWorldKernel(load_demo_definition())
    state = kernel.initial_state("world-3")
    with pytest.raises(CommandRejected) as caught:
        kernel.process(
            state,
            command(
                state,
                "set_state",
                {"variable_id": "world.storm_active", "value": True},
                issuer="director",
                actor=None,
            ),
        )
    assert caught.value.code == "unauthorized"

    next_state, receipt = kernel.process(
        state,
        command(
            state,
            "set_state",
            {"variable_id": "world.storm_active", "value": True},
            issuer="system",
            actor=None,
            key="system-state",
        ),
    )
    assert receipt.outcome == "succeeded"
    assert next_state.state_variables["world.storm_active"] is True


def test_projection_hides_truth_and_live_npc_positions():
    kernel = VisualWorldKernel(load_demo_definition())
    state = kernel.initial_state("world-4")
    player = player_projection(kernel, state, "player")
    dev = dev_projection(kernel, state)
    assert "loc_basement" not in {item["id"] for item in player["map"]["locations"]}
    assert "npc_caretaker" not in {item["id"] for item in player["visible_entities"]}
    assert "loc_basement" in {item["id"] for item in dev["map"]["locations"]}
    assert dev["runtime"]["entities"]["npc_caretaker"]["placement"]["target_id"] == "loc_foyer"


def test_projection_only_offers_take_for_portable_objects():
    kernel = VisualWorldKernel(load_demo_definition())
    state = kernel.initial_state("world-portable")
    for destination in ("loc_foyer", "loc_hall"):
        state, _ = kernel.process(
            state,
            command(state, "move", {"destination_id": destination}, key=f"move-{destination}"),
        )
    projection = player_projection(kernel, state, "player")
    labels = {item["label"] for item in projection["affordances"]}
    assert "拿起黄铜钥匙" in labels
    assert "拿起沉重书桌" not in labels
    assert "拿起墙边的拖痕" not in labels


def test_end_to_end_event_replay_interrupt_and_idempotency(tmp_path):
    service = VisualWorldService(db_path=tmp_path / "visual.db")
    created = service.create_session()
    session_id = created["session_id"]
    version = created["projection"]["world_version"]
    steps = [
        ("move", {"destination_id": "loc_foyer"}, "move-foyer"),
        ("move", {"destination_id": "loc_hall"}, "move-hall"),
        ("inspect", {"target_id": "obj_cellar_marks"}, "inspect-marks"),
        ("interact", {"target_id": "obj_brass_key", "verb": "take"}, "take-key"),
        ("interact", {"target_id": "conn_hall_basement", "verb": "unlock"}, "unlock-1"),
    ]
    result = None
    for kind, payload, key in steps:
        result = service.command(
            session_id,
            kind=kind,
            payload=payload,
            expected_state_version=version,
            idempotency_key=key,
        )
        version = result["projection"]["world_version"]
    assert result is not None
    assert result["receipt"]["outcome"] == "interrupted"
    assert result["projection"]["world"]["time"] == "1920-09-18T17:56:00"

    duplicate = service.command(
        session_id,
        kind="interact",
        payload={"target_id": "conn_hall_basement", "verb": "unlock"},
        expected_state_version=version,
        idempotency_key="unlock-1",
    )
    assert duplicate["receipt"]["outcome"] == "duplicate"
    assert duplicate["projection"]["world_version"] == version

    for kind, payload, key in [
        ("interact", {"target_id": "conn_hall_basement", "verb": "unlock"}, "unlock-2"),
        ("interact", {"target_id": "conn_hall_basement", "verb": "open"}, "open"),
        ("move", {"destination_id": "loc_basement"}, "move-basement"),
    ]:
        result = service.command(
            session_id,
            kind=kind,
            payload=payload,
            expected_state_version=version,
            idempotency_key=key,
        )
        version = result["projection"]["world_version"]
    assert result["projection"]["observer"]["location_id"] == "loc_basement"
    assert service.replay(session_id)["verified"] is True


def test_event_log_is_authoritative_when_snapshot_cache_is_corrupted(tmp_path):
    db_path = tmp_path / "event-authority.db"
    service = VisualWorldService(db_path=db_path)
    created = service.create_session()
    session_id = created["session_id"]
    service.command(
        session_id,
        kind="wait",
        payload={"minutes": 5},
        expected_state_version=0,
        idempotency_key="wait",
    )
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT snapshot_json FROM vwk_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        snapshot = json.loads(row[0])
        snapshot["clock"] = "1999-01-01T00:00:00"
        connection.execute(
            "UPDATE vwk_sessions SET snapshot_json = ? WHERE session_id = ?",
            (json.dumps(snapshot), session_id),
        )
    projection = service.projection(session_id)
    assert projection["world"]["time"] == "1920-09-18T17:50:00"
    assert service.replay(session_id)["verified"] is False


def test_definition_version_is_immutable_after_pin(tmp_path):
    service = VisualWorldService(db_path=tmp_path / "pin.db")
    service.create_session()
    modified = service.definition.model_copy(deep=True)
    modified.title = "同版本却被修改的世界"
    kernel = VisualWorldKernel(modified)
    with pytest.raises(DefinitionVersionConflict):
        service.repository.create_session(modified, kernel.initial_state("other-session"))


def test_visual_kernel_api_and_page(tmp_path):
    game = GameService(
        db_path=tmp_path / "game.db",
        llm_settings=LLMSettings(enabled=False, api_key=None),
    )
    worlds = VisualWorldService(db_path=tmp_path / "worlds.db")
    with TestClient(create_app(game, worlds)) as client:
        page = client.get("/world-kernel")
        assert page.status_code == 200
        assert "观察者认知地图" in page.text
        created = client.post("/api/world-kernel/sessions", json={})
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        acted = client.post(
            f"/api/world-kernel/sessions/{session_id}/commands",
            json={
                "kind": "move",
                "payload": {"destination_id": "loc_foyer"},
                "expected_state_version": 0,
                "idempotency_key": "api-move",
            },
        )
        assert acted.status_code == 200
        assert acted.json()["receipt"]["outcome"] == "succeeded"
        dev = client.get(
            f"/api/world-kernel/sessions/{session_id}/projection?view=dev"
        )
        assert dev.status_code == 200
        assert dev.json()["projection_type"] == "dev"
        replay = client.get(f"/api/world-kernel/sessions/{session_id}/replay")
        assert replay.json()["verified"] is True
    game.close()
