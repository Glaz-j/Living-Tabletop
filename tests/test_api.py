from __future__ import annotations

from threading import Event
from time import perf_counter, sleep
from types import SimpleNamespace

from fastapi.testclient import TestClient

from living_tabletop.api import create_app
from living_tabletop.llm import LLMSettings, OpenAICompatibleLLM, RoutedLLM
from living_tabletop.models import LLMResult
from living_tabletop.service import GameService


def make_client(tmp_path):
    service = GameService(
        db_path=tmp_path / "api.db",
        llm_settings=LLMSettings(enabled=False, api_key=None),
    )
    return TestClient(create_app(service)), service


def test_health_does_not_expose_secret(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["llm"]["api_key_configured"] is False
    assert response.json()["llm"]["max_retries"] == 1
    assert response.json()["action_model"] == "validated_agent_runtime_v2"
    assert response.json()["dialogue_mode"] == "llm_first_world_guarded_soft_canon_v1"
    assert "api_key" not in response.text.replace("api_key_configured", "")


def test_frontend_includes_inline_network_failure_feedback(tmp_path):
    client, _ = make_client(tmp_path)
    script = client.get("/static/app.js")
    page = client.get("/")
    styles = client.get("/static/styles.css")
    assert script.status_code == page.status_code == styles.status_code == 200
    assert "无法连接游戏服务" in script.text
    assert "行动未提交" in script.text
    assert 'window.addEventListener("offline"' in script.text
    assert 'id="clarification"' in page.text
    assert "你可以离开现场或偏离主线" in page.text
    assert 'id="continue-narrative"' in page.text
    assert 'id="skip-narrative"' in page.text
    assert 'id="interrupt-performance"' in page.text
    assert 'id="model-toggle"' in page.text
    assert 'id="model-mode"' in page.text
    assert "/api/llm/config" in script.text
    assert "测试当前模型" in page.text
    assert 'id="dialogue-options"' in page.text
    assert 'id="scene-stage"' in page.text
    assert 'id="scene-actors"' in page.text
    assert 'id="world-map-summary"' in page.text
    assert 'id="decision-divider"' in page.text
    assert "scheduleNarrativePoll" in script.text
    assert "renderedBeatId !== beat.id" in script.text
    assert "performanceActive" in script.text
    assert "scheduleNarrativeEndUnlock" not in script.text
    assert "decisionUnlocked" not in script.text
    assert "const sameSequence" in script.text
    assert "function renderScene(visual)" in script.text
    assert "data-scene-action-id" in script.text
    assert 'classList.toggle("performance-active", active)' in script.text
    assert "readNarrativeProgress" in script.text
    assert "saveNarrativeProgress" in script.text
    assert "sessionStorage" in script.text
    assert 'playback.status !== "pending"' in script.text
    assert "你要如何打断当前演出" in script.text
    assert "aria-live=\"polite\"" in page.text
    assert ".clarification.request-error" in styles.text


def _model_provider(name, models):
    class Models:
        def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id=model) for model in models])

    provider = OpenAICompatibleLLM(
        LLMSettings(enabled=True, api_key="test-only", model=models[0]),
        provider=name,
    )
    provider._client = SimpleNamespace(models=Models())
    return provider


def test_model_configuration_api_selects_and_persists_route(tmp_path):
    client, service = make_client(tmp_path)
    service.llm = RoutedLLM(
        local=_model_provider("local", ["qwen-local", "qwen-fast"]),
        remote=_model_provider("remote", ["gpt-remote", "gpt-creative"]),
        mode="auto",
    )

    discovered = client.get("/api/llm/config?refresh=true")
    assert discovered.status_code == 200
    assert discovered.json()["providers"]["local"]["models"] == ["qwen-fast", "qwen-local"]

    changed = client.put(
        "/api/llm/config",
        json={
            "mode": "remote",
            "local_model": "qwen-fast",
            "remote_model": "gpt-creative",
        },
    )
    assert changed.status_code == 200
    assert changed.json()["mode"] == "remote"
    assert changed.json()["providers"]["remote"]["model"] == "gpt-creative"
    assert (tmp_path / "llm_preferences.json").exists()


def test_model_configuration_rejects_unknown_discovered_model(tmp_path):
    client, service = make_client(tmp_path)
    service.llm = RoutedLLM(
        local=_model_provider("local", ["qwen-local"]),
        remote=_model_provider("remote", ["gpt-remote"]),
        mode="auto",
    )
    client.get("/api/llm/config?refresh=true")

    response = client.put(
        "/api/llm/config",
        json={"mode": "local", "local_model": "missing-model"},
    )

    assert response.status_code == 422
    assert "Unknown local model" in response.json()["detail"]


def test_session_action_and_developer_flow(tmp_path):
    client, _ = make_client(tmp_path)
    created = client.post("/api/sessions", json={"player_name": "林默", "seed": 19})
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    action = client.post(f"/api/sessions/{session_id}/actions", json={"action_id": "lobby_guestbook"})
    assert action.status_code == 200
    assert action.json()["version"] == 2
    developer = client.get(f"/api/sessions/{session_id}/developer")
    assert developer.status_code == 200
    assert "director" in developer.json()
    replay = client.get(f"/api/sessions/{session_id}/replay")
    assert replay.status_code == 200
    assert replay.json()["action_inputs"]


def test_reading_narrative_beats_does_not_advance_world_state(tmp_path):
    client, _ = make_client(tmp_path)
    created = client.post("/api/sessions", json={}).json()
    session_id = created["session_id"]
    assert len(created["narrative_sequence"]["beats"]) >= 2
    acted = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"action_id": "move__loc_lobby__loc_ward"},
    ).json()
    sequence = acted["narrative_sequence"]
    assert len(sequence["beats"]) >= 4
    version = acted["version"]
    world_time = acted["world_time"]

    for _ in range(3):
        status = client.get(
            f"/api/sessions/{session_id}/narrative/{sequence['id']}"
        )
        assert status.status_code == 200

    unchanged = client.get(f"/api/sessions/{session_id}").json()
    assert unchanged["version"] == version
    assert unchanged["world_time"] == world_time


def test_action_requires_exactly_one_input(tmp_path):
    client, _ = make_client(tmp_path)
    session_id = client.post("/api/sessions", json={}).json()["session_id"]
    response = client.post(f"/api/sessions/{session_id}/actions", json={})
    assert response.status_code == 422


def test_free_text_llm_failure_returns_503_without_mutating_session(tmp_path):
    client, _ = make_client(tmp_path)
    created = client.post("/api/sessions", json={}).json()
    session_id = created["session_id"]

    response = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"text": "他们在房子里住了多久？"},
    )

    assert response.status_code == 503
    assert "行动未提交" in response.json()["detail"]
    unchanged = client.get(f"/api/sessions/{session_id}").json()
    assert unchanged["version"] == created["version"]
    assert unchanged["world_time"] == created["world_time"]


def test_dialogue_utterance_can_accompany_authored_action(tmp_path):
    client, _ = make_client(tmp_path)
    session_id = client.post("/api/sessions", json={}).json()["session_id"]
    utterance = "“安娜，我想听你亲口说出昨晚发生的事。”"
    response = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"action_id": "lobby_talk_anna", "utterance": utterance},
    )
    assert response.status_code == 200
    assert response.json()["narrative_sequence"]["beats"][0]["text"] == utterance


def test_missing_session_returns_404(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/sessions/missing").status_code == 404


def test_interactive_rule_choice_survives_api_round_trip(tmp_path):
    client, _ = make_client(tmp_path)
    pending = None
    for seed in range(1, 200):
        created = client.post("/api/sessions", json={"seed": seed}).json()
        response = client.post(
            f"/api/sessions/{created['session_id']}/actions",
            json={"action_id": "lobby_guestbook", "interactive_rules": True},
        )
        if response.json().get("rule_prompt"):
            pending = response.json()
            break
    assert pending is not None
    version = pending["version"]
    resolved = client.post(
        f"/api/sessions/{pending['session_id']}/actions",
        json={"rule_choice": "accept_failure"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["rule_prompt"] is None
    assert resolved.json()["version"] == version + 1


def test_action_returns_before_background_narrator_and_stale_sequence_is_ignored(tmp_path):
    class BlockingNarratorLLM:
        enabled = True

        def __init__(self):
            self.started = Event()
            self.release = Event()

        def complete_json(self, **kwargs):
            self.started.set()
            self.release.wait(timeout=3)
            action = kwargs["user_payload"].get("resolved_action") or "未知行动"
            return LLMResult(
                data={"beats": [f"后台补充：{action}"]},
                latency_ms=1500,
                input_tokens=100,
                output_tokens=20,
            )

    client, service = make_client(tmp_path)
    fake = BlockingNarratorLLM()
    service.llm = fake
    for engine in service.engines.values():
        engine.llm = fake
        engine.keeper.llm = fake
        engine.narrator.llm = fake

    created = client.post("/api/sessions", json={}).json()
    session_id = created["session_id"]
    started = perf_counter()
    first = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"action_id": "lobby_guestbook"},
    ).json()
    elapsed = perf_counter() - started
    assert elapsed < 0.75
    assert first["narrative_sequence"]["status"] == "pending"
    assert fake.started.wait(timeout=1)

    second = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"action_id": "wait_five_minutes"},
    ).json()
    second_sequence = second["narrative_sequence"]
    assert second_sequence["state_version"] == 3
    fake.release.set()

    final = None
    for _ in range(100):
        final = client.get(
            f"/api/sessions/{session_id}/narrative/{second_sequence['id']}"
        ).json()
        if final["status"] != "pending":
            break
        sleep(0.02)
    assert final is not None and final["status"] == "ready"
    generated = [beat["text"] for beat in final["beats"] if beat["source"] == "generated"]
    assert generated == ["后台补充：停下来观察五分钟"]

    stale = client.get(
        f"/api/sessions/{session_id}/narrative/{first['narrative_sequence']['id']}"
    ).json()
    assert stale["superseded"] is True
    assert stale["state_version"] == 3
    saved = service.repository.load(session_id)
    assert any(
        entry.source == "generated" and "后台补充" in entry.text
        for entry in saved.visible_history
    )
    service.close()
