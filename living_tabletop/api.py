from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .llm import LLMUnavailable
from .models import RuleChoice
from .service import GameService, ScenarioNotFound
from .storage import SessionNotFound
from .visual_kernel import (
    CommandKind,
    CommandRejected,
    ConcurrencyConflict,
    VisualWorldService,
    VisualWorldSessionNotFound,
)


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(APIModel):
    player_name: str = Field(default="调查员", min_length=1, max_length=40)
    seed: int = Field(default=1927, ge=0, le=2**31 - 1)
    scenario_id: str | None = Field(default=None, min_length=1, max_length=100)


class ActionRequest(APIModel):
    action_id: str | None = None
    text: str | None = Field(default=None, max_length=1000)
    utterance: str | None = Field(default=None, max_length=500)
    rule_choice: RuleChoice | None = None
    interactive_rules: bool = False

    @model_validator(mode="after")
    def one_input(self) -> "ActionRequest":
        supplied = sum(
            (
                bool(self.action_id),
                bool(self.text and self.text.strip()),
                self.rule_choice is not None,
            )
        )
        if supplied != 1:
            raise ValueError("Provide exactly one of action_id, text, or rule_choice")
        if self.utterance and not self.action_id:
            raise ValueError("utterance can only accompany action_id")
        return self


class LLMConfigurationRequest(APIModel):
    mode: Literal["auto", "local", "remote"]
    local_model: str | None = Field(default=None, min_length=1, max_length=160)
    remote_model: str | None = Field(default=None, min_length=1, max_length=160)


class LLMProbeRequest(APIModel):
    provider: Literal["local", "remote"] | None = None


class CreateVisualWorldSessionRequest(APIModel):
    viewer_id: str = Field(default="player", min_length=1, max_length=100)


class VisualWorldCommandRequest(APIModel):
    kind: CommandKind
    payload: dict[str, Any] = Field(default_factory=dict)
    expected_state_version: int = Field(ge=0)
    issuer_id: str = Field(default="player", min_length=1, max_length=100)
    actor_id: str | None = Field(default="player", max_length=100)
    command_id: str | None = Field(default=None, min_length=1, max_length=120)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=120)


@lru_cache(maxsize=1)
def get_service() -> GameService:
    return GameService()


@lru_cache(maxsize=1)
def get_visual_world_service() -> VisualWorldService:
    return VisualWorldService()


ServiceDependency = Annotated[GameService, Depends(get_service)]
VisualWorldServiceDependency = Annotated[
    VisualWorldService, Depends(get_visual_world_service)
]


def create_app(
    service: GameService | None = None,
    visual_service: VisualWorldService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Living Tabletop V0",
        version="0.1.0",
        description="LLM-first, world-guarded AI TTRPG demo",
    )
    if service is not None:
        app.dependency_overrides[get_service] = lambda: service
    if visual_service is not None:
        app.dependency_overrides[get_visual_world_service] = lambda: visual_service

    static_dir = files("living_tabletop").joinpath("static")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(static_dir.joinpath("index.html")))

    @app.get("/world-kernel", include_in_schema=False)
    def visual_world_kernel_page():
        return FileResponse(str(static_dir.joinpath("world-kernel.html")))

    @app.get("/api/health")
    def health(game: ServiceDependency):
        return game.health()

    @app.get("/api/llm/config")
    def llm_configuration(game: ServiceDependency, refresh: bool = False):
        return game.llm_configuration(refresh=refresh)

    @app.put("/api/llm/config")
    def configure_llm(request: LLMConfigurationRequest, game: ServiceDependency):
        try:
            return game.configure_llm(
                mode=request.mode,
                local_model=request.local_model,
                remote_model=request.remote_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/llm/probe")
    def probe_llm(request: LLMProbeRequest, game: ServiceDependency):
        try:
            return game.probe_llm(provider_name=request.provider)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail=exc.public_message
                or "模型暂时不可用，请检查右上角的模型设置后重试。",
            ) from exc

    @app.get("/api/sessions")
    def list_sessions(game: ServiceDependency):
        return {"sessions": game.repository.list_sessions()}

    @app.get("/api/scenarios")
    def list_scenarios(game: ServiceDependency):
        return {"scenarios": game.scenario_catalog()}

    @app.post("/api/world-kernel/sessions", status_code=201)
    def create_visual_world_session(
        request: CreateVisualWorldSessionRequest,
        worlds: VisualWorldServiceDependency,
    ):
        try:
            return worlds.create_session(viewer_id=request.viewer_id)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/world-kernel/sessions/{session_id}/projection")
    def visual_world_projection(
        session_id: str,
        worlds: VisualWorldServiceDependency,
        viewer_id: str = "player",
        view: Literal["player", "dev"] = "player",
    ):
        try:
            return worlds.projection(session_id, viewer_id=viewer_id, view=view)
        except VisualWorldSessionNotFound as exc:
            raise HTTPException(status_code=404, detail="Visual world session not found") from exc
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/world-kernel/sessions/{session_id}/commands")
    def visual_world_command(
        session_id: str,
        request: VisualWorldCommandRequest,
        worlds: VisualWorldServiceDependency,
    ):
        try:
            return worlds.command(
                session_id,
                kind=request.kind,
                payload=request.payload,
                expected_state_version=request.expected_state_version,
                issuer_id=request.issuer_id,
                actor_id=request.actor_id,
                command_id=request.command_id,
                idempotency_key=request.idempotency_key,
            )
        except VisualWorldSessionNotFound as exc:
            raise HTTPException(status_code=404, detail="Visual world session not found") from exc
        except CommandRejected as exc:
            status = 409 if exc.code == "stale_version" else 422
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        except ConcurrencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/world-kernel/sessions/{session_id}/replay")
    def visual_world_replay(
        session_id: str,
        worlds: VisualWorldServiceDependency,
    ):
        try:
            return worlds.replay(session_id)
        except VisualWorldSessionNotFound as exc:
            raise HTTPException(status_code=404, detail="Visual world session not found") from exc

    @app.post("/api/sessions", status_code=201)
    def create_session(request: CreateSessionRequest, game: ServiceDependency):
        try:
            return game.create_session(
                player_name=request.player_name,
                seed=request.seed,
                scenario_id=request.scenario_id,
            )
        except ScenarioNotFound as exc:
            raise HTTPException(status_code=404, detail="Scenario not found") from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, game: ServiceDependency):
        try:
            return game.get_session(session_id)
        except (SessionNotFound, ScenarioNotFound) as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    @app.post("/api/sessions/{session_id}/actions")
    def act(session_id: str, request: ActionRequest, game: ServiceDependency):
        try:
            return game.act(
                session_id,
                action_id=request.action_id,
                text=request.text,
                utterance=request.utterance,
                rule_choice=request.rule_choice,
                interactive_rules=request.interactive_rules,
            )
        except (SessionNotFound, ScenarioNotFound) as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except LLMUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail=exc.public_message
                or "暂时无法连接 LLM，行动未提交，游戏状态没有改变。请稍后重试。",
            ) from exc

    @app.get("/api/sessions/{session_id}/developer")
    def developer(session_id: str, game: ServiceDependency):
        try:
            return game.developer_view(session_id)
        except (SessionNotFound, ScenarioNotFound) as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    @app.get("/api/sessions/{session_id}/narrative/{sequence_id}")
    def narrative_status(session_id: str, sequence_id: str, game: ServiceDependency):
        try:
            return game.narrative_status(session_id, sequence_id)
        except (SessionNotFound, ScenarioNotFound) as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    @app.get("/api/sessions/{session_id}/replay")
    def replay(session_id: str, game: ServiceDependency):
        try:
            return game.export_replay(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    return app


app = create_app()
