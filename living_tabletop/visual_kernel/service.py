from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path
import threading
import uuid
from typing import Any, Literal

from .kernel import VisualWorldKernel
from .models import CommandEnvelope, CommandKind, WorldDefinition
from .projection import dev_projection, player_projection
from .storage import VisualWorldRepository


def load_demo_definition(path: str | Path | None = None) -> WorldDefinition:
    definition_path = (
        Path(path)
        if path
        else Path(str(files("living_tabletop").joinpath("worlds", "visual_kernel_demo.json")))
    )
    return WorldDefinition.model_validate(json.loads(definition_path.read_text(encoding="utf-8")))


class VisualWorldService:
    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        definition: WorldDefinition | None = None,
    ):
        resolved_path = db_path or os.getenv(
            "LIVING_TABLETOP_VISUAL_DB_PATH", "data/visual_world_kernel.db"
        )
        self.repository = VisualWorldRepository(resolved_path)
        self.definition = definition or load_demo_definition()
        self._lock = threading.RLock()

    def create_session(self, *, viewer_id: str = "player") -> dict[str, Any]:
        with self._lock:
            session_id = f"vwk_{uuid.uuid4().hex}"
            kernel = VisualWorldKernel(self.definition)
            state = kernel.initial_state(session_id)
            self.repository.create_session(self.definition, state)
            projection = player_projection(kernel, state, viewer_id)
            return {"session_id": session_id, "projection": projection}

    def projection(
        self,
        session_id: str,
        *,
        viewer_id: str = "player",
        view: Literal["player", "dev"] = "player",
    ) -> dict[str, Any]:
        with self._lock:
            definition, state = self.repository.load(session_id)
            kernel = VisualWorldKernel(definition)
            if view == "dev":
                return dev_projection(kernel, state)
            return player_projection(kernel, state, viewer_id)

    def command(
        self,
        session_id: str,
        *,
        kind: CommandKind | str,
        payload: dict[str, Any],
        expected_state_version: int,
        issuer_id: str = "player",
        actor_id: str | None = "player",
        command_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            resolved_command_id = command_id or f"cmd_{uuid.uuid4().hex}"
            resolved_key = idempotency_key or resolved_command_id
            previous = self.repository.get_receipt(session_id, resolved_key)
            if previous is not None:
                definition, state = self.repository.load(session_id)
                kernel = VisualWorldKernel(definition)
                duplicate = previous.model_copy(
                    update={"outcome": "duplicate", "state_version": state.version}
                )
                return {
                    "receipt": duplicate.model_dump(mode="json"),
                    "projection": player_projection(kernel, state, issuer_id),
                }

            definition, state = self.repository.load(session_id)
            kernel = VisualWorldKernel(definition)
            command = CommandEnvelope(
                command_id=resolved_command_id,
                session_id=session_id,
                issuer_id=issuer_id,
                actor_id=actor_id,
                kind=kind,
                payload=payload,
                expected_state_version=expected_state_version,
                idempotency_key=resolved_key,
            )
            next_state, receipt = kernel.process(state, command)
            events = next_state.event_log[state.event_sequence :]
            self.repository.commit(
                expected_version=state.version,
                state=next_state,
                events=events,
                receipt=receipt,
            )
            viewer_id = issuer_id if issuer_id in next_state.entities else "player"
            return {
                "receipt": receipt.model_dump(mode="json"),
                "projection": player_projection(kernel, next_state, viewer_id),
            }

    def replay(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return self.repository.replay_report(session_id)
