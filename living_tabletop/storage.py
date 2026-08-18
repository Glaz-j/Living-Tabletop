from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import EventRecord, WorldState


class SessionNotFound(KeyError):
    pass


class SQLiteRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    canonical_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS event_log (
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, seq),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_event_log_session
                    ON event_log(session_id, seq);
                """
            )

    @staticmethod
    def _snapshot_payload(state: WorldState) -> dict[str, Any]:
        payload = state.model_dump(mode="json", exclude={"event_log"})
        for entity in payload["entities"].values():
            entity["tags"] = sorted(entity["tags"])
        for threat in payload["threats"].values():
            threat["crossed"] = sorted(threat["crossed"])
        for key in ("player_known_fact_ids", "discovered_clue_ids", "completed_actions"):
            payload[key] = sorted(payload[key])
        payload["player"]["checked_skills"] = sorted(payload["player"]["checked_skills"])
        return payload

    @classmethod
    def canonical_digest(cls, state: WorldState) -> str:
        payload = {
            "snapshot": cls._snapshot_payload(state),
            "event_log": [event.model_dump(mode="json") for event in state.event_log],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def simulation_digest(cls, state: WorldState) -> str:
        payload = cls._snapshot_payload(state)
        payload.pop("last_narrative", None)
        payload.pop("narrative_sequence", None)
        payload.pop("visible_history", None)
        payload.pop("agent_calls", None)
        payload.pop("turn_traces", None)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def save(self, state: WorldState) -> str:
        snapshot_json = json.dumps(self._snapshot_payload(state), ensure_ascii=False, sort_keys=True)
        digest = self.canonical_digest(state)
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT created_at FROM sessions WHERE session_id = ?", (state.session_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, scenario_id, state_version, snapshot_json,
                    canonical_digest, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    scenario_id = excluded.scenario_id,
                    state_version = excluded.state_version,
                    snapshot_json = excluded.snapshot_json,
                    canonical_digest = excluded.canonical_digest,
                    updated_at = excluded.updated_at
                """,
                (
                    state.session_id,
                    state.scenario_id,
                    state.version,
                    snapshot_json,
                    digest,
                    created_at,
                    now,
                ),
            )
            for event in state.event_log:
                event_json = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                connection.execute(
                    "INSERT OR IGNORE INTO event_log (session_id, seq, event_json) VALUES (?, ?, ?)",
                    (state.session_id, event.seq, event_json),
                )
        return digest

    def load(self, session_id: str) -> WorldState:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise SessionNotFound(session_id)
            event_rows = connection.execute(
                "SELECT event_json FROM event_log WHERE session_id = ? ORDER BY seq", (session_id,)
            ).fetchall()
        payload = json.loads(row["snapshot_json"])
        payload["event_log"] = [json.loads(event_row["event_json"]) for event_row in event_rows]
        return WorldState.model_validate(payload)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT session_id, scenario_id, state_version, canonical_digest, created_at, updated_at
                FROM sessions ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def export(self, session_id: str) -> dict[str, Any]:
        state = self.load(session_id)
        turn_inputs: list[dict[str, Any]] = []
        for event in state.event_log:
            if event.type == "action_started":
                turn_inputs.append(
                    {
                        "kind": "action",
                        "action_id": event.payload.get("action_id"),
                        "player_text": event.payload.get("player_text"),
                        "intent_source": event.payload.get("intent_source"),
                        "interactive_rules": bool(event.payload.get("interactive_rules", False)),
                        "open_plan": event.payload.get("open_plan"),
                    }
                )
            elif event.type == "rule_choice_made":
                turn_inputs.append(
                    {
                        "kind": "rule_choice",
                        "choice": event.payload.get("choice"),
                    }
                )
        return {
            "format": "living-tabletop-replay-v0",
            "scenario_id": state.scenario_id,
            "session_id": state.session_id,
            "rng_seed": state.rng_seed,
            "rng_draws": state.rng_draws,
            "state_version": state.version,
            "canonical_digest": self.canonical_digest(state),
            "simulation_digest": self.simulation_digest(state),
            "snapshot": self._snapshot_payload(state),
            "event_log": [event.model_dump(mode="json") for event in state.event_log],
            "recorded_agent_outputs": [call.model_dump(mode="json") for call in state.agent_calls],
            "turn_inputs": turn_inputs,
            "action_inputs": [
                {
                    "action_id": event.payload.get("action_id"),
                    "player_text": event.payload.get("player_text"),
                    "intent_source": event.payload.get("intent_source"),
                    "open_plan": event.payload.get("open_plan"),
                }
                for event in state.event_log
                if event.type == "action_started"
            ],
        }
