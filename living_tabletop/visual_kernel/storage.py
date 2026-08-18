from __future__ import annotations

from datetime import datetime
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .kernel import VisualWorldKernel
from .models import CommandReceipt, DomainEvent, WorldDefinition, WorldRuntime


class VisualWorldSessionNotFound(KeyError):
    pass


class ConcurrencyConflict(RuntimeError):
    pass


class DefinitionVersionConflict(RuntimeError):
    pass


class VisualWorldRepository:
    """SQLite event store with a non-authoritative snapshot cache."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vwk_definitions (
                    definition_id TEXT NOT NULL,
                    definition_version TEXT NOT NULL,
                    definition_digest TEXT NOT NULL,
                    definition_json TEXT NOT NULL,
                    PRIMARY KEY (definition_id, definition_version)
                );
                CREATE TABLE IF NOT EXISTS vwk_sessions (
                    session_id TEXT PRIMARY KEY,
                    definition_id TEXT NOT NULL,
                    definition_version TEXT NOT NULL,
                    definition_digest TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    event_sequence INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (definition_id, definition_version)
                        REFERENCES vwk_definitions(definition_id, definition_version)
                );
                CREATE TABLE IF NOT EXISTS vwk_events (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence),
                    FOREIGN KEY (session_id) REFERENCES vwk_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS vwk_command_receipts (
                    session_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    command_id TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, idempotency_key),
                    UNIQUE (session_id, command_id),
                    FOREIGN KEY (session_id) REFERENCES vwk_sessions(session_id)
                );
                """
            )

    @staticmethod
    def _json(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def snapshot_payload(cls, state: WorldRuntime) -> dict[str, Any]:
        return state.model_dump(mode="json", exclude={"event_log"})

    @classmethod
    def canonical_digest(cls, state: WorldRuntime) -> str:
        return sha256(cls._json(cls.snapshot_payload(state)).encode("utf-8")).hexdigest()

    def create_session(self, definition: WorldDefinition, state: WorldRuntime) -> None:
        now = datetime.now().isoformat()
        definition_json = self._json(definition)
        snapshot_json = self._json(self.snapshot_payload(state))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT definition_digest FROM vwk_definitions
                   WHERE definition_id = ? AND definition_version = ?""",
                (definition.id, definition.version),
            ).fetchone()
            if existing and existing["definition_digest"] != definition.content_digest:
                raise DefinitionVersionConflict(
                    "A definition version is immutable once a session has pinned it"
                )
            connection.execute(
                """INSERT OR IGNORE INTO vwk_definitions (
                       definition_id, definition_version, definition_digest, definition_json
                   ) VALUES (?, ?, ?, ?)""",
                (definition.id, definition.version, definition.content_digest, definition_json),
            )
            connection.execute(
                """INSERT INTO vwk_sessions (
                       session_id, definition_id, definition_version, definition_digest,
                       state_version, event_sequence, snapshot_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    state.session_id,
                    definition.id,
                    definition.version,
                    definition.content_digest,
                    state.version,
                    state.event_sequence,
                    snapshot_json,
                    now,
                    now,
                ),
            )

    def _session_row(self, connection: sqlite3.Connection, session_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM vwk_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise VisualWorldSessionNotFound(session_id)
        return row

    def load_definition(self, session_id: str) -> WorldDefinition:
        with self._connect() as connection:
            session = self._session_row(connection, session_id)
            row = connection.execute(
                """SELECT definition_json, definition_digest FROM vwk_definitions
                   WHERE definition_id = ? AND definition_version = ?""",
                (session["definition_id"], session["definition_version"]),
            ).fetchone()
        if row is None:
            raise DefinitionVersionConflict("Pinned world definition is missing")
        definition = WorldDefinition.model_validate_json(row["definition_json"])
        if definition.content_digest != row["definition_digest"]:
            raise DefinitionVersionConflict("Stored world definition digest is invalid")
        return definition

    def load(self, session_id: str) -> tuple[WorldDefinition, WorldRuntime]:
        definition = self.load_definition(session_id)
        kernel = VisualWorldKernel(definition)
        state = kernel.initial_state(session_id)
        with self._connect() as connection:
            session = self._session_row(connection, session_id)
            rows = connection.execute(
                "SELECT event_json FROM vwk_events WHERE session_id = ? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        events = [DomainEvent.model_validate_json(row["event_json"]) for row in rows]
        state = kernel.reduce_all(state, events)
        if state.version != session["state_version"] or state.event_sequence != session["event_sequence"]:
            raise ConcurrencyConflict("Session metadata disagrees with its authoritative event log")
        return definition, state

    def get_receipt(self, session_id: str, idempotency_key: str) -> CommandReceipt | None:
        with self._connect() as connection:
            self._session_row(connection, session_id)
            row = connection.execute(
                """SELECT receipt_json FROM vwk_command_receipts
                   WHERE session_id = ? AND idempotency_key = ?""",
                (session_id, idempotency_key),
            ).fetchone()
        return CommandReceipt.model_validate_json(row["receipt_json"]) if row else None

    def commit(
        self,
        *,
        expected_version: int,
        state: WorldRuntime,
        events: list[DomainEvent],
        receipt: CommandReceipt,
    ) -> None:
        if not events:
            return
        if state.version != expected_version + 1:
            raise ValueError("A committed command must advance exactly one world version")
        now = datetime.now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._session_row(connection, state.session_id)
            if session["state_version"] != expected_version:
                raise ConcurrencyConflict(
                    f"Expected version {expected_version}, found {session['state_version']}"
                )
            existing = connection.execute(
                """SELECT receipt_json FROM vwk_command_receipts
                   WHERE session_id = ? AND idempotency_key = ?""",
                (state.session_id, receipt.idempotency_key),
            ).fetchone()
            if existing:
                return
            for event in events:
                connection.execute(
                    "INSERT INTO vwk_events (session_id, sequence, event_json) VALUES (?, ?, ?)",
                    (state.session_id, event.sequence, self._json(event)),
                )
            connection.execute(
                """UPDATE vwk_sessions
                   SET state_version = ?, event_sequence = ?, snapshot_json = ?, updated_at = ?
                   WHERE session_id = ?""",
                (
                    state.version,
                    state.event_sequence,
                    self._json(self.snapshot_payload(state)),
                    now,
                    state.session_id,
                ),
            )
            connection.execute(
                """INSERT INTO vwk_command_receipts (
                       session_id, idempotency_key, command_id, receipt_json
                   ) VALUES (?, ?, ?, ?)""",
                (
                    state.session_id,
                    receipt.idempotency_key,
                    receipt.command_id,
                    self._json(receipt),
                ),
            )

    def replay_report(self, session_id: str) -> dict[str, Any]:
        definition, replayed = self.load(session_id)
        with self._connect() as connection:
            session = self._session_row(connection, session_id)
        cached_payload = json.loads(session["snapshot_json"])
        cached_digest = sha256(self._json(cached_payload).encode("utf-8")).hexdigest()
        replay_digest = self.canonical_digest(replayed)
        return {
            "verified": cached_digest == replay_digest,
            "session_id": session_id,
            "definition_id": definition.id,
            "definition_version": definition.version,
            "event_count": replayed.event_sequence,
            "world_version": replayed.version,
            "snapshot_digest": cached_digest,
            "replay_digest": replay_digest,
            "events": [event.model_dump(mode="json") for event in replayed.event_log],
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT session_id, definition_id, definition_version, state_version,
                          event_sequence, updated_at
                   FROM vwk_sessions ORDER BY updated_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]
