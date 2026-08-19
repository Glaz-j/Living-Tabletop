from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
import threading
from pathlib import Path
from typing import Any

from .context import remember_visible_beats
from .engine import GameEngine
from .llm import LLMSettings, LLMUnavailable, OpenAICompatibleLLM, RoutedLLM, RoutingMode
from .models import NarrativeBeat, RuleChoice, WorldState
from .scenario import DEFAULT_SCENARIO_ID, create_initial_state, load_scenarios, upgrade_world_state
from .storage import SQLiteRepository


class ScenarioNotFound(KeyError):
    pass


class GameService:
    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        llm_settings: LLMSettings | None = None,
    ):
        resolved_db = db_path or os.getenv("LIVING_TABLETOP_DB_PATH", "data/living_tabletop.db")
        self._llm_preferences_path = Path(resolved_db).parent / "llm_preferences.json"
        self.scenarios = load_scenarios()
        self.default_scenario_id = (
            DEFAULT_SCENARIO_ID if DEFAULT_SCENARIO_ID in self.scenarios else next(iter(self.scenarios))
        )
        self.scenario = self.scenarios[self.default_scenario_id]
        self.repository = SQLiteRepository(resolved_db)
        self.llm = (
            OpenAICompatibleLLM(llm_settings)
            if llm_settings is not None
            else RoutedLLM.from_env()
        )
        self._load_llm_preferences()
        self.engines = {
            scenario_id: GameEngine(scenario, self.llm)
            for scenario_id, scenario in self.scenarios.items()
        }
        self.engine = self.engines[self.default_scenario_id]
        self._lock = threading.RLock()
        self._narrator_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="living-narrator")

    def _load_llm_preferences(self) -> None:
        if not isinstance(self.llm, RoutedLLM) or not self._llm_preferences_path.exists():
            return
        try:
            payload = json.loads(self._llm_preferences_path.read_text(encoding="utf-8"))
            self.llm.configure(
                mode=payload.get("mode", "auto"),
                local_model=payload.get("local_model"),
                remote_model=payload.get("remote_model"),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    def _save_llm_preferences(self) -> None:
        if not isinstance(self.llm, RoutedLLM):
            return
        local = self.llm.providers.get("local")
        remote = self.llm.providers.get("remote")
        payload = {
            "mode": self.llm.mode,
            "local_model": local.settings.model if local else None,
            "remote_model": remote.settings.model if remote else None,
        }
        self._llm_preferences_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._llm_preferences_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self._llm_preferences_path)

    def _scenario_and_engine(self, scenario_id: str) -> tuple[Any, GameEngine]:
        try:
            return self.scenarios[scenario_id], self.engines[scenario_id]
        except KeyError as exc:
            raise ScenarioNotFound(scenario_id) from exc

    def scenario_catalog(self) -> list[dict[str, Any]]:
        catalog = []
        for scenario in self.scenarios.values():
            source = scenario.source.model_dump(mode="json") if scenario.source else None
            catalog.append(
                {
                    "id": scenario.id,
                    "title": scenario.title,
                    "subtitle": scenario.subtitle,
                    "minimum_evidence": scenario.minimum_evidence,
                    "presentation": scenario.presentation.model_dump(mode="json"),
                    "source": source,
                    "ruleset": "coc7_quickstart_subset_v1",
                    "default": scenario.id == self.default_scenario_id,
                }
            )
        return catalog

    def create_session(
        self,
        *,
        player_name: str = "调查员",
        seed: int = 1927,
        scenario_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            selected_id = scenario_id or self.default_scenario_id
            scenario, engine = self._scenario_and_engine(selected_id)
            state = create_initial_state(scenario, player_name=player_name, seed=seed)
            engine.director.recompute(state)
            engine.prepare_opening_sequence(state)
            self.repository.save(state)
            return engine.public_view(state)

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            state = self.repository.load(session_id)
            scenario, engine = self._scenario_and_engine(state.scenario_id)
            upgrade_world_state(state, scenario)
            return engine.public_view(state)

    def act(
        self,
        session_id: str,
        *,
        action_id: str | None = None,
        text: str | None = None,
        utterance: str | None = None,
        rule_choice: RuleChoice | str | None = None,
        interactive_rules: bool = False,
    ) -> dict[str, Any]:
        narration_job: tuple[GameEngine, WorldState] | None = None
        with self._lock:
            state = self.repository.load(session_id)
            scenario, engine = self._scenario_and_engine(state.scenario_id)
            upgrade_world_state(state, scenario)
            next_state, resolution = engine.play(
                state,
                action_id=action_id,
                text=text,
                utterance=utterance,
                rule_choice=rule_choice,
                interactive_rules=interactive_rules,
            )
            self.repository.save(next_state)
            view = engine.public_view(next_state, resolution)
            sequence = next_state.narrative_sequence
            if sequence is not None and sequence.status == "pending":
                narration_job = (engine, deepcopy(next_state))
        if narration_job is not None:
            self._narrator_executor.submit(self._complete_narration, *narration_job)
        return view

    def _complete_narration(self, engine: GameEngine, snapshot: WorldState) -> None:
        sequence = snapshot.narrative_sequence
        if sequence is None:
            return
        call_count = len(snapshot.agent_calls)
        generated = engine.narrator.expand_sequence(snapshot, sequence)
        call_record = snapshot.agent_calls[-1] if len(snapshot.agent_calls) > call_count else None

        with self._lock:
            current = self.repository.load(snapshot.session_id)
            upgrade_world_state(current, engine.scenario)
            current_sequence = current.narrative_sequence
            if (
                current.version != sequence.state_version
                or current_sequence is None
                or current_sequence.id != sequence.id
                or current_sequence.status != "pending"
            ):
                return

            existing = {beat.text.strip() for beat in current_sequence.beats}
            generated_beats: list[NarrativeBeat] = []
            for text in generated:
                if text in existing:
                    continue
                existing.add(text)
                beat = NarrativeBeat(
                    id=f"{current_sequence.id}_beat_{len(current_sequence.beats) + 1:02d}",
                    text=text,
                    source="generated",
                    skippable=True,
                )
                current_sequence.beats.append(beat)
                generated_beats.append(beat)
            current_sequence.status = "ready" if generated else "fallback"
            current_sequence.grounding_report = sequence.grounding_report
            if current_sequence.turn_trace_id and sequence.grounding_report is not None:
                for trace in reversed(current.turn_traces):
                    if trace.get("id") == current_sequence.turn_trace_id:
                        trace["grounding"] = sequence.grounding_report
                        break
            remember_visible_beats(current, current_sequence, generated_beats)
            if call_record is not None:
                current.agent_calls.append(call_record)
            self.repository.save(current)

    def narrative_status(self, session_id: str, sequence_id: str) -> dict[str, Any]:
        with self._lock:
            state = self.repository.load(session_id)
            scenario, engine = self._scenario_and_engine(state.scenario_id)
            upgrade_world_state(state, scenario)
            sequence = engine.public_view(state)["narrative_sequence"]
            return {
                **sequence,
                "superseded": sequence["id"] != sequence_id,
                "current_state_version": state.version,
            }

    def developer_view(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            state = self.repository.load(session_id)
            scenario, engine = self._scenario_and_engine(state.scenario_id)
            upgrade_world_state(state, scenario)
            return engine.developer_view(state)

    def llm_configuration(self, *, refresh: bool = False) -> dict[str, Any]:
        if isinstance(self.llm, RoutedLLM):
            return self.llm.configuration(refresh=refresh)
        provider = self.llm.status(refresh=refresh)
        return {
            "mode": "remote",
            "order": ["remote"],
            "providers": {"remote": provider},
            "last_used": None,
            "last_failures": {},
            "read_only": True,
        }

    def configure_llm(
        self,
        *,
        mode: RoutingMode,
        local_model: str | None = None,
        remote_model: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(self.llm, RoutedLLM):
            raise ValueError("Runtime model routing is unavailable")
        requested = {"local": local_model, "remote": remote_model}
        for name, model in requested.items():
            if model is None:
                continue
            provider = self.llm.providers.get(name)
            if provider is None:
                raise ValueError(f"{name.title()} provider is unavailable")
            known_models, _ = provider.list_models(refresh=False)
            if known_models and model not in known_models:
                raise ValueError(f"Unknown {name} model: {model}")
        self.llm.configure(
            mode=mode,
            local_model=local_model,
            remote_model=remote_model,
        )
        self._save_llm_preferences()
        return self.llm.configuration(refresh=False)

    def probe_llm(self, *, provider_name: str | None = None) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        }
        target = self.llm
        resolved_provider = provider_name
        if isinstance(self.llm, RoutedLLM) and provider_name is not None:
            target = self.llm.providers.get(provider_name)
            if target is None:
                raise ValueError("Unknown LLM provider")
        try:
            result = target.complete_json(
                system="You are a connectivity probe. Return only the requested JSON object.",
                user_payload={"request": "return status ok"},
                max_output_tokens=64,
                response_schema=schema,
                schema_name="ConnectivityProbe",
                reasoning_effort="none",
            )
        except LLMUnavailable:
            raise
        except Exception as exc:
            if isinstance(target, OpenAICompatibleLLM):
                failure = target._safe_error(exc)
                mode = provider_name or target.provider
                public = RoutedLLM._public_failure_message(
                    mode if mode in {"local", "remote"} else "auto"
                )
                raise LLMUnavailable(
                    f"{mode} LLM probe failed",
                    public_message=public,
                    failures={mode: failure},
                ) from exc
            raise
        if isinstance(self.llm, RoutedLLM) and provider_name is None and self.llm.last_used:
            resolved_provider = self.llm.last_used["provider"]
        if isinstance(target, OpenAICompatibleLLM):
            resolved_provider = target.provider
            model = target.settings.model
        else:
            model = self.llm.settings.model
        return {
            "status": "ok",
            "provider": resolved_provider,
            "model": model,
            "latency_ms": result.latency_ms,
        }

    def export_replay(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return self.repository.export(session_id)

    def health(self) -> dict[str, Any]:
        response = {
            "status": "ok",
            "scenario": self.default_scenario_id,
            "default_scenario": self.default_scenario_id,
            "scenarios": list(self.scenarios),
            "ruleset": "coc7_quickstart_subset_v1",
            "action_model": "validated_agent_runtime_v2",
            "narrative_mode": "outcome_grounded_async_beats_v2",
            "dialogue_mode": "llm_first_contextual_transcript_v2",
            "llm": {
                "enabled": self.llm.enabled,
                "model": self.llm.settings.model,
                "base_url": self.llm.settings.base_url,
                "api_key_configured": bool(self.llm.settings.api_key),
                "max_retries": self.llm.settings.max_retries,
            },
        }
        if isinstance(self.llm, RoutedLLM):
            response["llm"].update(
                {
                    "mode": self.llm.mode,
                    "order": [name for name, _ in self.llm._ordered_providers(include_unavailable=True)],
                    "last_used": self.llm.last_used,
                    "providers": {
                        name: {
                            "enabled": provider.enabled,
                            "configured": provider.configured,
                            "model": provider.settings.model,
                            "last_error": provider.last_error,
                        }
                        for name, provider in self.llm.providers.items()
                    },
                }
            )
        return response

    def close(self) -> None:
        self._narrator_executor.shutdown(wait=False, cancel_futures=True)
