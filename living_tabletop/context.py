from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

from .models import (
    ActionType,
    NarrativeBeat,
    NarrativeSequence,
    PlayerVisibleMemory,
    WorldState,
)


MAX_STORED_VISIBLE_MEMORIES = 80
DEFAULT_CONTEXT_ENTRIES = 20
DEFAULT_CONTEXT_CHARACTERS = 5000
QUERY_BIGRAM_STOPWORDS = {
    "一下",
    "与他",
    "与她",
    "今天",
    "和他",
    "和她",
    "看看",
    "想要",
    "我们",
    "我要",
    "我想",
    "现在",
    "继续",
    "聊聊",
    "谈论",
    "说说",
    "这个",
    "那个",
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def context_relevance_score(query: str, text: str) -> int:
    query_value = _normalized(query).lower()
    text_value = _normalized(text).lower()
    if not query_value or not text_value:
        return 0
    query_pairs = {
        query_value[index : index + 2]
        for index in range(len(query_value) - 1)
        if query_value[index : index + 2] not in QUERY_BIGRAM_STOPWORDS
    }
    text_pairs = {text_value[index : index + 2] for index in range(len(text_value) - 1)}
    pair_overlap = len(query_pairs & text_pairs)
    words = {
        word
        for word in re.findall(r"[a-z0-9_]{3,}", query_value)
        if word in text_value
    }
    return pair_overlap * 4 + len(words) * 3


def _looks_like_dialogue(text: str) -> bool:
    return ("“" in text and "”" in text) or text.count('"') >= 2


def substantially_repeats(text: str, earlier_texts: Iterable[str]) -> bool:
    """Reject presentation beats that replay an earlier beat verbatim or near-verbatim."""

    candidate = _normalized(text)
    if not candidate:
        return True
    for earlier_text in earlier_texts:
        earlier = _normalized(earlier_text)
        if not earlier:
            continue
        if candidate == earlier:
            return True
        shorter, longer = sorted((candidate, earlier), key=len)
        if len(shorter) >= 24 and shorter in longer:
            return True
        if min(len(candidate), len(earlier)) >= 24:
            if SequenceMatcher(None, candidate, earlier, autojunk=False).ratio() >= 0.84:
                return True
    return False


def _kind_for_beat(beat: NarrativeBeat, action_type: ActionType | None) -> str:
    if beat.source in {"keeper", "generated"}:
        if action_type in {ActionType.TALK, ActionType.DECEIVE} or _looks_like_dialogue(beat.text):
            return "dialogue_claim"
        return "soft_canon"
    if beat.source == "director":
        return "soft_canon"
    return "hard_canon"


def remember_visible_beats(
    state: WorldState,
    sequence: NarrativeSequence,
    beats: Iterable[NarrativeBeat] | None = None,
) -> None:
    """Persist only text that has become part of the player's performed scene."""

    selected = list(sequence.beats if beats is None else beats)
    existing = {
        (entry.sequence_id, _normalized(entry.text))
        for entry in state.visible_history
    }
    player = state.entities.get(state.player.entity_id)
    location_id = player.location if player else None
    for beat in selected:
        key = (sequence.id, _normalized(beat.text))
        if not key[1] or key in existing:
            continue
        existing.add(key)
        state.visible_history.append(
            PlayerVisibleMemory(
                id=f"visible_{state.version:06d}_{len(state.visible_history) + 1:04d}",
                state_version=sequence.state_version,
                world_time=state.world_time,
                location_id=location_id,
                sequence_id=sequence.id,
                kind=_kind_for_beat(beat, sequence.action_type),
                source=beat.source,
                action_type=sequence.action_type,
                text=beat.text,
            )
        )
    if len(state.visible_history) > MAX_STORED_VISIBLE_MEMORIES:
        state.visible_history = state.visible_history[-MAX_STORED_VISIBLE_MEMORIES:]


def _legacy_narrator_memories(state: WorldState) -> list[PlayerVisibleMemory]:
    """Recover player-visible generated prose from saves created before visible_history."""

    recovered: list[PlayerVisibleMemory] = []
    narrator_calls = [
        call
        for call in state.agent_calls
        if call.role == "narrator"
        and call.validation == "accepted"
        and call.structured_output
    ][-8:]
    for call in narrator_calls:
        output = call.structured_output or {}
        raw_texts = output.get("beats")
        if not isinstance(raw_texts, list):
            narrative = output.get("narrative")
            raw_texts = [narrative] if isinstance(narrative, str) else []
        for index, raw_text in enumerate(raw_texts):
            text = str(raw_text).strip()
            if not text:
                continue
            recovered.append(
                PlayerVisibleMemory(
                    id=f"legacy_visible_{call.id}_{index:02d}",
                    state_version=max(1, call.input_state_version),
                    world_time=call.created_at,
                    kind="dialogue_claim" if _looks_like_dialogue(text) else "soft_canon",
                    source="generated",
                    text=text,
                )
            )
    return recovered


def _legacy_keeper_outputs(state: WorldState) -> list[str]:
    texts: list[str] = []
    for call in state.agent_calls:
        if call.role != "keeper" or call.validation != "accepted" or not call.structured_output:
            continue
        plan = call.structured_output.get("open_plan")
        if not isinstance(plan, dict):
            continue
        for key in ("success_text", "failure_text"):
            value = plan.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(_normalized(value))
    return texts


def recent_visible_context(
    state: WorldState,
    *,
    max_entries: int = DEFAULT_CONTEXT_ENTRIES,
    max_characters: int = DEFAULT_CONTEXT_CHARACTERS,
    exclude_sequence_id: str | None = None,
    query: str | None = None,
    immediate_entries: int = 4,
) -> list[dict[str, object]]:
    """Return recent performed history within a predictable prompt budget."""

    indexed = [
        entry
        for entry in [*state.visible_history, *_legacy_narrator_memories(state)]
        if exclude_sequence_id is None or entry.sequence_id != exclude_sequence_id
    ]
    indexed.sort(key=lambda item: (item.state_version, item.world_time, item.id))

    unique: list[PlayerVisibleMemory] = []
    seen: set[str] = set()
    for entry in indexed:
        normalized = _normalized(entry.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(entry)

    if query and unique:
        recent = unique[-immediate_entries:] if immediate_entries > 0 else []
        recent_ids = {entry.id for entry in recent}
        relevant = sorted(
            (
                (context_relevance_score(query, entry.text), index, entry)
                for index, entry in enumerate(unique)
                if entry.id not in recent_ids
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        selected_ids = {
            *recent_ids,
            *(
                entry.id
                for score, _index, entry in relevant[: max(0, max_entries - len(recent))]
                if score > 0
            ),
        }
        unique = [entry for entry in unique if entry.id in selected_ids]

    chosen: list[PlayerVisibleMemory] = []
    used_characters = 0
    for entry in reversed(unique):
        size = len(entry.text)
        if chosen and (len(chosen) >= max_entries or used_characters + size > max_characters):
            break
        chosen.append(entry)
        used_characters += size
    chosen.reverse()
    legacy_keeper_texts = _legacy_keeper_outputs(state)
    serialized: list[dict[str, object]] = []
    for entry in chosen:
        normalized = _normalized(entry.text)
        is_legacy_keeper_text = any(
            len(normalized) >= 12 and normalized in generated
            for generated in legacy_keeper_texts
        )
        kind = entry.kind
        source = entry.source
        if is_legacy_keeper_text and entry.source == "authored":
            source = "keeper"
            kind = (
                "dialogue_claim"
                if entry.action_type in {ActionType.TALK, ActionType.DECEIVE}
                or _looks_like_dialogue(entry.text)
                else "soft_canon"
            )
        serialized.append(
            {
                "kind": kind,
                "text": entry.text,
                "state_version": entry.state_version,
                "location_id": entry.location_id,
                "source": source,
            }
        )
    return serialized
