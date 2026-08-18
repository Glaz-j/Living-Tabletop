from __future__ import annotations

from living_tabletop.director import Director
from living_tabletop.kernel import WorldKernel


def test_director_surfaces_opportunity_not_fact(scenario, state):
    kernel = WorldKernel(scenario)
    director = Director(scenario, kernel)
    state.director.actions_without_progress = 3
    intervention = director.decide(state)
    assert intervention is not None
    assert intervention.action == "surface_clue"
    assert state.flags["director_clue_opportunity"] is True
    assert "f_records_pattern" not in state.player_known_fact_ids


def test_director_adds_pressure_after_success_streak(scenario, state):
    kernel = WorldKernel(scenario)
    director = Director(scenario, kernel)
    state.director.experience.success_streak = 3
    intervention = director.decide(state)
    assert intervention is not None
    assert intervention.action == "increase_pressure"
    assert state.facts["f_wilson_alerted"].value is True
    assert state.director.experience.success_streak == 0


def test_director_offers_existing_respite_for_failure(scenario, state):
    kernel = WorldKernel(scenario)
    director = Director(scenario, kernel)
    state.player.stress = 5
    state.player.hp = 7
    state.director.experience.failure_streak = 2
    intervention = director.decide(state)
    assert intervention is not None
    assert intervention.action == "offer_respite"
    assert state.player.stress == 3
    assert state.player.hp == 8


def test_telemetry_responds_to_progress_and_danger(scenario, state):
    director = Director(scenario, WorldKernel(scenario))
    state.discovered_clue_ids.update({"clue_missing_note", "clue_cult_photo"})
    state.player.hp = 3
    state.player.stress = 6
    state.threats["threat_ritual"].progress = 80
    director.recompute(state)
    exp = state.director.experience
    assert exp.progress > 0
    assert exp.danger >= 70
    assert exp.time_pressure == 80
    assert state.director.phase.value in {"PEAK", "RELIEF"}

