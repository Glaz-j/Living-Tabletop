from __future__ import annotations

from collections import Counter

from living_tabletop.engine import GameEngine
from living_tabletop.scenario import create_initial_state, load_scenario
from living_tabletop.test_agent import (
    PERSONAS,
    HeuristicPlayerDriver,
    SyntheticComposerLLM,
    TestAgentRunner as _TestAgentRunner,
)
from scripts.test_agent import build_jobs


def test_large_campaign_builds_exactly_1024_balanced_games():
    scenarios = ["st_mary_hospital_v0", "the_haunting_corbitt_house_v1"]

    jobs = build_jobs(scenarios, total_runs=1024, runs_per_persona=1)

    assert len(jobs) == 1024
    assert len({job["run_id"] for job in jobs}) == 1024
    assert Counter(job["scenario_id"] for job in jobs) == {
        "st_mary_hospital_v0": 512,
        "the_haunting_corbitt_house_v1": 512,
    }
    assert set(Counter(job["persona"].id for job in jobs).values()) == {128}
    assert len(
        {
            (job["scenario_id"], job["persona"].id, job["seed"])
            for job in jobs
        }
    ) == 1024


def test_synthetic_location_answers_are_scoped_to_the_current_scenario():
    scenario = load_scenario(scenario_id="st_mary_hospital_v0")
    state = create_initial_state(scenario, seed=31)
    original_location = state.entities[state.player.entity_id].location
    engine = GameEngine(scenario, SyntheticComposerLLM(seed=31))

    resolved, resolution = engine.play(
        state,
        text="我只问路，暂时不去：医院地下室具体在什么地方？",
    )

    narrative = "\n".join(
        beat.text for beat in resolved.narrative_sequence.beats
    )
    assert resolution.accepted is True
    assert resolved.entities[resolved.player.entity_id].location == original_location
    assert "锅炉间" in narrative
    assert "罗克斯伯里" not in narrative


def test_repeated_source_followups_about_one_fact_are_not_false_duplicates():
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    seed = 2523
    persona = next(item for item in PERSONAS if item.id == "terse")
    engine = GameEngine(scenario, SyntheticComposerLLM(seed=seed))

    run = _TestAgentRunner(
        scenario,
        engine,
        HeuristicPlayerDriver(seed),
        set(),
    ).run(persona, seed=seed, max_turns=32)

    assert len(run.steps) == 32
    assert run.issues == []
