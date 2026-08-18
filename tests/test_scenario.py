from __future__ import annotations

from living_tabletop.models import EntityType


def test_scenario_has_v0_content(scenario):
    assert scenario.title == "圣玛丽医院"
    assert len([e for e in scenario.entities if e.type == EntityType.LOCATION]) == 9
    assert len([e for e in scenario.entities if e.type == EntityType.NPC]) == 6
    assert len(scenario.clues) == 12
    assert len(scenario.actions) >= 40
    assert len(scenario.endings) >= 6


def test_core_truth_is_hard_canon_and_immutable(scenario):
    facts = {fact.id: fact for fact in scenario.facts}
    assert facts["f_wren_leader"].canon == "hard_canon"
    assert facts["f_wren_leader"].immutable is True
    assert facts["f_ritual_time"].value == "1927-12-17T23:00:00"


def test_every_clue_points_to_an_existing_fact(scenario):
    facts = {fact.id for fact in scenario.facts}
    assert all(clue.fact_id in facts for clue in scenario.clues)
    assert sum(clue.critical for clue in scenario.clues) >= scenario.minimum_evidence


def test_location_graph_is_bidirectional_except_crypt(scenario):
    for origin, destinations in scenario.location_graph.items():
        for destination in destinations:
            assert origin in scenario.location_graph[destination]

