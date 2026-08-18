from __future__ import annotations

from copy import deepcopy

from living_tabletop.models import CheckOutcome
from living_tabletop.rules import RuleEngine


def test_d100_is_deterministic_from_seed_and_draw_count(state):
    other = deepcopy(state)
    rules = RuleEngine()
    first = rules.check(state, "observation", "regular", "test")
    second = rules.check(other, "observation", "regular", "test")
    assert first == second
    assert state.rng_draws == other.rng_draws == 1
    assert state.rolls[0].result == other.rolls[0].result


def test_automatic_action_does_not_draw_rng(state):
    result = RuleEngine().check(state, None, "regular", "move")
    assert result.outcome == CheckOutcome.AUTOMATIC
    assert result.succeeded is True
    assert state.rng_draws == 0


def test_hard_check_uses_half_skill(state):
    result = RuleEngine().check(state, "observation", "hard", "detail")
    assert result.target == state.player.skills["observation"] // 2


def test_coc_success_levels_and_fumble_boundaries():
    rules = RuleEngine()
    assert rules.outcome_for(60, 1) == CheckOutcome.CRITICAL
    assert rules.outcome_for(60, 2) == CheckOutcome.EXTREME
    assert rules.outcome_for(60, 30) == CheckOutcome.HARD
    assert rules.outcome_for(60, 60) == CheckOutcome.SUCCESS
    assert rules.outcome_for(60, 99) == CheckOutcome.FAILURE
    assert rules.outcome_for(60, 100) == CheckOutcome.FUMBLE
    assert rules.outcome_for(40, 96) == CheckOutcome.FUMBLE


def test_bonus_and_penalty_dice_keep_units_and_choose_best_or_worst(state, monkeypatch):
    rules = RuleEngine()
    draws = iter([64, 3, 64, 10])
    monkeypatch.setattr(rules, "draw_die", lambda _state, _sides: next(draws))
    bonus, bonus_candidates = rules.roll_percentile(state, bonus_dice=1)
    penalty, penalty_candidates = rules.roll_percentile(state, bonus_dice=-1)
    assert bonus_candidates == [64, 24]
    assert bonus == 24
    assert penalty_candidates == [64, 94]
    assert penalty == 94


def test_sanity_loss_can_trigger_temporary_and_indefinite_insanity(state, monkeypatch):
    rules = RuleEngine()
    state.player.sanity = 60
    state.player.starting_sanity = 30  # six points is one fifth of starting SAN
    percentile = iter([(80, [80]), (50, [50])])
    dice = iter([3, 2])  # duration, bout index
    monkeypatch.setattr(rules, "roll_percentile", lambda _state, _bonus=0: next(percentile))
    monkeypatch.setattr(rules, "draw_die", lambda _state, _sides: next(dice))

    result = rules.sanity_check(
        state,
        success_loss="0",
        failure_loss="6",
        reason="test horror",
    )

    assert result.succeeded is False
    assert result.loss == 6
    assert result.temporary_insanity_triggered is True
    assert result.indefinite_insanity_triggered is True
    assert state.player.sanity == 54
    assert state.player.temporary_insanity_until is not None
    assert state.player.bout_of_madness


def test_major_wound_and_instant_death_are_distinct(state, monkeypatch):
    rules = RuleEngine()
    state.player.hp = state.player.max_hp = 12
    state.player.characteristics["con"] = 60
    monkeypatch.setattr(rules, "roll_percentile", lambda _state, _bonus=0: (70, [70]))

    wound = rules.apply_damage(state, 6, source="test")
    assert wound.major_wound_triggered is True
    assert wound.instant_death is False
    assert state.player.major_wound is True
    assert state.player.unconscious is True
    assert state.flags.get("dead") is not True

    state.player.hp = 12
    death = rules.apply_damage(state, 12, source="test")
    assert death.instant_death is True
    assert state.flags["dead"] is True
