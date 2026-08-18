from __future__ import annotations

import math
import random
import re
from datetime import timedelta

from .models import (
    CheckOutcome,
    CheckResult,
    DamageResult,
    OpposedCheckDefinition,
    OpposedRollResult,
    RollRecord,
    SanityCheckResult,
    WorldState,
)


SUCCESS_OUTCOMES = {
    CheckOutcome.CRITICAL,
    CheckOutcome.EXTREME,
    CheckOutcome.HARD,
    CheckOutcome.SUCCESS,
    CheckOutcome.AUTOMATIC,
}

SUCCESS_RANK = {
    CheckOutcome.FUMBLE: 0,
    CheckOutcome.FAILURE: 0,
    CheckOutcome.SUCCESS: 1,
    CheckOutcome.HARD: 2,
    CheckOutcome.EXTREME: 3,
    CheckOutcome.CRITICAL: 4,
}

DIFFICULTY_RANK = {"regular": 1, "hard": 2, "extreme": 3}

_DICE_EXPRESSION = re.compile(r"^\s*(?:(\d+)d(\d+))?\s*([+-]\s*\d+)?\s*$", re.IGNORECASE)

_BOUTS = (
    "惊惧逃离：你本能地寻找最近的安全出口。",
    "短暂失神：现实像隔着厚玻璃，下一次行动承受惩罚骰。",
    "偏执凝视：你确信某个普通物件正监视着自己。",
    "失控怒意：你必须克制破坏眼前威胁象征的冲动。",
    "强迫确认：你反复检查门锁、影子或随身物，难以集中。",
    "记忆断片：刚才数分钟的细节变得破碎而不可靠。",
)


class RuleEngine:
    """A compact, deterministic CoC 7e-style rules layer.

    It implements the public quick-start mechanics needed by this solo demo rather
    than attempting to reproduce the complete commercial rulebook.
    """

    @staticmethod
    def _rng_for(state: WorldState) -> random.Random:
        rng = random.Random(state.rng_seed)
        # Draws absent from rng_sides come from pre-polyhedral snapshots and were d100s.
        legacy_draws = max(0, state.rng_draws - len(state.rng_sides))
        for _ in range(legacy_draws):
            rng.randint(1, 100)
        for sides in state.rng_sides:
            rng.randint(1, sides)
        return rng

    def draw_die(self, state: WorldState, sides: int) -> int:
        if sides < 1:
            raise ValueError("A die must have at least one side")
        rng = self._rng_for(state)
        result = rng.randint(1, sides)
        state.rng_draws += 1
        state.rng_sides.append(sides)
        return result

    def roll_expression(self, state: WorldState, expression: str | int) -> int:
        if isinstance(expression, int):
            return max(0, expression)
        value = expression.strip().lower().replace(" ", "")
        if value.isdigit():
            return int(value)
        match = _DICE_EXPRESSION.fullmatch(value)
        if not match or not match.group(1):
            raise ValueError(f"Unsupported dice expression: {expression}")
        count = int(match.group(1))
        sides = int(match.group(2))
        modifier = int((match.group(3) or "0").replace(" ", ""))
        if count > 20 or sides > 100:
            raise ValueError(f"Dice expression is too large: {expression}")
        return max(0, sum(self.draw_die(state, sides) for _ in range(count)) + modifier)

    def roll_percentile(self, state: WorldState, bonus_dice: int = 0) -> tuple[int, list[int]]:
        modifier = max(-2, min(2, int(bonus_dice)))
        base = self.draw_die(state, 100)
        candidates = [base]
        units = base % 10
        for _ in range(abs(modifier)):
            tens = self.draw_die(state, 10) - 1
            candidate = tens * 10 + units
            candidates.append(100 if candidate == 0 else candidate)
        result = min(candidates) if modifier > 0 else max(candidates) if modifier < 0 else base
        return result, candidates

    @staticmethod
    def skill_value(state: WorldState, skill: str) -> int:
        key = skill.lower()
        if key == "luck":
            return state.player.luck
        if key in state.player.characteristics:
            return int(state.player.characteristics[key])
        return int(state.player.skills.get(skill, state.player.skills.get(key, 0)))

    @staticmethod
    def threshold(base_value: int, difficulty: str) -> int:
        divisor = {"regular": 1, "hard": 2, "extreme": 5}[difficulty]
        return base_value // divisor

    @staticmethod
    def outcome_for(base_value: int, roll: int) -> CheckOutcome:
        if roll == 1:
            return CheckOutcome.CRITICAL
        if roll == 100 or (base_value < 50 and roll >= 96):
            return CheckOutcome.FUMBLE
        if roll <= base_value // 5:
            return CheckOutcome.EXTREME
        if roll <= base_value // 2:
            return CheckOutcome.HARD
        if roll <= base_value:
            return CheckOutcome.SUCCESS
        return CheckOutcome.FAILURE

    def check(
        self,
        state: WorldState,
        skill: str | None,
        difficulty: str,
        reason: str,
        *,
        bonus_dice: int = 0,
        modifier_labels: list[str] | None = None,
        pushed: bool = False,
    ) -> CheckResult:
        if skill is None:
            return CheckResult(
                required=False,
                outcome=CheckOutcome.AUTOMATIC,
                succeeded=True,
                difficulty=difficulty,
            )

        base_value = self.skill_value(state, skill)
        target = self.threshold(base_value, difficulty)
        roll, candidates = self.roll_percentile(state, bonus_dice)
        outcome = self.outcome_for(base_value, roll)
        succeeded = SUCCESS_RANK.get(outcome, 0) >= DIFFICULTY_RANK[difficulty]
        roll_id = f"roll_{len(state.rolls) + 1:05d}"
        state.rolls.append(
            RollRecord(
                id=roll_id,
                result=roll,
                skill=skill,
                target=target,
                outcome=outcome,
                reason=reason,
                timestamp=state.world_time,
                difficulty=difficulty,
                candidates=candidates,
                bonus_dice=bonus_dice,
                pushed=pushed,
            )
        )
        if succeeded and skill not in state.player.characteristics and skill != "luck":
            state.player.checked_skills.add(skill)
        return CheckResult(
            required=True,
            skill=skill,
            target=target,
            roll=roll,
            outcome=outcome,
            succeeded=succeeded,
            difficulty=difficulty,
            base_value=base_value,
            candidates=candidates,
            bonus_dice=bonus_dice,
            modifier_labels=modifier_labels or [],
            pushed=pushed,
            roll_id=roll_id,
        )

    def opposed_check(
        self,
        state: WorldState,
        player_check: CheckResult,
        definition: OpposedCheckDefinition,
    ) -> CheckResult:
        value = definition.value
        if value is None and definition.entity_id:
            entity = state.entities.get(definition.entity_id)
            if entity is not None:
                value = int(entity.attributes.get(definition.skill, 0))
        value = int(value or 0)
        roll, candidates = self.roll_percentile(state, definition.bonus_dice)
        outcome = self.outcome_for(value, roll)
        player_check.opponent = OpposedRollResult(
            entity_id=definition.entity_id,
            label=definition.label,
            skill=definition.skill,
            value=value,
            roll=roll,
            outcome=outcome,
            candidates=candidates,
            bonus_dice=definition.bonus_dice,
        )
        player_rank = SUCCESS_RANK.get(player_check.outcome, 0)
        opponent_rank = SUCCESS_RANK.get(outcome, 0)
        if player_rank == 0 and opponent_rank == 0 and definition.both_fail_no_winner:
            player_check.succeeded = False
        elif player_rank != opponent_rank:
            player_check.succeeded = player_rank > opponent_rank
        else:
            player_check.succeeded = definition.tie_favors == "initiator"
        return player_check

    @staticmethod
    def luck_cost(check: CheckResult) -> int | None:
        if (
            not check.required
            or check.succeeded
            or check.roll is None
            or check.target is None
            or check.opponent is not None
            or check.outcome == CheckOutcome.FUMBLE
            or check.pushed
        ):
            return None
        return max(1, check.roll - check.target)

    def spend_luck(self, state: WorldState, check: CheckResult, amount: int) -> CheckResult:
        expected = self.luck_cost(check)
        if expected is None or amount != expected or state.player.luck < amount:
            raise ValueError("Luck cannot be spent on this roll")
        state.player.luck -= amount
        check.outcome = self.outcome_for(int(check.base_value or check.target or 0), int(check.target))
        check.succeeded = True
        check.luck_spent = amount
        if check.skill and check.skill not in state.player.characteristics and check.skill != "luck":
            state.player.checked_skills.add(check.skill)
        if check.roll_id:
            record = next((item for item in state.rolls if item.id == check.roll_id), None)
            if record is not None:
                record.outcome = check.outcome
                record.luck_spent = amount
        return check

    def sanity_check(
        self,
        state: WorldState,
        *,
        success_loss: str,
        failure_loss: str,
        reason: str,
    ) -> SanityCheckResult:
        before = state.player.sanity
        roll, _ = self.roll_percentile(state)
        succeeded = roll <= before
        loss = self.roll_expression(state, success_loss if succeeded else failure_loss)
        state.player.sanity = max(0, before - loss)
        state.player.daily_sanity_loss += loss

        temporary = False
        indefinite = False
        bout: str | None = None
        if loss >= 5:
            intelligence = int(state.player.characteristics.get("int", 0))
            int_roll, _ = self.roll_percentile(state)
            if int_roll <= intelligence:
                temporary = True
                hours = self.draw_die(state, 10)
                state.player.temporary_insanity_until = state.world_time + timedelta(hours=hours)
                bout = _BOUTS[self.draw_die(state, len(_BOUTS)) - 1]
                state.player.bout_of_madness = bout

        threshold = max(1, state.player.starting_sanity // 5)
        if state.player.daily_sanity_loss >= threshold and not state.player.indefinite_insanity:
            state.player.indefinite_insanity = True
            indefinite = True
        if state.player.sanity == 0:
            state.flags["permanently_insane"] = True

        result = SanityCheckResult(
            id=f"sanity_{len(state.sanity_checks) + 1:05d}",
            reason=reason,
            roll=roll,
            target=before,
            succeeded=succeeded,
            loss=loss,
            sanity_before=before,
            sanity_after=state.player.sanity,
            temporary_insanity_triggered=temporary,
            indefinite_insanity_triggered=indefinite,
            bout=bout,
            timestamp=state.world_time,
        )
        state.sanity_checks.append(result)
        return result

    def apply_damage(self, state: WorldState, amount: int, *, source: str) -> DamageResult:
        amount = max(0, int(amount))
        before = state.player.hp
        after = max(0, before - amount)
        state.player.hp = after
        instant_death = amount >= state.player.max_hp
        major_wound_triggered = amount >= math.ceil(state.player.max_hp / 2)
        con_roll: int | None = None
        if major_wound_triggered and not instant_death:
            state.player.major_wound = True
            con_roll, _ = self.roll_percentile(state)
            if con_roll > int(state.player.characteristics.get("con", 0)):
                state.player.unconscious = True
        if instant_death:
            state.flags["dead"] = True
            state.player.unconscious = True
            state.player.dying = False
        elif after == 0:
            state.player.unconscious = True
            state.player.dying = state.player.major_wound

        result = DamageResult(
            id=f"damage_{len(state.damage_log) + 1:05d}",
            source=source,
            amount=amount,
            hp_before=before,
            hp_after=after,
            major_wound_triggered=major_wound_triggered,
            con_roll=con_roll,
            unconscious=state.player.unconscious,
            dying=state.player.dying,
            instant_death=instant_death,
            timestamp=state.world_time,
        )
        state.damage_log.append(result)
        return result

    @staticmethod
    def refresh_conditions(state: WorldState) -> None:
        until = state.player.temporary_insanity_until
        if until is not None and state.world_time >= until:
            state.player.temporary_insanity_until = None
            state.player.bout_of_madness = None
