from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

from pydantic import ValidationError

from ..harness import HarnessValidationError, StructuredHarness, strict_json_schema
from ..llm import LLMResult, LLMUnavailable, OpenAICompatibleLLM, record_agent_call
from ..models import ActionDefinition, ActionType, OpenActionPlan, ScenarioDefinition, WorldState
from .context import ContextAssembler
from .contracts import (
    AssembledTurnContext,
    KnowledgeQuery,
    PlannedOpenAction,
    PlayerIntentEnvelope,
    TurnCompositionOutput,
    TurnPlannerDecision,
)
from .dialogue import DialogueValidationError, SoftFactValidator
from .validation import PlanValidationError, PlanValidator
from .world_changes import ItemChangeValidationError, ItemChangeValidator


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ComposedTurn:
    decision: TurnPlannerDecision
    context: AssembledTurnContext
    composition: TurnCompositionOutput | None = None
    open_plan: OpenActionPlan | None = None
    legacy: bool = False


class TurnComposer:
    """V2 foreground actor: understand the turn and author its full performance."""

    _QUESTION = re.compile(
        r"[?？]|(?:什么|怎么|为何|为什么|哪里|哪儿|谁|多久|多少|吗|呢|where|what|who|how)",
        re.IGNORECASE,
    )

    def __init__(self, llm: OpenAICompatibleLLM, scenario: ScenarioDefinition):
        self.llm = llm
        self.scenario = scenario
        self.context_assembler = ContextAssembler()
        self.plan_validator = PlanValidator()
        self.fact_validator = SoftFactValidator()
        self.item_change_validator = ItemChangeValidator()

    @staticmethod
    def _performance_from_raw(data: dict[str, Any]) -> list[str]:
        def repair(raw: object) -> str:
            item = str(raw).strip()[:1500]
            while '"' in item:
                item = item.replace('"', "“", 1)
                if '"' in item:
                    item = item.replace('"', "”", 1)
            for opener, closer in (("“", "”"), ("‘", "’")):
                imbalance = item.count(opener) - item.count(closer)
                if imbalance > 0:
                    item = f"{item}{closer * imbalance}"
                elif imbalance < 0:
                    item = f"{opener * (-imbalance)}{item}"
            return item

        for key in ("performance", "beats", "response", "narrative", "text"):
            raw = data.get(key)
            if isinstance(raw, str) and raw.strip():
                return [repair(raw)]
            if isinstance(raw, list):
                beats = [repair(item) for item in raw if str(item).strip()]
                if beats:
                    return beats[:5]
        decision = data.get("decision") if isinstance(data.get("decision"), dict) else data
        open_plan = decision.get("open_plan") if isinstance(decision, dict) else None
        if isinstance(open_plan, dict):
            raw = open_plan.get("success_text")
            if isinstance(raw, str) and raw.strip():
                return [repair(raw)]
        return []

    def _fallback_decision(
        self,
        state: WorldState,
        envelope: PlayerIntentEnvelope,
    ) -> TurnPlannerDecision:
        location_id = state.entities[state.player.entity_id].location
        present_npcs = [
            entity
            for entity in state.entities.values()
            if entity.active
            and entity.location == location_id
            and entity.type.value == "NPC"
        ]
        addressee = present_npcs[0] if present_npcs else None
        question = bool(self._QUESTION.search(envelope.text))
        return TurnPlannerDecision(
            open_plan=PlannedOpenAction(
                label=(
                    f"回应{addressee.name}" if addressee else "回应玩家的自由行动"
                ),
                action_type=ActionType.TALK if addressee else ActionType.OTHER,
                goal=envelope.text,
                target_name=addressee.name if addressee else None,
                target_entity_id=addressee.id if addressee else None,
                duration_minutes=1,
                resolution="automatic",
                risk="safe",
                speech_act="question" if question else ("statement" if addressee else "none"),
                addressee_id=addressee.id if addressee else None,
            ),
            confidence=0.5,
        )

    def _validate_decision_fail_soft(
        self,
        state: WorldState,
        envelope: PlayerIntentEnvelope,
        decision: TurnPlannerDecision,
        available_actions: list[ActionDefinition],
    ) -> TurnPlannerDecision:
        try:
            return self.plan_validator.validate(state, envelope, decision, available_actions)
        except PlanValidationError as error:
            # Invalid authority metadata must never erase a useful response.  The
            # safe fallback is a non-mutating conversational/OTHER action; the
            # world kernel still sees no unauthorized movement or hard effect.
            logger.info("TurnComposer downgraded unsafe action metadata: %s", error)
            return self.plan_validator.validate(
                state,
                envelope,
                self._fallback_decision(state, envelope),
                available_actions,
            )

    @staticmethod
    def _legacy_decision(data: dict[str, Any]) -> TurnPlannerDecision | None:
        if "decision" in data:
            return None
        if "existing_action_id" not in data and "open_plan" not in data:
            return None
        try:
            return TurnPlannerDecision.model_validate(data)
        except ValidationError:
            return None

    def _normalize_legacy_dialogue(
        self,
        envelope: PlayerIntentEnvelope,
        decision: TurnPlannerDecision,
    ) -> TurnPlannerDecision:
        """Reproduce V1 planner conveniences only for migration fixtures."""

        planned = decision.open_plan
        if planned is None:
            return decision
        question = bool(self._QUESTION.search(envelope.text))
        has_speech = planned.action_type in {ActionType.TALK, ActionType.DECEIVE} or bool(
            planned.addressee_id
        )
        if not has_speech:
            return decision
        if question:
            planned.speech_act = "question"
        elif planned.speech_act == "none":
            planned.speech_act = "statement"
        addressee = planned.addressee_id or (
            planned.target_entity_id
            if planned.action_type in {ActionType.TALK, ActionType.DECEIVE}
            else None
        )
        planned.addressee_id = addressee
        if question and addressee and planned.knowledge_query is None:
            question_text = envelope.text
            spoken_clause = re.search(
                r"(?:说|问)(?:道)?[，,:：\s]*(?P<question>[^。；;]+)$",
                envelope.text,
            )
            if spoken_clause and self._QUESTION.search(spoken_clause.group("question")):
                question_text = spoken_clause.group("question").strip()
            planned.knowledge_query = KnowledgeQuery(
                query_text=question_text,
                asker_id=envelope.actor_id,
                addressee_id=addressee,
                subject_entity_ids=[
                    item.entity_id
                    for item in planned.referents
                    if item.entity_id is not None
                    and item.entity_id != addressee
                    and item.mention in question_text
                ],
            )
        return TurnPlannerDecision(open_plan=planned, confidence=decision.confidence)

    def _apply_composition(
        self,
        state: WorldState,
        envelope: PlayerIntentEnvelope,
        context: AssembledTurnContext,
        output: TurnCompositionOutput,
        decision: TurnPlannerDecision,
    ) -> OpenActionPlan | None:
        if decision.open_plan is None:
            return None
        plan = self.plan_validator.materialize(decision)
        success_beats = [
            beat.strip()[:1500] for beat in output.performance if beat.strip()
        ][:5]
        failure_beats = [
            beat.strip()[:1500] for beat in output.failure_performance if beat.strip()
        ][:4]
        if plan.resolution == "check" and not failure_beats:
            failure_beats = [
                "你的尝试没能取得预期结果，现场随之作出了真实而克制的反应。"
            ]
        if plan.resolution != "check":
            failure_beats = failure_beats or [plan.failure_text]

        allowed_fact_ids = set(state.player_known_fact_ids)
        allowed_entity_ids = {state.player.entity_id}
        allowed_entity_ids.update(item["id"] for item in context.present_entities)
        allowed_entity_ids.update(item["id"] for item in context.referenced_entities)
        allowed_entity_ids.update(item["id"] for item in context.inventory)
        if context.scene.get("id"):
            allowed_entity_ids.add(str(context.scene["id"]))

        planned = decision.open_plan
        speaker_id = planned.addressee_id or (
            planned.target_entity_id
            if planned.action_type in {ActionType.TALK, ActionType.DECEIVE}
            else None
        )
        if speaker_id:
            allowed_fact_ids.update(
                item["fact_id"]
                for item in context.present_npc_knowledge
                if item["knower_id"] == speaker_id
            )
        valid_proposals = []
        if speaker_id and speaker_id in state.entities:
            try:
                self.fact_validator.validate_prose(
                    state,
                    success_beats,
                    speaker_id=speaker_id,
                    allowed_entity_ids=allowed_entity_ids,
                    allowed_fact_ids=allowed_fact_ids,
                )
            except DialogueValidationError as error:
                # Hard-canon leakage is the one prose-level failure that remains
                # fail-closed.  Do not commit or display secret information.
                raise LLMUnavailable(
                    "Turn Composer exposed inaccessible hard canon",
                    public_message=(
                        "模型回复包含当前角色不应知道的关键事实，行动未提交，游戏状态没有改变。"
                        "请重试这一轮。"
                    ),
                ) from error
            valid_proposals = self.fact_validator.filter_proposals(
                state,
                output.proposed_facts,
                allowed_entity_ids=allowed_entity_ids,
            )[:6]
        else:
            try:
                self.fact_validator.validate_accessible_prose(
                    state,
                    success_beats,
                    allowed_entity_ids=allowed_entity_ids,
                    allowed_fact_ids=allowed_fact_ids,
                )
            except DialogueValidationError as error:
                raise LLMUnavailable(
                    "Turn Composer exposed inaccessible hard canon",
                    public_message=(
                        "模型回复包含当前角色不应知道的关键事实，行动未提交，游戏状态没有改变。"
                        "请重试这一轮。"
                    ),
                ) from error

        generated_facts = []
        reused_ids: list[str] = []
        if speaker_id and valid_proposals:
            generated_facts, reused_ids = self.fact_validator.materialize(
                state,
                valid_proposals,
                speaker_id=speaker_id,
            )
        used_fact_ids = list(
            dict.fromkeys(
                [
                    *(fact_id for fact_id in output.used_fact_ids if fact_id in allowed_fact_ids),
                    *reused_ids,
                    *(fact.id for fact in generated_facts),
                ]
            )
        )

        plan.success_beats = success_beats
        plan.failure_beats = failure_beats
        plan.success_text = "\n\n".join(success_beats)[:1200]
        plan.action_success_text = plan.success_text
        plan.failure_text = "\n\n".join(failure_beats)[:1200]
        plan.dialogue_complete = True
        plan.generated_facts = generated_facts
        try:
            item_changes = self.item_change_validator.materialize(
                state,
                output.proposed_item_changes,
                performance=success_beats,
                allowed_entity_ids=allowed_entity_ids,
            )
        except ItemChangeValidationError as error:
            logger.info("Turn Composer rejected an item world change: %s", error)
            raise LLMUnavailable(
                "Turn Composer proposed an invalid item world change",
                public_message=(
                    "模型描述了无法可靠写入世界状态的物品交接，本轮行动没有提交。"
                    "请重试这一轮。"
                ),
            ) from error
        plan.world_effects = item_changes.effects
        plan.approved_fact_ids = used_fact_ids
        plan.knowledge_source_id = speaker_id
        plan.disclosure_mode = "automatic" if used_fact_ids else None
        plan.knowledge_query_text = (
            envelope.text if planned.speech_act == "question" else None
        )
        plan.answered_query_parts = output.answered_query_parts
        plan.unanswered_query_parts = output.unresolved_query_parts
        return plan

    def compose(
        self,
        state: WorldState,
        envelope: PlayerIntentEnvelope,
        available_actions: list[ActionDefinition],
    ) -> ComposedTurn:
        if not self.llm.enabled:
            raise LLMUnavailable("Player text requires the Turn Composer LLM")
        context = self.context_assembler.assemble(state, envelope, available_actions)
        payload = {
            "current_turn": {
                "player_text_verbatim": envelope.text,
                "player_is_actor": True,
            },
            "context": context.model_dump(mode="json"),
            "authority": {
                "hard_state_is_read_only": True,
                "authored_action_ids": [action.id for action in available_actions],
                "soft_fact_subject_ids": [
                    item["id"] for item in context.referenced_entities
                ],
                "soft_fact_rule": (
                    "Unknown low-stakes details may be invented and proposed; never invent clues, "
                    "secret causes, stats, success at risky actions, or location changes. Inventory "
                    "may change only through proposed_item_changes."
                ),
                "item_change_rule": (
                    "If the visible success performance gives the player an item or removes one, "
                    "emit exactly one matching proposed_item_changes entry. Use a context entity id "
                    "for an existing item and null for a newly introduced item. Never emit effects."
                ),
            },
            "performance_contract": {
                "ordinary_turn": "write the complete visible response now",
                "mechanical_check": (
                    "performance is the success branch and failure_performance is the failure branch; "
                    "do not reveal that alternatives exist"
                ),
                "dialogue": (
                    "use direct first-person speech for important NPCs; answer the player's actual "
                    "question before atmosphere or topic changes"
                ),
            },
        }
        system = (
            "你是 Living Tabletop V2 的 Turn Composer，也是本轮唯一的前台叙事者。"
            "同时理解玩家意图并写出完整、可直接显示的剧情演出；玩家输入永远是本轮最高优先级。"
            "不要替玩家追加没有表达的移动、同意或行动。询问某地在哪里只是提问，绝不是前往。"
            "走到窗边、桌边等场景内动作也不是 MOVE；玩家说仍留在这里或不离开时绝不能改变地点。"
            "普通提问、议价、寒暄和请求都是 automatic；NPC 应结合身份、情绪、最近逐字对话直接回应，"
            "可以拒绝、犹豫、还价或不知道，但不得答非所问，也不得只写‘玩家把话说给对方听’。"
            "重要 NPC 使用第一人称直接对话体，并配少量动作和环境反应；每轮通常写 2 至 5 段。"
            "recent_visible_history 是原始对话主记忆，不能混淆 player 与 NPC 的说话者。"
            "present_npc_knowledge 是在场 NPC 可使用的事实；used_fact_ids 只能引用这里或玩家已知事实。"
            "如果地址、路线、营业时间、习惯等低风险细节尚未定义，可以自然编写并通过 proposed_facts 建议保存；"
            "可以让角色赠送、拾取、购买、交还或消耗普通物品，但每一处可见的物品得失都必须在 "
            "proposed_item_changes 中逐项声明；新物品的 item_entity_id 必须为 null，已有物品只能使用上下文给出的 id。"
            "不要直接输出内核 Effect，也不得虚构关键线索、幕后真相、检定成功、伤害、理智、时间或位置变化。"
            "若选择 existing_action_id，它必须完整匹配玩家意图；否则输出 open_plan。"
            "open_plan.goal 必须逐字等于 player_text_verbatim。只有玩家明确承诺移动时才可 MOVE/ESCAPE/带目的地的 REST。"
            "有真实风险时才选择 check，并同时写成功与失败演出；世界内核稍后只展示实际分支。"
            "只输出符合 JSON Schema 的对象。"
        )

        result: LLMResult | None = None
        try:
            outcome = StructuredHarness(self.llm, max_repairs=0).run(
                TurnCompositionOutput,
                system=system,
                user_payload=payload,
                max_output_tokens=1500,
                temperature=0.35,
                reasoning_effort="none",
                response_schema=strict_json_schema(TurnCompositionOutput),
            )
            result = outcome.llm_result
            output = outcome.value
        except HarnessValidationError as error:
            result = error.result
            data = result.data if result is not None else {}
            legacy = self._legacy_decision(data)
            if legacy is not None:
                try:
                    decision = self.plan_validator.validate(
                        state, envelope, legacy, available_actions
                    )
                except PlanValidationError:
                    decision = self._validate_decision_fail_soft(
                        state, envelope, legacy, available_actions
                    )
                    beats = self._performance_from_raw(data) or [
                        "你提出了这个问题。对方停顿片刻，确认自己听清了你的意思。"
                    ]
                    output = TurnCompositionOutput(
                        decision=decision,
                        performance=beats,
                        failure_performance=[],
                    )
                    record_agent_call(
                        state,
                        role="turn_composer",
                        result=result,
                        validation="fallback",
                    )
                    open_plan = self._apply_composition(
                        state, envelope, context, output, decision
                    )
                    return ComposedTurn(
                        decision=decision,
                        context=context,
                        composition=output,
                        open_plan=open_plan,
                    )
                decision = self._normalize_legacy_dialogue(envelope, decision)
                decision = self._validate_decision_fail_soft(
                    state, envelope, decision, available_actions
                )
                record_agent_call(
                    state,
                    role="turn_planner",
                    result=result,
                    validation="accepted",
                )
                return ComposedTurn(
                    decision=decision,
                    context=context,
                    legacy=True,
                )

            beats = self._performance_from_raw(data)
            if not beats:
                record_agent_call(
                    state,
                    role="turn_composer",
                    result=result,
                    validation="rejected",
                    error=True,
                )
                raise LLMUnavailable(
                    "Turn Composer returned no usable performance",
                    public_message=(
                        "模型已经响应，但没有生成可显示的完整剧情。行动未提交，游戏状态没有改变；请重试。"
                    ),
                ) from error
            decision_data = data.get("decision")
            try:
                decision = TurnPlannerDecision.model_validate(decision_data)
            except (ValidationError, TypeError):
                decision = self._fallback_decision(state, envelope)
            decision = self._validate_decision_fail_soft(
                state, envelope, decision, available_actions
            )
            output = TurnCompositionOutput(
                decision=decision,
                performance=beats,
                failure_performance=[],
            )
            record_agent_call(
                state,
                role="turn_composer",
                result=result,
                validation="fallback",
            )
        except LLMUnavailable:
            record_agent_call(
                state,
                role="turn_composer",
                result=result,
                validation="rejected",
                error=True,
            )
            raise
        except Exception as error:
            record_agent_call(
                state,
                role="turn_composer",
                result=result,
                validation="rejected",
                error=True,
            )
            logger.exception("Turn Composer call failed")
            raise LLMUnavailable(
                "Turn Composer could not complete the player's turn",
                public_message=(
                    "模型服务暂时没有完成这一轮，行动未提交，游戏状态没有改变。请稍后重试。"
                ),
            ) from error
        else:
            record_agent_call(
                state,
                role="turn_composer",
                result=result,
                validation="accepted",
            )

        decision = self._validate_decision_fail_soft(
            state, envelope, output.decision, available_actions
        )
        output.decision = decision
        open_plan = self._apply_composition(state, envelope, context, output, decision)
        return ComposedTurn(
            decision=decision,
            context=context,
            composition=output,
            open_plan=open_plan,
        )
