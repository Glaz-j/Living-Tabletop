from __future__ import annotations

import re

from ..harness import StructuredHarness, strict_json_schema
from ..llm import LLMUnavailable, OpenAICompatibleLLM, record_agent_call
from ..models import ActionDefinition, ActionType, ScenarioDefinition, WorldState
from .context import ContextAssembler
from .contracts import KnowledgeQuery, PlayerIntentEnvelope, TurnPlannerDecision
from .validation import PlanValidator


class TurnPlanner:
    """LLM semantic planner. It never emits effects, facts, or result prose."""

    def __init__(self, llm: OpenAICompatibleLLM, scenario: ScenarioDefinition):
        self.llm = llm
        self.scenario = scenario
        self.context_assembler = ContextAssembler()
        self.validator = PlanValidator()

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        return bool(
            re.search(r"[?？]", text)
            or any(token in text for token in ("什么", "怎么", "为何", "为什么", "多久", "多少", "哪里", "谁", "吗", "呢"))
        )

    def plan(
        self,
        state: WorldState,
        envelope: PlayerIntentEnvelope,
        available_actions: list[ActionDefinition],
    ) -> tuple[TurnPlannerDecision, object]:
        if not self.llm.enabled:
            raise LLMUnavailable("Player text requires the TurnPlanner LLM")
        context = self.context_assembler.assemble(state, envelope, available_actions)
        payload = {
            "player_intent": envelope.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "available_actions": context.available_actions,
            "output_contract": {
                "existing_action_id": "use only when an available action exactly matches the whole intent",
                "open_plan": {
                    "goal": "copy player_intent.text exactly",
                    "action_type": [item.value for item in ActionType],
                    "resolution": ["automatic", "check", "impossible"],
                    "speech_act": [
                        "none", "question", "statement", "request", "smalltalk", "deception", "threat"
                    ],
                    "knowledge_query": (
                        "for factual questions to a present NPC; describe what to retrieve, never answer it; "
                        "split every independently answerable sub-question into atoms; relation_types may only use "
                        "duration,time,location,status,identity,cause,quantity,history,historical_pattern,"
                        "experience,family,weakness,ownership,burial"
                    ),
                },
            },
        }
        system = (
            "你是 Living Tabletop 的 TurnPlanner，只负责理解玩家这一轮想做什么。"
            "玩家可以尝试任何行动，也可以永久离开预设主线；不要因为动作不在按钮列表中而拒绝。"
            "疑问句通常是 TALK，不是调查、说服或试探；普通闲聊通常 automatic。"
            "只有结果存在真实不确定性或对方明确抵抗时才选择 check；物理上不可能的结果选择 impossible，仍接受这次尝试。"
            "若玩家向在场 NPC 询问事实，输出 speech_act=question、addressee_id、referents 和 KnowledgeQuery。"
            "KnowledgeQuery 只描述要查什么，不得猜测答案或 fact_id；复合问题必须按语义拆成 atoms，"
            "每个 atom 单独保留 query_text、subject_entity_ids、predicate_hints 和 relation_types。"
            "NPC 是否知道以及是否披露由后续系统决定。"
            "不要输出成功/失败文本、NPC 回答、事实值、效果、世界状态修改或导演建议。"
            "open_plan.goal 必须逐字复制玩家原文。只输出符合 JSON Schema 的对象。"
        )
        try:
            outcome = StructuredHarness(self.llm).run(
                TurnPlannerDecision,
                system=system,
                user_payload=payload,
                response_schema=strict_json_schema(TurnPlannerDecision),
                post_validate=lambda value: self.validator.validate(
                    state, envelope, value, available_actions
                ),
            )
            decision = self.validator.validate(state, envelope, outcome.value, available_actions)
            plan = decision.open_plan
            if plan is not None and plan.action_type in {ActionType.TALK, ActionType.DECEIVE}:
                if plan.speech_act == "none":
                    plan.speech_act = "question" if self._looks_like_question(envelope.text) else "statement"
                addressee = plan.addressee_id or plan.target_entity_id
                if plan.speech_act == "question" and addressee and plan.knowledge_query is None:
                    plan.knowledge_query = KnowledgeQuery(
                        query_text=envelope.text,
                        asker_id=envelope.actor_id,
                        addressee_id=addressee,
                        subject_entity_ids=[
                            item.entity_id for item in plan.referents if item.entity_id is not None
                        ],
                    )
                decision = TurnPlannerDecision(open_plan=plan, confidence=decision.confidence)
            record_agent_call(
                state,
                role="turn_planner",
                result=outcome.llm_result,
                validation="accepted",
            )
            return decision, context
        except Exception as exc:
            record_agent_call(
                state,
                role="turn_planner",
                result=None,
                validation="rejected",
                error=True,
            )
            raise LLMUnavailable(
                "LLM could not plan the player's action",
                public_message=getattr(exc, "public_message", None),
                failures=getattr(exc, "failures", None),
            ) from exc
