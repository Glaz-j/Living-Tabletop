from __future__ import annotations

from typing import Any

from .context import context_relevance_score, recent_visible_context
from .harness import StructuredHarness, strict_json_schema
from .llm import LLMUnavailable, OpenAICompatibleLLM, record_agent_call
from .models import (
    ActionDefinition,
    ActionIntent,
    ActionType,
    KeeperDecision,
    ScenarioDefinition,
    WorldState,
)


class Keeper:
    """Adjudicates any player intent while keeping state mutation inside the kernel."""

    def __init__(self, llm: OpenAICompatibleLLM, scenario: ScenarioDefinition):
        self.llm = llm
        self.scenario = scenario

    @staticmethod
    def _decision_schema() -> dict:
        schema = strict_json_schema(KeeperDecision)
        return {
            "$defs": schema.get("$defs", {}),
            "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "existing_action_id": {"type": "string", "minLength": 1},
                    "open_plan": {"type": "null"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["existing_action_id", "open_plan", "confidence"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "existing_action_id": {"type": "null"},
                    "open_plan": {"$ref": "#/$defs/OpenActionPlan"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["existing_action_id", "open_plan", "confidence"],
            },
            ],
        }

    def adjudicate(
        self,
        state: WorldState,
        intent: ActionIntent,
        available_actions: list[ActionDefinition],
    ) -> KeeperDecision:
        text = (intent.content or intent.goal or "").strip()
        if not self.llm.enabled:
            raise LLMUnavailable("Player text requires the LLM intent interpreter")

        player_entity = state.entities[state.player.entity_id]
        location = state.entities.get(player_entity.location or "")
        location_description = location.attributes.get("description", "") if location else ""
        if context_relevance_score(text, str(location_description)) == 0:
            location_description = ""
        present_entities = [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type.value,
                "role": entity.attributes.get("role"),
            }
            for entity in state.entities.values()
            if entity.active and entity.location == player_entity.location and entity.id != state.player.entity_id
        ]
        present_entity_ids = {entity["id"] for entity in present_entities}
        known_facts = [
            {
                "id": fact.id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "value": fact.value,
                "canon": fact.canon,
            }
            for fact_id in sorted(state.player_known_fact_ids)
            if (fact := state.facts.get(fact_id)) is not None
        ]
        known_facts = [
            fact
            for fact in known_facts
            if context_relevance_score(
                text,
                f"{fact['subject']} {fact['predicate']} {fact['value']}",
            )
            > 0
        ]
        present_npc_knowledge = [
            {
                "knower_id": entry.knower_id,
                "fact_id": entry.fact_id,
                "fact": {
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "value": entry.belief_value if entry.belief_value is not None else fact.value,
                },
                "confidence": entry.confidence,
                "concealed": entry.concealed,
            }
            for entry in state.npc_knowledge
            if entry.knower_id in present_entity_ids
            and (fact := state.facts.get(entry.fact_id)) is not None
        ]
        present_npc_knowledge = [
            entry
            for entry in present_npc_knowledge
            if context_relevance_score(
                text,
                (
                    f"{entry['fact']['subject']} {entry['fact']['predicate']} "
                    f"{entry['fact']['value']}"
                ),
            )
            > 0
        ]
        recent_player_inputs = [
            {
                "kind": "player_intent",
                "action_id": event.payload.get("action_id"),
                "text": event.payload.get("player_text"),
            }
            for event in state.event_log[-30:]
            if event.type == "action_started" and event.payload.get("player_text")
        ][-8:]
        recent_player_inputs = [
            item
            for item in recent_player_inputs
            if context_relevance_score(text, str(item["text"])) > 0
        ]
        visible_history = recent_visible_context(
            state,
            query=text,
            immediate_entries=0,
        )
        context_actions = [
            action
            for action in available_actions
            if context_relevance_score(
                text,
                " ".join(
                    str(value)
                    for value in (
                        action.label,
                        action.target,
                        *action.aliases,
                        action.dialogue_text,
                    )
                    if value
                ),
            )
            > 0
        ]
        payload: dict[str, Any] = {
            "player_text": text,
            "world_time": state.world_time.isoformat(),
            "current_scene": {
                "id": location.id if location else None,
                "name": location.name if location else "未知地点",
                "description": location_description,
            },
            "present_entities": present_entities,
            "known_facts": known_facts,
            "present_npc_knowledge": present_npc_knowledge,
            "recent_context": {
                "visible_history": visible_history,
                "recent_player_inputs": recent_player_inputs,
                "trust_policy": {
                    "hard_canon": "确定发生且不可否认的世界事实或演出",
                    "soft_canon": "玩家已经看见的场景细节；后续必须承认其存在，但可在不矛盾的前提下解释",
                    "dialogue_claim": "角色说过的话；说话行为为真，但内容可能未知、错误或欺骗",
                    "player_intent": "玩家曾表达的意图，不代表行动结果已经成功",
                },
            },
            "inventory": [
                {"id": item_id, "name": state.entities[item_id].name}
                for item_id in state.player.inventory
                if item_id in state.entities
                and context_relevance_score(text, state.entities[item_id].name) > 0
            ],
            "player_capabilities": {
                "skills": state.player.skills,
                "characteristics": state.player.characteristics,
            },
            "available_actions": [
                {
                    "id": action.id,
                    "label": action.label,
                    "type": action.type.value,
                    "target": action.target,
                    "aliases": action.aliases,
                    "dialogue_text": action.dialogue_text,
                    "risk": action.risk,
                    "category": action.category,
                    "suggested_to_player": action.suggest,
                    "requires_explicit_intent": action.risk == "dangerous" or not action.suggest,
                }
                for action in context_actions
            ],
            "required_output": {
                "existing_action_id": (
                    "an available id only when its complete meaning exactly serves the intent; otherwise null"
                ),
                "confidence": "0 to 1",
                "open_plan": {
                    "label": "short player-facing action label",
                    "action_type": [item.value for item in ActionType],
                    "goal": "the literal player goal only; never add manipulation, testing, or investigation they did not state",
                    "target_name": "optional",
                    "target_entity_id": "optional known id",
                    "destination_name": "optional; may name a new off-script location",
                    "destination_entity_id": "optional known id",
                    "destination_description": "optional visible, soft-canon description",
                    "duration_minutes": "0 to 1440, excluding an overnight rest",
                    "resolution": ["automatic", "check", "impossible"],
                    "skill": "one listed player skill/characteristic or null",
                    "difficulty": ["regular", "hard", "extreme"],
                    "risk": ["safe", "uncertain", "dangerous"],
                    "rest_until_hour": "0-23 or null",
                    "rest_day_offset": "0-7",
                    "success_text": "visible consequence seed that stays on the player's exact current topic",
                    "failure_text": "visible failed-attempt seed that stays on the player's exact current topic",
                },
            },
            "current_request_contract": {
                "player_text_verbatim": text,
                "instruction": (
                    "This is the only action to adjudicate now. open_plan.goal must copy "
                    "player_text_verbatim exactly, character for character."
                ),
            },
        }
        system = (
            "你是 Living Tabletop 的守秘人（KP）仲裁器。玩家可以尝试任何行动，也可以永久偏离预设主线。"
            "绝不能仅因行动不在 available_actions 中而拒绝；预设动作只是快捷方式。"
            "这是玩家主动输入的原文，必须由你根据完整语义和对话上下文判断；严禁根据关键词、别名或局部词语重合选择动作。"
            "疑问句、追问、假设、否定和对某事的提及，都不等于玩家承诺执行该事。向在场 NPC 提问通常应生成 TALK open_plan。"
            "不得把普通寒暄、闲聊或字面提问擅自扩展成试探、套话、观察反应、缓和僵局、说服或调查；"
            "只有玩家原文明示这些目的时，goal 才能包含它们。与愿意交谈的在场 NPC 普通聊天通常是 automatic，"
            "只有玩家明确要说服、欺骗、威胁，或输入资料明确说明 NPC 会抵抗当前请求时才使用 check。"
            "success_text 也不得写成‘你试图借这个话题缓和/试探/观察’，只演出玩家字面说的话与 NPC 对该话题的直接反应。"
            "标记 requires_explicit_intent 的预设动作，只有当玩家明确表达了该动作完整目标及承诺时才能选择；"
            "例如询问房子是否安全，绝不等于向他人报告房子安全。"
            "玩家决定的是意图，不是结果：普通且可行的行为 automatic；有真实不确定性或风险时 check；"
            "recent_context.visible_history 是玩家已经实际看见或听见的演出记忆，必须按 trust_policy 延续，"
            "这些历史只用于解析当前原文明确提及的对象与避免矛盾，不能替当前原文选择话题或目的。"
            "known_facts 与 present_npc_knowledge 已按当前话题检索；不要把 available_actions 中的其他案件主题写入本轮演出。"
            "current_scene.description 与 available_actions 也已按当前话题检索；列表为空时正常生成 open_plan。"
            "不得仅因某个可见物未写入 entities、facts 或 available_actions 就否认它的存在。"
            "玩家拾取、翻看、阅读或触碰演出中明确可见且触手可及的物品时，表层互动通常应为 automatic；"
            "只有发现隐藏内容、解读专业信息、突破阻碍或承担真实风险时才进行 check。"
            "dialogue_claim 只证明角色说过这句话，不自动证明台词内容是客观事实。"
            "物理上不可能的结果用 impossible，但仍接受玩家的尝试并描述世界为何没有按其宣称改变。"
            "若现有动作准确覆盖意图可选择 existing_action_id，否则必须给出 open_plan。"
            "对话 open_plan 的 success_text 应使用现场直接对话体；只依据已知事实和对应 NPC 的知识作答，"
            "success_text 与 failure_text 都必须回应当前玩家原文的具体话题，不得借失败重新播放上一话题或强行转回主线。"
            "资料没有精确答案时让 NPC 坦率表示不知道，不得编造硬设定。concealed 知识不得无理由泄露。"
            "不要泄露隐藏事实，不要为了主线强迫玩家，不要直接提出任何世界状态写入或效果操作。"
            "最后再次核对 current_request_contract：你只处理 player_text_verbatim 这一轮输入；"
            "若生成 open_plan，其 goal 必须逐字复制 player_text_verbatim，不能概括或改写。"
            "只输出 JSON 对象。"
        )
        available_by_id = {action.id: action for action in available_actions}

        def validate_decision(decision: KeeperDecision) -> None:
            if (
                decision.existing_action_id is not None
                and decision.existing_action_id not in available_by_id
            ):
                raise ValueError(
                    f"existing_action_id is not currently available: {decision.existing_action_id}"
                )
            if decision.open_plan is not None and decision.open_plan.goal.strip() != text:
                raise ValueError(
                    "open_plan.goal must exactly copy current_request_contract.player_text_verbatim; "
                    "the previous output lost or rewrote the player's current intent"
                )

        try:
            outcome = StructuredHarness(self.llm).run(
                KeeperDecision,
                system=system,
                user_payload=payload,
                post_validate=validate_decision,
                response_schema=self._decision_schema(),
            )
            decision = outcome.value
            if decision.existing_action_id is not None:
                record_agent_call(
                    state,
                    role="keeper",
                    result=outcome.llm_result,
                    validation="accepted",
                )
                return decision

            plan = decision.open_plan
            if plan is None:
                raise ValueError("Keeper omitted open_plan")
            allowed_skills = {
                *state.player.skills,
                *state.player.characteristics,
                "luck",
            }
            if plan.skill not in allowed_skills:
                plan.skill = None
                if plan.resolution == "check":
                    plan.resolution = "automatic"
            decision = KeeperDecision(open_plan=plan, confidence=decision.confidence)
            record_agent_call(
                state,
                role="keeper",
                result=outcome.llm_result,
                validation="accepted",
            )
            return decision
        except Exception as exc:
            record_agent_call(state, role="keeper", result=None, validation="rejected", error=True)
            raise LLMUnavailable(
                "LLM could not interpret the player's action",
                public_message=getattr(exc, "public_message", None),
                failures=getattr(exc, "failures", None),
            ) from exc
