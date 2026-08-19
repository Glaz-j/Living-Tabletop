from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import random
import re
import time
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .engine import GameEngine
from .harness import StructuredHarness
from .llm import LLMResult, LLMUnavailable, OpenAICompatibleLLM
from .models import RuleChoice, ScenarioDefinition, SessionStatus, WorldState
from .scenario import create_initial_state


@dataclass(frozen=True, slots=True)
class TestPersona:
    id: str
    name: str
    style: str
    probe_weights: dict[str, int]


PERSONAS: tuple[TestPersona, ...] = (
    TestPersona(
        "immersive",
        "沉浸派演员",
        "始终用第一人称入戏，重视人物情绪和关系连续性。",
        {"roleplay": 5, "followup": 4, "smalltalk": 2, "mixed": 2, "location": 1},
    ),
    TestPersona(
        "mischief",
        "调皮捣蛋鬼",
        "喜欢碰边角物件、说怪话、临时改主意，但不会照按钮走。",
        {"mischief": 6, "ambiguous": 3, "derail": 3, "mixed": 2, "smalltalk": 1},
    ),
    TestPersona(
        "bug_hunter",
        "极限找茬者",
        "专门提出复合问题、指代追问和容易被误判的否定句。",
        {"multipart": 6, "location": 5, "memory": 4, "ambiguous": 3, "negotiation": 2},
    ),
    TestPersona(
        "negotiator",
        "精明谈判家",
        "反复确认报价、条件和承诺是否一致，不轻易接受模糊回答。",
        {"negotiation": 7, "followup": 4, "multipart": 2, "smalltalk": 1, "mixed": 2},
    ),
    TestPersona(
        "wanderer",
        "离题漫游者",
        "会聊生活、天气和城中琐事，也会主动永久偏离主线。",
        {"derail": 5, "smalltalk": 5, "roleplay": 2, "location": 2, "mischief": 2},
    ),
    TestPersona(
        "terse",
        "惜字如金者",
        "只说很短的话，常省略主语，考验指代和紧邻上下文。",
        {"ambiguous": 7, "followup": 5, "location": 2, "negotiation": 2, "mischief": 1},
    ),
    TestPersona(
        "chaotic",
        "混沌即兴者",
        "把动作、对话和临时目标揉在同一句里，期待世界自然承接。",
        {"mixed": 7, "mischief": 4, "derail": 3, "multipart": 3, "roleplay": 2},
    ),
    TestPersona(
        "continuity",
        "连续性审计员",
        "不断引用刚发生的原话、数字与对象，寻找说话者颠倒和失忆。",
        {"memory": 7, "followup": 5, "negotiation": 3, "multipart": 2, "ambiguous": 2},
    ),
)


@dataclass(slots=True)
class ProposedPlayerTurn:
    text: str
    probe: str
    invariants: list[str] = field(default_factory=list)


class PlayerDriver(Protocol):
    def propose(
        self,
        *,
        persona: TestPersona,
        turn: int,
        public_view: dict[str, Any],
        previous_inputs: list[str],
        previous_output: str,
    ) -> ProposedPlayerTurn: ...


class HeuristicPlayerDriver:
    """Seeded improviser. It uses probe families, never authored action ids."""

    def __init__(self, seed: int):
        self.random = random.Random(seed ^ 0xA61E)

    @staticmethod
    def _scene_parts(view: dict[str, Any]) -> tuple[str, str, str]:
        scene = view.get("scene", {})
        visual = scene.get("visual", {}) or {}
        actors = visual.get("actors", []) or []
        exits = visual.get("exits", []) or []
        npc = str(actors[0].get("name", "你")) if actors else "你"
        place = str(scene.get("name") or visual.get("location_name") or "这里")
        destination = (
            str(exits[0].get("destination_name") or exits[0].get("label") or "城里的疗养院")
            if exits
            else "城里的疗养院"
        )
        return npc, place, destination

    def _choose_probe(self, persona: TestPersona) -> str:
        names = list(persona.probe_weights)
        weights = [persona.probe_weights[name] for name in names]
        return self.random.choices(names, weights=weights, k=1)[0]

    def propose(
        self,
        *,
        persona: TestPersona,
        turn: int,
        public_view: dict[str, Any],
        previous_inputs: list[str],
        previous_output: str,
    ) -> ProposedPlayerTurn:
        npc, place, destination = self._scene_parts(public_view)
        probe = self._choose_probe(persona)
        tail = re.sub(r"\s+", "", previous_output)[-18:] if previous_output else "刚才那件事"
        variants: dict[str, list[tuple[str, list[str]]]] = {
            "roleplay": [
                (f"我把声音放轻，对{npc}说：我愿意听，你从最难开口的地方讲吧。", []),
                (f"我没有催促，只看着{npc}问：你现在最担心我发现什么？", []),
                (f"我在{place}找个不挡路的位置坐下，说：先别管委托，我们认真谈谈。", []),
            ],
            "smalltalk": [
                (f"我随口和{npc}聊起今天闷得反常的天气，问这种天气在本地常见吗？", ["nonempty_reply"]),
                (f"我问{npc}：附近有没有能喝到像样咖啡的地方？", ["nonempty_reply"]),
                (f"我笑着说这里空气真够呛，{npc}平时也这么觉得吗？", ["nonempty_reply"]),
            ],
            "followup": [
                (f"等等，{npc}，你刚才那句话具体是什么意思？", ["nonempty_reply"]),
                (f"那然后呢？别换话题，把刚才没说完的接着说。", ["nonempty_reply"]),
                (f"你提到的“{tail}”是亲眼看见的，还是听别人说的？", ["nonempty_reply"]),
            ],
            "location": [
                (f"我只问路，暂时不去：{destination}具体在什么地方？", ["no_location_change", "nonempty_reply", "location_answered"]),
                (f"{npc}，从这里到{destination}通常怎么走？我还没决定要不要出发。", ["no_location_change", "nonempty_reply", "location_answered"]),
                (f"{destination}是在城南还是城北？先告诉我位置就好。", ["no_location_change", "nonempty_reply", "location_answered"]),
            ],
            "negotiation": [
                (f"我对{npc}说：这个价钱不够，风险比你一开始讲的大，得加钱。", ["speaker_roles_preserved", "nonempty_reply"]),
                (f"我追问{npc}：你说‘可以商量’，那具体愿意再加多少？", ["speaker_roles_preserved", "nonempty_reply"]),
                (f"我摇头：别再重复委托内容，先正面回答报酬能不能提高。", ["nonempty_reply"]),
            ],
            "mischief": [
                (f"我突然敲了敲{place}里最近的木头表面，贴过去听它会不会回敲。", []),
                (f"我把桌上的杯子悄悄转了半圈，装作什么都没发生，观察{npc}。", []),
                (f"我一本正经地问{npc}：如果我现在学猫叫，最先把谁吓跑？", ["nonempty_reply"]),
            ],
            "derail": [
                ("我决定先不查这件案子了，去城里找一家还开门的书店逛逛。", []),
                (f"我想暂时离开{place}，回家吃饭睡一觉，明早再决定做什么。", []),
                (f"我问{npc}愿不愿意先陪我去吃点东西，案子晚些再谈。", ["nonempty_reply"]),
            ],
            "ambiguous": [
                ("那个呢？", ["nonempty_reply"]),
                ("再多一点？", ["nonempty_reply"]),
                ("不是去，我只是问问。", ["no_location_change", "nonempty_reply"]),
                ("他们后来呢？", ["nonempty_reply"]),
            ],
            "memory": [
                (f"刚才明明说到“{tail}”，请从这里接下去，不要重讲开头。", ["no_duplicate_performance", "nonempty_reply"]),
                (f"我复述一遍你刚才的关键词“{tail}”，问{npc}有没有记错。", ["nonempty_reply"]),
                ("你上一句是谁说的？别把我的话算到你自己头上。", ["speaker_roles_preserved", "nonempty_reply"]),
            ],
            "multipart": [
                (f"{npc}，请一次说清楚：事情何时开始、发生在哪、你亲眼见过什么、还有谁知道？", ["nonempty_reply"]),
                (f"我问三个问题：{destination}在哪儿、要走多久、现在能不能探视？我并没有说要去。", ["no_location_change", "nonempty_reply"]),
                ("先回答钱，再回答危险，最后告诉我你有没有隐瞒；别只挑一个说。", ["nonempty_reply"]),
            ],
            "mixed": [
                (f"我收好手边的东西，但不离开这里，同时问{npc}：还有什么没告诉我？", ["nonempty_reply"]),
                (f"我走到窗边看看天色，仍留在{place}，回头问{npc}刚才是不是撒谎。", ["no_location_change", "nonempty_reply"]),
                (f"我先向{npc}道歉，又立刻追问{destination}的地址，但明确说今天不去。", ["no_location_change", "nonempty_reply"]),
            ],
        }
        choices = variants[probe]
        offset = (turn + len(previous_inputs) + self.random.randrange(len(choices))) % len(choices)
        text, invariants = choices[offset]
        if text in previous_inputs:
            # Natural uniqueness: vary the utterance rather than appending a test id.
            qualifiers = ["换句话说，", "我想了想又补充：", "这次我说得更明确些：", "我压低声音："]
            text = f"{qualifiers[(turn + offset) % len(qualifiers)]}{text}"
        return ProposedPlayerTurn(text=text, probe=probe, invariants=list(invariants))


class LLMPlayerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    probe: str = Field(min_length=1, max_length=40)
    invariants: list[str] = Field(default_factory=list, max_length=6)


class LLMPlayerDriver:
    """Optional local/API player brain; duplicate outputs are repaired locally."""

    def __init__(self, llm: OpenAICompatibleLLM, seed: int):
        self.llm = llm
        self.seed = seed
        self.fallback = HeuristicPlayerDriver(seed)

    def propose(
        self,
        *,
        persona: TestPersona,
        turn: int,
        public_view: dict[str, Any],
        previous_inputs: list[str],
        previous_output: str,
    ) -> ProposedPlayerTurn:
        payload = {
            "persona": asdict(persona),
            "turn": turn,
            "public_view": public_view,
            "last_output": previous_output[-1200:],
            "recent_inputs": previous_inputs[-8:],
            "instruction": "invent one new free-text player utterance; do not select or copy a button",
        }
        try:
            result = StructuredHarness(self.llm, max_repairs=0).run(
                LLMPlayerProposal,
                system=(
                    "你是破坏性但公平的 TRPG 测试玩家。严格扮演 persona，依据当前公开画面自由输入，"
                    "不要沿默认脚本、不要输出 action_id、不要重复 recent_inputs。"
                    "probe 可用 location/smalltalk/followup/negotiation/memory/multipart/mischief/derail/mixed。"
                    "只在纯提问明确不授权移动时添加 no_location_change；所有对话添加 nonempty_reply。"
                ),
                user_payload=payload,
                max_output_tokens=300,
                temperature=0.9,
                reasoning_effort="none",
            ).value
        except Exception:
            return self.fallback.propose(
                persona=persona,
                turn=turn,
                public_view=public_view,
                previous_inputs=previous_inputs,
                previous_output=previous_output,
            )
        text = result.text.strip()
        if text in previous_inputs:
            return self.fallback.propose(
                persona=persona,
                turn=turn,
                public_view=public_view,
                previous_inputs=previous_inputs,
                previous_output=previous_output,
            )
        return ProposedPlayerTurn(text=text, probe=result.probe, invariants=result.invariants)


class SyntheticComposerLLM:
    """Fast dynamic V2 contract double used for high-volume invariant fuzzing."""

    enabled = True

    def __init__(self, seed: int = 1):
        self.random = random.Random(seed ^ 0xC0DEC0DE)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs) -> LLMResult:
        self.calls.append(kwargs)
        if kwargs.get("schema_name") != "TurnCompositionOutput":
            raise LLMUnavailable("Synthetic test model only implements TurnCompositionOutput")
        payload = kwargs["user_payload"]
        text = payload["current_turn"]["player_text_verbatim"]
        context = payload["context"]
        topic = re.sub(r"\s+", "", text)[:24]
        present = [
            item for item in context.get("present_entities", []) if item.get("type") == "NPC"
        ]
        npc = present[0] if present else None
        question = bool(re.search(r"[?？]|什么|怎么|哪里|哪儿|谁|多久|多少|吗|呢", text))
        explicit_move = bool(
            re.search(r"(?:我|我们).{0,12}(?:去|前往|离开|回家|返回|走到|进入)", text)
            and not re.search(
                r"(?:不去|没说要去|只是问|暂时不去|今天不去|并没有说要去|仍留在|不离开)",
                text,
            )
        )
        knowledge = context.get("present_npc_knowledge", [])
        used_fact_ids: list[str] = []
        proposed_facts: list[dict[str, Any]] = []
        answered: list[str] = []
        unresolved: list[str] = []

        if explicit_move:
            action_type = "REST" if "睡" in text or "休息" in text else "MOVE"
            performance = ["你按自己的决定离开眼前事务，街上的声响很快接替了室内的低语。"]
            destination = "调查员临时选择的去处"
            plan = {
                "label": "按玩家决定离开当前地点",
                "action_type": action_type,
                "goal": text,
                "destination_name": destination,
                "destination_description": "城中一个普通、安全且不涉及案件真相的去处。",
                "duration_minutes": 12,
                "resolution": "automatic",
                "risk": "safe",
                "rest_until_hour": 8 if action_type == "REST" else None,
                "rest_day_offset": 1 if action_type == "REST" else 0,
                "speech_act": "none",
            }
        elif npc:
            action_type = "TALK"
            npc_name = npc["name"]
            fact = knowledge[0] if knowledge and question else None
            if question and re.search(r"地址|哪里|哪儿|什么地方|怎么走|城南|城北", text):
                referenced = next(
                    (
                        item
                        for item in context.get("referenced_entities", [])
                        if item.get("type") == "LOCATION" and item.get("name") in text
                    ),
                    None,
                )
                if referenced:
                    route = "在罗克斯伯里区旧电车总站北侧，沿石墙走到第二个路口"
                    performance = [f"“{route}。你到了那里再问门房就不会走错。”{npc_name}回答。"]
                    proposed_facts = [
                        {
                            "subject_entity_id": referenced["id"],
                            "predicate": "route_description",
                            "value": route,
                            "confidence": 0.86,
                        }
                    ]
                    answered = [text]
                else:
                    performance = [f"“我不知道确切地址，不想拿猜测骗你。”{npc_name}坦率地说。"]
                    unresolved = [text]
            elif fact:
                used_fact_ids.append(fact["fact_id"])
                answer = str(fact["value"])
                performance = [f"{npc_name}认真听完，没有岔开话题。", f"“我能确定的是：{answer}。”"]
                answered = [text]
            elif "钱" in text or "报酬" in text or "价钱" in text:
                performance = [
                    f"{npc_name}把手指从杯沿移开，重新衡量你的条件。",
                    "“你说得对，风险比我最初描述的大。价钱可以再谈，但我要先知道你希望加到多少。”",
                ]
            elif question:
                performance = [
                    f"{npc_name}先确认你问的是“{topic}”，随后直接回答。",
                    "“这部分我并不确定；我能说的只有亲眼见过的事。”",
                ]
                unresolved = [text]
            else:
                performance = [
                    f"{npc_name}顺着你这次关于“{topic}”的话作出反应。",
                    "“我听明白了。你继续，我不会把你的意思改成别的。”",
                ]
            plan = {
                "label": f"与{npc_name}继续当前对话",
                "action_type": action_type,
                "goal": text,
                "target_name": npc_name,
                "target_entity_id": npc["id"],
                "duration_minutes": 1,
                "resolution": "automatic",
                "risk": "safe",
                "speech_act": "question" if question else "statement",
                "addressee_id": npc["id"],
            }
        else:
            plan = {
                "label": "处理玩家的自由行动",
                "action_type": "OTHER",
                "goal": text,
                "duration_minutes": 1,
                "resolution": "automatic",
                "risk": "safe",
                "speech_act": "none",
            }
            if question and re.search(r"地址|哪里|哪儿|什么地方|怎么走|城南|城北", text):
                performance = [
                    "现场没有人在场回答，而你掌握的资料也不足以确定具体地址。"
                ]
                unresolved = [text]
            else:
                performance = ["你的动作在当前场景里留下了清楚的痕迹，环境随之给出细微回应。"]

        output = {
            "decision": {
                "existing_action_id": None,
                "open_plan": plan,
                "confidence": 0.92,
            },
            "performance": performance,
            "failure_performance": [],
            "used_fact_ids": used_fact_ids,
            "proposed_facts": proposed_facts,
            "answered_query_parts": answered,
            "unresolved_query_parts": unresolved,
        }
        return LLMResult(data=output, latency_ms=1, input_tokens=100, output_tokens=80)


@dataclass(slots=True)
class TestAgentIssue:
    severity: str
    code: str
    message: str
    persona_id: str
    turn: int
    player_text: str
    probe: str
    state_version: int


@dataclass(slots=True)
class TestAgentStep:
    turn: int
    player_text: str
    probe: str
    invariants: list[str]
    location_before: str | None
    location_after: str | None
    version_before: int
    version_after: int
    accepted: bool
    narrative: str
    latency_ms: int


@dataclass(slots=True)
class TestAgentRun:
    scenario_id: str
    persona_id: str
    persona_name: str
    seed: int
    steps: list[TestAgentStep] = field(default_factory=list)
    issues: list[TestAgentIssue] = field(default_factory=list)
    probes: Counter[str] = field(default_factory=Counter)
    final_status: str = SessionStatus.ACTIVE.value

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probes"] = dict(self.probes)
        return payload


@dataclass(slots=True)
class TestAgentReport:
    scenario_id: str
    runs: list[TestAgentRun]

    @property
    def turn_count(self) -> int:
        return sum(len(run.steps) for run in self.runs)

    @property
    def issues(self) -> list[TestAgentIssue]:
        return [issue for run in self.runs for issue in run.issues]

    @property
    def unique_input_ratio(self) -> float:
        inputs = [step.player_text for run in self.runs for step in run.steps]
        return len(set(inputs)) / len(inputs) if inputs else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_count": len(self.runs),
            "turn_count": self.turn_count,
            "issue_count": len(self.issues),
            "unique_input_ratio": self.unique_input_ratio,
            "issue_codes": dict(Counter(issue.code for issue in self.issues)),
            "probe_coverage": dict(
                Counter(step.probe for run in self.runs for step in run.steps)
            ),
            "runs": [run.to_dict() for run in self.runs],
        }

    def to_markdown(self) -> str:
        data = self.to_dict()
        lines = [
            f"# Test Agent report: `{self.scenario_id}`",
            "",
            f"- Runs / turns: {data['run_count']} / {data['turn_count']}",
            f"- Issues: {data['issue_count']}",
            f"- Unique input ratio: {data['unique_input_ratio']:.1%}",
            f"- Probe coverage: {data['probe_coverage']}",
            "",
            "| Persona | Seed | Turns | Issues | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for run in self.runs:
            lines.append(
                f"| {run.persona_name} | {run.seed} | {len(run.steps)} | "
                f"{len(run.issues)} | {run.final_status} |"
            )
        if self.issues:
            lines.extend(["", "## Failure corpus", ""])
            for issue in self.issues:
                lines.append(
                    f"- **{issue.severity.upper()}** `{issue.code}` "
                    f"{issue.persona_id} turn {issue.turn}: {issue.message} — `{issue.player_text}`"
                )
        return "\n".join(lines) + "\n"


class TestAgentRunner:
    def __init__(
        self,
        scenario: ScenarioDefinition,
        engine: GameEngine,
        driver: PlayerDriver,
        shared_inputs: set[str] | None = None,
    ):
        self.scenario = scenario
        self.engine = engine
        self.driver = driver
        self.shared_inputs = shared_inputs if shared_inputs is not None else set()

    @staticmethod
    def _narrative(state: WorldState) -> str:
        sequence = state.narrative_sequence
        if sequence and sequence.beats:
            return "\n".join(beat.text for beat in sequence.beats).strip()
        return state.last_narrative.strip()

    @staticmethod
    def _normalized(text: str) -> str:
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()

    def _issue(
        self,
        run: TestAgentRun,
        *,
        code: str,
        message: str,
        turn: int,
        proposed: ProposedPlayerTurn,
        version: int,
        severity: str = "error",
    ) -> None:
        run.issues.append(
            TestAgentIssue(
                severity=severity,
                code=code,
                message=message,
                persona_id=run.persona_id,
                turn=turn,
                player_text=proposed.text,
                probe=proposed.probe,
                state_version=version,
            )
        )

    def run(
        self,
        persona: TestPersona,
        *,
        seed: int,
        max_turns: int,
    ) -> TestAgentRun:
        state = create_initial_state(self.scenario, seed=seed)
        run = TestAgentRun(
            scenario_id=self.scenario.id,
            persona_id=persona.id,
            persona_name=persona.name,
            seed=seed,
        )
        inputs: list[str] = []
        previous_output = ""
        for turn in range(1, max_turns + 1):
            if state.status != SessionStatus.ACTIVE:
                break
            proposed = self.driver.propose(
                persona=persona,
                turn=turn,
                public_view=self.engine.public_view(state),
                previous_inputs=inputs,
                previous_output=previous_output,
            )
            for _attempt in range(4):
                if proposed.text not in self.shared_inputs:
                    break
                proposed = self.driver.propose(
                    persona=persona,
                    turn=turn,
                    public_view=self.engine.public_view(state),
                    previous_inputs=[*inputs, *sorted(self.shared_inputs)],
                    previous_output=previous_output,
                )
            while proposed.text in self.shared_inputs:
                proposed.text = f"我换了个说法，{proposed.text}"
            run.probes[proposed.probe] += 1
            if proposed.text in inputs:
                self._issue(
                    run,
                    code="duplicate_player_input",
                    message="player driver repeated an earlier utterance",
                    turn=turn,
                    proposed=proposed,
                    version=state.version,
                )
                continue
            inputs.append(proposed.text)
            self.shared_inputs.add(proposed.text)
            location_before = state.entities[state.player.entity_id].location
            version_before = state.version
            started = time.perf_counter()
            try:
                next_state, resolution = self.engine.play(state, text=proposed.text)
                if resolution.awaiting_rule_choice:
                    next_state, resolution = self.engine.play(
                        next_state,
                        rule_choice=RuleChoice.ACCEPT_FAILURE,
                    )
            except LLMUnavailable as error:
                self._issue(
                    run,
                    code="llm_unavailable",
                    message=error.public_message or str(error),
                    turn=turn,
                    proposed=proposed,
                    version=state.version,
                )
                continue
            except Exception as error:
                self._issue(
                    run,
                    code="engine_exception",
                    message=f"{type(error).__name__}: {error}",
                    turn=turn,
                    proposed=proposed,
                    version=state.version,
                )
                continue
            latency_ms = round((time.perf_counter() - started) * 1000)
            location_after = next_state.entities[next_state.player.entity_id].location
            narrative = self._narrative(next_state)

            if not resolution.accepted:
                self._issue(
                    run,
                    code="free_text_rejected",
                    message=resolution.clarification or "free-text turn was rejected",
                    turn=turn,
                    proposed=proposed,
                    version=version_before,
                )
            if resolution.accepted and next_state.version <= version_before:
                self._issue(
                    run,
                    code="version_not_advanced",
                    message="accepted turn did not advance the state version",
                    turn=turn,
                    proposed=proposed,
                    version=version_before,
                )
            if "no_location_change" in proposed.invariants and location_after != location_before:
                self._issue(
                    run,
                    code="unauthorized_movement",
                    message=f"question moved player from {location_before} to {location_after}",
                    turn=turn,
                    proposed=proposed,
                    version=version_before,
                )
            if "nonempty_reply" in proposed.invariants and not narrative:
                self._issue(
                    run,
                    code="empty_performance",
                    message="dialogue turn produced no visible performance",
                    turn=turn,
                    proposed=proposed,
                    version=version_before,
                )
            if (
                "location_answered" in proposed.invariants
                and not resolution.interrupted
                and not re.search(
                r"(?:位于|地址|坐落|城南|城北|北侧|南侧|附近|旁边|街|路|大道|电车|"
                r"不知道|不清楚|不确定|说不准|没听说|无法确定)",
                narrative,
                )
            ):
                self._issue(
                    run,
                    code="location_question_not_answered",
                    message="reply neither located the place nor honestly said it was unknown",
                    turn=turn,
                    proposed=proposed,
                    version=version_before,
                )
            if "no_duplicate_performance" in proposed.invariants and previous_output:
                current = self._normalized(narrative)
                previous = self._normalized(previous_output)
                if current and previous and (current == previous or current in previous or previous in current):
                    self._issue(
                        run,
                        code="duplicate_performance",
                        message="current performance substantially repeated the previous turn",
                        turn=turn,
                        proposed=proposed,
                        version=version_before,
                    )
            if next_state.narrative_sequence and next_state.narrative_sequence.player_text != proposed.text:
                self._issue(
                    run,
                    code="player_text_not_preserved",
                    message="narrative sequence did not preserve the verbatim player utterance",
                    turn=turn,
                    proposed=proposed,
                    version=version_before,
                )

            run.steps.append(
                TestAgentStep(
                    turn=turn,
                    player_text=proposed.text,
                    probe=proposed.probe,
                    invariants=proposed.invariants,
                    location_before=location_before,
                    location_after=location_after,
                    version_before=version_before,
                    version_after=next_state.version,
                    accepted=resolution.accepted,
                    narrative=narrative,
                    latency_ms=latency_ms,
                )
            )
            state = next_state
            previous_output = narrative
        run.final_status = state.status.value
        return run


def run_test_agents(
    scenario: ScenarioDefinition,
    *,
    personas: Iterable[TestPersona] = PERSONAS,
    seeds: Iterable[int] = (11, 29),
    turns_per_run: int = 24,
    driver_factory=None,
    engine_factory=None,
) -> TestAgentReport:
    runs: list[TestAgentRun] = []
    shared_inputs: set[str] = set()
    for persona in personas:
        for seed in seeds:
            llm = SyntheticComposerLLM(seed) if engine_factory is None else None
            engine = (
                GameEngine(scenario, llm)
                if engine_factory is None
                else engine_factory(persona, seed)
            )
            driver = (
                HeuristicPlayerDriver(seed)
                if driver_factory is None
                else driver_factory(persona, seed)
            )
            runs.append(
                TestAgentRunner(scenario, engine, driver, shared_inputs).run(
                    persona,
                    seed=seed,
                    max_turns=turns_per_run,
                )
            )
    return TestAgentReport(scenario_id=scenario.id, runs=runs)
