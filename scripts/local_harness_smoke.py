from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from living_tabletop.engine import GameEngine
from living_tabletop.keeper import Keeper
from living_tabletop.llm import LLMSettings, OpenAICompatibleLLM
from living_tabletop.models import ActionIntent, PlayerVisibleMemory
from living_tabletop.scenario import create_initial_state, load_scenario


@dataclass(frozen=True)
class SmokeCase:
    name: str
    location_id: str
    player_text: str
    setup_action_id: str | None = None
    visible_text: str | None = None


CASES = {
    case.name: case
    for case in (
        SmokeCase("look_upstairs", "loc_house_ground", "看看楼上。"),
        SmokeCase(
            "overnight_rest",
            "loc_house_upper",
            "我想回家休息一下，然后第二天再来。",
        ),
        SmokeCase(
            "leave_city",
            "loc_cafe",
            "我去火车站买票离开这座城。",
        ),
        SmokeCase(
            "off_mainline",
            "loc_cafe",
            "我去唐人街吃碗面，再找个旅馆住下。",
        ),
        SmokeCase(
            "follow_up_duration",
            "loc_cafe",
            "他们在房子里住了多久？",
            setup_action_id="cafe_question_knott",
        ),
        SmokeCase(
            "visible_dossier",
            "loc_globe",
            "我去把地板上那份关于马卡里奥一家的简报拿来看看。",
            visible_text="地板上散落着一份关于马卡里奥一家的简报。",
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real Keeper inputs through the local structured harness.")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        dest="cases",
        help="Run one named case; repeat the option for multiple cases.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "max"],
        default="none",
    )
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--context-window", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = load_scenario(scenario_id="the_haunting_corbitt_house_v1")
    local_llm = OpenAICompatibleLLM(
        LLMSettings(
            enabled=True,
            api_key="ollama",
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=120,
            cooldown_seconds=0,
            max_retries=0,
            max_output_tokens=max(256, args.max_output_tokens),
            context_window=max(1024, args.context_window),
            temperature=0,
            reasoning_effort=args.reasoning_effort,
        ),
        provider="local",
    )
    offline_llm = OpenAICompatibleLLM(LLMSettings(enabled=False, api_key=None))
    engine = GameEngine(scenario, offline_llm)
    keeper = Keeper(local_llm, scenario)
    selected = args.cases or list(CASES)
    failed = False

    for case_name in selected:
        case = CASES[case_name]
        state = create_initial_state(scenario, seed=19)
        state.entities[state.player.entity_id].location = case.location_id
        if case.visible_text:
            state.visible_history.append(
                PlayerVisibleMemory(
                    id="visible_smoke_01",
                    state_version=state.version,
                    world_time=state.world_time,
                    location_id=case.location_id,
                    kind="soft_canon",
                    source="generated",
                    text=case.visible_text,
                )
            )
        if case.setup_action_id:
            state, resolution = engine.play(state, action_id=case.setup_action_id)
            if not resolution.accepted:
                raise RuntimeError(f"setup action failed: {case.setup_action_id}")
        intent = ActionIntent(
            content=case.player_text,
            goal=case.player_text,
            confidence=1,
            source="player_text",
        )
        try:
            decision = keeper.adjudicate(
                state,
                intent,
                engine.kernel.available_actions(state),
            )
            call = state.agent_calls[-1]
            output = {
                "case": case.name,
                "ok": True,
                "latency_ms": call.latency_ms,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "decision": decision.model_dump(mode="json"),
            }
        except Exception as error:
            failed = True
            output = {
                "case": case.name,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "cause": (
                    f"{type(error.__cause__).__name__}: {error.__cause__}"
                    if error.__cause__ is not None
                    else None
                ),
            }
        print(json.dumps(output, ensure_ascii=False))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
