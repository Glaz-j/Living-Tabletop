from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

from living_tabletop.engine import GameEngine
from living_tabletop.llm import RoutedLLM
from living_tabletop.models import SessionStatus
from living_tabletop.scenario import load_scenarios
from living_tabletop.test_agent import (
    HeuristicPlayerDriver,
    LLMPlayerDriver,
    PERSONAS,
    SyntheticComposerLLM,
    TestAgentIssue,
    TestAgentRun,
    TestAgentRunner,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def build_jobs(
    scenario_ids: list[str],
    *,
    total_runs: int | None,
    runs_per_persona: int,
) -> list[dict[str, Any]]:
    combinations = [
        (scenario_index, scenario_id, persona_index, persona)
        for persona_index, persona in enumerate(PERSONAS)
        for scenario_index, scenario_id in enumerate(scenario_ids)
    ]
    requested = (
        total_runs
        if total_runs is not None
        else len(combinations) * runs_per_persona
    )
    jobs: list[dict[str, Any]] = []
    for index in range(requested):
        cycle, combination_index = divmod(index, len(combinations))
        scenario_index, scenario_id, persona_index, persona = combinations[
            combination_index
        ]
        seed = 1009 + cycle * 10007 + persona_index * 101 + scenario_index * 1009
        jobs.append(
            {
                "index": index,
                "run_id": f"run-{index + 1:05d}",
                "scenario_id": scenario_id,
                "scenario_index": scenario_index,
                "persona": persona,
                "persona_index": persona_index,
                "cycle": cycle,
                "seed": seed,
            }
        )
    return jobs


def _run_path(output_dir: Path, job: dict[str, Any]) -> Path:
    return output_dir / "runs" / (
        f"{job['run_id']}__{job['scenario_id']}__{job['persona'].id}__{job['seed']}.json"
    )


def _load_completed_runs(output_dir: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "runs").glob("run-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(payload.get("run_id") or "")
        if run_id:
            completed[run_id] = payload
    return completed


def _campaign_summary(
    runs: list[dict[str, Any]],
    *,
    requested_runs: int,
    turns_per_run: int,
    game_backend: str,
    player_backend: str,
    started_at: str,
    status: str,
) -> dict[str, Any]:
    issues = [issue for run in runs for issue in run.get("issues", [])]
    steps = [step for run in runs for step in run.get("steps", [])]
    inputs = [str(step.get("player_text") or "") for step in steps]
    return {
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "requested_runs": requested_runs,
        "completed_runs": len(runs),
        "remaining_runs": max(0, requested_runs - len(runs)),
        "turns_per_run": turns_per_run,
        "completed_turns": len(steps),
        "expected_turn_capacity": requested_runs * turns_per_run,
        "game_backend": game_backend,
        "player_backend": player_backend,
        "issue_count": len(issues),
        "issue_codes": dict(Counter(str(issue.get("code")) for issue in issues)),
        "issue_severities": dict(
            Counter(str(issue.get("severity")) for issue in issues)
        ),
        "probe_coverage": dict(Counter(str(step.get("probe")) for step in steps)),
        "scenario_runs": dict(Counter(str(run.get("scenario_id")) for run in runs)),
        "persona_runs": dict(Counter(str(run.get("persona_id")) for run in runs)),
        "unique_input_count": len(set(inputs)),
        "unique_input_ratio": len(set(inputs)) / len(inputs) if inputs else 1.0,
        "mean_latency_ms": (
            round(sum(int(step.get("latency_ms") or 0) for step in steps) / len(steps), 2)
            if steps
            else 0
        ),
    }


def _summary_markdown(summary: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    lines = [
        "# Living Tabletop large-scale Test Agent report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Runs: {summary['completed_runs']} / {summary['requested_runs']}",
        f"- Turns: {summary['completed_turns']} / {summary['expected_turn_capacity']}",
        f"- Issues: {summary['issue_count']} — {summary['issue_codes']}",
        f"- Backends: game=`{summary['game_backend']}`, player=`{summary['player_backend']}`",
        f"- Unique utterances: {summary['unique_input_count']} ({summary['unique_input_ratio']:.1%})",
        f"- Mean engine latency: {summary['mean_latency_ms']} ms",
        f"- Scenario balance: {summary['scenario_runs']}",
        f"- Persona balance: {summary['persona_runs']}",
        f"- Probe coverage: {summary['probe_coverage']}",
        "",
    ]
    if issues:
        lines.extend(["## Failure corpus preview", ""])
        for issue in issues[:100]:
            lines.append(
                f"- **{str(issue.get('severity', 'error')).upper()}** "
                f"`{issue.get('code', 'unknown')}` {issue.get('scenario_id', '')} "
                f"{issue.get('persona_id', '')} turn {issue.get('turn', '?')}: "
                f"{issue.get('message', '')} — `{issue.get('player_text', '')}`"
            )
        if len(issues) > 100:
            lines.append(
                f"- …另有 {len(issues) - 100} 条，完整内容见 `failures.jsonl`。"
            )
    else:
        lines.extend(["## Failure corpus", "", "No invariant violations recorded."])
    return "\n".join(lines) + "\n"


def _persist_campaign(
    output_dir: Path,
    runs: list[dict[str, Any]],
    *,
    requested_runs: int,
    turns_per_run: int,
    game_backend: str,
    player_backend: str,
    started_at: str,
    status: str,
) -> dict[str, Any]:
    summary = _campaign_summary(
        runs,
        requested_runs=requested_runs,
        turns_per_run=turns_per_run,
        game_backend=game_backend,
        player_backend=player_backend,
        started_at=started_at,
        status=status,
    )
    issues: list[dict[str, Any]] = []
    for run in runs:
        for raw_issue in run.get("issues", []):
            issue = dict(raw_issue)
            issue["run_id"] = run.get("run_id")
            issue["scenario_id"] = run.get("scenario_id")
            issue["seed"] = run.get("seed")
            issues.append(issue)
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_text(output_dir / "summary.md", _summary_markdown(summary, issues))
    _atomic_text(
        output_dir / "failures.jsonl",
        "".join(json.dumps(issue, ensure_ascii=False) + "\n" for issue in issues),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpointed, resumable, multi-persona free-text Test Agents "
            "against Living Tabletop V2."
        )
    )
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--runs-per-persona", type=int, default=2)
    parser.add_argument(
        "--total-runs",
        type=int,
        help="Exact campaign run count across all selected scenarios and personas.",
    )
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument(
        "--game-backend",
        choices=("synthetic", "live"),
        default="synthetic",
        help="Synthetic is fast invariant fuzzing; live uses the configured model.",
    )
    parser.add_argument(
        "--player-backend",
        choices=("heuristic", "llm"),
        default="heuristic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-agent-reports"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed run checkpoints in output-dir (default: true).",
    )
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args()

    if args.runs_per_persona < 1:
        parser.error("--runs-per-persona must be positive")
    if args.total_runs is not None and args.total_runs < 1:
        parser.error("--total-runs must be positive")
    if args.turns < 1:
        parser.error("--turns must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")

    scenarios = load_scenarios()
    scenario_ids = list(scenarios) if args.scenario == "all" else [args.scenario]
    if any(item not in scenarios for item in scenario_ids):
        parser.error("unknown scenario id")

    jobs = build_jobs(
        scenario_ids,
        total_runs=args.total_runs,
        runs_per_persona=args.runs_per_persona,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "format_version": 2,
        "created_at": _utc_now(),
        "scenario_ids": scenario_ids,
        "persona_ids": [persona.id for persona in PERSONAS],
        "total_runs": len(jobs),
        "turns_per_run": args.turns,
        "game_backend": args.game_backend,
        "player_backend": args.player_backend,
        "jobs": [
            {
                key: value.id if key == "persona" else value
                for key, value in job.items()
            }
            for job in jobs
        ],
    }
    if manifest_path.exists() and args.resume:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = (
            "scenario_ids",
            "persona_ids",
            "total_runs",
            "turns_per_run",
            "game_backend",
            "player_backend",
        )
        if any(existing_manifest.get(key) != manifest.get(key) for key in comparable):
            parser.error(
                "output-dir contains an incompatible campaign; choose another directory "
                "or use --no-resume"
            )
        started_at = str(existing_manifest.get("created_at") or manifest["created_at"])
    elif manifest_path.exists():
        parser.error(
            "output-dir already contains a campaign; choose a new directory, "
            "or omit --no-resume to continue it"
        )
    else:
        started_at = str(manifest["created_at"])
        _atomic_json(manifest_path, manifest)

    completed = _load_completed_runs(args.output_dir) if args.resume else {}
    runs = [completed[job["run_id"]] for job in jobs if job["run_id"] in completed]
    _persist_campaign(
        args.output_dir,
        runs,
        requested_runs=len(jobs),
        turns_per_run=args.turns,
        game_backend=args.game_backend,
        player_backend=args.player_backend,
        started_at=started_at,
        status="running",
    )

    started_clock = time.perf_counter()
    initially_completed = len(runs)
    for job in jobs:
        if job["run_id"] in completed:
            continue
        scenario = scenarios[job["scenario_id"]]
        persona = job["persona"]
        seed = int(job["seed"])
        try:
            game_llm = (
                SyntheticComposerLLM(seed)
                if args.game_backend == "synthetic"
                else RoutedLLM.from_env()
            )
            engine = GameEngine(scenario, game_llm)
            driver = (
                LLMPlayerDriver(RoutedLLM.from_env(), seed)
                if args.player_backend == "llm"
                else HeuristicPlayerDriver(seed)
            )
            # Repeated input across independent games is useful Monte Carlo
            # coverage; only repetition inside one game is a driver bug.
            result = TestAgentRunner(scenario, engine, driver, set()).run(
                persona,
                seed=seed,
                max_turns=args.turns,
            )
        except Exception as error:  # keep an overnight campaign alive and reproducible
            result = TestAgentRun(
                scenario_id=scenario.id,
                persona_id=persona.id,
                persona_name=persona.name,
                seed=seed,
                issues=[
                    TestAgentIssue(
                        severity="error",
                        code="runner_crash",
                        message=f"{type(error).__name__}: {error}",
                        persona_id=persona.id,
                        turn=0,
                        player_text="",
                        probe="campaign",
                        state_version=0,
                    )
                ],
                final_status=SessionStatus.ACTIVE.value,
            )
        payload = result.to_dict()
        payload.update(
            {
                "run_id": job["run_id"],
                "run_index": job["index"],
                "cycle": job["cycle"],
                "completed_at": _utc_now(),
            }
        )
        _atomic_json(_run_path(args.output_dir, job), payload)
        completed[job["run_id"]] = payload
        runs.append(payload)

        if len(runs) % args.progress_every == 0 or len(runs) == len(jobs):
            summary = _persist_campaign(
                args.output_dir,
                runs,
                requested_runs=len(jobs),
                turns_per_run=args.turns,
                game_backend=args.game_backend,
                player_backend=args.player_backend,
                started_at=started_at,
                status="running" if len(runs) < len(jobs) else "analyzing",
            )
            elapsed = max(time.perf_counter() - started_clock, 0.001)
            new_runs = max(1, len(runs) - initially_completed)
            print(
                f"[{summary['completed_runs']}/{summary['requested_runs']}] "
                f"turns={summary['completed_turns']} issues={summary['issue_count']} "
                f"elapsed={elapsed:.1f}s rate={new_runs / elapsed:.2f} runs/s",
                flush=True,
            )

    runs.sort(key=lambda item: int(item.get("run_index", 0)))
    issues = [issue for run in runs for issue in run.get("issues", [])]
    status = "completed_with_issues" if issues else "completed"
    summary = _persist_campaign(
        args.output_dir,
        runs,
        requested_runs=len(jobs),
        turns_per_run=args.turns,
        game_backend=args.game_backend,
        player_backend=args.player_backend,
        started_at=started_at,
        status=status,
    )
    print(_summary_markdown(summary, issues), flush=True)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
