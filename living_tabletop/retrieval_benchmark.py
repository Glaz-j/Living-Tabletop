from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .agent_runtime.contracts import KnowledgeQuery
from .agent_runtime.knowledge import KnowledgeResolver, RetrievalStrategy
from .scenario import create_initial_state, load_scenario


DEFAULT_STRATEGIES: tuple[RetrievalStrategy, ...] = (
    "legacy",
    "bm25",
    "typed_hybrid_v1",
    "typed_hybrid_v2",
)


def load_eval_set(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported retrieval eval schema")
    if not payload.get("cases"):
        raise ValueError("Retrieval eval set contains no cases")
    ids = [item["id"] for item in payload["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Retrieval eval case IDs must be unique")
    for item in payload["cases"]:
        KnowledgeQuery.model_validate(item["query"])
        if not isinstance(item.get("gold_fact_ids"), list):
            raise ValueError(f"Eval case {item['id']} has no gold_fact_ids list")
    return payload


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _case_quality(retrieved: list[str], gold: set[str]) -> dict[str, float | bool]:
    retrieved_set = set(retrieved)
    intersection = len(retrieved_set & gold)
    precision = intersection / len(retrieved_set) if retrieved_set else (1.0 if not gold else 0.0)
    recall = intersection / len(gold) if gold else (1.0 if not retrieved_set else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": retrieved_set == gold,
        "top1": (retrieved[0] in gold) if gold and retrieved else (not gold and not retrieved),
    }


def benchmark_strategy(
    eval_payload: dict[str, Any],
    strategy: RetrievalStrategy,
    *,
    iterations: int = 100,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    scenario = load_scenario(scenario_id=eval_payload["scenario_id"])
    state = create_initial_state(scenario, seed=19, session_id=f"retrieval-eval-{strategy}")
    resolver = KnowledgeResolver(strategy=strategy)
    cases = eval_payload["cases"]
    queries = [(item, KnowledgeQuery.model_validate(item["query"])) for item in cases]

    # Warm caches and Python code paths without counting the warm-up.
    rss_before = _rss_bytes()
    for _item, query in queries:
        resolver.retrieve(state, query)

    start_cpu = time.process_time_ns()
    latencies_ms: list[float] = []
    latest_results: dict[str, list[str]] = {}
    for _ in range(iterations):
        for item, query in queries:
            started = time.perf_counter_ns()
            candidates = resolver.retrieve(state, query)
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            latest_results[item["id"]] = [candidate.fact_id for candidate in candidates]
    cpu_ms = (time.process_time_ns() - start_cpu) / 1_000_000

    # Measure a single-query peak separately so benchmark bookkeeping is excluded.
    gc.collect()
    transient_peak = 0
    tracemalloc.start()
    for _item, query in queries:
        before, _previous_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        measured_candidates = resolver.retrieve(state, query)
        _current, query_peak = tracemalloc.get_traced_memory()
        transient_peak = max(transient_peak, max(0, query_peak - before))
        del measured_candidates
    tracemalloc.stop()
    rss_after = _rss_bytes()
    cache_kib = resolver.cache_size_bytes() / 1024
    transient_kib = transient_peak / 1024

    case_results: list[dict[str, Any]] = []
    tag_metrics: dict[str, list[float]] = defaultdict(list)
    forbidden_hits = 0
    negative_cases = 0
    negative_correct = 0
    for item in cases:
        retrieved = latest_results[item["id"]]
        gold = set(item["gold_fact_ids"])
        forbidden = set(item.get("forbidden_fact_ids", []))
        quality = _case_quality(retrieved, gold)
        hit_forbidden = sorted(set(retrieved) & forbidden)
        forbidden_hits += len(hit_forbidden)
        if not gold:
            negative_cases += 1
            negative_correct += int(not retrieved)
        for tag in item.get("tags", []):
            tag_metrics[tag].append(float(quality["f1"]))
        case_results.append(
            {
                "id": item["id"],
                "gold_fact_ids": sorted(gold),
                "retrieved_fact_ids": retrieved,
                "forbidden_hits": hit_forbidden,
                **quality,
            }
        )

    count = len(case_results)
    aggregate = {
        "case_count": count,
        "exact_set_accuracy": sum(bool(item["exact"]) for item in case_results) / count,
        "top1_accuracy": sum(bool(item["top1"]) for item in case_results) / count,
        "macro_precision": statistics.fmean(float(item["precision"]) for item in case_results),
        "macro_recall": statistics.fmean(float(item["recall"]) for item in case_results),
        "macro_f1": statistics.fmean(float(item["f1"]) for item in case_results),
        "unknown_accuracy": negative_correct / negative_cases if negative_cases else 1.0,
        "forbidden_hit_count": forbidden_hits,
        "latency_ms_mean": statistics.fmean(latencies_ms),
        "latency_ms_p50": _percentile(latencies_ms, 0.50),
        "latency_ms_p95": _percentile(latencies_ms, 0.95),
        "throughput_queries_per_second": len(latencies_ms) / max(0.000001, sum(latencies_ms) / 1000),
        "cpu_ms_total": cpu_ms,
        "peak_python_alloc_kib": transient_kib,
        "cache_index_kib": cache_kib,
        "retrieval_memory_kib": transient_kib + cache_kib,
        "rss_delta_kib": (
            max(0, rss_after - rss_before) / 1024
            if rss_before is not None and rss_after is not None
            else None
        ),
    }
    return {
        "strategy": strategy,
        "aggregate": aggregate,
        "tag_macro_f1": {
            tag: statistics.fmean(scores) for tag, scores in sorted(tag_metrics.items())
        },
        "failures": [
            item
            for item in case_results
            if not item["exact"] or item["forbidden_hits"]
        ],
        "cases": case_results,
    }


def run_benchmark(
    eval_path: Path,
    *,
    strategies: Iterable[RetrievalStrategy] = DEFAULT_STRATEGIES,
    iterations: int = 100,
) -> dict[str, Any]:
    eval_payload = load_eval_set(eval_path)
    results = [
        benchmark_strategy(eval_payload, strategy, iterations=iterations)
        for strategy in strategies
    ]
    selected = max(
        results,
        key=lambda item: (
            item["aggregate"]["macro_f1"],
            item["aggregate"]["unknown_accuracy"],
            -item["aggregate"]["forbidden_hit_count"],
            -item["aggregate"]["latency_ms_p95"],
            -item["aggregate"]["retrieval_memory_kib"],
        ),
    )
    raw_eval = eval_path.read_bytes()
    return {
        "benchmark_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_path": str(eval_path),
        "eval_sha256": hashlib.sha256(raw_eval).hexdigest(),
        "scenario_id": eval_payload["scenario_id"],
        "iterations_per_case": iterations,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "selected_strategy": selected["strategy"],
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval benchmark",
        "",
        f"- Scenario: `{report['scenario_id']}`",
        f"- Eval SHA-256: `{report['eval_sha256']}`",
        f"- Iterations per case: {report['iterations_per_case']}",
        f"- Selected strategy: `{report['selected_strategy']}`",
        "",
        "| Strategy | Exact | Top-1 | Macro F1 | Unknown | Forbidden | P50 ms | P95 ms | Transient KiB | Cache KiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        aggregate = item["aggregate"]
        lines.append(
            "| {strategy} | {exact:.1%} | {top1:.1%} | {f1:.1%} | {unknown:.1%} | {forbidden} | {p50:.3f} | {p95:.3f} | {peak:.1f} | {cache:.1f} |".format(
                strategy=item["strategy"],
                exact=aggregate["exact_set_accuracy"],
                top1=aggregate["top1_accuracy"],
                f1=aggregate["macro_f1"],
                unknown=aggregate["unknown_accuracy"],
                forbidden=aggregate["forbidden_hit_count"],
                p50=aggregate["latency_ms_p50"],
                p95=aggregate["latency_ms_p95"],
                peak=aggregate["peak_python_alloc_kib"],
                cache=aggregate["cache_index_kib"],
            )
        )
    lines.extend(["", "## Remaining failures", ""])
    selected = next(
        item for item in report["results"] if item["strategy"] == report["selected_strategy"]
    )
    if not selected["failures"]:
        lines.append("None.")
    else:
        for failure in selected["failures"]:
            lines.append(
                f"- `{failure['id']}`: expected {failure['gold_fact_ids']}, got {failure['retrieved_fact_ids']}"
            )
    lines.extend(
        [
            "",
            "## Measurement boundary",
            "",
            "Latency and memory cover deterministic retrieval only. Planner and Narrator generation are intentionally excluded so model-routing changes do not distort algorithm comparison.",
            "`peak_python_alloc_kib` measures one retrieval call and excludes benchmark bookkeeping. Persistent cached-index payload is reported separately as `cache_index_kib`; RSS delta is also recorded in the JSON report when psutil is available.",
            "",
        ]
    )
    return "\n".join(lines)
