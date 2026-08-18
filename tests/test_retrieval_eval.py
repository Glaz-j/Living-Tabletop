from __future__ import annotations

from pathlib import Path

from living_tabletop.retrieval_benchmark import (
    benchmark_strategy,
    load_eval_set,
)


EVAL_PATH = Path("evals/retrieval/the_haunting_v1.json")


def test_retrieval_eval_set_is_valid_and_covers_regressions():
    payload = load_eval_set(EVAL_PATH)
    cases = payload["cases"]
    tags = {tag for case in cases for tag in case.get("tags", [])}

    assert len(cases) >= 35
    assert {"positive", "negative", "compound", "regression", "wrong_addressee"} <= tags
    assert any(case["id"] == "regression_duration_and_previous_incidents" for case in cases)


def test_optimized_retrieval_beats_legacy_on_accuracy_without_large_memory_cost():
    payload = load_eval_set(EVAL_PATH)
    legacy = benchmark_strategy(payload, "legacy", iterations=1)["aggregate"]
    optimized = benchmark_strategy(payload, "typed_hybrid_v2", iterations=1)["aggregate"]

    assert optimized["macro_f1"] >= 0.99
    assert optimized["unknown_accuracy"] == 1.0
    assert optimized["forbidden_hit_count"] == 0
    assert optimized["macro_f1"] - legacy["macro_f1"] >= 0.20
    assert optimized["latency_ms_p95"] < 25
    assert optimized["peak_python_alloc_kib"] < 2048
