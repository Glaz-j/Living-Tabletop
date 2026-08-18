from __future__ import annotations

import argparse
import json
from pathlib import Path

from living_tabletop.retrieval_benchmark import render_markdown, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local NPC knowledge retrieval")
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("evals/retrieval/the_haunting_v1.json"),
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/retrieval-benchmark.json"),
    )
    args = parser.parse_args()
    report = run_benchmark(args.eval, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
