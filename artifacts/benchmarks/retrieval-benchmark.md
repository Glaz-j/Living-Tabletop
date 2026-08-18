# Retrieval benchmark

- Scenario: `the_haunting_corbitt_house_v1`
- Eval SHA-256: `0b8308354ab38d384af104a620d8dc78e0c74f8cf0c34148a45dee3331bde87e`
- Iterations per case: 250
- Selected strategy: `typed_hybrid_v2`

| Strategy | Exact | Top-1 | Macro F1 | Unknown | Forbidden | P50 ms | P95 ms | Peak KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 61.5% | 79.5% | 71.8% | 72.2% | 4 | 0.572 | 1.140 | 349.0 |
| bm25 | 61.5% | 74.4% | 68.8% | 66.7% | 7 | 0.540 | 0.844 | 351.2 |
| typed_hybrid_v1 | 89.7% | 89.7% | 89.7% | 77.8% | 4 | 0.760 | 1.249 | 359.7 |
| typed_hybrid_v2 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 0.834 | 1.215 | 361.7 |

## Remaining failures

None.

## Measurement boundary

Latency and memory cover deterministic retrieval only. Planner and Narrator generation are intentionally excluded so model-routing changes do not distort algorithm comparison.
`peak_python_alloc_kib` is measured with `tracemalloc`; RSS delta is also recorded in the JSON report when psutil is available.
