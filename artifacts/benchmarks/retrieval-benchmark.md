# Retrieval benchmark

- Scenario: `the_haunting_corbitt_house_v1`
- Eval SHA-256: `0b8308354ab38d384af104a620d8dc78e0c74f8cf0c34148a45dee3331bde87e`
- Iterations per case: 250
- Selected strategy: `typed_hybrid_v2`

| Strategy | Exact | Top-1 | Macro F1 | Unknown | Forbidden | P50 ms | P95 ms | Transient KiB | Cache KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 61.5% | 79.5% | 71.8% | 72.2% | 4 | 0.144 | 0.258 | 26.6 | 0.0 |
| bm25 | 61.5% | 74.4% | 68.8% | 66.7% | 7 | 0.159 | 0.228 | 32.6 | 0.0 |
| typed_hybrid_v1 | 89.7% | 89.7% | 89.7% | 77.8% | 4 | 0.188 | 0.277 | 34.8 | 0.0 |
| typed_hybrid_v2 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 0.084 | 0.130 | 8.1 | 19.6 |

## Remaining failures

None.

## Measurement boundary

Latency and memory cover deterministic retrieval only. Planner and Narrator generation are intentionally excluded so model-routing changes do not distort algorithm comparison.
`peak_python_alloc_kib` measures one retrieval call and excludes benchmark bookkeeping. Persistent cached-index payload is reported separately as `cache_index_kib`; RSS delta is also recorded in the JSON report when psutil is available.
