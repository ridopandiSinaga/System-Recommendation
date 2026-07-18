# Catalog-size benchmark decision

- Status: **retain the deployed 1,000 interaction / 3,000 content catalog**
- Run date: 2026-07-17
- Raw artifact: `artifacts/evaluation/catalog_size_benchmark.json`
- Compact notebook evidence: `docs/evidence/catalog_size_benchmark_summary.json`
- Artifact SHA-256: `ffeef9e038f50e164097f45a5b8603896ee564d78b9a2b22d0d600913adbfc45`

## Question

Does a larger candidate catalog improve end-to-end recommendation quality and user reach
enough to justify its serving cost?

This is not an experiment that increases the test-set size. The evaluation cohorts remain
fixed; the treatment is the number of interaction and content books available to the model.

## Leakage-safe protocol

- One outer cohort of 200 users was reserved as fixed test data.
- Model selection used five development-only validation repeats of 200 users each.
- Every catalog configuration received exactly the same user/work holdout pairs within a
  repeat. Unavailable targets remained in end-to-end metrics as zero rather than disappearing.
- Hybrid weights, neighbors, dynamic weighting, candidate-pool size, rating threshold, and
  all data filters were held constant.
- Two challengers required a Bonferroni-adjusted 97.5% paired confidence interval, a positive
  lower-bound NDCG@10 delta, and a validation win rate of at least 60%.
- Reach could not decrease. P95 inference latency was limited to 3x baseline and serialized
  model size to 4x baseline.
- Fixed test was opened only after validation selection and only for baseline plus a selected
  challenger. Because no challenger passed validation, test evaluated baseline only.

## Validation results

All ranking and reach values below are macro averages over the five repeated splits. Discovery
metrics use one common, train-only catalog/popularity/TF-IDF reference per split, so their
denominators are comparable across configurations.

| Interaction / content | NDCG@10 | HitRate@10 | Content target retention | Personalized retention | Unique works recommended | Coverage vs common reference | Novelty@10 | P95 latency | Serialized model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 / 3,000 | 0.01872 | 0.028 | 24.7% | 12.6% | 391.0 | 2.17% | 11.30 | 105.1 ms | 9.65 MB |
| 3,000 / 10,000 | 0.02207 | 0.035 | 38.8% | 23.3% | 682.2 | 3.78% | 12.04 | 232.7 ms | 24.89 MB |
| 5,000 / 20,000 | 0.02180 | 0.035 | 46.4% | 29.0% | 818.2 | 4.53% | 12.46 | 408.9 ms | 41.86 MB |

The larger catalogs clearly improve reach and discovery. The 3,000/10,000 challenger retains
14.1 percentage points more targets and 10.7 points more personalized-evaluable holdouts than
baseline. The 5,000/20,000 challenger gains 21.7 and 16.4 points respectively.

The ranking improvement is not yet statistically reliable:

| Challenger | Mean paired NDCG@10 delta | Adjusted 97.5% CI | Positive-repeat rate | Resource gate | Decision |
|---|---:|---:|---:|---|---|
| 3,000 / 10,000 | +0.00335 | [-0.00311, 0.01081] | 80% | Pass (2.21x latency, 2.58x size) | Fail quality evidence |
| 5,000 / 20,000 | +0.00309 | [-0.00549, 0.01301] | 60% | Fail (3.89x latency, 4.34x size) | Fail quality and cost evidence |

The 3,000/10,000 configuration has the most promising trade-off: its NDCG point estimate is
about 17.9% above baseline and it passes both resource limits. However, its paired interval
still crosses zero. Promoting it now would turn an encouraging observation into an unsupported
production claim.

## Fixed-test and serving decision

Validation retained baseline, so the untouched fixed test evaluated only `ib1000_cb3000`:

- NDCG@10: 0.00631
- HitRate@10: 0.010
- content target retention: 23.0%
- personalized retention: 9.5%

These fixed-test numbers estimate baseline performance; they are **not** evidence that baseline
beat challengers on test, because the challengers were deliberately not inspected there.

The existing serving bundle already uses 1,000 interaction and 3,000 content books with fixed
weights 0.40 / 0.45 / 0.15. No serving artifact was rebuilt.

## Resource interpretation

Timing is environment-dependent and should be compared as a ratio within this run. Each model
was warmed once and measured on the same traffic-mix holdouts with native thread pools limited
to one thread. Serialized Joblib bytes and sparse/dense matrix payload bytes are portable model
size proxies; they are not presented as peak process RAM.

The current implementation sends the full content catalog into the collaborative model as well.
Content-only books therefore create zero rows in the CF item-user matrix and increase artifact
and inference cost. A follow-up optimization can fit CF only on interacting books and map its
scores onto the full content universe, but that algorithm change should be benchmarked
separately rather than mixed into this size experiment.

The additional content slots are currently filled by the project's deterministic catalog
selection rule. Improving candidate composition using train-only popularity, metadata quality,
or retrieval relevance may produce more value than increasing size alone.

## Reproduction

```powershell
uv run book-rec benchmark-catalog `
  --data-dir data/raw `
  --catalog-grid "1000,3000;3000,10000;5000,20000" `
  --baseline-catalog "1000,3000" `
  --hybrid-weights "0.40,0.45,0.15" `
  --validation-users 200 `
  --test-users 200 `
  --repetitions 5 `
  --k 10 `
  --neighbors 50 `
  --hybrid-candidate-pool-size 500 `
  --selection-metric ndcg `
  --min-selection-rate 0.60 `
  --max-latency-ratio 3 `
  --max-serialized-model-ratio 4 `
  --bootstrap-resamples 2000 `
  --output artifacts/evaluation/catalog_size_benchmark.json

uv run python scripts/catalog_benchmark_summary.py generate `
  artifacts/evaluation/catalog_size_benchmark.json `
  docs/evidence/catalog_size_benchmark_summary.json

uv run python scripts/catalog_benchmark_summary.py check `
  docs/evidence/catalog_size_benchmark_summary.json
```

The artifact reader verifies the schema, algorithm contract, source fingerprint, catalog IDs,
resource metrics, identical within-split cohorts, validation/test user disjointness, selection,
deployment decision, and raw-file SHA before lineage can be attached to a serving bundle.
The committed compact summary excludes per-user rows, raw timing samples, and repeated-split
payloads, while preserving the validated decision evidence consumed by the notebook. CI checks
its schema and stamps the saved notebook against this summary, the tracked serving bundle, and
its manifest. Raw data and heavyweight ignored benchmark artifacts are intentionally outside a
clean CI checkout; changing either requires a local Run All and a fresh notebook stamp.
