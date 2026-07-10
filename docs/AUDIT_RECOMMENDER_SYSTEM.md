# Audit Engineering Recommender System

Tanggal audit terakhir: 10 Juli 2026.

## Verdict

Project ini **bukan sekadar kulit atau notebook demo**. Ia sudah menunjukkan
kompetensi Machine Learning Engineer pada data contract, reproducible training,
offline evaluation, model packaging, serving boundary, observability dasar, test,
CI, dan container.

Namun status yang jujur adalah **production-oriented portfolio / strong offline
prototype**, belum production-proven. Yang belum terbukti bukan lagi “bisa membuat
model”, melainkan operasi setelah model menerima traffic: timestamped feedback,
online KPI, drift, registry, canary, rollback, security controls, dan SLA.

## Scorecard

| Area | Status | Penilaian |
| --- | --- | --- |
| Data contract dan preprocessing | Kuat | Schema, ID, range, implicit rating, duplicate policy, k-core, invariant, deterministic cap |
| Content recommendation | Kuat sebagai baseline | TF-IDF/cosine, bounded neighbor query, query-aware catalog, canonical-work dedup |
| Collaborative recommendation | Kuat sebagai baseline | Sparse adjusted-cosine, user bias centering, positive seed, popularity fallback |
| Hybrid ranking | Cukup kuat | Tiga sinyal, dynamic weights, strict ablation, cold-user fallback, explanations |
| Offline evaluation | Kuat | User-disjoint validation/test, work-level holdout, all-holdout end-to-end, bootstrap, segments |
| Reproducibility/lineage | Kuat | Lockfile, deterministic seeds, fingerprint, dependency/Git metadata, verified manifest |
| Serving | Cukup kuat | Load-once service, FastAPI, Streamlit, readiness, stable errors, request metrics |
| Testing/CI/package | Kuat | Negative contracts, coverage gate, multi-Python CI, wheel smoke, Docker build |
| Production operations | Belum | Tidak ada registry, online telemetry backend, canary, drift/quality alert, SLA |

## Perubahan yang sudah dikerjakan dari audit awal

### 1. Data semantics dan invariant

- Rating `0` Book-Crossing sekarang diperlakukan sebagai implicit/no explicit
  preference dan dibuang dari training explicit-feedback.
- Rating non-numeric, non-finite, atau di luar 0–10 ditolak.
- User ID harus integer positif; ISBN dinormalisasi.
- Duplicate `(user_id, isbn)` memakai mean deterministik.
- Rating tanpa metadata buku tidak boleh lolos.
- Minimum user/item frequency memakai convergent k-core, termasuk setelah
  `max_books` dan `max_ratings`.
- Interaction catalog dipisahkan dari content catalog. Semua item interaksi wajib
  ada, sedangkan cold-item dapat ditambahkan sampai `max_content_books`.

### 2. Model correctness

- Collaborative model memakai adjusted-cosine pada bias-adjusted explicit rating,
  bukan cosine mentah yang mudah didominasi rating scale user.
- Hanya rating pada/di atas positive threshold yang menjadi preference seed.
- Positive history dan profile seed dihitung per canonical work; beberapa edisi
  karya yang sama tidak menambah confidence berulang.
- Semua output Top-N dideduplikasi pada canonical work.
- Hybrid dynamic weight menghitung usable positive-work history, bukan seluruh
  rating termasuk dislike.
- Strict no-popularity/content ablation benar-benar fail closed tanpa filler leak.
- Legacy raw model dengan algorithm/model-format lama ditolak dan harus retrain.

### 3. Evaluasi dan leakage

- Validation user dan test user disjoint. Tuning hanya membaca validation;
  evaluasi final memakai test.
- Holdout memakai canonical work dan menghapus seluruh edisi target dari history.
- Filtering/candidate selection dilakukan sesudah holdout, dari train saja.
- Evaluator menjalankan **seluruh** holdout, bukan hanya intersection yang mudah.
- Scope `conditional` mensyaratkan target tersedia dan positive seed tersedia.
- Scope `end_to_end` menjalankan fallback aktual untuk cold-user dan menghitung
  target di luar katalog sebagai miss.
- Per-user rows menyimpan candidate/history/positive-seed flags, status, positive
  work count, dan segment.
- Baseline diberi nama eksplisit, misalnya
  `collaborative_with_popularity_fallback` dan `content_profile_strict`.
- Ranking metrics, bootstrap CI, catalog coverage, novelty, diversity, dan
  popularity concentration disimpan dengan scope yang eksplisit.
- Artifact evaluasi lama, wrong partition, wrong unit, wrong dataset/config,
  non-finite metrics, atau wrong algorithm version ditolak saat attachment.

Konsekuensi: angka audit lama berbasis ISBN atau hanya 74/200 evaluable holdout
tidak valid untuk dibandingkan dengan protocol aktif. Benchmark harus diregenerasi.

### 4. Artifact dan reproducibility

- Model disimpan bersama strict sidecar manifest berisi checksum, artifact ID,
  dataset fingerprint/schema/row count, config, metrics, Python/dependency version,
  Git SHA, dan dirty state.
- Writer memakai unique temporary name dan atomic replace.
- Loader menolak malformed/non-finite JSON, checksum mismatch, class/type mismatch,
  runtime incompatibility, dan model-format mismatch sebelum serving.
- Serving memeriksa nested model algorithm, catalog order, index mapping, sparse
  matrix shape/finite values, popularity table, ratings, manifest schema/rows, dan
  embedded dataset fingerprint.
- Hash dan deserialization memakai byte snapshot yang sama.

Catatan keamanan: checksum bukan tanda tangan. Pickle/joblib tetap harus datang
dari storage/pipeline terpercaya dan di-mount read-only.

### 5. Serving dan release engineering

- API dan Streamlit memakai satu verified `HybridRecommender` bundle; tidak ada
  tiga model yang berpotensi berasal dari training run berbeda.
- Serving load-only secara default. Runtime training hanya opt-in development.
- API memiliki liveness, readiness, model info, recommendation endpoints,
  in-process metrics, structured logging, sanitized request ID, bounded input,
  stable public error codes, dan generic 500.
- Streamlit tidak menampilkan internal exception/path dan membatasi cover ke host
  dataset yang dikenal.
- Download Kaggle development path memakai lock, stale-lock recovery, credential
  restoration, full staging parse/schema validation, checksum manifest, dan atomic
  directory publication.
- Docker berjalan non-root, membatasi BLAS thread, memiliki readiness healthcheck,
  serta mendokumentasikan read-only artifact mount.
- Wheel memasukkan package utama dan `app` adapters; CI smoke mengimpor wheel dari
  luar checkout.

## Risiko yang masih tersisa

### P1 — bukti performa dan stabilitas konfigurasi

Belum ada benchmark protocol terbaru pada cohort besar yang disimpan sebagai
evidence final. Grid tuning pada satu validation split dapat memilih bobot karena
noise. Lanjutkan dengan:

1. beberapa user-disjoint split atau repeated bootstrap;
2. paired-bootstrap delta antar konfigurasi;
3. laporan segment cold/no-positive/sparse/dense;
4. challenger seperti implicit ALS/BPR atau learned ranker;
5. acceptance threshold yang ditentukan sebelum membuka test partition.

### P1 — temporal dan online evaluation

Dataset tidak memiliki timestamp interaksi. User-disjoint leave-one-work-out bagus
untuk benchmark offline, tetapi tidak menguji future recommendation. Dibutuhkan
event schema bertimestamp dan pipeline:

```text
impression -> recommendation -> click/save/purchase -> delayed label
```

Kemudian ukur Recall/NDCG bersama coverage, novelty, latency, CTR/conversion,
long-tail exposure, dan guardrail.

### P1 — production MLOps

Belum ada:

- model registry dan promotion state;
- scheduled retraining/backfill orchestration;
- drift/data-quality alert;
- shared metrics/tracing backend;
- canary/shadow deployment dan automated rollback;
- authentication, authorization, rate limiting, dan abuse protection;
- load/soak test serta SLO latency/availability.

### P2 — model dan product quality

- Explanations masih kategori sinyal, belum kontribusi feature/seed yang dapat
  diaudit per rekomendasi.
- Popularity fallback masih dapat memperkuat exposure bias.
- Demographic features belum dipakai atau diaudit fairness-nya.
- External cover availability bukan bagian dari model SLA.

## Definition of done berikutnya

Project layak disebut production candidate bila semua ini tersedia:

- benchmark protocol aktif pada cohort representatif dan repeated split;
- threshold promosi challenger vs champion yang pre-registered;
- timestamped event/feedback contract dan offline-online metric parity;
- registry, immutable signed artifact, promotion/rollback flow;
- monitoring data drift, recommendation quality, segment fairness, latency, error;
- canary/load test dan runbook insiden;
- API auth/rate limit sesuai exposure layanan.

## Perintah verifikasi

```bash
uv lock --check
uv run ruff check src tests app notebooks/01_end_to_end_book_recommender.py
uv run pytest --cov=book_recommender --cov=app --cov-report=term-missing
uv build

uv run book-rec tune-hybrid \
  --data-dir data/raw --sample-users 200 --k 50 \
  --output artifacts/evaluation/hybrid_weight_tuning.json

uv run book-rec evaluate-collab \
  --data-dir data/raw --sample-users 200 --k 50 \
  --output artifacts/evaluation/recommender_evaluation.json

uv run book-rec train-hybrid \
  --data-dir data/raw \
  --evaluation-artifact artifacts/evaluation/recommender_evaluation.json \
  --output models/hybrid.joblib

uv run book-rec artifact-info models/hybrid.joblib
```
