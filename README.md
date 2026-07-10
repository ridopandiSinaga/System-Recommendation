# Book Recommender System

Portfolio project Machine Learning Engineering untuk rekomendasi buku berbasis
Book-Crossing/Kaggle. Project ini sudah melampaui notebook prototype: ada data
contract, model content/collaborative/hybrid, evaluasi bebas leakage, artifact
lineage, CLI, API, Streamlit, test, CI, dan container.

Statusnya tetap **production-oriented offline system**, bukan layanan production
yang sudah terbukti oleh traffic nyata. Belum ada event timestamp, online feedback,
drift monitoring, canary, atau SLA operasional.

## Arsitektur

```text
Books/Ratings/Users CSV
        |
        v
cleaning + schema validation + convergent k-core
        |
        +--> interaction catalog ------> adjusted-cosine item CF
        |
        +--> content catalog ----------> TF-IDF metadata neighbors
                         \               /
                          hybrid ranking + popularity fallback
                                      |
                         verified joblib + manifest JSON
                                      |
                         Streamlit UI / FastAPI API
```

Komponen penting:

- `src/book_recommender/data.py`: cleaning, invariant, deterministic sampling,
  interaction catalog, dan cold content catalog.
- `content_based.py`: TF-IDF/cosine dengan deduplikasi canonical work.
- `collaborative.py`: adjusted-cosine item CF dengan positive-preference seed.
- `hybrid.py`: penggabungan collaborative, content profile, dan popularity.
- `evaluation.py` dan `cli.py`: split validation/test, metrik ranking, bootstrap,
  segment, diagnostics, dan artifact evaluasi.
- `artifacts.py` dan `service.py`: lineage, checksum, compatibility, dan serving
  invariant.
- `app/api.py` dan `app/streamlit_app.py`: adapter serving yang memuat satu bundle.

Audit teknis dan backlog tersisa ada di
[`docs/AUDIT_RECOMMENDER_SYSTEM.md`](docs/AUDIT_RECOMMENDER_SYSTEM.md).

## Setup

Project mendukung Python 3.10–3.12 dan memakai lockfile `uv`.

```bash
uv sync --frozen --dev --extra app --extra api
```

Verifikasi environment:

```bash
uv run book-rec --help
uv run pytest
uv run ruff check src tests app notebooks/01_end_to_end_book_recommender.py
```

`requirements.txt` adalah export production yang ter-pin dan ber-hash, bukan
sumber dependency utama:

```bash
uv export --frozen --no-dev --extra app --extra api \
  --format requirements-txt --output-file requirements.txt
```

## Dataset dan data contract

Dataset: [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

File berikut wajib berada di `data/raw/`:

- `Books.csv`
- `Ratings.csv`
- `Users.csv`

```bash
uv run kaggle datasets download \
  -d arashnic/book-recommendation-dataset \
  -p data/raw --unzip
uv run book-rec check-data --data-dir data/raw
uv run book-rec summary --data-dir data/raw
```

Kontrak preprocessing utama:

- ISBN dan user ID dinormalisasi serta divalidasi.
- Rating harus finite pada rentang 0–10; nilai `0` adalah implicit/no-rating dan
  tidak masuk training explicit-feedback.
- Duplikat `(user_id, isbn)` diagregasi dengan mean yang deterministik.
- Semua rating setelah cleaning wajib memiliki metadata buku.
- Filter minimum user/item dijalankan sampai konvergen.
- Batas `max_books`/`max_ratings` diterapkan deterministik lalu invariant dicek ulang.
- Interaction catalog dan content catalog dipisahkan. Content catalog dapat memuat
  buku cold-item tanpa rating; seluruh item interaksi selalu dipertahankan.

Detail sumber dan manifest checksum ada di [`docs/DATASET.md`](docs/DATASET.md).
Data mentah dan artifact hasil training tidak disimpan di git.

## Menjalankan rekomendasi

```bash
# Buku serupa; query dipertahankan walau catalog dibatasi.
uv run book-rec recommend-content \
  --data-dir data/raw \
  --title "Adventures of Huckleberry Finn"

# Personalized recommendation.
uv run book-rec recommend-collab --data-dir data/raw --user-id 5709
uv run book-rec recommend-hybrid --data-dir data/raw --user-id 5709
```

Semua output dideduplikasi pada canonical work, sehingga edisi paperback/hardcover
dari karya yang sama tidak memenuhi beberapa slot Top-N. Rating di bawah
`positive_rating_threshold` tetap tercatat sebagai feedback tetapi tidak dipakai
sebagai sinyal suka.

Unknown user atau user tanpa positive seed memakai popularity fallback pada mode
serving. Ablation `strict` di evaluator mematikan fallback agar kontribusi sinyal
personalized tidak tersamarkan.

## Evaluasi yang benar

Jalankan tuning hanya pada partition validation:

```bash
uv run book-rec tune-hybrid \
  --data-dir data/raw \
  --sample-users 200 \
  --k 50 \
  --output artifacts/evaluation/hybrid_weight_tuning.json
```

Setelah memilih konfigurasi, jalankan test partition yang tidak disentuh tuning:

```bash
uv run book-rec evaluate-collab \
  --data-dir data/raw \
  --sample-users 200 \
  --k 50 \
  --hybrid-collaborative-weight 0.40 \
  --hybrid-content-weight 0.45 \
  --hybrid-popularity-weight 0.15 \
  --output artifacts/evaluation/recommender_evaluation.json
```

Protokol aktif:

- user validation dan test disjoint;
- satu canonical work relevan ditahan dan seluruh edisinya dihapus dari history;
- filter dan pemilihan candidate dilakukan dari train saja;
- `conditional` memakai target yang tersedia dan user dengan positive seed;
- `end_to_end` memakai seluruh holdout, menjalankan cold-user fallback sungguhan,
  dan menghitung unavailable target sebagai miss;
- per-user row menyimpan status availability dan segment positive-work history;
- confidence interval bootstrap dipisahkan per scope;
- relevance, coverage, dan candidate count memakai canonical work.

Metrik: Precision/Recall/HitRate/MAP/MRR/NDCG@K, catalog coverage, novelty,
intra-list diversity, serta popularity share top 1/5/10%.

Angka dari protocol evaluasi lama berbasis ISBN atau hanya `evaluable_holdouts`
tidak comparable dan sengaja tidak dicantumkan di README. Regenerasi artifact
sebelum membuat klaim performa.

## Training dan artifact lineage

Bundle serving utama:

```bash
uv run book-rec train-hybrid \
  --data-dir data/raw \
  --evaluation-artifact artifacts/evaluation/recommender_evaluation.json \
  --output models/hybrid.joblib

uv run book-rec artifact-info models/hybrid.joblib
```

Attachment artifact evaluasi bersifat fail-closed. Protocol, partition, unit
relevance/split, source dataset fingerprint, algorithm version, preprocessing,
neighbor count, candidate-pool size, bobot, threshold, dan random seed harus
kompatibel dengan model yang dilatih.

Setiap `.joblib` memiliki sidecar `.joblib.manifest.json` berisi:

- SHA-256 artifact dan unique artifact ID;
- source/training dataset fingerprint, row count, dan schema;
- Python, project, serta dependency versions;
- Git SHA dan dirty-worktree flag;
- training config dan metrik evaluasi terverifikasi.

Checksum mendeteksi korupsi, bukan membuktikan authenticity. Joblib/pickle dapat
mengeksekusi kode: hanya muat artifact dari pipeline/storage terpercaya dan mount
artifact read-only saat serving.

## FastAPI

```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Environment variable:

```text
BOOK_RECOMMENDER_ARTIFACT=/absolute/path/models/hybrid.joblib
```

Endpoint:

- `GET /healthz`: process liveness.
- `GET /readyz`: readiness dan artifact identity; `503` bila model gagal dimuat.
- `GET /model-info`: lineage ringkas.
- `GET /metrics`: in-process request/error/latency counters.
- `GET /v1/recommendations/users/{user_id}?top_n=10`.
- `GET /v1/recommendations/similar?isbn=...&top_n=10` atau `title=...`.

Response error publik memakai stable error code; detail internal hanya ditulis ke
structured log. `x-request-id` divalidasi/dibuat oleh middleware.

## Docker

```bash
docker build -t book-recommender .
docker run --rm -p 8000:8000 \
  --mount type=bind,src="$(pwd)/models",dst=/opt/app/models,readonly \
  book-recommender
```

Container berjalan sebagai non-root user, membatasi thread BLAS, dan healthcheck
memakai `/readyz`. Directory yang di-mount harus memuat `hybrid.joblib` dan
sidecar manifest-nya.

## Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

UI load-only secara default dan memakai bundle yang sama dengan API. Untuk
eksperimen lokal saja, runtime training dapat diaktifkan:

```text
BOOK_RECOMMENDER_ALLOW_RUNTIME_TRAINING=1
```

Jika data belum ada, jalur development dapat mengambil Kaggle credential dari
Streamlit secrets, memvalidasi schema/isi di staging, membuat checksum manifest,
dan mem-publish dataset directory secara atomik. Jangan mengaktifkan jalur ini
sebagai pola production serving.

## Quality gates

CI menjalankan Python 3.10, 3.11, dan 3.12:

- frozen dependency/requirements check;
- Ruff;
- pytest dengan branch coverage minimal 75% untuk package dan adapters;
- wheel build serta import/CLI smoke dari directory di luar checkout;
- Docker build.

Verifikasi lokal ekuivalen:

```bash
uv lock --check
uv run ruff check src tests app notebooks/01_end_to_end_book_recommender.py
uv run pytest --cov=book_recommender --cov=app --cov-report=term-missing
uv build
```

## Batasan dan langkah berikutnya

- Dataset tidak memiliki timestamp interaksi; user-disjoint holdout bukan pengganti
  temporal backtest.
- Tuning masih grid search pada satu validation split. Tambahkan repeated split
  atau paired-bootstrap delta sebelum menganggap satu konfigurasi unggul stabil.
- Adjusted-cosine item CF dan TF-IDF adalah baseline interpretable, bukan learned
  ranker/state-of-the-art recommender.
- Cold-user sudah punya fallback dan cold-item masuk content catalog, tetapi item
  baru tetap sulit mendapat exposure pada jalur collaborative.
- Belum ada online feedback, business KPI, fairness slices, drift alert, model
  registry, canary, rollback automation, rate limiting, atau shared metrics backend.
- Cover hanya diizinkan dari host image dataset yang dikenal dan tetap bergantung
  pada availability pihak ketiga.

Notebook naratif tersedia di
[`notebooks/01_end_to_end_book_recommender.ipynb`](notebooks/01_end_to_end_book_recommender.ipynb).
