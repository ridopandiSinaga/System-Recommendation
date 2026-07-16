# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# ruff: noqa: E402

# %% [markdown]
# # Dari Buku Populer ke Hybrid Recommender yang Bisa Dipertanggungjawabkan
#
# Bayangkan pengguna baru selesai membaca sebuah buku dan bertanya, “selanjutnya
# baca apa?” Secara bisnis, sistem perlu mengisi rak rekomendasi Top-N dengan buku
# yang cukup relevan untuk mendorong klik, simpan, atau mulai membaca—tanpa terus
# mengulang judul populer dan tanpa menampilkan beberapa edisi dari karya yang sama.
#
# Notebook ini mengikuti perjalanan keputusan tersebut:
#
# 1. mulai dari popularity baseline yang selalu tersedia;
# 2. mencoba collaborative filtering untuk personalisasi;
# 3. menambahkan content-based ketika pola co-rating lemah;
# 4. menggabungkan sinyal tersebut dalam hybrid recommender;
# 5. memisahkan demo, diagnostic validation, dan benchmark formal;
# 6. memverifikasi bahwa model eksperimen terhubung ke artifact serving.
#
# Kode produksi tetap berada di `src/book_recommender`. Notebook berfungsi sebagai
# narasi, visualisasi evidence, dan decision record—bukan salinan pipeline kedua.

# %% [markdown]
# ## 0. Apa yang Bisa dan Tidak Bisa Dibuktikan
#
# Dataset Book-Crossing menyediakan rating eksplisit, tetapi tidak menyediakan
# impression, click, save, purchase, inventory, maupun timestamp interaksi. Karena
# itu, ranking metric di sini adalah **offline proxy**. Kita dapat menguji apakah
# model menemukan kembali karya yang ditahan dari histori pengguna, tetapi belum
# dapat mengklaim peningkatan penjualan atau engagement tanpa eksperimen online.
#
# Target keputusan offline:
#
# - relevansi: HitRate, MAP/MRR, dan NDCG@K;
# - reach: candidate dan personalization availability;
# - discovery: catalog coverage, novelty, dan diversity;
# - risiko: popularity concentration, cold-start, dan ketidakpastian antar-split.

# %%
from __future__ import annotations

import os

# Batasi native thread sebelum NumPy/sklearn diimpor. Ini membuat notebook lebih
# stabil pada laptop dan runner CI dengan resource kecil.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import gc
import json
from pathlib import Path
from time import perf_counter

import matplotlib

try:
    get_ipython
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from threadpoolctl import threadpool_limits

from book_recommender.artifacts import fingerprint_dataset
from book_recommender.cli import read_benchmark_metrics
from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender, book_work_key
from book_recommender.data import (
    BookDataset,
    clean_dataset as clean_book_dataset,
    dataset_summary,
    load_raw_data,
    prepare_interaction_dataset,
    select_content_catalog,
    validate_data_dir,
)
from book_recommender.evaluation import (
    prepare_validation_test_evaluation,
    ranking_metrics_at_k,
)
from book_recommender.hybrid import HybridRecommender
from book_recommender.service import RecommenderService

try:
    from IPython.display import display
except ImportError:
    display = print

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 90)


def find_project_root(start: Path) -> Path:
    """Find the checkout root without assuming where Jupyter was launched."""

    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root containing pyproject.toml.")


def configured_path(name: str, default: Path, project_root: Path) -> Path:
    value = Path(os.environ.get(name, str(default)))
    return value if value.is_absolute() else project_root / value


def display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def work_key_map(books: pd.DataFrame) -> dict[str, str]:
    return {
        str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
        for row in books.itertuples(index=False)
    }


def recommendation_work_keys(
    recommendations: pd.DataFrame,
    isbn_to_work: dict[str, str],
) -> list[str]:
    return list(
        dict.fromkeys(
            isbn_to_work[str(isbn)]
            for isbn in recommendations["isbn"].astype(str)
            if str(isbn) in isbn_to_work
        )
    )


def weight_label(record: dict[str, object]) -> str:
    return (
        f"C={float(record['collaborative_weight']):.2f}, "
        f"T={float(record['content_weight']):.2f}, "
        f"P={float(record['popularity_weight']):.2f}"
    )


# %% [markdown]
# ## 1. Konfigurasi dan Reproducibility
#
# Default notebook diselaraskan dengan bundle serving: 1.000 interaction books,
# 3.000 content books, 50 neighbors, rating positif minimal 8, dan Top-10. Nilai
# dapat dioverride melalui environment variable untuk smoke test CI tanpa membuat
# cabang logika khusus di notebook.

# %%
PROJECT_ROOT = find_project_root(Path.cwd())
DATA_DIR = configured_path(
    "BOOK_RECOMMENDER_DATA_DIR",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT,
)
MODEL_ARTIFACT = configured_path(
    "BOOK_RECOMMENDER_MODEL_ARTIFACT",
    PROJECT_ROOT / "models" / "hybrid.joblib",
    PROJECT_ROOT,
)
BENCHMARK_ARTIFACT = configured_path(
    "BOOK_RECOMMENDER_BENCHMARK_ARTIFACT",
    PROJECT_ROOT / "artifacts" / "evaluation" / "repeated_split_hybrid_benchmark.json",
    PROJECT_ROOT,
)

MAX_BOOKS = int(os.environ.get("BOOK_RECOMMENDER_NOTEBOOK_MAX_BOOKS", "1000"))
MAX_CONTENT_BOOKS = int(
    os.environ.get("BOOK_RECOMMENDER_NOTEBOOK_MAX_CONTENT_BOOKS", "3000")
)
MIN_USER_RATINGS = 2
MIN_BOOK_RATINGS = 2
N_NEIGHBORS = 50
POSITIVE_RATING_THRESHOLD = 8.0
TOP_K = 10
EVALUATION_SAMPLE_USERS = int(
    os.environ.get("BOOK_RECOMMENDER_NOTEBOOK_EVAL_USERS", "40")
)
RANDOM_STATE = 42

validate_data_dir(DATA_DIR)
configuration = pd.DataFrame(
    {
        "parameter": [
            "data_dir",
            "max_interaction_books",
            "max_content_books",
            "neighbors",
            "positive_rating_threshold",
            "top_k",
            "diagnostic_sample_users",
            "random_state",
        ],
        "value": [
            display_path(DATA_DIR, PROJECT_ROOT),
            MAX_BOOKS,
            MAX_CONTENT_BOOKS,
            N_NEIGHBORS,
            POSITIVE_RATING_THRESHOLD,
            TOP_K,
            EVALUATION_SAMPLE_USERS,
            RANDOM_STATE,
        ],
    }
)
display(configuration)


# %% [markdown]
# ## 2. Data Understanding: Sinyalnya Memang Sulit
#
# Dataset terdiri dari katalog buku, rating user–ISBN, dan metadata user. ISBN
# merepresentasikan edisi, bukan selalu karya unik. Nanti, unit output dan unit
# relevance evaluasi akan menggunakan canonical work berbasis judul–penulis.

# %%
raw = load_raw_data(DATA_DIR)
raw_shapes = pd.DataFrame(
    {
        "dataset": ["Books", "Ratings", "Users"],
        "rows": [len(raw.books), len(raw.ratings), len(raw.users)],
        "columns": [raw.books.shape[1], raw.ratings.shape[1], raw.users.shape[1]],
    }
)
display(raw_shapes)
display(raw.books.head(3))
display(raw.ratings.head(3))
display(raw.users.head(3))


# %% [markdown]
# ### Missing Value dan Semantik Rating 0
#
# Rating `0` pada Book-Crossing berarti tidak ada rating eksplisit. Nilai tersebut
# tidak boleh diperlakukan sebagai ketidaksukaan, sebab itu akan mengubah arti
# label. Kolom umur memiliki banyak missing value, tetapi tidak dipakai model.

# %%
missing_summary = pd.concat(
    {
        "books": raw.books.isna().sum(),
        "ratings": raw.ratings.isna().sum(),
        "users": raw.users.isna().sum(),
    },
    axis=1,
).fillna(0).astype("int64")
display(missing_summary)

rating_counts = raw.ratings["Book-Rating"].value_counts().sort_index()
implicit_rows = int((raw.ratings["Book-Rating"] == 0).sum())
explicit_raw = raw.ratings[raw.ratings["Book-Rating"] > 0].copy()

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(x=rating_counts.index.astype(str), y=rating_counts.values, color="#4C78A8", ax=ax)
ax.set(title="Distribusi Rating: Nilai 0 Mendominasi", xlabel="Rating", ylabel="Interaksi")
plt.tight_layout()
plt.show()

display(
    pd.DataFrame(
        {
            "metric": ["implicit/no-rating rows", "explicit rating rows", "implicit share"],
            "value": [
                implicit_rows,
                len(explicit_raw),
                implicit_rows / len(raw.ratings),
            ],
        }
    )
)


# %% [markdown]
# ### Sparsity dan Long Tail
#
# Collaborative filtering hanya melihat interaksi yang tersedia. Jika sebagian
# besar user dan buku hanya memiliki sedikit rating, masalah utama bukan sekadar
# algoritme: banyak target bahkan tidak memiliki cukup sinyal untuk dipelajari.

# %%
raw_user_counts = explicit_raw["User-ID"].value_counts()
raw_book_counts = explicit_raw["ISBN"].astype(str).value_counts()
matrix_density = len(explicit_raw) / (
    explicit_raw["User-ID"].nunique() * explicit_raw["ISBN"].nunique()
)

ranked_book_counts = raw_book_counts.sort_values(ascending=False).reset_index(drop=True)
cumulative_share = ranked_book_counts.cumsum() / ranked_book_counts.sum()
top_share_rows = []
for fraction in (0.01, 0.05, 0.10):
    cutoff = max(1, int(np.ceil(len(ranked_book_counts) * fraction)))
    top_share_rows.append(
        {
            "top_catalog_fraction": fraction,
            "interaction_share": float(ranked_book_counts.head(cutoff).sum() / ranked_book_counts.sum()),
        }
    )

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(raw_user_counts.clip(upper=raw_user_counts.quantile(0.99)), bins=40, color="#4C78A8")
axes[0].set(title="Rating per User (hingga P99)", xlabel="Jumlah rating", ylabel="User")
axes[1].loglog(np.arange(1, len(ranked_book_counts) + 1), ranked_book_counts, color="#F58518")
axes[1].set(title="Long Tail Buku", xlabel="Rank popularitas", ylabel="Jumlah rating")
axes[2].plot(
    np.arange(1, len(cumulative_share) + 1) / len(cumulative_share),
    cumulative_share,
    color="#54A24B",
)
axes[2].axhline(0.80, color="black", linestyle="--", linewidth=1)
axes[2].set(
    title="Kumulatif Interaksi",
    xlabel="Proporsi katalog",
    ylabel="Proporsi interaksi",
)
plt.tight_layout()
plt.show()

display(
    pd.DataFrame(
        {
            "metric": ["explicit matrix density", "unique explicit users", "rated ISBN"],
            "value": [matrix_density, len(raw_user_counts), len(raw_book_counts)],
        }
    )
)
display(pd.DataFrame(top_share_rows))


# %% [markdown]
# ## 3. Data Preparation dan Kontrak
#
# Pipeline membersihkan identifier, memvalidasi rating, mengagregasi pasangan
# user–ISBN duplikat secara deterministik, menghapus orphan rating, menjalankan
# k-core sampai konvergen, lalu memisahkan interaction catalog dari content catalog.
# Buku tanpa rating masih dapat dipertahankan untuk pencarian berbasis metadata.

# %%
raw_rating_rows = len(raw.ratings)
raw_explicit_rows = len(explicit_raw)
cleaned = clean_book_dataset(raw)

del raw, explicit_raw
gc.collect()

interaction_dataset = prepare_interaction_dataset(
    books=cleaned.books,
    ratings=cleaned.ratings,
    users=cleaned.users,
    min_user_ratings=MIN_USER_RATINGS,
    min_book_ratings=MIN_BOOK_RATINGS,
    max_books=MAX_BOOKS,
    random_state=RANDOM_STATE,
)
content_books = select_content_catalog(
    cleaned.books,
    required_books=interaction_dataset.books,
    max_content_books=MAX_CONTENT_BOOKS,
)
dataset = BookDataset(
    books=content_books,
    ratings=interaction_dataset.ratings,
    users=cleaned.users,
)

funnel = pd.DataFrame(
    {
        "stage": [
            "raw rating rows",
            "explicit ratings",
            "cleaned + deduplicated",
            "interaction core",
        ],
        "rows": [
            raw_rating_rows,
            raw_explicit_rows,
            len(cleaned.ratings),
            len(dataset.ratings),
        ],
    }
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=funnel, x="rows", y="stage", color="#4C78A8", ax=ax)
ax.set(title="Data Preparation Funnel", xlabel="Jumlah baris rating", ylabel="")
for container in ax.containers:
    ax.bar_label(container, fmt="{:,.0f}", padding=4)
plt.tight_layout()
plt.show()

display(funnel)
display(pd.Series(dataset_summary(dataset), name="value").to_frame())


# %%
user_degree = dataset.ratings.groupby("user_id").size()
book_degree = dataset.ratings.groupby("isbn").size()
orphan_ratings = ~dataset.ratings["isbn"].isin(dataset.books["isbn"])
duplicate_pairs = dataset.ratings.duplicated(["user_id", "isbn"])
contract_checks = pd.DataFrame(
    {
        "contract": [
            "rating finite and in [1, 10]",
            "no duplicate user–ISBN",
            "every rating has book metadata",
            f"minimum user degree >= {MIN_USER_RATINGS}",
            f"minimum item degree >= {MIN_BOOK_RATINGS}",
            "interaction catalog is contained in content catalog",
        ],
        "passed": [
            bool(
                np.isfinite(dataset.ratings["book_rating"]).all()
                and dataset.ratings["book_rating"].between(1, 10).all()
            ),
            not bool(duplicate_pairs.any()),
            not bool(orphan_ratings.any()),
            bool(user_degree.min() >= MIN_USER_RATINGS),
            bool(book_degree.min() >= MIN_BOOK_RATINGS),
            set(interaction_dataset.books["isbn"]).issubset(set(dataset.books["isbn"])),
        ],
    }
)
display(contract_checks)
assert contract_checks["passed"].all(), "Prepared dataset violates a documented contract."


# %% [markdown]
# ### ISBN Bukan Karya
#
# Satu karya dapat memiliki beberapa ISBN karena edisi atau format berbeda. Tanpa
# deduplikasi work-level, Top-N dapat dipenuhi oleh paperback dan hardcover dari
# buku yang sama, sementara holdout ISBN lain bocor sebagai kandidat ekuivalen.

# %%
catalog_with_work = dataset.books[["isbn", "book_title", "book_author"]].copy()
catalog_with_work["work_key"] = [
    book_work_key(row.book_title, row.book_author, row.isbn)
    for row in catalog_with_work.itertuples(index=False)
]
edition_counts = catalog_with_work.groupby("work_key").size().sort_values(ascending=False)
multi_edition_keys = edition_counts[edition_counts > 1].head(5).index

display(
    pd.DataFrame(
        {
            "unit": ["ISBN/edition", "canonical work", "multi-edition works"],
            "count": [
                len(catalog_with_work),
                catalog_with_work["work_key"].nunique(),
                int((edition_counts > 1).sum()),
            ],
        }
    )
)
display(
    catalog_with_work[catalog_with_work["work_key"].isin(multi_edition_keys)]
    .sort_values(["work_key", "isbn"])
    .head(15)
)


# %% [markdown]
# ## 4. Baseline Pertama: Popularity
#
# Sebelum personalisasi, sistem perlu baseline yang murah dan selalu tersedia.
# Skor popularity menggunakan `mean_rating × log1p(rating_count)`, sehingga sebuah
# buku dengan satu rating 10 tidak otomatis mengalahkan buku yang konsisten disukai
# banyak pengguna.

# %%
popularity = (
    dataset.ratings.groupby("isbn")
    .agg(mean_rating=("book_rating", "mean"), rating_count=("book_rating", "size"))
    .reset_index()
)
popularity["score"] = popularity["mean_rating"] * np.log1p(popularity["rating_count"])
popularity = (
    popularity.merge(dataset.books, on="isbn", how="left")
    .sort_values(["score", "rating_count", "isbn"], ascending=[False, False, True])
    .reset_index(drop=True)
)
display(
    popularity[
        ["score", "mean_rating", "rating_count", "isbn", "book_title", "book_author"]
    ].head(TOP_K)
)


# %% [markdown]
# Popularity tidak pernah mengalami cold-user dan latency-nya rendah, tetapi tidak
# personal. Ia juga berisiko memperkuat pola *rich get richer*. Karena itu popularity
# cocok sebagai fallback dan baseline pembanding, bukan otomatis jawaban akhir.

# %% [markdown]
# ## 5. Collaborative Filtering: Belajar dari Pola Bersama
#
# Item-based collaborative filtering membangun sparse matrix buku × user. Rating
# dikoreksi terhadap rata-rata user yang diregularisasi, lalu model mencari item
# dengan pola rating serupa. Hanya rating `>= 8` yang menjadi positive seed.

# %%
collab_model = ItemBasedCollaborativeRecommender(
    n_neighbors=N_NEIGHBORS,
    positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
).fit(dataset.books, dataset.ratings)

isbn_to_work = work_key_map(dataset.books)
positive_history = dataset.ratings[
    dataset.ratings["book_rating"] >= POSITIVE_RATING_THRESHOLD
].copy()
positive_history["work_key"] = positive_history["isbn"].astype(str).map(isbn_to_work)
positive_work_counts = positive_history.groupby("user_id")["work_key"].nunique()

# Pilih user secara deterministik dengan histori cukup kaya namun tetap mudah dibaca.
moderate_users = positive_work_counts[(positive_work_counts >= 3) & (positive_work_counts <= 20)]
candidate_pool = moderate_users if not moderate_users.empty else positive_work_counts
candidate_user_id = int(
    candidate_pool.rename("positive_works")
    .reset_index()
    .sort_values(["positive_works", "user_id"], ascending=[False, True])
    .iloc[0]["user_id"]
)

user_history = (
    dataset.ratings[dataset.ratings["user_id"] == candidate_user_id]
    .sort_values(["book_rating", "isbn"], ascending=[False, True])
    .merge(dataset.books, on="isbn", how="left")
)
display(user_history[["isbn", "book_title", "book_author", "book_rating"]].head(15))


# %%
strict_collaborative = collab_model.score_for_user(
    candidate_user_id,
    seed_items=5,
    candidate_pool_size=100,
)
served_collaborative = collab_model.recommend_for_user(candidate_user_id, top_n=TOP_K)

display(
    pd.DataFrame(
        {
            "user_id": [candidate_user_id],
            "positive_canonical_works": [int(positive_work_counts[candidate_user_id])],
            "strict_collaborative_candidates": [len(strict_collaborative)],
            "served_recommendations": [len(served_collaborative)],
            "fallback_or_filler_possible": [len(strict_collaborative) < TOP_K],
        }
    )
)
display(
    served_collaborative[
        ["score", "isbn", "book_title", "book_author", "publisher"]
    ].head(TOP_K)
)


# %% [markdown]
# Collaborative filtering memberi personalisasi ketika positive seed dan co-rating
# cukup. Namun `recommend_for_user` sengaja dapat memakai popularity fallback. Karena
# itu, jumlah strict candidate perlu ditampilkan agar fallback tidak keliru disebut
# sebagai keberhasilan personalisasi.

# %% [markdown]
# ## 6. Content-Based: Ketika Co-rating Tidak Cukup
#
# Metadata tetap tersedia meskipun sebuah item belum memiliki banyak interaksi.
# Model content mengubah `title + author + publisher` menjadi TF-IDF unigram/bigram,
# lalu mencari cosine-neighbor dan mendeduplikasi canonical work.

# %%
content_model = ContentBasedRecommender().fit(dataset.books)
query_title = "Adventures of Huckleberry Finn"
if query_title not in set(dataset.books["book_title"]):
    query_title = str(dataset.books.sort_values("isbn").iloc[0]["book_title"])

content_recommendations = content_model.recommend_similar(
    title=query_title,
    top_n=TOP_K,
)
display(
    content_recommendations[
        ["score", "isbn", "book_title", "book_author", "publisher"]
    ]
)


# %% [markdown]
# Content-based kuat untuk “buku mirip” dan cold item, tetapi metadata proyek ini
# masih dangkal. Tanpa genre, sinopsis, tema, atau embedding semantik, kemiripan
# publisher dapat terlalu dominan. Keterbatasan itu yang memotivasi penggabungan
# sinyal, bukan asumsi bahwa content model selalu lebih baik.

# %% [markdown]
# ## 7. Hybrid Muncul sebagai Konsekuensi
#
# Hybrid menggabungkan collaborative, content profile, dan popularity setelah
# masing-masing dinormalisasi. Bobot dasar adalah 0,40 / 0,45 / 0,15, lalu dapat
# bergeser berdasarkan jumlah positive canonical works milik user.

# %%
hybrid_model = HybridRecommender(
    n_neighbors=N_NEIGHBORS,
    collaborative_weight=0.40,
    content_weight=0.45,
    popularity_weight=0.15,
    dynamic_weights=True,
    positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
).fit(dataset.books, dataset.ratings)

hybrid_recommendations = hybrid_model.recommend_for_user(
    candidate_user_id,
    top_n=TOP_K,
)
display(
    hybrid_recommendations[
        [
            "score",
            "collaborative_score",
            "content_score",
            "popularity_score",
            "source",
            "reason",
            "isbn",
            "book_title",
            "book_author",
        ]
    ]
)


# %% [markdown]
# ### Failure Mode, Bukan Hanya Happy Path
#
# Tiga skenario berikut memperlihatkan routing sinyal. Cold user seharusnya jatuh
# ke popularity; sparse user lebih mengandalkan metadata; rich-history user lebih
# mungkin memperoleh collaborative candidate. Ini adalah diagnostic behavior,
# bukan ranking benchmark.

# %%
sparse_user_id = int(
    positive_work_counts.rename("positive_works")
    .reset_index()
    .sort_values(["positive_works", "user_id"], ascending=[True, True])
    .iloc[0]["user_id"]
)
cold_user_id = int(dataset.ratings["user_id"].max()) + 1_000_000
scenario_rows = []
for scenario, user_id in (
    ("cold", cold_user_id),
    ("sparse-positive", sparse_user_id),
    ("richer-positive", candidate_user_id),
):
    recommendations = hybrid_model.recommend_for_user(user_id, top_n=TOP_K)
    source_counts = recommendations["source"].value_counts()
    scenario_rows.append(
        {
            "scenario": scenario,
            "user_id": user_id,
            "history_rows": int((dataset.ratings["user_id"] == user_id).sum()),
            "positive_works": int(positive_work_counts.get(user_id, 0)),
            "recommendations": len(recommendations),
            "dominant_source": source_counts.index[0] if not source_counts.empty else "empty",
            "popularity_share": float(source_counts.get("popularity", 0) / max(1, len(recommendations))),
        }
    )
display(pd.DataFrame(scenario_rows))


# %% [markdown]
# ## 8. Evaluasi Ranking: Test Tidak Boleh Menjadi Tempat Bereksperimen
#
# Diagnostic notebook menggunakan split kecil untuk menjelaskan mekanisme. Split
# dilakukan pada canonical work sebelum filtering; semua edisi target dihapus dari
# histori; candidate catalog dibentuk dari train saja; user validation dan test
# dipisahkan. Cell berikut hanya membaca **validation holdout**. Test holdout tetap
# tidak disentuh dan bukan sumber keputusan bobot di notebook.

# %%
evaluation_data = prepare_validation_test_evaluation(
    cleaned,
    sample_users=EVALUATION_SAMPLE_USERS,
    validation_fraction=0.5,
    min_user_ratings=MIN_USER_RATINGS,
    min_book_ratings=MIN_BOOK_RATINGS,
    max_books=MAX_BOOKS,
    max_content_books=MAX_CONTENT_BOOKS,
    random_state=RANDOM_STATE,
)
train_dataset = evaluation_data.train_dataset
validation_holdouts = evaluation_data.validation_holdout_ratings.copy()

eval_isbn_to_work = work_key_map(train_dataset.books)
eval_catalog_works = set(eval_isbn_to_work.values())
eval_history_users = set(train_dataset.ratings["user_id"].astype(int))
eval_positive_users = set(
    train_dataset.ratings.loc[
        train_dataset.ratings["book_rating"] >= POSITIVE_RATING_THRESHOLD,
        "user_id",
    ].astype(int)
)

validation_holdouts["candidate_available"] = validation_holdouts["work_key"].isin(
    eval_catalog_works
)
validation_holdouts["history_available"] = validation_holdouts["user_id"].isin(
    eval_history_users
)
validation_holdouts["positive_seed_available"] = validation_holdouts["user_id"].isin(
    eval_positive_users
)
validation_holdouts["personalization_evaluable"] = (
    validation_holdouts["candidate_available"]
    & validation_holdouts["history_available"]
    & validation_holdouts["positive_seed_available"]
)

availability_checks = pd.DataFrame(
    {
        "check": [
            "sampled validation",
            "target in catalog",
            "user history survives",
            "positive seed survives",
            "personalization evaluable",
        ],
        "users": [
            len(validation_holdouts),
            int(validation_holdouts["candidate_available"].sum()),
            int(validation_holdouts["history_available"].sum()),
            int(validation_holdouts["positive_seed_available"].sum()),
            int(validation_holdouts["personalization_evaluable"].sum()),
        ],
    }
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=availability_checks, x="users", y="check", color="#E45756", ax=ax)
ax.set(title="Validation Availability Checks", xlabel="Holdout user", ylabel="")
for container in ax.containers:
    ax.bar_label(container, padding=4)
plt.tight_layout()
plt.show()

display(availability_checks)
display(
    pd.DataFrame(
        {
            "partition": ["validation diagnostic", "reserved test (untouched)"],
            "holdouts": [
                len(evaluation_data.validation_holdout_ratings),
                len(evaluation_data.test_holdout_ratings),
            ],
        }
    )
)


# %%
eval_hybrid = HybridRecommender(
    n_neighbors=N_NEIGHBORS,
    collaborative_weight=0.40,
    content_weight=0.45,
    popularity_weight=0.15,
    dynamic_weights=True,
    positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
).fit(train_dataset.books, train_dataset.ratings)
eval_collaborative = eval_hybrid.collaborative_model

validation_metric_rows = []
for row in validation_holdouts.itertuples(index=False):
    user_id = int(row.user_id)
    relevant = [str(row.work_key)]
    train_history = train_dataset.ratings[train_dataset.ratings["user_id"] == user_id]
    recommendations_by_model = {
        "hybrid": recommendation_work_keys(
            eval_hybrid.recommend_for_user(user_id, top_n=TOP_K),
            eval_isbn_to_work,
        ),
        "collaborative + fallback": recommendation_work_keys(
            eval_collaborative.recommend_for_user(user_id, top_n=TOP_K),
            eval_isbn_to_work,
        ),
        "popularity": recommendation_work_keys(
            eval_collaborative.recommend_popular(
                top_n=TOP_K,
                exclude_isbns=set(train_history["isbn"].astype(str)),
            ),
            eval_isbn_to_work,
        ),
    }
    for model_name, recommended in recommendations_by_model.items():
        validation_metric_rows.append(
            {
                "model": model_name,
                "user_id": user_id,
                "candidate_available": bool(row.candidate_available),
                "positive_seed_available": bool(row.positive_seed_available),
                **ranking_metrics_at_k(recommended, relevant, TOP_K),
            }
        )

validation_metrics = pd.DataFrame(validation_metric_rows)
metric_columns = [
    f"precision_at_{TOP_K}",
    f"recall_at_{TOP_K}",
    f"hit_rate_at_{TOP_K}",
    f"map_at_{TOP_K}",
    f"mrr_at_{TOP_K}",
    f"ndcg_at_{TOP_K}",
]
end_to_end_validation = validation_metrics.groupby("model")[metric_columns].mean()
conditional_validation_rows = validation_metrics[
    validation_metrics["candidate_available"]
    & validation_metrics["positive_seed_available"]
]
conditional_validation = conditional_validation_rows.groupby("model")[metric_columns].mean()

validation_summary = end_to_end_validation.reset_index()[
    ["model", f"hit_rate_at_{TOP_K}", f"ndcg_at_{TOP_K}"]
].melt(id_vars="model", var_name="metric", value_name="value")
if bool((validation_summary["value"] > 0).any()):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=validation_summary, x="model", y="value", hue="metric", ax=ax)
    ax.set(
        title=f"Validation Diagnostic Only (n={len(validation_holdouts)})",
        xlabel="",
        ylabel="Metric",
    )
    ax.tick_params(axis="x", rotation=12)
    plt.tight_layout()
    plt.show()
else:
    print(
        "No Top-K hits occurred in this small validation diagnostic. "
        "Use the repeated benchmark below for model-selection evidence."
    )

display(
    {
        "end_to_end": end_to_end_validation,
        "conditional": conditional_validation,
        "conditional_users": int(conditional_validation_rows["user_id"].nunique()),
    }
)


# %% [markdown]
# Diagnostic di atas membuktikan implementasi split dan metric dapat dijalankan,
# bukan bahwa satu model menang. Sampelnya kecil, hanya satu split, dan tidak memiliki
# confidence interval. Keputusan bobot harus datang dari repeated validation; fixed
# test baru boleh dibuka setelah keputusan dibekukan.

# %% [markdown]
# ## 9. Evidence dari Bundle Serving: Baseline dan Trade-off
#
# Manifest bundle tersimpan di repository bersama model. Selain checksum dan lineage,
# manifest membawa ringkasan evaluasi yang digunakan saat artifact dibangun. Evidence
# ini masih pilot; grafik berikut menunjukkan apa yang diukur, bukan klaim uplift.

# %%
manifest_path = MODEL_ARTIFACT.with_name(f"{MODEL_ARTIFACT.name}.manifest.json")
manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
deployment_evidence = manifest_payload.get("metrics", {})
deployment_availability = deployment_evidence.get("availability", {})

evidence_strength = pd.DataFrame(
    {
        "metric": [
            "sampled holdouts",
            "target available",
            "history evaluable",
            "personalization evaluable",
            "bootstrap resamples",
        ],
        "value": [
            deployment_availability.get("sampled_holdouts", 0),
            deployment_availability.get("candidate_available_holdouts", 0),
            deployment_availability.get("history_evaluable_holdouts", 0),
            deployment_availability.get("personalization_evaluable_holdouts", 0),
            next(
                (
                    interval.get("n_resamples", 0)
                    for model_intervals in deployment_evidence.get(
                        "confidence_intervals",
                        {},
                    ).values()
                    for scope_intervals in model_intervals.values()
                    for interval in scope_intervals.values()
                ),
                0,
            ),
        ],
    }
)
display(evidence_strength)
print("Evidence label: PILOT — useful for pipeline verification, not a performance claim.")


# %%
pilot_summary = pd.DataFrame(deployment_evidence.get("summary", {}).get("end_to_end", []))
if not pilot_summary.empty:
    pilot_metric_columns = [
        column
        for column in (f"hit_rate_at_{TOP_K}", f"ndcg_at_{TOP_K}")
        if column in pilot_summary.columns
    ]
    pilot_plot = pilot_summary.melt(
        id_vars=["model", "sample_size"],
        value_vars=pilot_metric_columns,
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=pilot_plot, x="model", y="value", hue="metric", ax=ax)
    ax.set(title="Deployment Pilot: End-to-End Ranking", xlabel="", ylabel="Metric")
    ax.tick_params(axis="x", rotation=18)
    plt.tight_layout()
    plt.show()
    display(pilot_summary[["model", "sample_size", *pilot_metric_columns]])


# %%
diagnostics = pd.DataFrame(deployment_evidence.get("diagnostics", {})).T
if not diagnostics.empty:
    rate_columns = [
        column
        for column in (
            f"catalog_coverage_at_{TOP_K}",
            f"intra_list_diversity_at_{TOP_K}",
            f"top_10pct_popularity_share_at_{TOP_K}",
        )
        if column in diagnostics.columns
    ]
    novelty_column = f"novelty_at_{TOP_K}"
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    diagnostics[rate_columns].plot(kind="bar", ax=axes[0])
    axes[0].set(title="Coverage, Diversity, dan Popularity Share", xlabel="", ylabel="Rate")
    axes[0].tick_params(axis="x", rotation=18)
    if novelty_column in diagnostics.columns:
        diagnostics[[novelty_column]].plot(kind="bar", color="#54A24B", ax=axes[1])
    axes[1].set(title="Novelty (Skala Terpisah)", xlabel="", ylabel="Self-information")
    axes[1].tick_params(axis="x", rotation=18)
    plt.tight_layout()
    plt.show()
    display(diagnostics)


# %% [markdown]
# ## 10. Benchmark Formal: Repeated Validation lalu Fixed Test
#
# CLI benchmark menjalankan alur berikut:
#
# ```text
# fixed test disisihkan sekali
#          ↓
# repeated validation pada development users
#          ↓
# complete canonical-work holdout + train-only filtering
#          ↓
# paired hierarchical bootstrap + selection gate
#          ↓
# baseline dipertahankan / challenger dipilih
#          ↓
# fixed test dibuka sekali untuk konfigurasi yang sudah beku
# ```
#
# Artifact penuh berada di `artifacts/evaluation/` dan sengaja tidak masuk Git karena
# memuat per-user rows. Notebook memvalidasi schema artifact sebelum membaca field
# visual. Jika artifact belum tersedia, bagian ini dilewati tanpa merusak Run All.

# %%
benchmark_candidates = [
    BENCHMARK_ARTIFACT,
    PROJECT_ROOT
    / "artifacts"
    / "verification"
    / "repeated_split_hybrid_benchmark_schema_v2_smoke.json",
]
benchmark_path = next((path for path in benchmark_candidates if path.exists()), None)
benchmark_payload = None

if benchmark_path is None:
    print(
        "Formal benchmark artifact is not available. Run `uv run book-rec benchmark-hybrid` "
        "to populate this section."
    )
else:
    verified_benchmark = read_benchmark_metrics(benchmark_path)
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    parameters = benchmark_payload["parameters"]
    is_portfolio_scale = (
        int(parameters["validation_users"]) >= 200
        and int(parameters["test_users"]) >= 200
        and int(parameters["repetitions"]) >= 5
        and int(parameters["bootstrap_resamples"]) >= 2_000
    )
    display(
        pd.DataFrame(
            {
                "field": [
                    "artifact",
                    "protocol",
                    "validation users / repeat",
                    "fixed-test users",
                    "repetitions",
                    "bootstrap resamples",
                    "evidence label",
                ],
                "value": [
                    display_path(benchmark_path, PROJECT_ROOT),
                    benchmark_payload["protocol"],
                    parameters["validation_users"],
                    parameters["test_users"],
                    parameters["repetitions"],
                    parameters["bootstrap_resamples"],
                    "PORTFOLIO-SCALE" if is_portfolio_scale else "SMOKE/PILOT",
                ],
            }
        )
    )


# %%
if benchmark_payload is not None:
    validation_records = []
    for configuration_record in benchmark_payload["validation"]["aggregate_by_config"]:
        label = weight_label(configuration_record)
        for split in configuration_record["per_split"]:
            validation_records.append(
                {
                    "split_id": split["split_id"],
                    "configuration": label,
                    "ndcg": split["end_to_end_metrics"][f"ndcg_at_{TOP_K}"],
                }
            )
    validation_by_split = pd.DataFrame(validation_records)
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.lineplot(
        data=validation_by_split,
        x="split_id",
        y="ndcg",
        hue="configuration",
        marker="o",
        ax=ax,
    )
    ax.set(title="NDCG per Repeated Validation Split", xlabel="", ylabel=f"NDCG@{TOP_K}")
    plt.tight_layout()
    plt.show()
    display(validation_by_split)


# %%
if benchmark_payload is not None:
    selection = benchmark_payload["selection"]
    selection_rows = []
    paired = benchmark_payload["validation"]["paired_deltas_vs_baseline"]
    for config_id, evidence in paired.items():
        interval = evidence["end_to_end"][f"ndcg_at_{TOP_K}"]
        selection_rows.append(
            {
                "config_id": config_id,
                "mean_delta": interval["mean"],
                "lower": interval["lower"],
                "upper": interval["upper"],
                "selection_rate": evidence["selection_rate"],
                "passes_gate": config_id == selection["selected_config_id"]
                and selection["decision"] != "retain_baseline",
            }
        )
    selection_frame = pd.DataFrame(selection_rows)
    config_labels = {
        record["config_id"]: weight_label(record)
        for record in benchmark_payload["parameters"]["weight_configs"]
    }
    selection_frame["configuration"] = selection_frame["config_id"].map(config_labels)
    if not selection_frame.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        means = selection_frame["mean_delta"].to_numpy(dtype=float)
        lower = selection_frame["lower"].to_numpy(dtype=float)
        upper = selection_frame["upper"].to_numpy(dtype=float)
        ax.errorbar(
            means,
            selection_frame["configuration"],
            xerr=np.vstack([means - lower, upper - means]),
            fmt="o",
            color="#4C78A8",
            capsize=5,
        )
        ax.axvline(selection["min_delta"], color="black", linestyle="--", linewidth=1)
        ax.set(
            title=f"Paired Delta NDCG@{TOP_K} terhadap Baseline",
            xlabel="Mean delta dan confidence interval",
            ylabel="",
        )
        plt.tight_layout()
        plt.show()
    display(selection_frame)
    display(
        pd.DataFrame(
            {
                "decision": [selection["decision"]],
                "baseline_config_id": [selection["baseline_config_id"]],
                "selected_config_id": [selection["selected_config_id"]],
                "minimum_delta": [selection["min_delta"]],
                "minimum_selection_rate": [selection["min_selection_rate"]],
            }
        )
    )


# %%
if benchmark_payload is not None:
    fixed_test = benchmark_payload["fixed_test"]
    fixed_availability = fixed_test["availability"]
    fixed_funnel = pd.DataFrame(
        {
            "stage": [
                "sampled",
                "target available",
                "history evaluable",
                "personalization evaluable",
            ],
            "holdouts": [
                fixed_availability["sampled_holdouts"],
                fixed_availability["candidate_available_holdouts"],
                fixed_availability["history_evaluable_holdouts"],
                fixed_availability["personalization_evaluable_holdouts"],
            ],
        }
    )
    fixed_metrics = pd.DataFrame(
        [
            {
                "configuration": weight_label(record),
                "sampled_holdouts": record["sampled_holdouts"],
                "conditional_holdouts": record["conditional_holdouts"],
                **record["end_to_end_metrics"],
            }
            for record in fixed_test["aggregate_by_config"]
        ]
    )
    display(fixed_funnel)
    display(fixed_metrics)


# %% [markdown]
# Decision record harus dibaca apa adanya. Jika lower bound paired delta tidak
# melewati gate atau selection rate tidak stabil, keputusan yang benar adalah
# **retain baseline**. Menambahkan model yang lebih kompleks tanpa evidence bukan
# kemajuan engineering.

# %% [markdown]
# ## 11. Experiment-to-Serving: Model yang Sama, Bukan Demo Terpisah
#
# `RecommenderService.load` memverifikasi manifest, checksum SHA-256, tipe model,
# kompatibilitas runtime, algorithm version, schema, dan dataset fingerprint sebelum
# memberi akses inference. Ini adalah boundary yang sama dengan API dan Streamlit.

# %%
service = RecommenderService.load(MODEL_ARTIFACT)
service_info = service.model_info()
service_manifest = service.manifest

lineage = pd.DataFrame(
    {
        "field": [
            "artifact_id",
            "model_type",
            "created_at",
            "code_sha",
            "code_dirty",
            "dataset_fingerprint",
            "books",
            "ratings",
            "artifact_size_mb",
        ],
        "value": [
            service_info["artifact_id"],
            service_info["model_type"],
            service_info["created_at"],
            service_info["code_sha"],
            service_manifest.code_dirty,
            service_info["dataset_fingerprint"],
            service_info["books"],
            service_info["ratings"],
            round(MODEL_ARTIFACT.stat().st_size / (1024**2), 2),
        ],
    }
)
display(lineage)


# %%
notebook_training_fingerprint = fingerprint_dataset(
    dataset.books,
    dataset.ratings,
    dataset.users,
).fingerprint
serving_config = service_manifest.training_config
parity = pd.DataFrame(
    {
        "parameter": [
            "max_books",
            "max_content_books",
            "neighbors",
            "positive_rating_threshold",
            "collaborative_weight",
            "content_weight",
            "popularity_weight",
            "training_dataset_fingerprint",
        ],
        "notebook": [
            MAX_BOOKS,
            MAX_CONTENT_BOOKS,
            N_NEIGHBORS,
            POSITIVE_RATING_THRESHOLD,
            0.40,
            0.45,
            0.15,
            notebook_training_fingerprint,
        ],
        "serving": [
            serving_config.get("max_books"),
            serving_config.get("max_content_books"),
            serving_config.get("neighbors"),
            serving_config.get("positive_rating_threshold"),
            serving_config.get("collaborative_weight"),
            serving_config.get("content_weight"),
            serving_config.get("popularity_weight"),
            service_manifest.dataset.get("fingerprint"),
        ],
    }
)
parity["matches"] = parity["notebook"].astype(str) == parity["serving"].astype(str)
display(parity)


# %% [markdown]
# ### Warm-Process Serving Smoke Test
#
# Ini bukan load test produksi. Tujuannya hanya memastikan artifact terverifikasi
# dapat memberi rekomendasi berulang dan mencatat latency warm-process pada mesin
# yang menjalankan notebook.

# %%
serving_user_id = int(service.ratings["user_id"].value_counts().index[0])
with threadpool_limits(limits=1):
    service.recommend_for_user(serving_user_id, top_n=TOP_K)
    latency_ms = []
    for _ in range(10):
        started = perf_counter()
        serving_recommendations = service.recommend_for_user(serving_user_id, top_n=TOP_K)
        latency_ms.append((perf_counter() - started) * 1_000)

serving_smoke = pd.DataFrame(
    {
        "metric": ["runs", "result rows", "mean latency ms", "p95 latency ms"],
        "value": [
            len(latency_ms),
            len(serving_recommendations),
            float(np.mean(latency_ms)),
            float(np.quantile(latency_ms, 0.95)),
        ],
    }
)
display(serving_smoke)
display(
    serving_recommendations[
        ["score", "source", "reason", "isbn", "book_title", "book_author"]
    ].head(5)
)


# %% [markdown]
# ## 12. Trade-off dan Langkah Online
#
# | Pilihan | Kekuatan | Risiko utama |
# | --- | --- | --- |
# | Popularity | murah, selalu tersedia | nonpersonal, popularity bias |
# | Collaborative | personal dari perilaku bersama | sparse/cold-start |
# | Content | cold item dan item similarity | metadata dangkal/overspecialization |
# | Hybrid | fallback dan beberapa sumber evidence | bobot serta operasi lebih kompleks |
#
# Jika traffic nyata tersedia, unit randomisasi A/B test harus user-level. Primary
# metric dapat berupa save-to-list atau read-start; CTR menjadi leading indicator;
# guardrail mencakup latency, error/empty rate, fallback rate, long-tail exposure,
# dan konsentrasi popularitas. Purchase hanya relevan jika produk memang mencatat
# transaksi dan attribution window didefinisikan lebih dahulu.

# %% [markdown]
# ## 13. Kesimpulan Jujur
#
# Yang sudah dibuktikan:
#
# - data contract, convergent k-core, dan pemisahan interaction/content catalog;
# - popularity, collaborative, content, dan hybrid berjalan pada pipeline yang sama;
# - work-level holdout dan train-only filtering mencegah leakage edisi;
# - bundle serving diverifikasi dengan checksum, lineage, schema, dan fingerprint;
# - benchmark formal memiliki fixed test, repeated validation, paired bootstrap,
#   selection gate, availability, segment, dan discovery diagnostics.
#
# Yang belum boleh diklaim:
#
# - uplift bisnis tanpa impression/click/save/purchase;
# - generalisasi temporal tanpa timestamp;
# - kemenangan hybrid jika paired confidence interval dan selection gate tidak lolos;
# - skalabilitas katalog jutaan item dari eksperimen 1.000–3.000 buku.
#
# Engineering yang robust bukan model paling rumit. Ia adalah keputusan yang dapat
# diulang, evidence yang tidak bocor, failure mode yang terlihat, dan model serving
# yang identitasnya sama dengan model yang dievaluasi.
