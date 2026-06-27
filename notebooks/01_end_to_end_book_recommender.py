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

# %% [markdown]
# # End-to-End Book Recommender System
#
# Notebook ini memperlihatkan proses dari awal sampai akhir untuk membangun sistem
# rekomendasi buku berbasis dataset Kaggle Book Recommendation Dataset.
#
# Tujuan notebook:
#
# - memahami struktur dan kualitas data
# - membersihkan data buku, rating, dan user
# - membangun rekomendasi content-based dari metadata buku
# - membangun rekomendasi collaborative filtering dari histori rating user
# - mengevaluasi hasil secara sederhana dan menjelaskan keterbatasannya
#
# Kode utama sengaja disimpan di `src/book_recommender` supaya notebook ini tetap
# fokus pada narasi, eksplorasi, dan hasil.

# %%
from pathlib import Path

import matplotlib
try:
    get_ipython
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import dataset_summary, load_raw_data, prepare_dataset, validate_data_dir
from book_recommender.evaluation import precision_at_k, recall_at_k

try:
    from IPython.display import display
except ImportError:
    display = print

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)


# %% [markdown]
# ## 1. Konfigurasi
#
# Dataset tidak disimpan di git. Unduh dataset dari Kaggle:
#
# https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
#
# Lalu letakkan file berikut di `data/raw/`:
#
# - `Books.csv`
# - `Ratings.csv`
# - `Users.csv`

# %%
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Parameter ini sengaja dibuat eksplisit agar eksperimen mudah diulang.
# Naikkan MAX_BOOKS jika ingin eksperimen yang lebih besar.
MAX_BOOKS = 5_000
MIN_USER_RATINGS = 2
MIN_BOOK_RATINGS = 2
RANDOM_STATE = 42

validate_data_dir(DATA_DIR)
DATA_DIR


# %% [markdown]
# ## 2. Data Understanding
#
# Dataset terdiri dari tiga file utama:
#
# - `Books.csv`: metadata buku
# - `Ratings.csv`: rating buku dari user
# - `Users.csv`: metadata user

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


# %%
display(raw.books.head())
display(raw.ratings.head())
display(raw.users.head())


# %% [markdown]
# ### Missing Value
#
# Missing value paling jelas berada pada kolom `Age` di data user. Untuk model
# rekomendasi yang dibangun di notebook ini, fitur umur tidak dipakai langsung,
# tetapi pengecekan tetap dilakukan agar kualitas data terlihat.

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


# %% [markdown]
# ### Distribusi Rating
#
# Rating `0` pada dataset ini diperlakukan sebagai belum memberi rating eksplisit.
# Untuk recommender berbasis rating eksplisit, rating `0` dibuang pada tahap
# preprocessing.

# %%
rating_counts = raw.ratings["Book-Rating"].value_counts().sort_index()

plt.figure(figsize=(9, 4))
sns.barplot(x=rating_counts.index.astype(str), y=rating_counts.values, color="#4C78A8")
plt.title("Distribusi Rating Buku")
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()

display(rating_counts.rename("count").reset_index().rename(columns={"index": "rating"}))


# %% [markdown]
# ## 3. Data Preparation
#
# Tahap preparation dilakukan melalui fungsi di `book_recommender.data`:
#
# - rename kolom agar konsisten
# - hapus rating `0`
# - hapus record yang tidak punya ISBN atau judul
# - filter interaksi sparse
# - ambil `MAX_BOOKS` buku yang paling banyak dirating
#
# Pengambilan buku berdasarkan jumlah rating lebih defensible daripada slicing
# `books[:10000]`, karena tidak bergantung pada urutan file mentah.

# %%
dataset = prepare_dataset(
    DATA_DIR,
    min_user_ratings=MIN_USER_RATINGS,
    min_book_ratings=MIN_BOOK_RATINGS,
    max_books=MAX_BOOKS,
    random_state=RANDOM_STATE,
)

summary = pd.Series(dataset_summary(dataset), name="value").to_frame()
display(summary)


# %%
display(dataset.books.head())
display(dataset.ratings.head())


# %% [markdown]
# ### Buku dan User Paling Aktif

# %%
book_popularity = (
    dataset.ratings.groupby("isbn")
    .agg(mean_rating=("book_rating", "mean"), rating_count=("book_rating", "size"))
    .reset_index()
    .merge(dataset.books, on="isbn", how="left")
    .sort_values(["rating_count", "mean_rating"], ascending=False)
)

display(
    book_popularity[
        ["isbn", "book_title", "book_author", "mean_rating", "rating_count"]
    ].head(10)
)


# %%
user_activity = dataset.ratings["user_id"].value_counts().head(10)
display(user_activity.rename_axis("user_id").reset_index(name="rating_count"))


# %% [markdown]
# ## 4. Content-Based Recommendation
#
# Pendekatan content-based memakai teks metadata:
#
# ```text
# book_title + book_author + publisher
# ```
#
# Teks tersebut diubah menjadi TF-IDF vector, lalu buku terdekat dicari dengan
# cosine distance.

# %%
content_model = ContentBasedRecommender().fit(dataset.books)

query_title = "Adventures of Huckleberry Finn"
content_recommendations = content_model.recommend_similar(title=query_title, top_n=10)

display(
    content_recommendations[
        ["score", "isbn", "book_title", "book_author", "publisher", "image_m_url"]
    ]
)


# %% [markdown]
# Hasil di atas bisa dibaca sebagai buku yang metadata-nya paling mirip dengan
# query. Untuk contoh ini, hasil yang baik seharusnya berisi edisi lain dari
# `Huckleberry Finn`, buku lain dari Mark Twain, atau buku dari publisher/seri
# yang relevan.

# %% [markdown]
# ## 5. Collaborative Filtering
#
# Pendekatan collaborative filtering memakai histori rating user. Model yang
# dipakai di sini adalah item-item collaborative filtering:
#
# - membuat sparse matrix `book x user`
# - mencari buku yang memiliki pola rating mirip
# - merekomendasikan buku mirip dengan buku yang disukai user
#
# Pendekatan ini lebih ringan daripada deep learning dan cocok untuk baseline
# project yang reproducible.

# %%
collab_model = ItemBasedCollaborativeRecommender(n_neighbors=50).fit(
    dataset.books,
    dataset.ratings,
)

candidate_user_id = 5709
if candidate_user_id not in set(dataset.ratings["user_id"]):
    candidate_user_id = int(dataset.ratings["user_id"].value_counts().index[0])

user_history = (
    dataset.ratings[dataset.ratings["user_id"] == candidate_user_id]
    .sort_values("book_rating", ascending=False)
    .merge(dataset.books, on="isbn", how="left")
)

display(user_history[["isbn", "book_title", "book_author", "book_rating"]].head(10))


# %%
collab_recommendations = collab_model.recommend_for_user(candidate_user_id, top_n=10)

display(
    collab_recommendations[
        ["score", "isbn", "book_title", "book_author", "publisher", "image_m_url"]
    ]
)


# %% [markdown]
# ## 6. Evaluasi Sederhana
#
# Untuk gambaran awal, notebook ini memakai evaluasi leave-one-out kecil:
#
# - pilih sebagian user yang punya minimal 3 rating
# - tahan satu buku dengan rating tinggi sebagai data relevan
# - latih recommender tanpa rating yang ditahan
# - cek apakah buku yang ditahan muncul di Top-K rekomendasi
#
# Evaluasi ini belum sekuat eksperimen produksi, tetapi lebih baik daripada
# klaim akurasi manual tanpa target yang jelas. Karena dataset sangat sparse,
# Top-50 dipakai untuk membaca sinyal baseline awal.

# %%
def build_leave_one_out_sample(
    ratings: pd.DataFrame,
    sample_users: int = 30,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = ratings.groupby("user_id").filter(lambda frame: len(frame) >= min_interactions)
    eligible = eligible[eligible["book_rating"] >= min_relevant_rating]

    users = (
        eligible["user_id"]
        .drop_duplicates()
        .sample(n=min(sample_users, eligible["user_id"].nunique()), random_state=random_state)
    )

    holdout_rows = []
    for user_id in users:
        user_rows = eligible[eligible["user_id"] == user_id]
        holdout_rows.append(user_rows.sample(n=1, random_state=random_state))

    holdout = pd.concat(holdout_rows).reset_index(drop=True)
    train = ratings.merge(
        holdout[["user_id", "isbn"]],
        on=["user_id", "isbn"],
        how="left",
        indicator=True,
    )
    train = train[train["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)
    return train, holdout


train_ratings, holdout_ratings = build_leave_one_out_sample(
    dataset.ratings,
    sample_users=10,
    random_state=RANDOM_STATE,
)

eval_model = ItemBasedCollaborativeRecommender(n_neighbors=100).fit(
    dataset.books,
    train_ratings,
)

k = 50
metric_rows = []
for row in holdout_ratings.itertuples(index=False):
    recommended = eval_model.recommend_for_user(int(row.user_id), top_n=k)["isbn"].tolist()
    relevant = [row.isbn]
    metric_rows.append(
        {
            "user_id": int(row.user_id),
            "heldout_isbn": row.isbn,
            f"precision_at_{k}": precision_at_k(recommended, relevant, k),
            f"recall_at_{k}": recall_at_k(recommended, relevant, k),
            f"hit_at_{k}": float(row.isbn in recommended),
        }
    )

metrics = pd.DataFrame(metric_rows)
display(metrics.head())
metric_columns = [f"precision_at_{k}", f"recall_at_{k}", f"hit_at_{k}"]
display(metrics[metric_columns].mean().to_frame("mean"))


# %% [markdown]
# ## 7. Kesimpulan
#
# Dari notebook ini:
#
# - Dataset dapat diproses ulang dari file mentah Kaggle.
# - Content-based recommendation cocok untuk mencari buku mirip berdasarkan
#   metadata, termasuk saat user belum punya histori rating.
# - Collaborative filtering cocok ketika user sudah memiliki histori rating,
#   karena rekomendasi berasal dari pola kesukaan user lain.
# - Evaluasi awal sudah memakai target holdout, tetapi masih terbatas karena
#   hanya mengevaluasi sampel kecil dan belum memakai split temporal.
#
# Pengembangan lanjutan yang masuk akal:
#
# - simpan model sebagai artifact di `models/`
# - tambah Streamlit demo dengan cover buku
# - eksperimen hybrid recommender yang menggabungkan metadata dan rating
# - evaluasi dengan Precision@K, Recall@K, MAP@K, atau NDCG@K pada split yang
#   lebih sistematis
