# Laporan Proyek Machine Learning: Book Recommender System

Sistem rekomendasi buku berbasis dataset Book-Crossing. Project ini mengeksplorasi
tiga pendekatan rekomendasi: content-based filtering, item-based collaborative
filtering, dan hybrid recommender. Notebook end-to-end tersedia di
[notebooks/01_end_to_end_book_recommender.ipynb](notebooks/01_end_to_end_book_recommender.ipynb).

## Domain Proyek

Katalog buku yang besar membuat proses menemukan bacaan yang relevan menjadi
lebih sulit. Pengguna dapat mengetahui judul buku yang disukai, tetapi belum
tentu dapat menemukan karya lain dengan karakteristik serupa. Di sisi lain,
riwayat rating dari banyak pengguna dapat memuat pola preferensi yang berguna
untuk personalisasi.

## Business Understanding

### Masalah yang Diselesaikan

Terdapat tiga tantangan utama dalam data rekomendasi buku ini:

1. Metadata buku tersebar pada katalog yang besar, sehingga pengguna sulit
   menemukan judul yang relevan secara manual.
2. Interaksi rating bersifat sparse; tidak setiap pengguna memberi cukup rating
   untuk membangun rekomendasi personal.
3. Satu karya dapat muncul dalam beberapa ISBN atau edisi, sehingga rekomendasi
   dapat berulang bila setiap ISBN diperlakukan sebagai buku yang berbeda.

### Tujuan

Project ini bertujuan untuk:

1. Menyiapkan data buku, rating, dan pengguna menjadi data yang konsisten untuk
   pemodelan rekomendasi.
2. Membuat rekomendasi berbasis metadata buku dan riwayat rating pengguna.
3. Membandingkan pendekatan rekomendasi dengan evaluasi ranking yang menghindari
   kebocoran data.

### Solusi yang Dicoba

| Pendekatan | Sinyal utama | Peran dalam sistem |
| --- | --- | --- |
| Content-based filtering | Judul, penulis, dan penerbit | Mencari buku yang mirip dengan sebuah buku query. |
| Collaborative filtering | Pola rating pengguna terhadap buku | Memberi rekomendasi personal dari pola preferensi bersama. |
| Hybrid recommender | Content, collaborative, dan popularitas | Menggabungkan sinyal agar sistem tetap memberi hasil saat histori pengguna terbatas. |

## Data Understanding

Data berasal dari [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset),
yang bersumber dari Book-Crossing. Dataset mentah terdiri dari tiga file CSV.

| File | Jumlah baris | Informasi yang digunakan |
| --- | ---: | --- |
| Books.csv | 271.360 | ISBN, judul, penulis, tahun terbit, penerbit, dan URL cover. |
| Ratings.csv | 1.149.780 | User ID, ISBN, dan rating buku pada skala 0-10. |
| Users.csv | 278.858 | User ID, lokasi, dan usia. |

Beberapa observasi penting dari data:

- Sebanyak 716.109 baris rating bernilai 0. Pada Book-Crossing, nilai tersebut
  menunjukkan tidak ada rating eksplisit, bukan preferensi netral.
- Kolom Age memiliki missing value. Kolom ini tetap dipertahankan sebagai
  metadata pengguna, tetapi tidak dipakai sebagai fitur model.
- Buku yang sama dapat memiliki beberapa ISBN karena perbedaan edisi atau
  format. Hal ini perlu ditangani saat membuat daftar rekomendasi.
- Data mentah tidak disimpan di repository. Data perlu diunduh dari Kaggle
  sebelum notebook, training, atau evaluasi dijalankan.

## Data Preprocessing

Preprocessing dilakukan pada fungsi yang sama untuk notebook, CLI, training, dan
serving agar arti data tidak berubah antarproses.

| Tahap | Keputusan | Alasan |
| --- | --- | --- |
| Normalisasi skema | Kolom sumber diganti ke nama yang konsisten; ISBN dan user ID divalidasi. | Mencegah identifier yang kosong atau tidak valid masuk ke model. |
| Validasi rating | Rating harus berupa angka finite pada rentang 0-10. | Menjaga kualitas sinyal eksplisit. |
| Penanganan rating 0 | Rating 0 dikeluarkan dari training explicit-feedback. | Nilai 0 bukan skor suka atau tidak suka. |
| Duplikasi interaksi | Duplikat pasangan user dan ISBN diagregasi dengan rata-rata rating. | Dataset tidak memiliki timestamp untuk menentukan rating terakhir secara jujur. |
| Metadata buku | Judul wajib tersedia; author dan publisher yang kosong diisi Unknown. | Model content membutuhkan teks metadata yang konsisten. |
| Kecocokan data | Rating tanpa metadata buku tidak diteruskan ke data training. | Setiap kandidat interaksi harus dapat dipetakan ke buku. |

## Data Preparation

Setelah cleaning, interaksi sparse difilter menggunakan convergent k-core. User
dan buku yang tidak memenuhi minimum interaksi dihapus berulang sampai tidak ada
lagi node yang melanggar batas. Pendekatan ini lebih aman daripada filter satu
kali karena penghapusan satu user atau buku dapat membuat entitas lain menjadi
sparse.

Notebook end-to-end menggunakan konfigurasi eksperimen berikut:

| Parameter | Nilai |
| --- | ---: |
| Minimum rating per user | 2 |
| Minimum rating per buku | 2 |
| Maksimum buku pada eksperimen | 5.000 |
| Random state | 42 |

Katalog 5.000 buku dipilih secara deterministik berdasarkan jumlah rating,
bukan dengan mengambil 5.000 atau 10.000 baris pertama dari file mentah.
Pipeline juga memisahkan interaction catalog dari content catalog. Dengan
pemisahan ini, buku tanpa rating masih dapat tersedia untuk pencarian berbasis
metadata, sedangkan seluruh buku yang memiliki interaksi tetap dipertahankan.

## Modeling

### Content-Based Filtering

Model content-based membentuk teks gabungan dari:

~~~text
book_title + book_author + publisher
~~~

Teks tersebut diubah menjadi vektor TF-IDF dengan unigram dan bigram. Kemudian
cosine distance digunakan untuk mencari buku yang paling dekat dengan buku query.
Pendekatan ini tidak membutuhkan riwayat rating pengguna, sehingga sesuai untuk
fitur buku serupa.

Setiap hasil Top-N dideduplikasi menggunakan canonical work, yaitu identitas
yang dibentuk dari judul dan penulis yang telah dinormalisasi. Dengan demikian,
edisi paperback dan hardcover dari karya yang sama tidak memenuhi beberapa
posisi rekomendasi.

### Item-Based Collaborative Filtering

Collaborative filtering menggunakan matriks sparse buku x pengguna. Kemiripan
antarbuku dihitung dengan adjusted cosine: rating dikoreksi terlebih dahulu
terhadap rata-rata rating pengguna yang diregularisasi. Pendekatan ini dipilih
sebagai baseline yang ringan dan dapat dijelaskan, bukan model deep learning.

Hanya rating positif, dengan default rating minimal 8, yang dipakai sebagai seed
preferensi. Buku yang sudah dibaca pengguna dan karya yang setara dengan buku
tersebut dikeluarkan dari hasil. Jika user tidak dikenal atau tidak memiliki seed
positif, sistem menggunakan popularity fallback.

### Hybrid Recommender

Hybrid recommender menggabungkan tiga skor yang telah dinormalisasi:

| Sinyal | Sumber |
| --- | --- |
| Collaborative score | Buku dengan pola rating yang mirip dengan histori pengguna. |
| Content profile score | Kemiripan metadata terhadap buku yang diberi rating positif oleh pengguna. |
| Popularity score | Kombinasi rata-rata rating dan jumlah rating sebagai fallback. |

Bobot default adalah 0,40 untuk collaborative, 0,45 untuk content, dan 0,15
untuk popularitas. Saat histori positif pengguna sangat sedikit, bobot dapat
bergeser ke metadata. Bila tidak ada histori positif, hybrid sengaja kembali ke
popularitas agar sistem tetap mengembalikan rekomendasi.

Hybrid di sini adalah weighted-ranking baseline. Ia bukan learned ranker dan
belum dapat dipandang sebagai model terbaik sebelum benchmark yang lebih besar
selesai dilakukan.

## Evaluasi

Evaluasi recommender tidak menggunakan RMSE atau akurasi manual berdasarkan
penulis yang sama. Sistem dievaluasi sebagai masalah ranking: apakah karya yang
ditahan dari histori pengguna dapat muncul kembali pada daftar rekomendasi.

Protokol evaluasi yang digunakan:

1. Pilih satu canonical work sebagai holdout untuk setiap user yang memenuhi
   syarat; seluruh edisi karya tersebut dikeluarkan dari histori training.
2. Pisahkan user validation dan user test agar pemilihan konfigurasi tidak
   membaca test set.
3. Jalankan filtering interaksi dan pemilihan kandidat hanya dari train split.
4. Evaluasi seluruh holdout test, termasuk cold-user dan target yang tidak
   tersedia pada katalog.
5. Bandingkan hybrid, collaborative dengan popularity fallback, content profile
   strict, dan popularity baseline.

Metrik utama adalah Precision@K, Recall@K, HitRate@K, MAP@K, MRR@K, dan NDCG@K.
Evaluator juga dapat melaporkan catalog coverage, novelty, intra-list diversity,
serta konsentrasi popularitas.

Ada dua sudut pandang hasil:

- Conditional hanya menghitung user dengan positive seed dan target yang tersedia
  di katalog. Scope ini membantu mendiagnosis kualitas sinyal personal.
- End-to-end menghitung seluruh holdout. Target yang tidak tersedia tetap menjadi
  miss, dan cold-user tetap melalui fallback seperti pada kondisi serving.

Repository ini belum menyimpan benchmark terbaru pada cohort besar sebagai bukti
performa final. Karena itu, README sengaja tidak mengklaim satu model sebagai
pemenang atau menampilkan angka dari eksperimen lama yang memakai algoritme dan
protokol berbeda.

## Kesimpulan

Sistem berhasil membangun tiga baseline recommender yang dapat dijalankan ulang:

- Content-based filtering membantu menemukan buku serupa meskipun tidak ada
  riwayat pengguna.
- Item-based collaborative filtering memanfaatkan pola rating pengguna untuk
  personalisasi, tetapi kualitasnya bergantung pada histori positif yang cukup.
- Hybrid menggabungkan kedua sinyal dan popularity fallback untuk menghadapi
  kondisi data pengguna yang berbeda.

Pemisahan katalog, deduplikasi canonical work, dan evaluasi ranking bebas
leakage membuat hasil eksperimen lebih representatif daripada evaluasi berbasis
ISBN atau split acak sederhana.

## Batasan dan Pengembangan Lanjutan

| Batasan | Pengembangan berikutnya |
| --- | --- |
| Dataset tidak memiliki timestamp interaksi. | Menggunakan event bertimestamp untuk temporal backtest. |
| Data rating sparse dan popularitas dapat mendominasi hasil. | Mengukur dan mengurangi popularity bias serta long-tail exposure. |
| Cold-start belum terselesaikan sepenuhnya. | Menambah strategi onboarding dan fitur untuk user atau buku baru. |
| Hybrid masih berupa weighted baseline. | Membandingkan implicit ALS/BPR, matrix factorization, atau learned-to-rank. |
| Belum ada feedback dan KPI online. | Mencatat impression, click, save, dan purchase untuk eksperimen online. |
| Belum ada monitoring model produksi. | Menambahkan data-quality check, drift monitoring, canary, dan rollback. |

---

## Menjalankan Project

### Prasyarat

- Python 3.10-3.12
- uv
- Kaggle API credential untuk mengunduh dataset

### Instalasi

~~~bash
uv sync --frozen --dev --extra app --extra api
~~~

### Menyiapkan Dataset

Letakkan Books.csv, Ratings.csv, dan Users.csv pada folder data/raw/. Dataset
dapat diunduh dengan Kaggle CLI:

~~~bash
uv run kaggle datasets download \
  -d arashnic/book-recommendation-dataset \
  -p data/raw --unzip

uv run book-rec check-data --data-dir data/raw
~~~

Credential Kaggle tidak boleh disimpan di repository.

### Menjalankan Notebook

Buka notebooks/01_end_to_end_book_recommender.ipynb pada VS Code atau lingkungan
Jupyter setelah environment dan dataset siap. Notebook menjalankan alur data
understanding, preparation, modeling, dan evaluasi contoh.

### Menjalankan Rekomendasi dari CLI

~~~bash
# Mencari buku serupa
uv run book-rec recommend-content \
  --data-dir data/raw \
  --title "Adventures of Huckleberry Finn"

# Rekomendasi personal
uv run book-rec recommend-hybrid --data-dir data/raw --user-id 5709
~~~

Gunakan bantuan CLI untuk command training dan evaluasi lainnya:

~~~bash
uv run book-rec --help
~~~

### Menjalankan Benchmark

~~~bash
uv run book-rec benchmark-hybrid \
  --data-dir data/raw \
  --validation-users 200 \
  --test-users 200 \
  --repetitions 3 \
  --k 50 \
  --output artifacts/evaluation/repeated_split_hybrid_benchmark.json
~~~

Untuk smoke test, gunakan jumlah user, repetition, dan bootstrap resample yang
lebih kecil. Hasil smoke test tidak boleh digunakan sebagai klaim performa.

### Menjalankan Aplikasi

~~~bash
# Streamlit
uv run streamlit run app/streamlit_app.py

# FastAPI
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
~~~

Streamlit dan FastAPI memuat bundle model dari models/hybrid.joblib. Jalankan
training terlebih dahulu atau gunakan bundle model yang tersedia.

### Quality Check

~~~bash
uv run ruff check src tests app notebooks/01_end_to_end_book_recommender.py
uv run pytest --cov=book_recommender --cov=app --cov-report=term-missing
~~~

## Referensi

1. [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
2. [Scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
3. [Scikit-learn: cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
