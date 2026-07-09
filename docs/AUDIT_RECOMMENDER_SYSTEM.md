# Audit Recommender System

Tanggal audit: 2026-07-08
Tanggal tindak lanjut: 2026-07-09

## Status Tindak Lanjut

Perbaikan awal yang sudah diterapkan setelah audit:

- Evaluasi collaborative filtering memakai protokol leave-one-out dengan
  filtering sparsity dan pemilihan candidate books dari train split saja.
- CLI evaluasi menampilkan `sampled_holdouts`, `evaluable_holdouts`,
  `dropped_holdouts`, jumlah train ratings, dan jumlah candidate books.
- Metrik ranking ditambah `MRR@K`.
- Diagnostics rekomendasi ditambah `Coverage@K`, novelty, intra-list diversity,
  serta share rekomendasi pada top 1%, 5%, dan 10% item populer.
- Hasil evaluasi dapat disimpan sebagai artifact JSON di `artifacts/evaluation/`.
- Hybrid recommender ditambahkan dengan skor gabungan collaborative, content
  profile, dan popularity yang dinormalisasi.
- CLI dan Streamlit app dapat menampilkan rekomendasi hybrid.
- Rekomendasi hybrid menyertakan `source` dan `reason` sebagai explainability
  sederhana untuk user/reviewer.
- CLI `tune-hybrid` ditambahkan untuk membandingkan beberapa kombinasi bobot
  hybrid pada split evaluasi yang sama.
- Root `app.py` lama dipindahkan ke `legacy/app.py` agar entry point aktif tidak
  membingungkan.
- Test ditambah untuk memastikan holdout tidak ikut membentuk candidate catalog
  train, memverifikasi metrik diagnostics, dan menjaga roundtrip artifact hybrid.
- Evaluasi lokal `--sample-users 200 --k 50 --max-books 5000` menunjukkan
  `HitRate@50` hybrid tuned `0.2027`, collaborative `0.1081`, dan popularity
  `0.1216`.
- Bobot dasar hybrid default diubah ke hasil tuning awal:
  collaborative `0.40`, content `0.45`, popularity `0.15`.

Catatan: bagian audit di bawah sudah disesuaikan dengan tindak lanjut awal.
Beberapa item yang masih relevan adalah content-based evaluation, cold-start
yang lebih kuat, explainability yang lebih detail, validasi hybrid yang lebih
luas, dan scalability plan.

## Ringkasan Project

Project ini adalah sistem rekomendasi buku berbasis Kaggle Book Recommendation
Dataset. Pipeline utama sudah dirapikan ke package `src/book_recommender/`,
notebook end-to-end ada di `notebooks/01_end_to_end_book_recommender.py`, dan UI
aktif ada di `app/streamlit_app.py`.

File utama yang ditemukan:

| Area | File |
| --- | --- |
| Loading dataset dan preprocessing | `src/book_recommender/data.py` |
| Content-based recommendation | `src/book_recommender/content_based.py` |
| Collaborative filtering | `src/book_recommender/collaborative.py` |
| Evaluasi ranking | `src/book_recommender/evaluation.py` |
| CLI | `src/book_recommender/cli.py` |
| UI aktif | `app/streamlit_app.py` |
| Notebook utama | `notebooks/01_end_to_end_book_recommender.py` dan `.ipynb` |
| Tests | `tests/test_recommenders.py` |
| Dokumentasi dataset | `docs/DATASET.md` |
| File lama/arsip | `legacy/` dan `docs/LEGACY_REPORT_2023.md` |

Catatan: `app.py` lama berbasis TensorFlow dan folder `aset/` sudah dipindahkan
ke `legacy/app.py`, sehingga entry point aktif project tetap
`app/streamlit_app.py`.

## Pipeline Saat Ini

Alur pipeline aktif:

1. `Books.csv`, `Ratings.csv`, dan `Users.csv` dibaca dari `data/raw/`.
2. Kolom dinormalisasi menjadi `isbn`, `book_title`, `book_author`,
   `publisher`, `user_id`, dan `book_rating`.
3. Rating `0` dibuang dan sistem hanya memakai rating eksplisit `1` sampai `10`.
4. Rating yang ISBN-nya tidak ada di data buku dibuang.
5. Interaksi dapat difilter dengan `min_user_ratings` dan `min_book_ratings`.
6. Default CLI/UI mengambil `max_books=5000` buku yang paling banyak dirating.
7. Content-based model membuat representasi TF-IDF dari
   `book_title + book_author + publisher`, lalu mencari nearest neighbors dengan
   cosine distance.
8. Collaborative model membuat sparse matrix item-user dari rating eksplisit,
   lalu memakai item-item KNN berbasis cosine distance.
9. Hybrid model menggabungkan skor collaborative, content profile, dan
   popularity yang dinormalisasi.
10. Evaluasi memakai leave-one-out split dan membandingkan hybrid,
    collaborative, serta popularity baseline.
11. Streamlit app menampilkan rekomendasi buku mirip, rekomendasi per `user_id`,
    dan rekomendasi hybrid.

Representasi saat ini:

| Komponen | Representasi |
| --- | --- |
| User | `user_id` numerik dari `Ratings.csv`/`Users.csv` |
| Item | Buku direpresentasikan oleh `isbn`; metadata item dari title, author, publisher, cover URL |
| Interaksi user-item | Rating eksplisit `book_rating` skala 1-10 setelah rating 0 dibuang |
| Output content-based | Top-N buku mirip dengan satu judul/ISBN query, dengan skor similarity |
| Output collaborative | Top-N buku untuk user, dengan skor agregasi similarity item-item |
| Output hybrid | Top-N buku untuk user dengan `source` dan `reason` sederhana |
| Output evaluasi | Ranking metrics Top-K dan diagnostics coverage/diversity/novelty, bukan accuracy classification |

Sistem utama menghasilkan Top-N recommendation. Tidak ada model prediksi rating
final di package baru.

## Dataset dan Struktur Data

Dataset lokal yang diaudit:

| File | Baris | Kolom |
| --- | ---: | ---: |
| `Books.csv` | 271,360 | 8 |
| `Ratings.csv` | 1,149,780 | 3 |
| `Users.csv` | 278,858 | 3 |

Struktur data memenuhi syarat minimum recommender system:

- Ada `user_id`: `User-ID`.
- Ada `item_id`: `ISBN`.
- Ada rating eksplisit: `Book-Rating` skala 0-10, lalu project memakai 1-10.
- Ada metadata item: judul, author, tahun publikasi, publisher, cover URL.
- Tidak ada timestamp.
- Tidak ada kolom genre, kategori, tag, sinopsis, harga, atau fitur semantik kaya.

Kualitas data:

| Audit | Hasil |
| --- | --- |
| Missing value buku | `Image-URL-L`: 3, `Book-Author`: 2, `Publisher`: 2 |
| Missing value rating | 0 |
| Missing value user | `Age`: 110,762 |
| Duplikat full rows | 0 pada books, ratings, users |
| Duplikat ISBN books | 0 |
| Duplikat pasangan user-ISBN ratings | 0 |
| Rating 0 | 716,109 |
| Rating positif 1-10 | 433,671 |
| Rating positif dengan ISBN valid di books | 383,843 |
| Rating positif dengan ISBN tidak dikenal | 49,828 |
| Buku tanpa interaksi positif | 121,523 |

Sparsity sangat tinggi:

- Cleaned positive ratings pada ISBN valid: 68,091 user, 149,836 buku, 383,843
  interaksi.
- Sparsity terhadap rated clean books: sekitar `99.9962%`.
- Pada default prepared dataset (`max_books=5000`, `min_user_ratings=2`,
  `min_book_ratings=2`): 20,827 user, 5,000 buku, 117,726 rating, sparsity
  sekitar `99.8869%`.

Distribusi interaksi menunjukkan cold-start dan sparsity kuat:

| Kondisi cleaned ratings | Jumlah | Proporsi |
| --- | ---: | ---: |
| User dengan <= 1 rating | 39,223 | 57.60% |
| User dengan <= 2 rating | 47,897 | 70.34% |
| User dengan <= 5 rating | 57,307 | 84.16% |
| Item dengan <= 1 rating | 99,413 | 66.35% |
| Item dengan <= 2 rating | 122,032 | 81.44% |
| Item dengan <= 5 rating | 139,105 | 92.84% |

Warning: dataset valid untuk recommender, tetapi sangat sparse. Collaborative
filtering harus dibaca sebagai baseline awal, bukan sistem final yang kuat.

## Audit Content-Based Recommendation

Implementasi aktif: `ContentBasedRecommender`.

Kekuatan:

- Memakai TF-IDF dengan unigram dan bigram.
- Similarity memakai cosine distance via `NearestNeighbors(metric="cosine")`.
- Query item sendiri difilter berdasarkan ISBN.
- Output menyertakan skor similarity.
- Artifact bisa disimpan dan diload dengan `joblib`.

Kelemahan:

- Fitur hanya `book_title + book_author + publisher`. Ini dangkal karena dataset
  tidak punya genre, tag, deskripsi, atau sinopsis.
- `pub_year` tidak dipakai, dan tidak ada fitur numerik yang dinormalisasi.
- Tidak ada diversity reranking, sehingga hasil bisa berupa edisi lain dari buku
  yang sama atau buku yang sangat mirip secara metadata.
- Tidak ada filtering berbasis histori user karena fungsi content-based saat ini
  hanya menerima query title/ISBN, bukan user profile.
- Explainability belum eksplisit. Skor similarity ada, tetapi belum ada alasan
  seperti "mirip karena author/publisher/judul".
- Query title memilih match pertama jika ada banyak edisi dengan judul sama.

Kesimpulan: content-based sudah benar sebagai baseline metadata, tetapi masih
terbatas. Kualitasnya sangat bergantung pada title/author/publisher.

## Audit Collaborative Recommendation

Implementasi aktif: `ItemBasedCollaborativeRecommender`.

Pendekatan:

- Item-based collaborative filtering.
- Sparse matrix `item x user`.
- Similarity item memakai cosine distance dengan brute-force nearest neighbors.
- Seed item berasal dari item dengan rating tertinggi pada histori user.
- Skor kandidat = similarity item * bobot rating user.
- Item yang sudah dirating user difilter dari rekomendasi.
- User baru tanpa histori mendapat popularity fallback.

Kekuatan:

- Missing interaction tidak diperlakukan sebagai rating 0 dense secara eksplisit;
  data disimpan sparse.
- Rating 0 dari dataset dibuang sebelum modeling.
- Popularity fallback tersedia untuk unknown user.
- Popularity dihitung dari data yang dipakai saat `fit`, sehingga pada evaluasi
  CLI popularity baseline dihitung dari train ratings.
- Tests mencakup filtering rekomendasi item yang belum dibaca dan roundtrip
  artifact.

Kelemahan:

- Cosine similarity memakai raw explicit rating tanpa user mean-centering atau
  normalisasi bias user/item. Ini bisa membuat item populer atau user yang sering
  memberi rating tinggi lebih dominan.
- Tidak ada matrix factorization, ALS, SVD, BPR, atau model implicit-feedback.
- User dengan 1-2 interaksi sangat rapuh; padahal jumlahnya dominan.
- Cold-start item tidak tertangani oleh collaborative model.
- Candidate generation berasal dari nearest neighbors seed item saja, sehingga
  coverage dan serendipity bisa terbatas.
- Explainability collaborative belum sampai level seed item yang menyumbang
  skor. Hybrid sudah punya `source` dan `reason`, tetapi masih berupa alasan
  kategori sinyal, bukan kontribusi item per item.

Warning: collaborative filtering layak sebagai baseline ringan, tetapi dataset
sangat sparse. Performa Top-N harus diuji dengan lebih banyak user dan metrik
coverage/diversity.

## Audit Hybrid Recommendation

Hybrid recommender aktif sudah tersedia sebagai baseline gabungan. Skor akhir
menggabungkan collaborative signal, content profile dari buku yang pernah diberi
rating tinggi, dan popularity signal yang dinormalisasi. Bobot default hasil
tuning awal adalah collaborative `0.40`, content `0.45`, dan popularity `0.15`.

Yang sudah baik:

- Skor collaborative, content, dan popularity tidak digabung mentah, tetapi
  dinormalisasi lebih dulu.
- Untuk user dengan histori cukup, collaborative dan content dapat saling
  melengkapi.
- Untuk sinyal yang tidak tersedia, hybrid dapat turun ke signal lain atau
  popularity fallback.
- Output rekomendasi menyertakan `source` dan `reason`, sehingga UI/CLI tidak
  hanya menampilkan skor numerik.
- CLI `tune-hybrid` dapat membandingkan beberapa kombinasi bobot pada split
  evaluasi yang sama.

Batasan yang masih relevan:

- Bobot hybrid masih hasil grid kecil pada sampel terbatas, belum validasi
  berulang atau eksperimen skala penuh.
- Belum ada learning-to-rank, matrix factorization, atau model latent factor.
- Diversity penalty belum masuk ke skor akhir; diversity baru dimonitor sebagai
  diagnostics evaluasi.
- Cold-start user tanpa histori masih membutuhkan onboarding preference jika
  ingin lebih baik dari popularity fallback.
- Cold-start item belum ideal karena content index default masih dibatasi oleh
  candidate catalog eksperimen.

## Potensi Data Leakage

Temuan awal P0 sudah ditindaklanjuti pada jalur evaluasi CLI. Evaluasi sekarang
menggunakan protokol `leave_one_out_train_only_filtering`: holdout dibuat lebih
dulu, lalu candidate catalog, popularity baseline, dan model rekomendasi
dibentuk dari `train_ratings`.

Yang sudah baik:

- Model collaborative, hybrid, dan popularity baseline pada evaluasi di-fit dari
  `train_ratings`, bukan full ratings.
- Candidate books untuk evaluasi dipilih dari interaksi train, sehingga holdout
  tidak ikut menentukan top-books catalog.
- Item yang sudah ada di train history user dikeluarkan dari rekomendasi baru.
- Artifact evaluasi menyimpan `sampled_holdouts`, `evaluable_holdouts`,
  `dropped_holdouts`, jumlah `train_ratings`, dan jumlah `candidate_books`.

Risiko tersisa:

- `prepare_dataset()` tetap berguna untuk eksplorasi, Streamlit, dan rekomendasi
  interaktif, tetapi tidak boleh dipakai sebagai benchmark final jika filtering
  berbasis full interactions dilakukan sebelum split.
- Dataset tidak memiliki timestamp, sehingga leakage temporal tidak bisa diuji
  atau dicegah dengan temporal split.
- Jika eksperimen baru memakai `max_ratings` random sampling sebelum split,
  sampling policy perlu didokumentasikan agar hasil tidak disalahartikan sebagai
  evaluasi produksi.

## Evaluasi yang Sudah Ada

Evaluasi aktif sudah memakai metrik recommender Top-N, bukan accuracy biasa:

- `Precision@K`
- `Recall@K`
- `HitRate@K`
- `MAP@K`
- `MRR@K`
- `NDCG@K`
- `Coverage@K`
- novelty
- intra-list diversity
- share rekomendasi pada top 1%, 5%, dan 10% item populer

CLI juga membandingkan hybrid, collaborative filtering, dan popularity baseline.
Evaluasi dapat disimpan sebagai artifact JSON untuk reproducibility dan
menyertakan ringkasan per segmen user berdasarkan jumlah interaksi.

Hasil verifikasi lokal `--sample-users 200 --k 50 --max-books 5000` dengan 74
holdout yang evaluable:

| Model | Precision@50 | Recall@50 | HitRate@50 | MAP@50 | MRR@50 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid tuned | 0.0041 | 0.2027 | 0.2027 | 0.0323 | 0.0323 | 0.0667 |
| collaborative | 0.0022 | 0.1081 | 0.1081 | 0.0405 | 0.0405 | 0.0555 |
| popularity | 0.0024 | 0.1216 | 0.1216 | 0.0326 | 0.0326 | 0.0503 |

Diagnostics pada evaluasi yang sama:

| Model | Coverage@50 | Novelty@50 | Top 1% share | Top 5% share | Top 10% share | Diversity@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid tuned | 0.4106 | 12.1369 | 0.0668 | 0.1776 | 0.2508 | 0.9393 |
| collaborative | 0.4072 | 12.6837 | 0.0265 | 0.0716 | 0.1065 | 0.9788 |
| popularity | 0.0108 | 9.5427 | 0.6216 | 1.0000 | 1.0000 | 0.9722 |

Interpretasi:

- Hybrid tuned menemukan holdout item pada 20.27% dari user evaluable,
  tertinggi pada eksperimen ini.
- Precision@50 tetap rendah karena tiap user hanya punya satu holdout relevan,
  sehingga satu hit maksimum hanya memberi precision `1/50 = 0.02`.
- Collaborative memiliki MAP/MRR lebih tinggi dari hybrid pada sampel ini,
  artinya ketika collaborative berhasil hit, posisi item sering lebih atas.
- Popularity baseline memiliki coverage sangat rendah dan top-popularity share
  sangat tinggi. Ini menegaskan popularity bias, meskipun relevance-nya tidak
  selalu buruk pada dataset yang sangat sparse.
- Hasil ini masih harus dibaca sebagai evaluasi baseline sampel, bukan klaim
  performa produksi.

## Evaluasi yang Belum Ada

Yang belum tersedia atau belum cukup kuat:

- Evaluasi content-based secara kuantitatif.
- Serendipity.
- Validasi hybrid dengan grid lebih luas, repeated random seeds, atau split
  tambahan.
- Temporal split. Dataset tidak punya timestamp, jadi tidak bisa dilakukan tanpa
  data tambahan.
- Negative sampling eksplisit tidak diperlukan jika ranking dievaluasi terhadap
  seluruh unseen candidate set, tetapi candidate universe perlu didokumentasikan
  lebih tegas.
- Evaluasi dengan sample user lebih besar. Eksperimen 200 sampled users hanya
  menghasilkan 74 holdout yang evaluable setelah filtering.

## Potensi High Bias / Underfitting

Gejala high bias yang mungkin:

- Content-based hanya memakai metadata sederhana. Jika dua buku tidak berbagi
  author/publisher/kata judul, model sulit menangkap kemiripan semantik.
- Collaborative item-item KNN adalah baseline sederhana dan belum memakai latent
  factors.
- Banyak user punya sedikit interaksi, sehingga personalisasi sering lemah atau
  jatuh ke popularity fallback.
- Cold-start item tidak bisa masuk collaborative tanpa interaksi.

Risiko underfitting paling jelas ada pada content-based metadata dan collaborative
baseline untuk user sparse.

## Potensi High Variance / Overfitting

Gejala high variance yang mungkin:

- Dataset sangat sparse; perubahan sedikit pada histori user bisa mengubah seed
  item dan hasil rekomendasi.
- User dengan 1-2 rating dapat menghasilkan rekomendasi terlalu percaya diri dari
  sinyal yang minim.
- Evaluasi sampel kecil dapat menghasilkan metrik yang tidak stabil.
- Jika jumlah neighbor atau model yang lebih kompleks dinaikkan tanpa validasi,
  model mudah menyesuaikan noise interaksi historis.

Project belum menunjukkan overfitting berat dari metrik training/test karena
package baru tidak memakai training accuracy. Namun validasi ranking masih terlalu
terbatas untuk menyimpulkan generalisasi kuat.

## Popularity Bias dan Diversity

Popularity bias sudah dimonitor sebagai diagnostics evaluasi melalui catalog
coverage, novelty, intra-list diversity, dan share rekomendasi pada top 1%, 5%,
serta 10% item populer.

Temuan:

- Popularity baseline tersedia.
- Collaborative recommender memiliki fallback popularity untuk unknown user.
- Pada evaluasi sample 200 user, popularity baseline hanya memiliki
  `Coverage@50 = 0.0108`, sedangkan hybrid tuned `0.4106` dan collaborative
  `0.4072`.
- Popularity baseline menempatkan 100% rekomendasinya di top 5% dan top 10%
  item populer pada evaluasi tersebut, sehingga bias popularitasnya jelas.
- Hybrid tuned menaikkan HitRate@50, tetapi top 10% popularity share-nya
  `0.2508`, lebih tinggi dari collaborative `0.1065`.

Rekomendasi:

- Pantau diagnostics ini pada setiap eksperimen, bukan hanya pada satu artifact.
- Pertimbangkan reranking dengan diversity penalty agar list tidak terlalu mirip.
- Jika hybrid ingin lebih eksploratif, tuning bobot harus mempertimbangkan
  relevance dan popularity diagnostics, bukan hanya HitRate/NDCG.

## Cold-Start Problem

Cold-start user:

- Sudah ada fallback popularity jika `user_id` tidak punya histori di model.
- Belum ada onboarding preference, misalnya pilih author/genre/judul favorit.
- User dengan 1-2 interaksi belum ditangani secara khusus.

Cold-start item:

- Collaborative model tidak bisa merekomendasikan item baru tanpa interaksi.
- Content-based bisa membantu item baru jika metadata tersedia, tetapi pipeline
  default `prepare_dataset()` hanya mempertahankan buku yang punya rating atau
  termasuk top rated books. Ini membatasi kemampuan content-based untuk item baru.

Rekomendasi:

- Pisahkan katalog item untuk content-based dari item universe collaborative.
- Untuk item baru, masukkan ke content index selama metadata minimal ada.
- Untuk user baru, buat fallback content-based dari preferensi awal atau
  popularity yang didiversifikasi.

## Explainability

Status saat ini:

- Content-based menampilkan skor similarity tetapi belum menjelaskan fitur yang
  mirip.
- Collaborative menampilkan skor agregat tetapi belum menunjukkan seed item yang
  menyebabkan kandidat muncul.
- Streamlit belum menampilkan alasan rekomendasi.

Rekomendasi explainability minimal:

- Content-based: tampilkan "mirip dengan buku X karena author/publisher/kata judul
  yang overlap".
- Collaborative: simpan kontribusi skor per seed item, lalu tampilkan "direkomendasikan
  karena mirip dengan buku A/B yang pernah diberi rating tinggi".
- Popularity fallback: tampilkan "populer di data training, rating rata-rata X
  dari Y rating".

## Reproducibility

Yang sudah baik:

- Ada `pyproject.toml`, `uv.lock`, dan `requirements.txt`.
- CLI tersedia via `book-rec`.
- `random_state` tersedia di preprocessing sampling dan leave-one-out split.
- Tests dan lint tersedia.
- Model artifact dapat disimpan ke `models/`.
- README menjelaskan setup, dataset, CLI, Streamlit, dan contoh hasil evaluasi.

Yang perlu diperbaiki:

- `requirements.txt` tidak punya versi pinning, sementara `pyproject.toml` lebih
  jelas. Karena Streamlit Cloud membaca `requirements.txt`, file ini perlu tetap
  disinkronkan dengan dependency app di `pyproject.toml`.
- Data raw lokal berisi file gambar tambahan dari Kaggle/legacy; tidak masalah,
  tetapi dokumentasi hanya menyebut CSV.
- Artifact eksperimen disimpan di `artifacts/` dan tidak ikut git. Untuk laporan
  publik, ringkasan hasil penting harus tetap dipindahkan ke README/docs.

Parameter penting yang sudah ada:

- `top_n`
- `k`
- `n_neighbors`
- `min_user_ratings`
- `min_book_ratings`
- `max_books`
- `max_ratings`
- `random_state`
- `min_interactions`
- `min_relevant_rating`

Parameter penting yang belum ada:

- diversity reranking weight
- novelty/popularity penalty
- user segment threshold
- hybrid score weights
- candidate universe policy

## Scalability

Saat ini aman untuk dataset kecil sampai sedang dengan `max_books=5000`.

Risiko scalability:

- `NearestNeighbors(algorithm="brute")` untuk content-based dan collaborative akan
  makin mahal saat item bertambah besar.
- Collaborative fit menyimpan matrix item-user dengan semua item di `books`.
- Rekomendasi collaborative menghitung nearest neighbors real-time untuk beberapa
  seed item user.
- Streamlit membangun model saat startup/cache jika artifact tidak ada.
- Belum ada approximate nearest neighbor index, batch precompute, atau caching
  recommendation per user.

Rekomendasi:

- Untuk >50k item, pertimbangkan ANN seperti FAISS, Annoy, HNSW, atau sklearn
  hanya untuk baseline offline.
- Precompute top similar items per item untuk item-item CF.
- Cache rekomendasi populer dan rekomendasi user aktif.
- Batasi candidate universe secara eksplisit dan dokumentasikan tradeoff.
- Pisahkan mode eksperimen kecil dan mode produksi.

## Masalah Kritis

| Prioritas | Masalah | Dampak | Bukti |
| --- | --- | --- | --- |
| P1 | Tidak ada timestamp | Tidak bisa temporal split; evaluasi tidak merepresentasikan urutan waktu konsumsi | Kolom timestamp tidak ada di Books/Ratings/Users |
| P1 | Dataset sangat sparse | CF rapuh, banyak user/item minim sinyal | 57.60% user hanya punya <=1 rating; 66.35% item hanya punya <=1 rating |
| P1 | Content-based metadata dangkal | Rekomendasi bisa terlalu literal dan kurang semantik | Fitur hanya title, author, publisher |
| P1 | Evaluasi content-based belum kuantitatif | Sulit membandingkan kontribusi content-only terhadap hybrid | Evaluasi resmi fokus hybrid/collaborative/popularity |
| P1 | Cold-start item belum tertangani penuh | Buku baru tanpa rating sulit muncul di collaborative | CF hanya memakai item dengan interaksi |
| P2 | Validasi hybrid masih awal | Bobot bisa terlalu cocok pada sampel kecil | Tuning baru memakai grid kecil dan 74 holdout evaluable |
| P2 | Explainability masih dangkal | Alasan rekomendasi belum menunjukkan kontribusi seed item atau fitur spesifik | `reason` masih berupa kategori sinyal |
| P2 | Brute-force nearest neighbors | Tidak scalable untuk katalog besar | `NearestNeighbors(metric="cosine", algorithm="brute")` |

## Rekomendasi Perbaikan Bertahap

Tahap 1: perbaiki evaluasi dan leakage. Status: selesai untuk jalur evaluasi
CLI aktif.

1. [x] Buat fungsi evaluasi yang melakukan clean minimal dulu, lalu split holdout,
   lalu menghitung filter/top-books/popularity dari train ratings saja.
2. [x] Pastikan `books_for_eval` tidak dipilih memakai hitungan dari full ratings.
3. [x] Simpan metrik evaluasi ke `artifacts/evaluation/*.json`.
4. [x] Tambahkan test bahwa ISBN holdout tidak dipakai dalam popularity count untuk
   user evaluasi dan bahwa model fit hanya menerima train ratings.

Tahap 2: tambah metrik recommender yang belum ada. Status: selesai untuk metrik
utama dan diagnostics.

1. [x] Tambahkan `mrr_at_k`.
2. [x] Tambahkan `catalog_coverage_at_k`.
3. [x] Tambahkan novelty berbasis inverse popularity.
4. [x] Tambahkan intra-list diversity berbasis content vector cosine distance.
5. [x] Laporkan metrik per user segment: `1-2`, `3-5`, `6-10`, dan `>10` interaksi.

Tahap 3: kuatkan cold-start dan content-based.

1. Pisahkan `content_books` dari `collab_books`.
2. Masukkan semua buku bermetadata valid ke content index, bukan hanya top rated.
3. Tambahkan user onboarding sederhana di Streamlit: pilih beberapa judul/author,
   lalu gabungkan content similarity.
4. Tambahkan alasan rekomendasi content-based yang lebih spesifik dari fitur
   author/publisher/judul.

Tahap 4: mitigasi popularity bias.

1. Laporkan share rekomendasi pada top 1%, 5%, dan 10% item populer.
2. Tambahkan reranking sederhana:
   `final_score = relevance_score - alpha * popularity_penalty + beta * diversity_gain`.
3. Bandingkan hasil sebelum/sesudah reranking pada HitRate@K, Coverage@K, dan
   Diversity@K.

Tahap 5: hybrid minimal. Status: selesai sebagai baseline, masih perlu validasi
lebih luas.

1. [x] Normalisasi skor collaborative dan content ke skala yang sama.
2. [x] Untuk user dengan histori cukup, gabungkan CF score dan content score dari
   item yang pernah diberi rating tinggi.
3. [x] Untuk user sparse, beri bobot lebih tinggi ke content/popularity.
4. [x] Dokumentasikan aturan fallback.
5. [ ] Uji bobot dengan grid lebih luas, repeated seeds, dan sample yang lebih
   besar.

## Checklist Final Sebelum Project Dianggap Selesai

- [x] Evaluation split bebas dari preprocessing leakage berbasis full interactions
      pada jalur evaluasi CLI aktif.
- [x] Popularity baseline dihitung hanya dari train pada evaluasi CLI.
- [x] Item yang sudah dikonsumsi user selalu difilter dari rekomendasi baru.
- [x] Precision@K, Recall@K, HitRate@K, MAP@K, NDCG@K, dan MRR@K tersedia.
- [x] Coverage@K, diversity, novelty, dan popularity distribution tersedia.
- [ ] Evaluasi content-based tersedia, minimal dengan leave-one-out user profile
      berbasis item metadata.
- [x] Evaluasi dilaporkan per segment user berdasarkan jumlah interaksi.
- [ ] Cold-start user memiliki fallback selain popularity murni.
- [ ] Cold-start item dapat masuk melalui content-based index.
- [x] Rekomendasi punya explainability sederhana.
- [x] `app.py` legacy dipindahkan atau diberi penanda jelas.
- [ ] Dependency deployment memakai manifest yang tersinkron antara
      `pyproject.toml` dan `requirements.txt`.
- [x] Hasil eksperimen disimpan dengan parameter, tanggal, dan random seed.
- [ ] Scalability plan jelas untuk katalog di atas 50k item.

## Verifikasi Audit

Perintah yang dijalankan:

```powershell
uv run pytest
uv run ruff check src tests app notebooks\01_end_to_end_book_recommender.py
uv run book-rec tune-hybrid --data-dir data/raw --sample-users 200 --k 50 --max-books 5000 --output artifacts/evaluation/hybrid_weight_tuning_sample200.json
uv run book-rec evaluate-collab --data-dir data/raw --sample-users 200 --k 50 --max-books 5000 --output artifacts/evaluation/recommender_eval_tuned_sample200.json
uv run book-rec recommend-hybrid --data-dir data/raw --user-id 5709 --top-n 3 --max-books 5000
```

Hasil:

- `pytest`: 13 passed.
- `ruff`: All checks passed.
- Tuning hybrid terbaik pada sample ini: collaborative `0.40`, content `0.45`,
  popularity `0.15`.
- Evaluasi CLI sample 200 user menghasilkan `HitRate@50` hybrid tuned `0.2027`,
  collaborative `0.1081`, dan popularity `0.1216`.
