# Audit Recommender System

Tanggal audit: 2026-07-08

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
- Root `app.py` lama dipindahkan ke `legacy/app.py` agar entry point aktif tidak
  membingungkan.
- Test ditambah untuk memastikan holdout tidak ikut membentuk candidate catalog
  train, memverifikasi metrik diagnostics, dan menjaga roundtrip artifact hybrid.
- Evaluasi lokal `--sample-users 200 --k 50 --max-books 5000` menunjukkan
  `HitRate@50` hybrid `0.1757`, collaborative `0.1081`, dan popularity `0.1216`.

Catatan: bagian audit di bawah tetap menggambarkan temuan awal. Beberapa item
yang masih relevan adalah content-based evaluation, cold-start yang lebih kuat,
explainability, tuning hybrid, dan scalability plan.

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
| File lama/arsip | `legacy/`, `docs/LEGACY_REPORT_2023.md`, dan `app.py` |

Catatan: `app.py` di root tampak sebagai UI lama berbasis TensorFlow dan folder
`aset/`. Jalur ini tidak konsisten dengan package baru, membutuhkan dependency
yang tidak ada di `pyproject.toml`, dan sebaiknya dipindahkan ke `legacy/` atau
diberi penanda jelas bahwa bukan entry point aktif.

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
9. Evaluasi collaborative memakai leave-one-out split dan membandingkan model
   collaborative dengan popularity baseline.
10. Streamlit app menampilkan rekomendasi buku mirip dan rekomendasi per `user_id`.

Representasi saat ini:

| Komponen | Representasi |
| --- | --- |
| User | `user_id` numerik dari `Ratings.csv`/`Users.csv` |
| Item | Buku direpresentasikan oleh `isbn`; metadata item dari title, author, publisher, cover URL |
| Interaksi user-item | Rating eksplisit `book_rating` skala 1-10 setelah rating 0 dibuang |
| Output content-based | Top-N buku mirip dengan satu judul/ISBN query, dengan skor similarity |
| Output collaborative | Top-N buku untuk user, dengan skor agregasi similarity item-item |
| Output evaluasi | Ranking metrics Top-K, bukan accuracy classification |

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
- Explainability belum dibuat eksplisit, meskipun secara teknis bisa dijelaskan
  dari seed item yang menyumbang skor.

Warning: collaborative filtering layak sebagai baseline ringan, tetapi dataset
sangat sparse. Performa Top-N harus diuji dengan lebih banyak user dan metrik
coverage/diversity.

## Audit Hybrid Recommendation

Belum ada hybrid recommender aktif. Project menjalankan content-based dan
collaborative sebagai dua pendekatan terpisah.

Ini bukan bug jika scope project adalah membandingkan dua baseline, tetapi jika
targetnya sistem rekomendasi yang lebih siap pakai, hybrid perlu ditambahkan
secara bertahap.

Rekomendasi hybrid realistis:

- User baru: content-based dari preferensi onboarding atau popularity baseline.
- User lama: collaborative sebagai sumber utama.
- Item baru: content-based karena belum punya interaksi.
- Skor akhir: normalisasi skor collaborative, content similarity, popularity, dan
  diversity penalty sebelum digabung.
- Jangan menggabungkan skor mentah tanpa normalisasi karena skala similarity,
  popularity, dan skor model berbeda.

## Potensi Data Leakage

Temuan P0:

1. `prepare_dataset()` melakukan filter sparsity dan pemilihan `max_books` dari
   seluruh rating sebelum evaluasi leave-one-out. Pada CLI, `evaluate_collab_model`
   memanggil `load_prepared_dataset()` lebih dulu, baru menjalankan
   `build_leave_one_out_split()`. Artinya item universe dan top-books selection
   dapat dipengaruhi interaksi yang nanti menjadi holdout. Ini leakage evaluasi
   berbasis interaksi/popularity.
2. `select_most_rated_books()` memilih buku berdasarkan `ratings["isbn"].value_counts()`
   dari seluruh dataset prepared. Untuk benchmark offline yang fair, popularity
   atau item eligibility berbasis interaksi seharusnya dihitung dari train split.
3. Opsi `max_ratings` melakukan random sampling rating sebelum split. Jika dipakai
   untuk evaluasi, sampling ini juga terjadi sebelum train-test split dan bisa
   mengubah distribusi test dengan cara yang tidak eksplisit.

Temuan yang sudah baik:

- Pada evaluasi collaborative, model di-fit memakai `train_ratings`, bukan full
  ratings.
- Popularity baseline pada evaluasi dihitung dari recommender yang di-fit pada
  train ratings.
- Item yang sudah ada di train history user dikeluarkan dari rekomendasi
  collaborative dan popularity baseline.
- Holdout row dihapus dari train oleh `build_leave_one_out_split()`.

P0 ini tidak berarti semua angka evaluasi tidak berguna, tetapi angka tersebut
belum sepenuhnya bebas dari bias item-selection. Perbaikannya tidak perlu rewrite:
buat jalur evaluasi yang split lebih dulu, lalu fit preprocessing/model dari
train interactions.

## Evaluasi yang Sudah Ada

Evaluasi aktif sudah memakai metrik recommender Top-N, bukan accuracy biasa:

- `Precision@K`
- `Recall@K`
- `HitRate@K`
- `MAP@K`
- `NDCG@K`

CLI juga membandingkan collaborative filtering dengan popularity baseline.

Hasil verifikasi lokal:

| Model | Precision@50 | Recall@50 | HitRate@50 | MAP@50 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| collaborative | 0.0020 | 0.1000 | 0.1000 | 0.0033 | 0.0200 |
| popularity | 0.0010 | 0.0500 | 0.0500 | 0.0167 | 0.0250 |

Interpretasi:

- Collaborative menemukan holdout item pada 10% dari 20 user evaluasi.
- Popularity baseline menemukan holdout item pada 5% dari 20 user.
- Precision@50 rendah karena tiap user hanya punya satu item relevan, sehingga
  satu hit maksimum hanya memberi precision `1/50 = 0.02`.
- MAP/NDCG popularity lebih tinggi pada sampel ini ketika popularity berhasil
  hit di posisi lebih atas.

## Evaluasi yang Belum Ada

Yang belum tersedia atau belum cukup kuat:

- Evaluasi content-based secara kuantitatif.
- MRR@K.
- Coverage@K sebagai metrik resmi.
- Intra-list diversity.
- Novelty.
- Popularity distribution dari item rekomendasi.
- Serendipity.
- Evaluasi per segment user, misalnya user dengan 1-2 interaksi vs user aktif.
- Temporal split. Dataset tidak punya timestamp, jadi tidak bisa dilakukan tanpa
  data tambahan.
- Negative sampling eksplisit tidak diperlukan jika ranking dievaluasi terhadap
  seluruh unseen candidate set, tetapi candidate universe perlu didokumentasikan
  lebih tegas.
- Evaluasi dengan sample user lebih besar. Notebook memakai 20 user; CLI default
  100 user. Ini masih kecil untuk klaim kuat.

Probe tambahan pada 200 user, Top-10 collaborative:

- 2,000 rekomendasi total.
- 1,261 unique item direkomendasikan.
- Sample catalog coverage@10: `25.22%` dari 5,000 item.
- Share rekomendasi pada top 1% item populer: `2.25%`.
- Share rekomendasi pada top 10% item populer: `10.85%`.
- Mean pairwise Jaccard antar list 100 user pertama: `0.0044`.

Angka probe ini cukup sehat untuk sampel kecil, tetapi belum menjadi metrik resmi
karena belum ada di pipeline evaluasi dan belum diuji lint/test.

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

Popularity bias sudah disadari di README, tetapi belum dimonitor sebagai metrik
pipeline.

Temuan:

- Popularity baseline tersedia.
- Collaborative recommender memiliki fallback popularity untuk unknown user.
- Belum ada Coverage@K, novelty, intra-list diversity, atau distribusi popularitas
  rekomendasi sebagai output evaluasi resmi.
- Probe 200 user menunjukkan coverage sampel cukup baik, tetapi ini belum
  menggantikan evaluasi resmi.

Rekomendasi:

- Tambahkan `catalog_coverage_at_k`.
- Tambahkan `intra_list_diversity_at_k` berbasis cosine distance content vectors.
- Tambahkan novelty sederhana: `-log2(popularity(item) / total_interactions)`.
- Simpan distribusi popularity bucket untuk item yang direkomendasikan.
- Pertimbangkan reranking dengan diversity penalty agar list tidak terlalu mirip.

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

- `app.py` root masih mengarah ke dependency/aset lama dan bisa membingungkan.
- Hasil eksperimen belum disimpan otomatis sebagai file metrik.
- Parameter evaluasi dan model belum disimpan bersama hasil eksperimen.
- `requirements.txt` tidak punya versi pinning, sementara `pyproject.toml` lebih
  jelas. Pilih satu sumber dependency utama untuk deployment.
- Data raw lokal berisi file gambar tambahan dari Kaggle/legacy; tidak masalah,
  tetapi dokumentasi hanya menyebut CSV.

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
| P0 | Preprocessing berbasis interaksi dilakukan sebelum evaluation split | Metrik holdout bisa bias karena item universe/top-books dipengaruhi data holdout | `prepare_dataset()` dipanggil sebelum `build_leave_one_out_split()` |
| P0 | Pemilihan top books memakai seluruh ratings sebelum split | Popularity/item eligibility bocor ke evaluasi | `select_most_rated_books()` memakai `value_counts()` seluruh rating |
| P1 | Tidak ada timestamp | Tidak bisa temporal split; evaluasi tidak merepresentasikan urutan waktu konsumsi | Kolom timestamp tidak ada di Books/Ratings/Users |
| P1 | Dataset sangat sparse | CF rapuh, banyak user/item minim sinyal | 57.60% user hanya punya <=1 rating; 66.35% item hanya punya <=1 rating |
| P1 | Evaluasi belum mencakup coverage/diversity/novelty | Risiko popularity bias dan filter bubble tidak termonitor | Metrik resmi hanya ranking relevance |
| P1 | Content-based metadata dangkal | Rekomendasi bisa terlalu literal dan kurang semantik | Fitur hanya title, author, publisher |
| P1 | Cold-start item belum tertangani penuh | Buku baru tanpa rating sulit muncul di collaborative | CF hanya memakai item dengan interaksi |
| P1 | `app.py` root adalah jalur legacy yang tidak konsisten | Membingungkan entry point dan dependency | Mengimpor TensorFlow dan `./aset/*`, sementara package baru tidak |
| P2 | Explainability belum eksplisit | Sulit menjelaskan alasan rekomendasi ke user | Output hanya skor dan metadata |
| P2 | Belum ada hybrid recommender | Sistem belum menggabungkan kekuatan content dan collaborative | Dua tab/model berjalan terpisah |
| P2 | Brute-force nearest neighbors | Tidak scalable untuk katalog besar | `NearestNeighbors(metric="cosine", algorithm="brute")` |

## Rekomendasi Perbaikan Bertahap

Tahap 1: perbaiki evaluasi dan leakage.

1. Buat fungsi evaluasi yang melakukan clean minimal dulu, lalu split holdout,
   lalu menghitung filter/top-books/popularity dari train ratings saja.
2. Pastikan `books_for_eval` tidak dipilih memakai hitungan dari full ratings.
3. Simpan metrik evaluasi ke `artifacts/evaluation/*.csv` atau `.json`.
4. Tambahkan test bahwa ISBN holdout tidak dipakai dalam popularity count untuk
   user evaluasi dan bahwa model fit hanya menerima train ratings.

Tahap 2: tambah metrik recommender yang belum ada.

1. Tambahkan `mrr_at_k`.
2. Tambahkan `catalog_coverage_at_k`.
3. Tambahkan novelty berbasis inverse popularity.
4. Tambahkan intra-list diversity berbasis content vector cosine distance.
5. Laporkan metrik per user segment: `1-2`, `3-5`, `6-10`, dan `>10` interaksi.

Tahap 3: kuatkan cold-start dan content-based.

1. Pisahkan `content_books` dari `collab_books`.
2. Masukkan semua buku bermetadata valid ke content index, bukan hanya top rated.
3. Tambahkan user onboarding sederhana di Streamlit: pilih beberapa judul/author,
   lalu gabungkan content similarity.
4. Tambahkan alasan rekomendasi content-based.

Tahap 4: mitigasi popularity bias.

1. Laporkan share rekomendasi pada top 1%, 5%, dan 10% item populer.
2. Tambahkan reranking sederhana:
   `final_score = relevance_score - alpha * popularity_penalty + beta * diversity_gain`.
3. Bandingkan hasil sebelum/sesudah reranking pada HitRate@K, Coverage@K, dan
   Diversity@K.

Tahap 5: hybrid minimal.

1. Normalisasi skor collaborative dan content ke skala yang sama.
2. Untuk user dengan histori cukup, gabungkan CF score dan content score dari
   item yang pernah diberi rating tinggi.
3. Untuk user sparse, beri bobot lebih tinggi ke content/popularity.
4. Dokumentasikan aturan fallback.

## Checklist Final Sebelum Project Dianggap Selesai

- [ ] Evaluation split bebas dari preprocessing leakage berbasis full interactions.
- [ ] Popularity baseline dihitung hanya dari train pada setiap eksperimen.
- [ ] Item yang sudah dikonsumsi user selalu difilter dari rekomendasi baru.
- [ ] Precision@K, Recall@K, HitRate@K, MAP@K, NDCG@K, dan MRR@K tersedia.
- [ ] Coverage@K, diversity, novelty, dan popularity distribution tersedia.
- [ ] Evaluasi content-based tersedia, minimal dengan leave-one-out user profile
      berbasis item metadata.
- [ ] Evaluasi dilaporkan per segment user berdasarkan jumlah interaksi.
- [ ] Cold-start user memiliki fallback selain popularity murni.
- [ ] Cold-start item dapat masuk melalui content-based index.
- [ ] Rekomendasi punya explainability sederhana.
- [ ] `app.py` legacy dipindahkan atau diberi penanda jelas.
- [ ] Dependency deployment memakai satu sumber yang konsisten.
- [ ] Hasil eksperimen disimpan dengan parameter, tanggal, dan random seed.
- [ ] Scalability plan jelas untuk katalog di atas 50k item.

## Verifikasi Audit

Perintah yang dijalankan:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests app notebooks\01_end_to_end_book_recommender.py
.\.venv\Scripts\book-rec.exe evaluate-collab --data-dir data/raw --sample-users 20 --k 50 --max-books 5000
```

Hasil:

- `pytest`: 6 passed.
- `ruff`: All checks passed.
- Evaluasi CLI: hasil sama dengan README untuk sample 20 user.
