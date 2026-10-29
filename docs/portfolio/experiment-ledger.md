# Register lengkap eksperimen

Delapan fase ini membentuk satu perjalanan penelitian: membuat jaringan dapat belajar, menguji penyebab hasil lemah, memperkuat pembanding, lalu menyelaraskan model dengan tujuan rekomendasi. Semua angka di halaman ini diturunkan dari delapan JSON bukti; model tidak dilatih ulang ketika register dibuat.

**121 catatan konfigurasi: 54 training deep neural, 18 training MF embedding dangkal, dan 49 kontrol/konfigurasi klasik; total 490 epoch training embedding.** Ini bukan 121 arsitektur unik. Konfigurasi yang sengaja diulang pada run berbeda tetap dicatat. Iterasi solver bias, diagnosis mini-batch, model yang dimuat ulang, dan evaluasi pertukaran riwayat tidak ditambahkan sebagai epoch atau fit benchmark baru.

Versi terstruktur lengkap: [experiment-ledger.json](experiment-ledger.json). Sumber penghitungan: [notebook 03](../../notebooks/03_neural_experiment_history.ipynb). Mekanisme belajar diperagakan di [notebook 02](../../notebooks/02_neural_book_recommender.ipynb); smoke test notebook 02 memakai subset berbeda dan tidak masuk benchmark ini.

| Fase | Pertanyaan | Catatan | Deep | MF | Kontrol/klasik | Epoch |
|---|---|---:|---:|---:|---:|---:|
| [1](#fase-1) | MLP awal: membuktikan mekanisme belajar | 1 | 1 | 0 | 0 | 5 |
| [2](#fase-2) | Bias + residual: apakah neural memberi koreksi berguna? | 4 | 3 | 0 | 1 | 12 |
| [3](#fase-3) | Ablation L2 dan output: mencari penyebab koreksi kecil | 8 | 6 | 0 | 2 | 24 |
| [4](#fase-4) | Warm-up 12 epoch: apakah berhenti terlalu cepat? | 2 | 2 | 0 | 0 | 24 |
| [5](#fase-5) | Dua split × tiga seed: menilai kestabilan prediksi rating | 18 | 12 | 0 | 6 | 75 |
| [6](#fase-6) | Bias dituning dan MF: apakah benchmark cukup kuat? | 50 | 0 | 18 | 32 | 108 |
| [7](#fase-7) | Objective ranking dan metadata: mengejar posisi buku relevan | 18 | 18 | 0 | 0 | 106 |
| [8](#fase-8) | Encoder riwayat dan cosine: menguji personalisasi yang berguna | 20 | 12 | 0 | 8 | 136 |
| **Total** | | **121** | **54** | **18** | **49** | **490** |

## Cara membaca angka

RMSE mengukur kesalahan menebak rating; lebih kecil lebih baik. N adalah NDCG@10 dan R adalah Recall@10; lebih besar lebih baik. Hit berarti satu karya target relevan muncul di sepuluh rekomendasi. Angka ranking di tabel memakai cohort semua pengguna relevan. Semua cohort, MAE/MRR, coverage dan metrik lain yang tersedia tetap disimpan tanpa pembulatan dalam JSON ledger.

Kolom epoch berbentuk **terbaik/selesai**, bukan jumlah pembaruan Adam. Tanda — berarti tidak ada training embedding. Selection memilih checkpoint atau konfigurasi; assessment/test melaporkan hasil sesudah pemilihan. Nama kandidat dapat diklik untuk membuka sumber bukti. JSON Pointer dalam ledger menunjukkan letak baris dan metrik secara tepat di file tersebut.

Skor antarprotokol tidak boleh dianggap satu papan peringkat. Fase 1–4 memakai validation lama yang kecil; fase 5 membuat pembagian internal dari train lama; fase 6–8 memakai ulang assessment internal. Rating memakai pasangan pengguna/ISBN yang dikenal, sedangkan ranking memakai target karya dan fallback yang dijelaskan per fase. Tabel ini tidak menyatakan test independen atau promosi model ke aplikasi.

Assessment pada tabel adalah evaluasi pada **fase asal** kandidat. Hasil model yang dipakai ulang di fase berikutnya tersimpan pada `later_evaluations`, sehingga riwayat lengkap dapat ditelusuri tanpa menggandakan jumlah fit. `selected_primary` berarti pilihan dalam protokol fase itu, bukan model terbaik universal atau model produksi.

<a id="fase-1"></a>

## Fase 1 MLP awal membuktikan mekanisme belajar

**Pertanyaan.** Bisakah embedding pengguna dan ISBN, forward propagation, MSE, backpropagation, dan Adam bekerja dalam proyek rekomendasi nyata?

**Perubahan.** Embedding 16 dimensi pengguna/buku, concatenation beserta interaksi perkalian, Dense ReLU 32/16, lalu sigmoid untuk rating ternormalisasi. Training memakai MSE + L2, bukan objective ranking.

**Hasil.** Lima epoch selesai; validation memilih epoch 2. Pada 135 pasangan test dengan kedua ID dikenal, RMSE neural 1,504167, sedangkan rata-rata rating pengguna 1,422509. Ranking neural dengan fallback menemukan 2 dari 272 target, pembanding item-based dan popularitas masing-masing 5.

**Batas kesimpulan.** Hanya 135/500 pasangan test memiliki kedua ID dikenal. Ranking 272 target dan RMSE 135 pasangan adalah kelompok berbeda; angka tidak boleh digabung. Berhasil melatih jaringan belum berarti rekomendasinya lebih baik.

**Langkah berikutnya.** Pisahkan pola rating umum melalui bias pengguna/buku, lalu minta jaringan mempelajari koreksinya.

**Pengaturan bersama.** MSE rating; embedding 16; hidden 32/16 ReLU; Adam lr 0,001; batch 1.024; L2 1e-5; seed 42; maksimal 8 epoch; patience 3.

**Sumber dan ruang lingkup.** [JSON fase 1](../evidence/neural_colab_t4_results.json); [notebook penjelasan](../../notebooks/03_neural_experiment_history.ipynb). Benchmark awal; checkpoint dipilih dari validation. Test dilihat pada run ini, sehingga pemakaiannya pada run berikutnya bukan test baru.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [mlp_original](../evidence/neural_colab_t4_results.json) | MLP; seed 42; L2 1e-5 | 2/5 | RMSE 1.621018 | RMSE 1.504167; 2/272 hit; N 0.004836 | kandidat tunggal; checkpoint terbaik validation |

<a id="fase-2"></a>

## Fase 2 Bias dan residual apakah neural memberi koreksi berguna

**Pertanyaan.** Apakah jaringan lebih mudah belajar setelah kecenderungan rating pengguna dan buku dijelaskan oleh baseline?

**Perubahan.** Bias dari train dibekukan. MLP LeakyReLU menambahkan koreksi dengan skala 0,25 atau 1; L2 1e-5 dibandingkan 1e-4. Bias-only menjadi kontrol tanpa training neural.

**Hasil.** Residual skala 0,25 dan L2 1e-4 dipilih: validation RMSE 1,746604. Pada 135 pasangan test dikenal, RMSE 1,404086, sedikit lebih buruk dari bias-only 1,400129. Ranking neural masih 2/272 target.

**Batas kesimpulan.** Validation hanya 140 pasangan dikenal. Test yang sama sudah dilihat pada fase 1. Dua residual yang tidak dipilih tidak diberi skor test; kosong berarti tidak dievaluasi, bukan nol. Label serving_candidate dalam bukti bukan deployment aplikasi.

**Langkah berikutnya.** Periksa apakah L2 atau inisialisasi output terlalu menekan koreksi neural, dengan kontrol analitik sederhana.

**Pengaturan bersama.** MSE rating raw + L2; embedding 16; hidden 32/16 LeakyReLU; Adam lr 0,001; batch 1.024; shrinkage pengguna/buku 0/20; seed 42; maksimal 10 epoch; patience 3.

**Sumber dan ruang lingkup.** [JSON fase 2](../evidence/neural_colab_residual_results.json); [notebook penjelasan](../../notebooks/03_neural_experiment_history.ipynb). Pengembangan pada validation lama dan test yang dipakai ulang; bukan konfirmasi independen.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [bias_only_u0_i20](../evidence/neural_colab_residual_results.json) | bias-only; shrink u/i 0/20; residual tidak aktif | — | RMSE 1.747573 | RMSE 1.400129 | kontrol |
| [residual_u0_i20_s025](../evidence/neural_colab_residual_results.json) | residual; L2 1e-05; skala 0.25; shrink u/i 0/20 | 1/4 | RMSE 1.746608 | tidak dievaluasi | neural tidak terpilih |
| [residual_u0_i20_s100](../evidence/neural_colab_residual_results.json) | residual; L2 1e-05; skala 1; shrink u/i 0/20 | 1/4 | RMSE 1.746786 | tidak dievaluasi | neural tidak terpilih |
| [residual_u0_i20_s025_l2e4](../evidence/neural_colab_residual_results.json) | residual; L2 0.0001; skala 0.25; shrink u/i 0/20 | 1/4 | RMSE 1.746604 | RMSE 1.404086; 2/272 hit; N 0.004836 | neural terpilih melalui validation |

<a id="fase-3"></a>

## Fase 3 Ablation L2 dan output mencari penyebab koreksi kecil

**Pertanyaan.** Apakah koreksi kecil terjadi karena regularisasi, output awal nol, atau jaringan memang tidak mampu menyesuaikan data?

**Perubahan.** L2 diuji dari 1e-4 sampai nol, disertai output acak kecil pada L2 1e-6. Kontrol bias-only dan bias + satu scalar yang dihitung dari train memeriksa apakah perbaikan sekadar pergeseran rata-rata.

**Hasil.** Keenam neural memilih epoch 1 dan berhenti setelah 4 epoch. Tanpa L2 menjadi kandidat terbaik: validation RMSE 1,746581; kontrol scalar 1,747000. Diagnosis terpisah pada 128 pasangan train menurunkan MSE 0,026764 menjadi 0,00004253 setelah 150 update.

**Batas kesimpulan.** Perbaikan validation sangat kecil pada 140 pasangan. Menghafal mini-batch membuktikan kapasitas optimisasi, bukan generalisasi. Diagnosis 150 update tidak masuk 24 epoch benchmark atau hitungan kandidat. Beberapa konfigurasi sengaja mengulang fase 2.

**Langkah berikutnya.** Periksa kemungkinan sinyal baru muncul terlambat dengan menjalankan dua kandidat selama 12 epoch penuh.

**Pengaturan bersama.** Residual skala 0,25; MSE + L2 per baris; embedding 16; hidden 32/16 LeakyReLU; Adam lr 0,001; batch 1.024; shrinkage 0/20; seed 42; maksimal 10 epoch; patience 3.

**Sumber dan ruang lingkup.** [JSON fase 3](../evidence/neural_colab_ablation_results.json); [notebook penjelasan](../../notebooks/03_neural_experiment_history.ipynb). Diagnosis validation yang sama; tidak membuka test dan tidak mengganti model aplikasi.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [bias_only](../evidence/neural_colab_ablation_results.json) | bias-only; shrink u/i 0/20; residual tidak aktif | — | RMSE 1.747573 | tidak dievaluasi | kontrol |
| [bias_plus_train_fitted_scalar](../evidence/neural_colab_ablation_results.json) | bias + scalar fit dari train; shrink u/i 0/20 | — | RMSE 1.747000 | tidak dievaluasi | kontrol |
| [residual_l2_1e4_zero_head](../evidence/neural_colab_ablation_results.json) | residual; L2 0.0001; head nol | 1/4 | RMSE 1.746604 | tidak dievaluasi | neural tidak terpilih |
| [residual_l2_1e5_zero_head](../evidence/neural_colab_ablation_results.json) | residual; L2 1e-05; head nol | 1/4 | RMSE 1.746608 | tidak dievaluasi | neural tidak terpilih |
| [residual_l2_1e6_zero_head](../evidence/neural_colab_ablation_results.json) | residual; L2 1e-06; head nol | 1/4 | RMSE 1.746599 | tidak dievaluasi | neural tidak terpilih |
| [residual_l2_1e7_zero_head](../evidence/neural_colab_ablation_results.json) | residual; L2 1e-07; head nol | 1/4 | RMSE 1.746584 | tidak dievaluasi | neural tidak terpilih |
| [residual_l2_0_zero_head](../evidence/neural_colab_ablation_results.json) | residual; L2 0; head nol | 1/4 | RMSE 1.746581 | tidak dievaluasi | neural terpilih melalui validation |
| [residual_l2_1e6_tiny_random_head](../evidence/neural_colab_ablation_results.json) | residual; L2 1e-06; head acak kecil | 1/4 | RMSE 1.746591 | tidak dievaluasi | neural tidak terpilih |

<a id="fase-4"></a>

## Fase 4 Latihan 12 epoch apakah berhenti terlalu cepat

**Pertanyaan.** Apakah early stopping sebelumnya menghentikan pembelajaran sebelum koreksi neural menjadi berguna?

**Perubahan.** Residual L2 nol dan 1e-7 dilatih tepat 12 epoch, early stopping dimatikan. Checkpoint tetap dipilih dari validation; kontrol fase 3 tidak dilatih ulang.

**Hasil.** Kedua model menyelesaikan 12 epoch, tetapi checkpoint terbaik tetap epoch 1. Train membaik sementara validation memburuk; kandidat tanpa L2 mempertahankan validation RMSE 1,746581.

**Batas kesimpulan.** Hasil ini menolak penjelasan 'kurang lama' untuk dua konfigurasi dalam batas percobaan ini; bukan bukti bahwa semua arsitektur pasti gagal jika dilatih lebih lama. Tidak ada evaluasi test baru.

**Langkah berikutnya.** Perluas pemeriksaan ke dua pembagian pengembangan dan tiga seed agar kesimpulan tidak bergantung pada satu pembagian kecil.

**Pengaturan bersama.** Residual skala 0,25; embedding 16; hidden 32/16; Adam lr 0,001; batch 1.024; shrinkage 0/20; seed 42; tepat 12 epoch; output awal nol.

**Sumber dan ruang lingkup.** [JSON fase 4](../evidence/neural_colab_warmup_results.json); [notebook penjelasan](../../notebooks/03_neural_experiment_history.ipynb). Tindak lanjut adaptif setelah melihat ablation, memakai validation lama; tidak membuka test.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [residual_l2_0_zero_head_fixed12](../evidence/neural_colab_warmup_results.json) | residual; L2 0; head nol | 1/12 | RMSE 1.746581 | tidak dievaluasi | neural terpilih melalui validation |
| [residual_l2_1e7_zero_head_fixed12](../evidence/neural_colab_warmup_results.json) | residual; L2 1e-07; head nol | 1/12 | RMSE 1.746584 | tidak dievaluasi | neural tidak terpilih |

<a id="fase-5"></a>

## Fase 5 Dua split kali tiga seed menilai kestabilan prediksi rating

**Pertanyaan.** Apakah MLP dan residual tetap bekerja ketika pembagian pengembangan dan inisialisasi diubah?

**Perubahan.** Train lama dibagi menjadi fit, selection, dan assessment internal; split 17/29, seed 42/2026/7. MLP L2 1e-5 dibanding residual tanpa L2, serta kontrol global mean, user mean, dan bias per split.

**Hasil.** Dua belas neural selesai dalam 75 epoch. Rata-rata assessment RMSE MLP antarseed 1,633008/1,597189 pada split 17/29, dibanding residual 1,920037/1,837842. MLP utama dipilih seed 42/7 berdasarkan selection.

**Batas kesimpulan.** Ini pembagian ulang data train yang sudah digunakan, bukan test independen. Seed mengubah inisialisasi dan urutan mini-batch, bukan memberi tiga dataset independen. Kemenangan terhadap kontrol bias awal belum membuktikan keunggulan terhadap bias yang dituning.

**Langkah berikutnya.** Berikan pembanding klasik kesempatan tuning yang jelas, lalu ukur ranking katalog penuh selain RMSE.

**Pengaturan bersama.** MSE rating; embedding 16; hidden 32/16; Adam lr 0,001; batch 1.024; maksimal 12 epoch; patience 3; split 17/29; seed 42/2026/7; fallback rating sama.

**Sumber dan ruang lingkup.** [JSON fase 5](../evidence/neural_colab_multiseed_results.json); [notebook penjelasan](../../notebooks/03_neural_experiment_history.ipynb). Assessment pengembangan internal; seluruh 12 jaringan dikunci sebelum assessment run ini. Original validation/test tidak digunakan.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [split17_mlp_seed42](../evidence/neural_colab_multiseed_results.json) | split 17; seed 42; mlp; L2 1e-05 | 2/5 | RMSE 1.618725 | RMSE 1.627100 | utama split melalui selection |
| [split17_residual_seed42](../evidence/neural_colab_multiseed_results.json) | split 17; seed 42; residual; L2 0 | 4/7 | RMSE 1.863247 | RMSE 1.916731 | seed/arsitektur pembanding |
| [split17_mlp_seed2026](../evidence/neural_colab_multiseed_results.json) | split 17; seed 2026; mlp; L2 1e-05 | 2/5 | RMSE 1.628013 | RMSE 1.635077 | seed/arsitektur pembanding |
| [split17_residual_seed2026](../evidence/neural_colab_multiseed_results.json) | split 17; seed 2026; residual; L2 0 | 1/4 | RMSE 1.865228 | RMSE 1.922754 | seed/arsitektur pembanding |
| [split17_mlp_seed7](../evidence/neural_colab_multiseed_results.json) | split 17; seed 7; mlp; L2 1e-05 | 2/5 | RMSE 1.622358 | RMSE 1.636848 | seed/arsitektur pembanding |
| [split17_residual_seed7](../evidence/neural_colab_multiseed_results.json) | split 17; seed 7; residual; L2 0 | 4/7 | RMSE 1.864050 | RMSE 1.920624 | seed/arsitektur pembanding |
| [split29_mlp_seed42](../evidence/neural_colab_multiseed_results.json) | split 29; seed 42; mlp; L2 1e-05 | 2/5 | RMSE 1.711005 | RMSE 1.591829 | seed/arsitektur pembanding |
| [split29_residual_seed42](../evidence/neural_colab_multiseed_results.json) | split 29; seed 42; residual; L2 0 | 5/8 | RMSE 1.929414 | RMSE 1.834486 | seed/arsitektur pembanding |
| [split29_mlp_seed2026](../evidence/neural_colab_multiseed_results.json) | split 29; seed 2026; mlp; L2 1e-05 | 3/6 | RMSE 1.710937 | RMSE 1.598703 | seed/arsitektur pembanding |
| [split29_residual_seed2026](../evidence/neural_colab_multiseed_results.json) | split 29; seed 2026; residual; L2 0 | 6/9 | RMSE 1.929594 | RMSE 1.835114 | seed/arsitektur pembanding |
| [split29_mlp_seed7](../evidence/neural_colab_multiseed_results.json) | split 29; seed 7; mlp; L2 1e-05 | 3/6 | RMSE 1.708185 | RMSE 1.601035 | utama split melalui selection |
| [split29_residual_seed7](../evidence/neural_colab_multiseed_results.json) | split 29; seed 7; residual; L2 0 | 5/8 | RMSE 1.930751 | RMSE 1.843927 | seed/arsitektur pembanding |
| [split17_global_mean](../evidence/neural_colab_multiseed_results.json) | split 17; global_mean | — | RMSE 1.774471 | RMSE 1.791449 | kontrol |
| [split17_user_mean](../evidence/neural_colab_multiseed_results.json) | split 17; user_mean | — | RMSE 1.914803 | RMSE 1.970989 | kontrol |
| [split17_bias](../evidence/neural_colab_multiseed_results.json) | split 17; bias | — | RMSE 1.865513 | RMSE 1.922917 | kontrol |
| [split29_global_mean](../evidence/neural_colab_multiseed_results.json) | split 29; global_mean | — | RMSE 1.884707 | RMSE 1.757864 | kontrol |
| [split29_user_mean](../evidence/neural_colab_multiseed_results.json) | split 29; user_mean | — | RMSE 1.988832 | RMSE 1.888518 | kontrol |
| [split29_bias](../evidence/neural_colab_multiseed_results.json) | split 29; bias | — | RMSE 1.933780 | RMSE 1.837243 | kontrol |

<a id="fase-6"></a>

## Fase 6 Bias dituning dan MF apakah benchmark cukup kuat

**Pertanyaan.** Apakah keuntungan MLP bertahan terhadap bias yang dioptimalkan bersama dan matrix factorization, dan apakah RMSE rendah berarti ranking baik?

**Perubahan.** Grid shrinkage bias pengguna/buku {0,5,20,100} menghasilkan 32 konfigurasi. MF 16 dimensi memakai tiga lambda × tiga seed × dua split, total 18 training. Enam MLP fase 5 dimuat ulang, tidak dilatih lagi.

**Hasil.** Bias terpilih 5/20 pada split 17 dan 5/5 pada split 29; lambda MF 0,01 pada keduanya. Assessment RMSE bias 1,634786/1,579396, MLP utama 1,627100/1,601035, MF utama 1,636338/1,595686. MLP utama hanya menemukan 33/1.776 dan 21/1.806 target; popularitas 75/1.776 dan 66/1.806.

**Batas kesimpulan.** Ranking tidak dipakai memilih checkpoint pada fase ini: pemilihannya masih RMSE. MF adalah embedding dangkal, bukan deep MLP. Iterasi solver bias bukan epoch neural. Assessment internal sudah dilihat dan tidak memberi konfirmasi independen.

**Langkah berikutnya.** Selaraskan target latihan dan pemilihan checkpoint dengan ranking; uji BPR serta tambahan metadata.

**Pengaturan bersama.** Bias: squared error + ridge, maksimum 1.000 iterasi, toleransi 1e-10. MF: MSE rating + penalti sampled-pair; embedding 16; Adam lr 0,005; batch 1.024; maksimum 30 epoch; patience 5. Lambda dipilih dari mean RMSE selection tiga seed.

**Sumber dan ruang lingkup.** [JSON fase 6](../evidence/neural_colab_comparator_results.json); [notebook penjelasan](../../notebooks/03_neural_experiment_history.ipynb). Pengembangan adaptif pada assessment fase 5 yang dipakai ulang; original validation/test tidak digunakan.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [split17_bias_u0_i0](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 0/0 | — | RMSE 1.840170 | tidak dievaluasi | grid tidak terpilih; 70 iterasi solver |
| [split17_bias_u0_i5](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 0/5 | — | RMSE 1.818091 | tidak dievaluasi | grid tidak terpilih; 90 iterasi solver |
| [split17_bias_u0_i20](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 0/20 | — | RMSE 1.819285 | tidak dievaluasi | grid tidak terpilih; 36 iterasi solver |
| [split17_bias_u0_i100](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 0/100 | — | RMSE 1.845626 | tidak dievaluasi | grid tidak terpilih; 18 iterasi solver |
| [split17_bias_u5_i0](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 5/0 | — | RMSE 1.631921 | tidak dievaluasi | grid tidak terpilih; 51 iterasi solver |
| [split17_bias_u5_i5](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 5/5 | — | RMSE 1.609442 | tidak dievaluasi | grid tidak terpilih; 32 iterasi solver |
| [split17_bias_u5_i20](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 5/20 | — | RMSE 1.603481 | RMSE 1.634786; 31/1776 hit; N 0.008038 | bias terpilih; 19 iterasi solver |
| [split17_bias_u5_i100](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 5/100 | — | RMSE 1.616103 | tidak dievaluasi | grid tidak terpilih; 13 iterasi solver |
| [split17_bias_u20_i0](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 20/0 | — | RMSE 1.676916 | tidak dievaluasi | grid tidak terpilih; 25 iterasi solver |
| [split17_bias_u20_i5](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 20/5 | — | RMSE 1.655690 | tidak dievaluasi | grid tidak terpilih; 20 iterasi solver |
| [split17_bias_u20_i20](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 20/20 | — | RMSE 1.649582 | tidak dievaluasi | grid tidak terpilih; 13 iterasi solver |
| [split17_bias_u20_i100](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 20/100 | — | RMSE 1.662300 | tidak dievaluasi | grid tidak terpilih; 10 iterasi solver |
| [split17_bias_u100_i0](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 100/0 | — | RMSE 1.716790 | tidak dievaluasi | grid tidak terpilih; 14 iterasi solver |
| [split17_bias_u100_i5](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 100/5 | — | RMSE 1.695872 | tidak dievaluasi | grid tidak terpilih; 11 iterasi solver |
| [split17_bias_u100_i20](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 100/20 | — | RMSE 1.689610 | tidak dievaluasi | grid tidak terpilih; 10 iterasi solver |
| [split17_bias_u100_i100](../evidence/neural_colab_comparator_results.json) | split 17; shrink u/i 100/100 | — | RMSE 1.702796 | tidak dievaluasi | grid tidak terpilih; 8 iterasi solver |
| [split29_bias_u0_i0](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 0/0 | — | RMSE 1.905820 | tidak dievaluasi | grid tidak terpilih; 76 iterasi solver |
| [split29_bias_u0_i5](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 0/5 | — | RMSE 1.893377 | tidak dievaluasi | grid tidak terpilih; 90 iterasi solver |
| [split29_bias_u0_i20](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 0/20 | — | RMSE 1.901352 | tidak dievaluasi | grid tidak terpilih; 36 iterasi solver |
| [split29_bias_u0_i100](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 0/100 | — | RMSE 1.929930 | tidak dievaluasi | grid tidak terpilih; 18 iterasi solver |
| [split29_bias_u5_i0](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 5/0 | — | RMSE 1.715694 | tidak dievaluasi | grid tidak terpilih; 52 iterasi solver |
| [split29_bias_u5_i5](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 5/5 | — | RMSE 1.706251 | RMSE 1.579396; 10/1806 hit; N 0.002184 | bias terpilih; 32 iterasi solver |
| [split29_bias_u5_i20](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 5/20 | — | RMSE 1.711387 | tidak dievaluasi | grid tidak terpilih; 19 iterasi solver |
| [split29_bias_u5_i100](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 5/100 | — | RMSE 1.732908 | tidak dievaluasi | grid tidak terpilih; 13 iterasi solver |
| [split29_bias_u20_i0](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 20/0 | — | RMSE 1.760622 | tidak dievaluasi | grid tidak terpilih; 25 iterasi solver |
| [split29_bias_u20_i5](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 20/5 | — | RMSE 1.751097 | tidak dievaluasi | grid tidak terpilih; 20 iterasi solver |
| [split29_bias_u20_i20](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 20/20 | — | RMSE 1.756155 | tidak dievaluasi | grid tidak terpilih; 13 iterasi solver |
| [split29_bias_u20_i100](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 20/100 | — | RMSE 1.778117 | tidak dievaluasi | grid tidak terpilih; 10 iterasi solver |
| [split29_bias_u100_i0](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 100/0 | — | RMSE 1.801030 | tidak dievaluasi | grid tidak terpilih; 14 iterasi solver |
| [split29_bias_u100_i5](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 100/5 | — | RMSE 1.790784 | tidak dievaluasi | grid tidak terpilih; 11 iterasi solver |
| [split29_bias_u100_i20](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 100/20 | — | RMSE 1.794979 | tidak dievaluasi | grid tidak terpilih; 10 iterasi solver |
| [split29_bias_u100_i100](../evidence/neural_colab_comparator_results.json) | split 29; shrink u/i 100/100 | — | RMSE 1.816923 | tidak dievaluasi | grid tidak terpilih; 8 iterasi solver |
| [split17_mf_lam1e4_seed42](../evidence/neural_colab_comparator_results.json) | split 17; seed 42; λ 0.0001 | 1/6 | RMSE 1.623685 | RMSE 1.649604 | grid tidak terpilih |
| [split17_mf_lam1e4_seed2026](../evidence/neural_colab_comparator_results.json) | split 17; seed 2026; λ 0.0001 | 1/6 | RMSE 1.619221 | RMSE 1.646871 | grid tidak terpilih |
| [split17_mf_lam1e4_seed7](../evidence/neural_colab_comparator_results.json) | split 17; seed 7; λ 0.0001 | 1/6 | RMSE 1.618662 | RMSE 1.638987 | grid tidak terpilih |
| [split17_mf_lam1e3_seed42](../evidence/neural_colab_comparator_results.json) | split 17; seed 42; λ 0.001 | 1/6 | RMSE 1.623043 | RMSE 1.649025 | grid tidak terpilih |
| [split17_mf_lam1e3_seed2026](../evidence/neural_colab_comparator_results.json) | split 17; seed 2026; λ 0.001 | 1/6 | RMSE 1.618796 | RMSE 1.646392 | grid tidak terpilih |
| [split17_mf_lam1e3_seed7](../evidence/neural_colab_comparator_results.json) | split 17; seed 7; λ 0.001 | 1/6 | RMSE 1.618304 | RMSE 1.638589 | grid tidak terpilih |
| [split17_mf_lam1e2_seed42](../evidence/neural_colab_comparator_results.json) | split 17; seed 42; λ 0.01 | 1/6 | RMSE 1.618393 | RMSE 1.644750; 2/1776 hit; N 0.000518 | lambda terpilih; seed pembanding |
| [split17_mf_lam1e2_seed2026](../evidence/neural_colab_comparator_results.json) | split 17; seed 2026; λ 0.01 | 1/6 | RMSE 1.615680 | RMSE 1.642520; 3/1776 hit; N 0.000645 | lambda terpilih; seed pembanding |
| [split17_mf_lam1e2_seed7](../evidence/neural_colab_comparator_results.json) | split 17; seed 7; λ 0.01 | 1/6 | RMSE 1.615551 | RMSE 1.636338; 9/1776 hit; N 0.001811 | MF utama |
| [split29_mf_lam1e4_seed42](../evidence/neural_colab_comparator_results.json) | split 29; seed 42; λ 0.0001 | 1/6 | RMSE 1.715523 | RMSE 1.595283 | grid tidak terpilih |
| [split29_mf_lam1e4_seed2026](../evidence/neural_colab_comparator_results.json) | split 29; seed 2026; λ 0.0001 | 1/6 | RMSE 1.721891 | RMSE 1.601725 | grid tidak terpilih |
| [split29_mf_lam1e4_seed7](../evidence/neural_colab_comparator_results.json) | split 29; seed 7; λ 0.0001 | 1/6 | RMSE 1.713368 | RMSE 1.598492 | grid tidak terpilih |
| [split29_mf_lam1e3_seed42](../evidence/neural_colab_comparator_results.json) | split 29; seed 42; λ 0.001 | 1/6 | RMSE 1.715011 | RMSE 1.594740 | grid tidak terpilih |
| [split29_mf_lam1e3_seed2026](../evidence/neural_colab_comparator_results.json) | split 29; seed 2026; λ 0.001 | 1/6 | RMSE 1.721274 | RMSE 1.601093 | grid tidak terpilih |
| [split29_mf_lam1e3_seed7](../evidence/neural_colab_comparator_results.json) | split 29; seed 7; λ 0.001 | 1/6 | RMSE 1.712942 | RMSE 1.598127 | grid tidak terpilih |
| [split29_mf_lam1e2_seed42](../evidence/neural_colab_comparator_results.json) | split 29; seed 42; λ 0.01 | 1/6 | RMSE 1.711634 | RMSE 1.590970; 2/1806 hit; N 0.000547 | lambda terpilih; seed pembanding |
| [split29_mf_lam1e2_seed2026](../evidence/neural_colab_comparator_results.json) | split 29; seed 2026; λ 0.01 | 1/6 | RMSE 1.716965 | RMSE 1.596797; 7/1806 hit; N 0.001555 | lambda terpilih; seed pembanding |
| [split29_mf_lam1e2_seed7](../evidence/neural_colab_comparator_results.json) | split 29; seed 7; λ 0.01 | 1/6 | RMSE 1.710114 | RMSE 1.595686; 5/1806 hit; N 0.001142 | MF utama |

<a id="fase-7"></a>

## Fase 7 Objective ranking dan metadata mengejar posisi buku relevan

**Pertanyaan.** Apakah BPR memperbaiki ranking dibanding MSE, dan apakah judul/penulis/penerbit menambah informasi?

**Perubahan.** Dua menara MLP dengan embedding 32 membandingkan MSE ID, BPR ID, dan BPR ID + TF-IDF metadata 1.024 fitur. Unit kandidat menjadi karya kanonik. Checkpoint dan seed utama dipilih memakai NDCG@10 selection.

**Hasil.** BPR ID utama menemukan 72/1.776 dan 72/1.806 target, mendekati popularitas 75/1.776 dan 66/1.806. Keenam pasangan seed BPR ID memiliki NDCG lebih tinggi dari MSE ID pada cohort positif, tetapi hanya dua interval seluruhnya di atas nol. Metadata memberi dua selisih positif dan empat negatif terhadap BPR ID.

**Batas kesimpulan.** MSE memakai rerata rating per karya; BPR memakai karya dengan salah satu rating >=8. Supervisi dan jumlah update ikut berubah, sehingga ini bukan isolasi loss saja. Dibanding MLP lama, arsitektur, unit ISBN/karya, fallback dan aturan checkpoint juga berbeda. Interval BPR utama versus popularitas melintasi nol pada kedua split; itu bukan bukti kesetaraan.

**Langkah berikutnya.** Daftar BPR ID masih sangat mirip popularitas. Uji representasi yang membaca riwayat positif secara langsung, dengan cosine pada riwayat sama sebagai pembanding.

**Pengaturan bersama.** Embedding 32; user tower Dense64 ReLU→32; item tower ID+proyeksi metadata→Dense64 ReLU→32; dot/sqrt32 + bias karya + global logit tetap. Adam lr 0,001; batch 1.024; lambda 1e-5; maksimum 12 epoch; patience 3. Metadata nol pada kedua arm ID; bobot awal berpasangan.

**Sumber dan ruang lingkup.** [JSON fase 7](../evidence/neural_colab_ranking_results.json); [notebook penjelasan](../../notebooks/04_neural_ranking_experiments.ipynb). Assessment pengembangan yang dipakai ulang; semua 18 fit dan hash checkpoint dikunci sebelum assessment run ini. Tidak ada promosi aplikasi.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [split17_mse_id_seed42](../evidence/neural_colab_ranking_results.json) | split 17; seed 42; mse_id | 2/5 | N 0.011027; R 0.021727 | 48/1776 hit; N 0.012673 | checkpoint terpilih; seed pembanding |
| [split17_bpr_id_seed42](../evidence/neural_colab_ranking_results.json) | split 17; seed 42; bpr_id | 1/4 | N 0.017489; R 0.033426 | 72/1776 hit; N 0.020088 | neural utama per arm |
| [split17_bpr_metadata_seed42](../evidence/neural_colab_ranking_results.json) | split 17; seed 42; bpr_metadata | 3/6 | N 0.016860; R 0.031198 | 77/1776 hit; N 0.020964 | checkpoint terpilih; seed pembanding |
| [split17_mse_id_seed2026](../evidence/neural_colab_ranking_results.json) | split 17; seed 2026; mse_id | 1/4 | N 0.013197; R 0.028969 | 57/1776 hit; N 0.015551 | checkpoint terpilih; seed pembanding |
| [split17_bpr_id_seed2026](../evidence/neural_colab_ranking_results.json) | split 17; seed 2026; bpr_id | 10/12 | N 0.016987; R 0.033983 | 75/1776 hit; N 0.020529 | checkpoint terpilih; seed pembanding |
| [split17_bpr_metadata_seed2026](../evidence/neural_colab_ranking_results.json) | split 17; seed 2026; bpr_metadata | 6/9 | N 0.016717; R 0.032312 | 70/1776 hit; N 0.019746 | checkpoint terpilih; seed pembanding |
| [split17_mse_id_seed7](../evidence/neural_colab_ranking_results.json) | split 17; seed 7; mse_id | 1/4 | N 0.013869; R 0.026184 | 57/1776 hit; N 0.017638 | neural utama per arm |
| [split17_bpr_id_seed7](../evidence/neural_colab_ranking_results.json) | split 17; seed 7; bpr_id | 3/6 | N 0.017101; R 0.032869 | 68/1776 hit; N 0.019451 | checkpoint terpilih; seed pembanding |
| [split17_bpr_metadata_seed7](../evidence/neural_colab_ranking_results.json) | split 17; seed 7; bpr_metadata | 3/6 | N 0.017070; R 0.032869 | 67/1776 hit; N 0.019132 | neural utama per arm |
| [split29_mse_id_seed42](../evidence/neural_colab_ranking_results.json) | split 29; seed 42; mse_id | 1/4 | N 0.014457; R 0.029977 | 59/1806 hit; N 0.017201 | neural utama per arm |
| [split29_bpr_id_seed42](../evidence/neural_colab_ranking_results.json) | split 29; seed 42; bpr_id | 3/6 | N 0.018941; R 0.032805 | 71/1806 hit; N 0.021169 | checkpoint terpilih; seed pembanding |
| [split29_bpr_metadata_seed42](../evidence/neural_colab_ranking_results.json) | split 29; seed 42; bpr_metadata | 3/6 | N 0.018685; R 0.032240 | 68/1806 hit; N 0.020483 | checkpoint terpilih; seed pembanding |
| [split29_mse_id_seed2026](../evidence/neural_colab_ranking_results.json) | split 29; seed 2026; mse_id | 1/4 | N 0.013067; R 0.025452 | 69/1806 hit; N 0.016890 | checkpoint terpilih; seed pembanding |
| [split29_bpr_id_seed2026](../evidence/neural_colab_ranking_results.json) | split 29; seed 2026; bpr_id | 4/7 | N 0.019743; R 0.036199 | 72/1806 hit; N 0.020434 | neural utama per arm |
| [split29_bpr_metadata_seed2026](../evidence/neural_colab_ranking_results.json) | split 29; seed 2026; bpr_metadata | 4/7 | N 0.019112; R 0.035633 | 70/1806 hit; N 0.019878 | neural utama per arm |
| [split29_mse_id_seed7](../evidence/neural_colab_ranking_results.json) | split 29; seed 7; mse_id | 1/4 | N 0.013502; R 0.025452 | 67/1806 hit; N 0.017832 | checkpoint terpilih; seed pembanding |
| [split29_bpr_id_seed7](../evidence/neural_colab_ranking_results.json) | split 29; seed 7; bpr_id | 3/6 | N 0.019415; R 0.034502 | 59/1806 hit; N 0.018532 | checkpoint terpilih; seed pembanding |
| [split29_bpr_metadata_seed7](../evidence/neural_colab_ranking_results.json) | split 29; seed 7; bpr_metadata | 3/6 | N 0.018921; R 0.033371 | 62/1806 hit; N 0.019213 | checkpoint terpilih; seed pembanding |

<a id="fase-8"></a>

## Fase 8 Encoder riwayat dan cosine menguji personalisasi yang berguna

**Pertanyaan.** Apakah membaca kumpulan buku yang disukai meningkatkan kualitas, dan apakah jaringan neural lebih berguna dari kemiripan klasik pada riwayat sama?

**Perubahan.** History-BPR dan full-softmax memakai embedding masukan/keluaran terpisah, pooling sum/sqrt(count), Dense64 tanh→32, serta bias karya. Target dikeluarkan dari riwayat latihan. Pengguna training perlu >=2 positif; inferensi boleh satu positif. Grid cosine k50/200 × shrink0/10 menjadi kontrol.

**Hasil.** Cosine terpilih k50/shrink10 pada kedua split menemukan 191/1.776 dan 198/1.806 target. Softmax utama 109/1.776 dan 66/1.806; history-BPR 70/1.776 dan 67/1.806. Semua neural utama masih lebih rendah dari cosine terpilih pada NDCG cohort positif. Pertukaran riwayat membuat hit softmax utama split 17 turun 109→48, sedangkan split 29 tetap 66.

**Batas kesimpulan.** Dibanding ID-BPR lama, representasi, populasi target training, dan regularisasi berubah. Perbandingan BPR/softmax dalam fase ini lebih terkontrol. Donor mengubah isi dan panjang riwayat sekaligus; diagnosis bukan bukti kausal isi saja. Lima cohort saling tumpang tindih. Assessment tetap pengembangan yang dipakai ulang.

**Langkah berikutnya.** Jadikan cosine riwayat sebagai benchmark kuat untuk pengujian berikutnya; perbaikan neural harus konsisten dan diperiksa pada konfirmasi yang belum dipakai merancang eksperimen. Ini arah tindak lanjut, bukan hasil training tambahan atau deployment.

**Pengaturan bersama.** Embedding 32; encoder tanpa parameter ID pengguna; Adam lr 0,001; batch 1.024; reg 1e-5 × mean semua parameter trainable²; maksimum 20 epoch; patience 4. BPR/softmax memakai target dan bobot awal sama. Cosine: Cij/sqrt(ci*cj) × Cij/(Cij+shrink), diagonal nol, top-k per baris sumber; seri skor dipecah popularitas fit lalu leksikal, neural leksikal.

**Sumber dan ruang lingkup.** [JSON fase 8](../evidence/neural_colab_history_results.json); [notebook penjelasan](../../notebooks/05_neural_history_experiments.ipynb). Dua belas neural dan delapan cosine dikunci sebelum assessment. Empat diagnosis pertukaran riwayat tidak melatih ulang. Tidak ada promosi otomatis ke aplikasi.

| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |
|---|---|---:|---|---|---|
| [split17_history_bpr_seed42](../evidence/neural_colab_history_results.json) | split 17; seed 42; history_bpr | 6/10 | N 0.017179; R 0.035097 | 70/1776 hit; N 0.020275 | checkpoint terpilih; seed pembanding |
| [split17_history_softmax_seed42](../evidence/neural_colab_history_results.json) | split 17; seed 42; history_softmax | 4/8 | N 0.016904; R 0.030641 | 72/1776 hit; N 0.022626 | checkpoint terpilih; seed pembanding |
| [split17_history_bpr_seed2026](../evidence/neural_colab_history_results.json) | split 17; seed 2026; history_bpr | 3/7 | N 0.017925; R 0.035097 | 72/1776 hit; N 0.020573 | checkpoint terpilih; seed pembanding |
| [split17_history_softmax_seed2026](../evidence/neural_colab_history_results.json) | split 17; seed 2026; history_softmax | 19/20 | N 0.028223; R 0.057382 | 109/1776 hit; N 0.033191 | neural utama per arm |
| [split17_history_bpr_seed7](../evidence/neural_colab_history_results.json) | split 17; seed 7; history_bpr | 6/10 | N 0.019409; R 0.038440 | 70/1776 hit; N 0.021220 | neural utama per arm |
| [split17_history_softmax_seed7](../evidence/neural_colab_history_results.json) | split 17; seed 7; history_softmax | 20/20 | N 0.026674; R 0.049582 | 89/1776 hit; N 0.030436 | checkpoint terpilih; seed pembanding |
| [split29_history_bpr_seed42](../evidence/neural_colab_history_results.json) | split 29; seed 42; history_bpr | 16/20 | N 0.022360; R 0.041855 | 67/1806 hit; N 0.019279 | neural utama per arm |
| [split29_history_softmax_seed42](../evidence/neural_colab_history_results.json) | split 29; seed 42; history_softmax | 2/6 | N 0.020085; R 0.037330 | 66/1806 hit; N 0.018624 | neural utama per arm |
| [split29_history_bpr_seed2026](../evidence/neural_colab_history_results.json) | split 29; seed 2026; history_bpr | 8/12 | N 0.020121; R 0.036765 | 64/1806 hit; N 0.019181 | checkpoint terpilih; seed pembanding |
| [split29_history_softmax_seed2026](../evidence/neural_colab_history_results.json) | split 29; seed 2026; history_softmax | 3/7 | N 0.019574; R 0.035068 | 69/1806 hit; N 0.020363 | checkpoint terpilih; seed pembanding |
| [split29_history_bpr_seed7](../evidence/neural_colab_history_results.json) | split 29; seed 7; history_bpr | 6/10 | N 0.022008; R 0.038462 | 65/1806 hit; N 0.019605 | checkpoint terpilih; seed pembanding |
| [split29_history_softmax_seed7](../evidence/neural_colab_history_results.json) | split 29; seed 7; history_softmax | 2/6 | N 0.019911; R 0.037896 | 73/1806 hit; N 0.020273 | checkpoint terpilih; seed pembanding |
| [split17_itemcos_k50_shrink0](../evidence/neural_colab_history_results.json) | split 17; k 50; shrink 0 | — | N 0.048083; R 0.075209 | 149/1776 hit; N 0.053777 | cosine grid pembanding |
| [split17_itemcos_k50_shrink10](../evidence/neural_colab_history_results.json) | split 17; k 50; shrink 10 | — | N 0.065337; R 0.102507 | 191/1776 hit; N 0.067522 | cosine terpilih |
| [split17_itemcos_k200_shrink0](../evidence/neural_colab_history_results.json) | split 17; k 200; shrink 0 | — | N 0.047727; R 0.073538 | 140/1776 hit; N 0.053393 | cosine grid pembanding |
| [split17_itemcos_k200_shrink10](../evidence/neural_colab_history_results.json) | split 17; k 200; shrink 10 | — | N 0.065046; R 0.100836 | 192/1776 hit; N 0.067985 | cosine grid pembanding |
| [split29_itemcos_k50_shrink0](../evidence/neural_colab_history_results.json) | split 29; k 50; shrink 0 | — | N 0.057481; R 0.084276 | 142/1806 hit; N 0.050029 | cosine grid pembanding |
| [split29_itemcos_k50_shrink10](../evidence/neural_colab_history_results.json) | split 29; k 50; shrink 10 | — | N 0.074762; R 0.115950 | 198/1806 hit; N 0.066156 | cosine terpilih |
| [split29_itemcos_k200_shrink0](../evidence/neural_colab_history_results.json) | split 29; k 200; shrink 0 | — | N 0.056909; R 0.083145 | 145/1806 hit; N 0.051236 | cosine grid pembanding |
| [split29_itemcos_k200_shrink10](../evidence/neural_colab_history_results.json) | split 29; k 200; shrink 10 | — | N 0.074436; R 0.114819 | 206/1806 hit; N 0.068261 | cosine grid pembanding |

## Diagnosis dan pembanding yang tidak menambah jumlah kandidat

Diagnosis kapasitas fase 3 memakai 128 pasangan train selama 150 update, dengan MSE 0,026764486 → 0,000042529. Ini latihan tambahan khusus diagnosis pada batch kecil, dicatat terpisah dari 72 fit embedding benchmark. Fase 4 hanya merujuk hasil diagnosis tersebut. Empat evaluasi donor riwayat fase 8 mempertahankan bobot model dan tidak melakukan training lagi.

Skor popularitas, rata-rata rating, dan item-based pada benchmark awal tetap penting sebagai referensi; aturan hitungan notebook 03 tidak menjadikannya tambahan baris kandidat terdaftar. Enam MLP yang dimuat ulang pada fase 6, serta model historis yang dinilai ulang pada fase 7–8, tetap merujuk fit asal. Tes perangkat lunak dan preflight GPU membuktikan aspek operasional, bukan meningkatkan jumlah eksperimen kualitas model.

## Mereproduksi register

Dari root repository, jalankan perintah berikut. Hanya Python standard library yang diperlukan; tidak mengakses Colab, dataset privat, atau TensorFlow.

```bash
python scripts/build_portfolio_ledger.py
python scripts/build_portfolio_ledger.py --check
```

Pemeriksaan memastikan keanggotaan dan jumlah per fase, total epoch, nama kandidat unik dalam register, kecocokan metrik dengan JSON Pointer sumber, dan kesamaan dokumen yang dihasilkan. SHA-256 setiap sumber tercatat dalam JSON ledger. Tabel Markdown membulatkan enam desimal; JSON mempertahankan presisi bukti sumber.
