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
# # Belajar menyusun rekomendasi: objective ranking dan metadata buku
#
# **Pertanyaan eksperimen:** apakah neural menghasilkan daftar buku yang lebih
# relevan jika dilatih untuk menyusun urutan, dan apakah informasi buku membantu?
# Kita menguji kedua pertanyaan itu melalui tiga kelompok model pada pembagian
# data yang sama. Semua percobaan, termasuk yang kalah, ditampilkan di sini.
#
# Pada percobaan sebelumnya, neural belajar memperkirakan angka rating. Buku
# dengan prediksi rating tertinggi kemudian menjadi rekomendasi. Kali ini, satu
# kelompok tetap belajar rating sebagai pembanding. Dua kelompok lain belajar
# menaikkan skor buku yang disukai dibandingkan buku pembanding yang belum
# tercatat dalam riwayat fit pengguna. Salah satunya juga menerima informasi judul, penulis, dan
# penerbit. Metrik urutan rekomendasi memilih checkpoint setiap kelompok.
#
# Notebook ini membaca **dua JSON bukti tersimpan**. Restart & Run All tidak
# melatih model, menghubungi Colab, membuka CSV privat, atau memerlukan GPU dan
# TensorFlow. Riwayat enam run sebelumnya tetap tersedia di
# [notebook 03](03_neural_experiment_history.ipynb). Demonstrasi dasar lima konsep
# neural tersedia di [notebook 02](02_neural_book_recommender.ipynb).
#
# **Batas utama:** ini pengembangan lanjutan setelah hasil lama diketahui.
# Assessment internal yang sama dipakai ulang; hasilnya bukan konfirmasi pada
# test independen baru. Pilihan pada run ini dikunci sebelum assessment, tetapi
# penguncian itu tidak menghapus pemakaian assessment pada run sebelumnya.

# %%
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

root_override = os.environ.get("BOOK_RECOMMENDER_PROJECT_ROOT")
search_root = Path(root_override).resolve() if root_override else Path.cwd().resolve()
ROOT = next(
    (
        path
        for path in (search_root, *search_root.parents)
        if (path / "docs/evidence/neural_colab_ranking_results.json").is_file()
    ),
    None,
)
if ROOT is None:
    raise FileNotFoundError("Buka dari checkout proyek atau set BOOK_RECOMMENDER_PROJECT_ROOT.")
EVIDENCE = ROOT / "docs/evidence"
EVIDENCE_FILES = {
    "results": "neural_colab_ranking_results.json",
    "manifest": "neural_colab_ranking_run_manifest.json",
}
evidence = {
    key: json.loads((EVIDENCE / filename).read_text(encoding="utf-8"))
    for key, filename in EVIDENCE_FILES.items()
}
results = evidence["results"]
manifest = evidence["manifest"]
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 250)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.precision", 8)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.2})
display(
    pd.DataFrame(
        [
            {
                "sumber": filename,
                "SHA256 (newline dinormalisasi)": hashlib.sha256(
                    (EVIDENCE / filename).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                ).hexdigest(),
            }
            for filename in EVIDENCE_FILES.values()
        ]
    )
)

# %% [markdown]
# ## Hasil utama terlebih dahulu
#
# **Ada peningkatan terhadap MLP lama, tetapi keunggulan atas popularitas belum
# terbukti.** BPR dengan ID yang dipilih melalui selection menemukan target bagi
# 72 dari 1.776 pengguna pada split 17 dan 72 dari 1.806 pada split 29.
# MLP lama menemukan 33 dan 21 target; popularitas menemukan 75 dan 66.
# Jadi neural sekarang mendekati popularitas, dengan arah selisih yang berbeda
# pada kedua split. Interval selisih Recall/NDCG BPR utama terhadap popularitas
# masih melintasi nol pada kedua split; skor yang dekat bukan bukti kesetaraan.
#
# Perbaikan BPR atas MLP lama juga terlihat pada cohort riwayat positif bersama,
# sehingga bukan semata-mata hasil fallback baru. Namun, pembandingan lama/baru
# tetap mengubah arsitektur, unit karya, dan pemilihan checkpoint. Pada kontrol
# baru yang dipasangkan per seed, BPR ID memberi NDCG lebih tinggi daripada MSE
# ID di seluruh enam pasangan; hanya dua interval berada seluruhnya di atas nol.
# Ini mendukung arah eksperimen ranking, dengan ketidakpastian yang masih nyata.
#
# **Metadata belum memberi manfaat konsisten:** BPR dengan metadata menemukan
# 67 dan 70 target pada model utamanya. Pada enam pasangan seed dengan BPR ID,
# selisih NDCG metadata positif pada dua pasangan dan negatif pada empat.
# TF-IDF judul/penulis/penerbit dalam konfigurasi ini belum menunjukkan tambahan
# yang dapat diulang. Kesimpulan ini tetap pengembangan internal pada assessment
# yang sudah dipakai ulang, belum konfirmasi dari test independen.
#
# Tabel ini memakai model utama yang dipilih lewat selection, bukan model dengan
# hasil assessment tertinggi. “Target ditemukan” berarti karya yang disembunyikan
# muncul di 10 rekomendasi. Kolom riwayat positif memakai pengguna yang mempunyai
# sedikitnya satu karya dengan rating minimal 8 di fit, sehingga hasil neural
# dapat dilihat tanpa kontribusi fallback bagi pengguna tanpa positif.

# %%
quick_rows = []
for ranking in results["ranking_results"]:
    chosen_models = [
        *ranking["primary_models"].items(),
        ("MLP lama", ranking["old_primary_mlp"]),
        ("Popularitas", "popularity"),
    ]
    for label, name in chosen_models:
        overall = ranking["metrics"][name]["all_relevant"]
        positive = ranking["metrics"][name]["fit_positive_history"]
        quick_rows.append(
            {
                "split": ranking["split_seed"],
                "model utama": label,
                "kandidat terpilih": name,
                "target ditemukan (semua)": f"{overall['hits_at_10']}/{overall['users']}",
                "Recall@10 semua (%)": 100 * overall["recall_at_10"],
                "NDCG@10 semua": overall["ndcg_at_10"],
                "target ditemukan (riwayat positif)": f"{positive['hits_at_10']}/{positive['users']}",
                "Recall@10 riwayat positif (%)": 100 * positive["recall_at_10"],
                "NDCG@10 riwayat positif": positive["ndcg_at_10"],
            }
        )
display(pd.DataFrame(quick_rows))
display(
    Markdown(
        "Angka lebih tinggi berarti lebih banyak target ditemukan atau target "
        "mendapat posisi lebih baik. Seluruh seed, ketidakpastian selisih, dan "
        "batas perbandingan dijelaskan di bawah; tabel ringkas ini belum cukup "
        "untuk menyatakan kemenangan yang akan berlaku pada data baru."
    )
)

# %% [markdown]
# ## Apa yang dibandingkan?
#
# | Kelompok | Masukan jaringan | Sasaran latihan |
# |---|---|---|
# | `mse_id` | ID pengguna dan ID karya buku | Mendekatkan prediksi dengan rating |
# | `bpr_id` | ID pengguna dan ID karya buku | Menaikkan skor buku disukai terhadap kandidat pembanding |
# | `bpr_metadata` | ID pengguna, ID karya, judul, penulis, penerbit | Sasaran ranking yang sama dengan `bpr_id` |
#
# Ketiganya memakai jaringan dua menara dengan dimensi yang sama. Menara
# pengguna mengolah embedding pengguna melalui Dense ReLU 64 lalu Dense 32.
# Menara buku mengolah embedding karya 32 dan proyeksi metadata 32 melalui
# Dense ReLU 64 lalu Dense 32. Skor menggabungkan dot product kedua keluaran,
# dibagi akar 32, bias karya yang dilatih, dan offset global tetap. Input
# metadata pada dua kelompok ID berisi nol; parameter jaringan tetap memiliki
# bentuk yang sama, tetapi jalur metadata yang aktif hanya ada pada kelompok
# metadata. Ini bukan klaim kapasitas fungsional ketiganya identik.
#
# MSE memakai `sigmoid(skor)` dan target `(rata-rata rating karya - 1) / 9`.
# BPR memakai `softplus(skor pembanding - skor buku positif)`. Menurunkan loss
# BPR mendorong skor buku positif lebih tinggi. Regularisasi menyertai kedua
# objective; angka loss dari objective berbeda tidak boleh dibandingkan sebagai
# ukuran kualitas yang sama.
#
# Perbandingan `mse_id` dan `bpr_id` mengubah **objective sekaligus supervisi**:
# MSE melihat rating karya yang diamati, sedangkan BPR memakai karya dengan
# salah satu rating edisinya minimal 8 dan kandidat pembanding yang disampling.
# Karena itu, selisih hasil bukan efek loss saja. Perbandingan `bpr_metadata`
# dengan `bpr_id` lebih langsung menguji tambahan fitur pada protokol BPR sama.
# MLP lama juga merupakan arsitektur dan unit buku berbeda; hasilnya menjadi
# benchmark historis, bukan kontrol yang hanya berbeda satu komponen.
# Checkpoint MLP lama dipilih berdasarkan RMSE, sedangkan model baru memakai
# NDCG@10. Selisih lintas model lama/baru juga mencakup perubahan cara seleksi.
#
# Popularitas dan MLP lama ikut dinilai pada pengguna dan katalog yang sama.
# Hasil rekomendasi MLP lama dimuat dari bukti terdahulu. Popularitas dihitung
# ulang dari fit, lalu kecocokannya dengan hasil tersimpan diperiksa. Keduanya
# tidak dihitung sebagai training baru.

# %% [markdown]
# ## Data, metadata, dan buku pembanding
#
# Setiap karya menggabungkan edisi dengan judul/penulis kanonik yang sama.
# Rating fit diringkas per pengguna/karya. Seluruh edisi karya yang tercatat
# dalam riwayat fit dikeluarkan dari kandidat rekomendasi. Pemilihan buku pembanding BPR
# dilakukan seragam dari karya yang **tidak ada dalam seluruh riwayat fit**
# pengguna. Label selection/assessment tidak menentukan sampel pembanding.
#
# Buku yang belum diamati **belum tentu tidak disukai**. Ia hanya kandidat
# pembanding untuk latihan. Rating 0 juga tidak diubah menjadi bukti tidak suka.
# Pengguna tanpa karya positif pada fit memakai fallback popularitas yang sama
# di semua kelompok. Hasil ditampilkan pada seluruh pengguna relevan dan pada
# subset bersama yang mempunyai setidaknya satu karya positif di fit.
# MLP historis mempertahankan hasil dan kebijakan fallback lamanya; ia tidak
# diam-diam diganti dengan popularitas pada cohort tanpa positif. Karena itu,
# cohort dengan riwayat positif bersama lebih tepat untuk melihat perubahan
# neural terpisah dari pengaruh kebijakan fallback baru.
#
# Teks judul, penulis, dan penerbit dikumpulkan pada tingkat karya dari metadata
# fit. TF-IDF dibatasi paling banyak 1.024 fitur dan hanya di-fit dari katalog
# fit. Metadata ini tidak memasukkan rating atau target holdout. Pemakaian
# metadata pada katalog yang sudah dikenal belum membuktikan kemampuan
# merekomendasikan karya baru yang tidak tersedia saat training.
#
# Audit matriks menemukan **13 dari 4.364 karya** pada masing-masing split
# memiliki fitur TF-IDF seluruhnya nol: seluruh term karya tersebut berada di
# luar vocabulary 1.024 fitur. Pada batas frekuensi vocabulary terdapat term
# dengan frekuensi sama; versi library berbeda dapat memilih anggota berbeda
# dari kelompok tie itu. Reproduksi matriks menggunakan **vocabulary runtime
# yang disimpan**, bukan mengandalkan pemilihan term otomatis yang baru.

# %%
display(pd.DataFrame(results["dataset"]["training_work_summaries"]))
split_rows = []
for split in results["dataset"]["splits"]:
    # Tabel datar mempertahankan field skema input lama tanpa memuat CSV privat.
    split_rows.append(
        {
            key: value
            for key, value in split.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    )
display(pd.DataFrame(split_rows))
display(
    pd.DataFrame(
        [
            {
                "split": int(split),
                "karya tanpa fitur TF-IDF": count,
                **manifest["verification"]["metadata_vocabulary_cutoffs"][split],
            }
            for split, count in manifest["verification"]["metadata_zero_feature_works"].items()
        ]
    )
)
assert results["dataset"]["reused_prior_assessment"]
assert not results["original_test_used"]
assert not results["original_validation_used"]
assert not results["model_promotion_to_application"]

# %% [markdown]
# ## Protokol dan pilihan checkpoint
#
# Dua split internal (17 dan 29), tiga seed training (42, 2026, 7), dan tiga
# kelompok menghasilkan **18 percobaan training yang direncanakan**. Adam
# memakai learning rate 0,001 dan batch 1.024. Batas setiap percobaan 12 epoch,
# dengan patience 3. Seleksi checkpoint memakai NDCG@10 atas katalog penuh
# pada selection; angka rating pada assessment tidak dipakai untuk seleksi.
#
# Semua kandidat dan pilihan model utama per kelompok ditetapkan sebelum
# assessment run ini dibuka. Seluruh seed tetap dilaporkan. Model tidak dipilih
# ulang berdasarkan assessment yang terlihat lebih baik. Jika anggaran
# menghentikan sebagian percobaan, status dan alasannya harus tampak; hasil
# seleksi tidak boleh dipromosikan sebagai perbandingan lengkap 18 model.

# %%
configuration = results["configuration"]
display(
    pd.DataFrame(
        [
            {"pengaturan": key, "nilai": value}
            for key, value in configuration.items()
            if key != "candidates"
        ]
    )
)
display(
    pd.DataFrame(
        [
            {
                "status run": results["status"],
                "selesai (Asia/Jakarta)": pd.Timestamp(results["completed_utc"])
                .tz_convert("Asia/Jakarta")
                .isoformat(),
                "semua kandidat selesai": results["all_candidates_completed"],
                "assessment dilakukan": results["assessment_performed"],
                "durasi eksperimen (detik)": results["total_seconds"],
                "epoch neural total": results["training"]["total_neural_epochs"],
            }
        ]
    )
)

# %% [markdown]
# ## Seluruh konfigurasi dan hasil selection

# %%
candidates = results["candidate_validation_results"]
candidate_by_name = {candidate["name"]: candidate for candidate in candidates}
assert len(candidate_by_name) == len(candidates), "Nama kandidat harus unik."
assert all(
    len(candidate["history"]) == candidate["epochs_completed"] for candidate in candidates
), "Jumlah epoch harus cocok dengan history tersimpan."
candidate_rows = []
for candidate in candidates:
    row = {
        "kandidat": candidate["name"],
        "kelompok": candidate["arm"],
        "objective": candidate["objective"],
        "split": candidate["split_seed"],
        "seed training": candidate["initialization_seed"],
        "status": candidate["status"],
        "epoch selesai": candidate["epochs_completed"],
        "checkpoint epoch": candidate["best_epoch"],
        "checkpoint di batas epoch": candidate["best_at_epoch_cap"],
        "training mencapai batas epoch": candidate["stopped_at_epoch_cap"],
        "contoh training per epoch": candidate["training_examples_per_epoch"],
        "update optimizer total": candidate["optimizer_iterations"],
        "save/load terverifikasi": candidate["model_roundtrip_verified"],
        "SHA256 parameter awal": candidate["initial_trainable_sha256"],
    }
    for cohort, metrics in candidate["selection"].items():
        for metric in ("users", "hits_at_10", "recall_at_10", "ndcg_at_10"):
            row[f"selection {cohort}: {metric}"] = metrics[metric]
    candidate_rows.append(row)
candidate_table = pd.DataFrame(candidate_rows)
display(candidate_table)
display(
    Markdown(
        f"Tercatat **{len(candidates)} kandidat**, "
        f"**{sum(candidate['epochs_completed'] > 0 for candidate in candidates)}** "
        "menyelesaikan setidaknya satu epoch. Seluruh seed ditampilkan; "
        "nilai selection tidak dicampur dengan hasil assessment. "
        "Hash parameter awal membantu memeriksa pasangan ablation yang memakai "
        "bobot awal sama; ia bukan ukuran kemiripan hasil model."
    )
)

# %%
selection_rows = []
for split in results["primary_selection"]:
    assert split["selection_only"]
    for arm in split["arms"]:
        selection_rows.append(
            {
                "split": split["split_seed"],
                "kelompok": arm["arm"],
                "model utama dari selection": arm["primary_candidate"],
                "jumlah seed": arm["n_seeds"],
                "mean selection NDCG@10": arm["selection_ndcg_at_10_mean"],
                "SD selection NDCG@10": arm["selection_ndcg_at_10_std"],
                "NDCG@10 model utama": arm["primary_selection"]["all_relevant"]["ndcg_at_10"],
            }
        )
display(pd.DataFrame(selection_rows))
initialization_check = candidate_table.groupby(["split", "seed training"])[
    "SHA256 parameter awal"
].nunique()
assert initialization_check.eq(1).all(), "Bobot awal kelompok yang dipasangkan berbeda."
display(
    Markdown(
        "Bobot trainable awal **identik di ketiga kelompok** untuk setiap pasangan "
        "split/seed, berdasarkan hash seluruh nilai trainable. Fitur metadata berbeda "
        "sesuai kelompok. Jumlah contoh dan update per epoch MSE/BPR berbeda; tabel "
        "konfigurasi dan history mencatat keduanya. Checkpoint yang terbaik tepat "
        "pada batas epoch belum membuktikan model sudah konvergen."
    )
)

# %% [markdown]
# ## Seluruh epoch dan kurva pembelajaran
#
# Titik terbaik ditentukan oleh NDCG selection, bukan oleh loss train terkecil.
# Train loss turun hanya membuktikan objective train membaik. Kualitas
# rekomendasi pada data lain tetap dinilai terpisah.

# %%
epoch_rows = []
for candidate in candidates:
    for epoch in candidate["history"]:
        epoch_rows.append(
            {
                "candidate": candidate["name"],
                "arm": candidate["arm"],
                "split_seed": candidate["split_seed"],
                "training_seed": candidate["initialization_seed"],
                **epoch,
                "selected_checkpoint": epoch["epoch"] == candidate["best_epoch"],
            }
        )
epochs = pd.DataFrame(epoch_rows)
assert len(epochs) == results["training"]["total_neural_epochs"]
display(epochs)
arms = ["mse_id", "bpr_id", "bpr_metadata"]
split_seeds = sorted({candidate["split_seed"] for candidate in candidates})
training_seeds = [42, 2026, 7]
seed_colors = {42: "tab:blue", 2026: "tab:orange", 7: "tab:green"}
fig, axes = plt.subplots(len(split_seeds), len(arms), figsize=(15, 8), squeeze=False)
for row_index, split_seed in enumerate(split_seeds):
    for column_index, arm in enumerate(arms):
        ax = axes[row_index, column_index]
        for training_seed in training_seeds:
            group = epochs[
                (epochs["split_seed"] == split_seed)
                & (epochs["arm"] == arm)
                & (epochs["training_seed"] == training_seed)
            ].sort_values("epoch")
            ax.plot(
                group["epoch"],
                group["selection_ndcg_at_10"],
                marker=".",
                label=f"seed {training_seed}",
                color=seed_colors[training_seed],
            )
            best = group[group["selected_checkpoint"]]
            ax.scatter(
                best["epoch"],
                best["selection_ndcg_at_10"],
                marker="*",
                s=110,
                color=seed_colors[training_seed],
            )
        ax.set(title=f"Split {split_seed} — {arm}", xlabel="Epoch", ylabel="Selection NDCG@10")
        ax.legend(fontsize=8)
fig.suptitle("Semua epoch; tanda bintang menunjukkan checkpoint pilihan selection")
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(len(split_seeds), len(arms), figsize=(15, 8), squeeze=False)
for row_index, split_seed in enumerate(split_seeds):
    for column_index, arm in enumerate(arms):
        ax = axes[row_index, column_index]
        for training_seed in training_seeds:
            group = epochs[
                (epochs["split_seed"] == split_seed)
                & (epochs["arm"] == arm)
                & (epochs["training_seed"] == training_seed)
            ].sort_values("epoch")
            ax.plot(
                group["epoch"],
                group["train_data_loss"],
                marker=".",
                label=f"data seed {training_seed}",
                color=seed_colors[training_seed],
            )
            ax.plot(
                group["epoch"],
                group["train_total_loss"],
                linestyle="--",
                label=f"total seed {training_seed}",
                color=seed_colors[training_seed],
            )
        ax.set(title=f"Split {split_seed} — {arm}", xlabel="Epoch", ylabel="Train loss")
        ax.legend(fontsize=7)
fig.suptitle("Loss train menurut objective masing-masing; skala MSE dan BPR berbeda")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bukti embedding, forward propagation, loss, backpropagation, dan Adam
#
# Lima konsep ini membentuk satu proses: embedding berubah menjadi representasi
# pengguna/buku; forward propagation menghasilkan skor; loss membandingkan skor
# dengan sasaran; backpropagation menghitung gradien; Adam memperbarui parameter.
# Diagnosis di bawah berasal dari langkah training yang tercatat. Gradien atau
# perubahan bobot yang bukan nol membuktikan jalur belajar aktif, belum
# membuktikan rekomendasi lebih baik daripada pembanding.

# %%
trace_rows, update_rows, score_rows = [], [], []
for candidate in candidates:
    trace = candidate["first_step_trace"]
    trace_rows.append(
        {
            "kandidat": candidate["name"],
            "objective": trace["objective"],
            "batch": trace["batch_size"],
            "iterasi Adam setelah langkah": trace["optimizer_iterations"],
            **trace["loss_before_update"],
        }
    )
    for variable, update_norm in trace["variable_update_norms"].items():
        update_rows.append(
            {
                "kandidat": candidate["name"],
                "parameter": variable,
                "norma perubahan setelah Adam": update_norm,
            }
        )
    for index, (before, after) in enumerate(
        zip(trace["raw_positive_scores_before"], trace["raw_positive_scores_after"]), start=1
    ):
        negative = trace["raw_negative_scores_before"]
        score_rows.append(
            {
                "kandidat": candidate["name"],
                "contoh dalam batch": index,
                "skor karya sebelum": before,
                "skor pembanding sebelum (BPR)": negative[index - 1] if negative else None,
                "skor karya sesudah": after,
                "perubahan skor karya": after - before,
            }
        )
display(pd.DataFrame(trace_rows))
display(pd.DataFrame(update_rows))
display(pd.DataFrame(score_rows).fillna("tidak dipakai pada MSE"))
display(
    Markdown(
        "Gradien pada tabel pertama adalah norma global **sebelum update**. Tabel "
        "kedua berisi perubahan parameter **sesudah Adam**, bukan gradien tiap "
        "parameter. Skor adalah skor mentah; untuk MSE skor melewati sigmoid saat "
        "menghitung loss. Contoh MSE tidak wajib memiliki rating tinggi. Parameter "
        "jalur metadata pada kelompok ID juga dapat berubah akibat regularisasi "
        "meskipun inputnya nol; perubahan itu tidak berarti metadata dipakai."
    )
)

# %% [markdown]
# ## Assessment: seluruh model pada pengguna yang sama
#
# Satu karya yang disembunyikan menjadi target untuk setiap pengguna. Target
# relevan jika salah satu rating edisinya minimal 8. Recall@10 adalah proporsi
# pengguna yang targetnya masuk 10 rekomendasi; dengan satu target per pengguna,
# nilainya sama dengan Hit Rate@10. NDCG@10 juga memperhatikan posisi target.
# MRR@10 memakai kebalikan posisi target dan nol jika target di luar top 10.
#
# Metrik ini mengukur penemuan target yang diamati. Buku lain yang tidak diberi
# rating bukan otomatis rekomendasi buruk. Cakupan katalog dan entropy mengukur
# sebaran paparan karya, bukan otomatis keberagaman semantik atau kepuasan.

# %%
ranking_results = results["ranking_results"]
ranking_rows = []
cohort_rows = []
for split in ranking_results:
    assert split["status"] == "completed"
    cohort_rows.append(
        {
            "split": split["split_seed"],
            "karya katalog": split["catalogue_works"],
            **split["cohort_counts"],
        }
    )
    for name, cohorts in split["metrics"].items():
        kind = candidate_by_name[name]["arm"] if name in candidate_by_name else "benchmark historis"
        if name == "popularity":
            kind = "popularitas fit"
        for cohort, metrics in cohorts.items():
            assert metrics["users"] == split["cohort_counts"][cohort]
            ranking_rows.append(
                {
                    "split": split["split_seed"],
                    "model": name,
                    "kelompok": kind,
                    "cohort": cohort,
                    "model utama": name in split["primary_models"].values()
                    or name in (split["old_primary_mlp"], "popularity"),
                    **metrics,
                }
            )
ranking_table = pd.DataFrame(ranking_rows)
display(pd.DataFrame(cohort_rows))
for cohort in ("all_relevant", "fit_positive_history", "no_fit_positive_history"):
    display(Markdown(f"**Cohort `{cohort}` — seluruh model:**"))
    display(ranking_table[ranking_table["cohort"] == cohort].reset_index(drop=True))

# %%
primary_ranking = ranking_table[ranking_table["model utama"]].copy()
fig, axes = plt.subplots(len(split_seeds), 2, figsize=(14, 9), squeeze=False)
for row_index, split_seed in enumerate(split_seeds):
    for column_index, cohort in enumerate(("all_relevant", "fit_positive_history")):
        ax = axes[row_index, column_index]
        group = primary_ranking[
            (primary_ranking["split"] == split_seed) & (primary_ranking["cohort"] == cohort)
        ]
        group.set_index("model")["ndcg_at_10"].plot.barh(ax=ax, color="tab:blue")
        ax.set(title=f"Split {split_seed} — {cohort}", xlabel="Assessment NDCG@10", ylabel="")
        ax.tick_params(axis="y", labelsize=8)
fig.suptitle("Model utama dipilih melalui selection sebelum assessment run ini")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Variasi seed dan ketidakpastian perbandingan
#
# Rata-rata dan standar deviasi tiga seed menggambarkan variasi inisialisasi
# serta mini-batch dalam split yang sama. Ketiganya bukan tiga dataset
# independen. Dua split dilaporkan terpisah karena pengguna/data dapat tumpang
# tindih. Interval bootstrap melakukan resampling pengguna berpasangan; nilai
# positif pada selisih kandidat − referensi menguntungkan kandidat untuk metrik
# ranking. Interval bersyarat pada assessment internal yang dipakai ulang dan
# tidak mengoreksi banyak perbandingan.

# %%
new_ranking = ranking_table[ranking_table["model"].isin(candidate_by_name)]
seed_summary = (
    new_ranking.groupby(["split", "kelompok", "cohort"], sort=False)
    .agg(
        jumlah_seed=("model", "count"),
        recall_mean=("recall_at_10", "mean"),
        recall_sd=("recall_at_10", "std"),
        ndcg_mean=("ndcg_at_10", "mean"),
        ndcg_sd=("ndcg_at_10", "std"),
        mrr_mean=("mrr_at_10", "mean"),
        mrr_sd=("mrr_at_10", "std"),
    )
    .reset_index()
)
display(seed_summary)
bootstrap_rows = []
for split in ranking_results:
    for comparison in split["paired_bootstrap"]:
        bootstrap_rows.append(
            {
                "split": split["split_seed"],
                "kandidat": comparison["model"],
                "referensi": comparison["reference"],
                "cohort": comparison["cohort"],
                "metrik": comparison["metric"],
                "selisih kandidat - referensi": comparison["difference"],
                "batas bawah 95%": comparison["ci95"][0],
                "batas atas 95%": comparison["ci95"][1],
                "pengguna": comparison["users"],
                "bootstrap draws": comparison["draws"],
                "seed bootstrap": comparison["seed"],
            }
        )
bootstrap_table = pd.DataFrame(bootstrap_rows)
for cohort in ("all_relevant", "fit_positive_history"):
    for metric in ("recall_at_10", "ndcg_at_10", "mrr_at_10"):
        display(Markdown(f"**Semua perbandingan `{metric}` pada `{cohort}`:**"))
        display(
            bootstrap_table[
                (bootstrap_table["cohort"] == cohort) & (bootstrap_table["metrik"] == metric)
            ].reset_index(drop=True)
        )

# %% [markdown]
# ## Ringkasan keputusan berdasarkan bukti
#
# Perubahan baru boleh disebut peningkatan jika dibandingkan pada cohort dan
# aturan yang sama. Selisih terhadap MLP lama, terhadap kontrol MSE baru, dan
# terhadap popularitas menjawab pertanyaan berbeda. Konfirmasi independen
# masih diperlukan sebelum menganggap temuan pengembangan ini akan bertahan
# pada pengguna atau data baru. Tidak ada model yang otomatis dipromosikan
# menjadi model aplikasi hanya karena menang pada assessment ini.

# %%
display(
    primary_ranking[primary_ranking["cohort"].isin(["all_relevant", "fit_positive_history"])][
        [
            "split",
            "cohort",
            "model",
            "users",
            "hits_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "mrr_at_10",
        ]
    ].reset_index(drop=True)
)
for split in ranking_results:
    seed_matched = bootstrap_table[
        (bootstrap_table["split"] == split["split_seed"])
        & (bootstrap_table["cohort"] == "fit_positive_history")
        & (bootstrap_table["metrik"] == "ndcg_at_10")
    ]
    display(Markdown(f"**Split {split['split_seed']} — tiga seed pada riwayat positif bersama:**"))
    for arm, reference_arm, interpretation in (
        ("bpr_id", "mse_id", "BPR ID dibanding kontrol MSE ID (objective + supervisi berubah)"),
        ("bpr_metadata", "bpr_id", "Tambahan metadata pada BPR yang sama"),
    ):
        paired = seed_matched[
            seed_matched["kandidat"].str.startswith(f"split{split['split_seed']}_{arm}_seed")
            & seed_matched["referensi"].str.startswith(
                f"split{split['split_seed']}_{reference_arm}_seed"
            )
        ]
        assert len(paired) == len(training_seeds)
        higher = int((paired["selisih kandidat - referensi"] > 0).sum())
        above_zero = int((paired["batas bawah 95%"] > 0).sum())
        below_zero = int((paired["batas atas 95%"] < 0).sum())
        display(
            Markdown(
                f"{interpretation}: NDCG assessment lebih tinggi pada **{higher}/{len(paired)} "
                f"pasangan seed**; **{above_zero}** interval seluruhnya di atas nol dan "
                f"**{below_zero}** interval seluruhnya di bawah nol. Ini ringkasan deskriptif "
                "interval bersyarat tanpa koreksi banyak perbandingan, bukan enam replikasi "
                "dataset independen."
            )
        )
        display(paired.reset_index(drop=True))
    chosen_names = set(split["primary_models"].values())
    display(Markdown("**Model utama dari selection dibanding popularitas dan MLP lama:**"))
    display(
        seed_matched[
            seed_matched["kandidat"].isin(chosen_names)
            & seed_matched["referensi"].isin(["popularity", split["old_primary_mlp"]])
        ].reset_index(drop=True)
    )
display(
    Markdown(
        "**Batas keputusan:** angka di atas berlaku pada katalog dan cohort ini. "
        "BPR ID memperbaiki hasil MLP lama dan mendekati popularitas, tetapi belum "
        "terbukti mengungguli popularitas secara konsisten. Metadata belum memberi "
        "tambahan yang konsisten pada konfigurasi ini. Penambahan metadata belum "
        "menguji karya baru/cold-start; tiga seed belum merupakan pencarian menyeluruh "
        "kapasitas dan learning rate. Hasil pengembangan ini tetap perlu dikonfirmasi "
        "pada data independen sebelum klaim umum atau promosi aplikasi."
    )
)

# %% [markdown]
# ## Penyimpanan hasil dan reproduksi
#
# Model, checkpoint optimizer, state RNG, log, prediksi, dan rekomendasi rinci
# disimpan di workspace. Manifest di repository mencatat sidik SHA256, status
# backup lokal, pengujian perangkat lunak, dan pemeriksaan akhir status sesi.
# Status ini bukti tersimpan, bukan pemeriksaan runtime secara langsung.
# Setelah backup dan audit terverifikasi, perintah stop menerima `Not Found`.
# Pemeriksaan server berikutnya memastikan tidak ada sesi aktif dan menghapus
# catatan lokal yang sudah tidak berlaku. Jadi tidak ada runtime aktif pada
# pemeriksaan akhir, tetapi perintah stop tidak dilaporkan berhasil; waktu dan
# penyebab runtime sebelumnya berakhir tidak diketahui.
#
# Untuk memeriksa notebook, jalankan `scripts/notebook_quality.py check-pair`
# pada pasangan `.ipynb`/`.py` dan `check-saved` dengan kedua JSON pada
# `EVIDENCE_FILES` sebagai `--dependency`. Menjalankan ulang notebook hanya
# membaca bukti; menjalankan ulang training membutuhkan input dan runtime yang
# sesuai dengan protokol, bukan sekadar menekan Run All pada laporan ini.

# %%
display(
    pd.DataFrame(
        [
            {"catatan": "Sesi khusus proyek", "nilai": manifest["session_name"]},
            {
                "catatan": "Akun Free dikonfirmasi",
                "nilai": manifest["account_free_without_paid_balance_confirmed_by_user"],
            },
            {
                "catatan": "Sumber daya berbayar diizinkan",
                "nilai": manifest["paid_resources_authorized"],
            },
            {"catatan": "Tes perangkat lunak lulus", "nilai": manifest["runtime_tests"]["passed"]},
            {"catatan": "Jumlah arsip terverifikasi", "nilai": len(manifest["backups"])},
            {
                "catatan": "Server mengonfirmasi tidak ada sesi aktif",
                "nilai": manifest["shutdown"]["server_reports_no_active_sessions"],
            },
            {
                "catatan": "Pemeriksaan akhir sesudah verifikasi backup/hasil",
                "nilai": manifest["shutdown"][
                    "inactive_state_verified_after_backup_and_result_verification"
                ],
            },
            {
                "catatan": "Perintah stop berhasil",
                "nilai": manifest["shutdown"]["stop_command_succeeded"],
            },
            {
                "catatan": "Perintah stop menerima Not Found",
                "nilai": manifest["shutdown"]["stop_returned_not_found"],
            },
            {
                "catatan": "Waktu tepat runtime berakhir diketahui",
                "nilai": manifest["shutdown"]["exact_termination_time_known"],
            },
        ]
    )
)
display(pd.DataFrame(manifest["backups"]))
assert manifest["shutdown"]["stopped"]
assert manifest["shutdown"]["server_reports_no_active_sessions"]
assert manifest["shutdown"]["stop_attempt_after_backup_and_result_verification"]
assert manifest["shutdown"]["inactive_state_verified_after_backup_and_result_verification"]
assert not manifest["paid_resources_authorized"]
display(pd.DataFrame({"batas interpretasi dari hasil run": results["limitations"]}))
