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
# # Apakah riwayat buku membuat rekomendasi lebih personal?
#
# **Pertanyaan utama:** apakah model yang membaca kumpulan buku yang disukai
# pengguna memberi rekomendasi lebih baik dibandingkan BPR berbasis ID dan
# pembanding popularitas? Eksperimen ini menguji dua objective neural
# berbasis riwayat, pembanding klasik yang memakai riwayat sama, serta diagnosis
# perubahan rekomendasi ketika riwayat ditukar dengan riwayat pengguna lain.
#
# Alasan pengujian berasal dari hasil sebelumnya. BPR berbasis ID sudah
# meningkatkan penemuan target dibanding MLP lama, tetapi daftar rekomendasinya
# sangat banyak bertumpang tindih dengan popularitas. Kemiripan tersebut perlu
# diperiksa lebih lanjut; ia sendiri belum membuktikan model mengabaikan pengguna.
#
# Notebook ini membaca **tiga JSON bukti tersimpan**. Restart & Run All tidak
# melatih model, menghubungi Colab, membuka CSV privat, atau memerlukan GPU dan
# TensorFlow. Riwayat keseluruhan tersedia di [notebook 03](03_neural_experiment_history.ipynb),
# hasil eksperimen BPR/metadata sebelumnya di [notebook 04](04_neural_ranking_experiments.ipynb).
#
# **Batas evaluasi:** run ini dirancang setelah hasil sebelumnya diketahui.
# Assessment internal dipakai ulang, sehingga hasilnya masih pengembangan
# adaptif, bukan konfirmasi pada test independen. Pemilihan checkpoint hanya
# melalui selection dalam run ini tidak menghapus pemakaian assessment terdahulu.

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
        if (path / "docs/evidence/neural_colab_history_results.json").is_file()
    ),
    None,
)
if ROOT is None:
    raise FileNotFoundError("Buka dari checkout proyek atau set BOOK_RECOMMENDER_PROJECT_ROOT.")
EVIDENCE = ROOT / "docs/evidence"
EVIDENCE_FILES = {
    "results": "neural_colab_history_results.json",
    "manifest": "neural_colab_history_run_manifest.json",
    "diagnosis": "neural_personalization_diagnosis.json",
}
evidence = {
    key: json.loads((EVIDENCE / filename).read_text(encoding="utf-8"))
    for key, filename in EVIDENCE_FILES.items()
}
results, manifest, diagnosis = (evidence[key] for key in ("results", "manifest", "diagnosis"))
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_colwidth", 110)
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
# **Data ini mempunyai sinyal rekomendasi yang berguna, tetapi neural belum
# memanfaatkannya secara konsisten.** Pembanding cosine riwayat yang dipilih
# melalui selection menemukan **191/1.776** dan **198/1.806** target. Popularitas
# menemukan **75/1.776** dan **66/1.806**. Pada dua split ini, pembanding klasik
# berbasis hubungan antarkarya memberi hasil terkuat di antara model utama.
# Ini bukan hasil evaluasi aplikasi yang sudah dipromosikan atau test baru.
#
# Neural full softmax utama menemukan **109** target pada split 17, meningkat
# dari BPR ID lama **72**, tetapi hanya **66** pada split 29 dibanding BPR ID
# lama **72**. Neural history-BPR utama menemukan **70** dan **67** target.
# Pada cohort positif, semua model neural utama masih memiliki NDCG lebih
# rendah daripada cosine terpilih, dengan interval selisih seluruhnya negatif.
# Jadi perubahan neural belum memberi peningkatan yang konsisten pada kedua split.
#
# Diagnosis riwayat juga berbeda: softmax utama split 17 turun dari **109** ke
# **48** hit ketika diberi riwayat donor, sedangkan split 29 tetap **66** hit.
# Ini mendukung kegunaan riwayat asli pada checkpoint split 17, tetapi tidak
# membuktikan semua model sudah memakai riwayat secara bermanfaat. Pada split 29,
# hit softmax sama dengan popularitas namun posisi target lebih buruk menurut
# NDCG cohort positif. Seluruh temuan tetap bersyarat pada assessment pengembangan
# yang telah dipakai ulang; belum ada konfirmasi independen.
#
# Tabel ini menampilkan model utama dari selection. “Target ditemukan” berarti
# karya relevan yang disembunyikan muncul pada 10 rekomendasi. Hasil seluruh
# seed tetap tersedia di bagian rinci; model tidak dipilih ulang menggunakan
# assessment. Kelompok pengguna dibedakan menurut jumlah karya positif di fit.

# %%
COHORTS = (
    "all_relevant",
    "fit_positive_history",
    "no_fit_positive_history",
    "single_fit_positive",
    "multiple_fit_positive",
)
COMPARISON_COHORTS = ("all_relevant", "fit_positive_history", "multiple_fit_positive")
quick_rows = []
for ranking in results["ranking_results"]:
    chosen = [
        *ranking["primary_models"].items(),
        ("Cosine riwayat", ranking["baseline_name"]),
        ("BPR ID lama", ranking["previous_primary_models"]["bpr_id"]),
        ("Popularitas", "popularity"),
    ]
    for label, name in chosen:
        overall = ranking["metrics"][name]["all_relevant"]
        positive = ranking["metrics"][name]["fit_positive_history"]
        multiple = ranking["metrics"][name]["multiple_fit_positive"]
        quick_rows.append(
            {
                "split": ranking["split_seed"],
                "model utama": label,
                "kandidat": name,
                "target ditemukan (semua)": f"{overall['hits_at_10']}/{overall['users']}",
                "Recall@10 semua (%)": 100 * overall["recall_at_10"],
                "NDCG@10 semua": overall["ndcg_at_10"],
                "NDCG@10 riwayat positif": positive["ndcg_at_10"],
                "target ditemukan (minimal 2 positif)": f"{multiple['hits_at_10']}/{multiple['users']}",
                "NDCG@10 minimal 2 positif": multiple["ndcg_at_10"],
            }
        )
display(pd.DataFrame(quick_rows))

# %% [markdown]
# ## Diagnosis sebelum run ini: seberapa mirip BPR ID dengan popularitas?
#
# Diagnosis berikut dihitung sesudah melihat hasil run ketujuh, dari rekomendasi
# dan rating fit yang sudah tersimpan. Tidak ada training atau pemilihan ulang
# checkpoint. Pada pengguna dengan sedikitnya satu karya positif di fit, buku
# peringkat pertama BPR ID sama dengan popularitas pada sekitar **99,74%**
# pengguna split 17 dan **97,60%** pengguna split 29. Kedua daftar top 10 rata-rata
# memiliki **7,13** dan **6,92** karya yang sama.
#
# Kemiripan ini adalah deskripsi konsentrasi rekomendasi, bukan bukti kausal
# bahwa jaringan tidak memakai pengguna. Daftar yang berbeda juga belum tentu
# lebih relevan. Karena itu, run baru mengukur kualitas sekaligus sensitivitas
# terhadap masukan riwayat, tetap memakai pembanding popularitas.

# %%
diagnosis_rows, diagnosis_bucket_rows, exposure_rows = [], [], []
for split in diagnosis["splits"]:
    for cohort, metrics in split["overlap"].items():
        diagnosis_rows.append(
            {
                "split": split["split_seed"],
                "BPR utama": split["primary_bpr"],
                "cohort": cohort,
                "pengguna": metrics["users"],
                "top-1 sama dengan popularitas (%)": 100 * metrics["same_top1"] / metrics["users"],
                "rata-rata karya sama di top-10": metrics["mean_common_top10"],
                "rata-rata Jaccard": metrics["mean_jaccard"],
                "daftar identik urutan": metrics["exact_ordered_lists"],
                "himpunan karya identik": metrics["identical_item_sets"],
            }
        )
    for bucket in split["metrics_by_bucket"]:
        for name, metrics in bucket["models"].items():
            diagnosis_bucket_rows.append(
                {
                    "split": split["split_seed"],
                    "jumlah karya positif": bucket["positive_fit_works"],
                    "pengguna": bucket["users"],
                    "model": name,
                    **metrics,
                }
            )
    for group, metrics in split["exposure"].items():
        exposure_rows.append({"split": split["split_seed"], "kelompok/model": group, **metrics})
diagnosis_table = pd.DataFrame(diagnosis_rows)
display(Markdown(diagnosis["scope"]))
display(diagnosis_table)
display(pd.DataFrame(diagnosis_bucket_rows))
display(pd.DataFrame(exposure_rows))
positive_diagnosis = diagnosis_table[diagnosis_table["cohort"] == "positive"].set_index("split")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
positive_diagnosis["top-1 sama dengan popularitas (%)"].plot.bar(ax=axes[0], rot=0)
positive_diagnosis["rata-rata karya sama di top-10"].plot.bar(ax=axes[1], rot=0)
axes[0].set(ylim=(0, 100), ylabel="Pengguna (%)", title="BPR ID: kesamaan buku peringkat pertama")
axes[1].set(
    ylim=(0, 10),
    ylabel="Karya dari 10 rekomendasi",
    title="BPR ID: tumpang tindih dengan popularitas",
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Apa yang berubah pada run ini?
#
# Model baru membentuk representasi pengguna dari **himpunan karya positif**
# pada fit: karya dengan salah satu rating edisinya minimal 8. Embedding karya
# dalam riwayat dijumlahkan, dibagi akar jumlah karya, lalu melalui Dense 64
# dengan tanh dan Dense 32. Representasi pengguna tersebut dibandingkan dengan
# embedding keluaran setiap kandidat karya; skor juga mempunyai bias karya.
# Embedding masukan riwayat dan embedding keluaran kandidat adalah parameter
# terpisah. Identitas pengguna tidak menjadi embedding yang dilatih pada model ini.
#
# Riwayat adalah **himpunan tanpa urutan waktu**. Tidak ada klaim bahwa jaringan
# mempelajari urutan pembacaan atau memprediksi buku berikutnya secara temporal.
# Gagasan memakai embedding riwayat sebagai masukan juga muncul pada contoh
# [retrieval TensorFlow Recommenders](https://www.tensorflow.org/recommenders/examples/sequential_retrieval),
# tetapi contoh itu menggunakan urutan dan GRU. Implementasi proyek ini bukan
# reproduksi model sekuensial tersebut.
#
# Dua objective memakai arsitektur, inisialisasi berpasangan, dan contoh target
# training yang sama:
#
# | Kelompok | Sasaran latihan |
# |---|---|
# | `history_bpr` | Menaikkan skor target positif terhadap satu kandidat pembanding yang disampling |
# | `history_softmax` | Menaikkan probabilitas target terhadap seluruh kandidat yang diizinkan dalam katalog |
#
# BPR menggunakan `softplus(skor pembanding - skor target)`. Full softmax memakai
# cross-entropy target pada skor katalog. Keduanya mengejar ranking melalui
# bentuk perbandingan berbeda; loss dan jumlah kandidat pembandingnya tidak
# berada pada skala yang dapat dibandingkan langsung.
# Pemakaian likelihood multinomial untuk rekomendasi juga dibahas dalam
# [Liang dkk., Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814).
# Model proyek ini memakai encoder deterministik; tidak memiliki encoder
# variational atau loss KL dan bukan reproduksi VAE dari paper tersebut.
#
# Perbandingan dengan BPR ID lama mengubah representasi **serta pengguna/contoh
# training yang memenuhi syarat dan definisi regularisasi**. Selisih hasilnya
# tidak mengisolasi manfaat encoder riwayat saja. Perbandingan `history_bpr`
# dengan `history_softmax` lebih terkontrol untuk objective karena memakai
# arsitektur, contoh target, konteks, dan inisialisasi yang sama.

# %% [markdown]
# ## Mencegah target masuk ke riwayat training
#
# Untuk setiap target positif pada fit, **karya target dikeluarkan dari riwayat
# sebelum pooling**. Pengguna training harus mempunyai sedikitnya dua karya
# positif agar masih ada konteks setelah satu target dikeluarkan. Satu pengguna
# dapat menyumbang beberapa contoh training; satu contoh adalah satu karya
# target dan riwayat positif lainnya. Ini bukan pembuatan split assessment baru.
# Pengguna dengan lebih banyak karya positif menyumbang lebih banyak target;
# pembobotan training dilakukan per contoh, bukan sama rata per pengguna.
#
# Ketika selection/assessment, model memakai seluruh riwayat positif fit.
# Pengguna dengan satu karya positif tetap bisa menerima skor neural, walaupun
# pengguna dengan satu positif tidak menyumbang target latihan berbasis riwayat.
# Pengguna tanpa positif memakai fallback popularitas. Semua karya yang ada di
# fit pengguna, termasuk yang ratingnya rendah, dikeluarkan dari rekomendasi.
#
# Kandidat negatif BPR disampling dari karya yang tidak ada dalam seluruh fit
# pengguna. Softmax menutup skor semua karya fit selain target training;
# target dibuka kembali sebagai label yang sah. Dengan demikian, karya positif
# lain dalam riwayat tidak dihukum sebagai alternatif negatif pada contoh itu.
# Preferensi terhadap kandidat yang belum diamati tetap tidak diketahui:
# **belum tercatat bukan berarti tidak disukai**. Label selection/assessment
# tidak dipakai untuk menyusun contoh training atau memilih negatif.

# %%
display(pd.DataFrame(results["dataset"]["training_work_summaries"]))
training_bucket_rows = []
for split in diagnosis["splits"]:
    training_bucket_rows.append(
        {
            "split": split["split_seed"],
            **{
                f"pengguna fit {bucket} positif": count
                for bucket, count in split["train_users_by_positive_bucket"].items()
            },
            "pengguna minimal 2 positif": split["train_eligible_leave_one_positive_out_users"],
            "contoh leave-one-positive-out": split["train_eligible_leave_one_positive_out_pairs"],
        }
    )
display(pd.DataFrame(training_bucket_rows))
assert results["dataset"]["reused_prior_assessment"]
assert not results["original_test_used"]
assert not results["original_validation_used"]
assert not results["model_promotion_to_application"]

# %% [markdown]
# ## Pembanding klasik yang memakai riwayat positif sama
#
# Pembanding klasik menghitung cosine antar-karya dari interaksi positif fit.
# Skor kandidat dikumpulkan dari kemiripannya dengan karya positif dalam
# riwayat. Grid membandingkan 50/200 tetangga dan shrinkage 0/10, sehingga ada
# empat konfigurasi per split, delapan catatan konfigurasi klasik keseluruhan.
# Shrinkage mengurangi kemiripan yang didukung sedikit pengguna bersama:
# cosine dikalikan `jumlah pengguna bersama / (jumlah pengguna bersama + shrinkage)`.
# Jadi konfigurasi dengan shrinkage 10 bukan cosine tanpa pembobotan tambahan.
# Checkpoint neural dan konfigurasi klasik sama-sama dipilih melalui NDCG@10
# selection, dengan Recall@10 sebagai pemecah seri.
#
# Pembanding ini membantu membedakan manfaat membaca riwayat dengan manfaat
# jaringan neural. Ia tidak dilatih melalui backpropagation dan tidak dihitung
# sebagai training neural baru. Popularitas dan hasil neural terdahulu juga
# dipakai sebagai benchmark dengan pilihan model historis tetap.
# Pada skor cosine yang persis sama, popularitas fit lalu urutan leksikal karya
# menjadi pemecah seri; pada neural, seri diselesaikan lewat urutan leksikal.
# Ini kebijakan yang ditetapkan sebelum penilaian, bukan perbedaan yang
# disembunyikan. Cosine ini berbasis interaksi positif karya kanonik, berbeda
# dari adjusted-cosine rating ISBN pada aplikasi awal.

# %%
baselines = results["baseline_grid_results"]
baseline_rows = []
for baseline in baselines:
    baseline_rows.append(
        {
            "kandidat": baseline["name"],
            "split": baseline["split_seed"],
            "tetangga k": baseline["k"],
            "shrinkage": baseline["shrink"],
            "status": baseline["status"],
            "terpilih dari selection": baseline["selected"],
            "selection NDCG@10": baseline["selection"]["all_relevant"]["ndcg_at_10"],
            "selection Recall@10": baseline["selection"]["all_relevant"]["recall_at_10"],
            "selection NDCG@10 minimal 2 positif": baseline["selection"]["multiple_fit_positive"][
                "ndcg_at_10"
            ],
            "kemiripan bukan nol": baseline["similarity_nonzero"],
            "save/load terverifikasi": baseline["model_roundtrip_verified"],
            "durasi (detik)": baseline["seconds"],
        }
    )
display(pd.DataFrame(baseline_rows))

# %% [markdown]
# ## Protokol dan seluruh konfigurasi
#
# Dua split (17 dan 29), tiga seed (42, 2026, 7), dan dua objective menghasilkan
# 12 rencana training neural. Adam memakai learning rate 0,001, batch 1.024,
# maksimal 20 epoch, dan patience 4. Regularisasi `1e-5 × mean(parameter²)`
# diterapkan pada seluruh parameter trainable. Definisi ini berbeda dari
# regularisasi sampled-pair pada run sebelumnya; angka lambda saja tidak
# menunjukkan kekuatan penyusutan yang sama.
#
# Semua kandidat harus selesai dan seluruh model terpilih dikunci melalui
# sidik hash sebelum assessment run ini dibuka. Seluruh seed dilaporkan;
# penilaian pada assessment tidak memilih ulang model utama.

# %%
display(
    pd.DataFrame(
        [
            {"pengaturan": key, "nilai": value}
            for key, value in results["configuration"].items()
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
                "seluruh kandidat selesai": results["all_candidates_completed"],
                "assessment dilakukan": results["assessment_performed"],
                "durasi eksperimen (detik)": results["total_seconds"],
                "epoch neural": results["training"]["total_neural_epochs"],
            }
        ]
    )
)
candidates = results["candidate_validation_results"]
candidate_by_name = {candidate["name"]: candidate for candidate in candidates}
assert len(candidate_by_name) == len(candidates)
candidate_rows = []
for candidate in candidates:
    candidate_rows.append(
        {
            "kandidat": candidate["name"],
            "split": candidate["split_seed"],
            "kelompok": candidate["arm"],
            "seed": candidate["initialization_seed"],
            "status": candidate["status"],
            "epoch selesai": candidate["epochs_completed"],
            "epoch terbaik": candidate["best_epoch"],
            "checkpoint terbaik di batas epoch": candidate["best_at_epoch_cap"],
            "contoh per epoch": candidate["training_examples_per_epoch"],
            "update Adam": candidate["optimizer_iterations"],
            "selection NDCG@10": candidate["selection"]["all_relevant"]["ndcg_at_10"],
            "selection Recall@10": candidate["selection"]["all_relevant"]["recall_at_10"],
            "selection NDCG@10 minimal 2 positif": candidate["selection"]["multiple_fit_positive"][
                "ndcg_at_10"
            ],
            "save/load terverifikasi": candidate["model_roundtrip_verified"],
            "SHA256 parameter awal": candidate["initial_trainable_sha256"],
        }
    )
candidate_table = pd.DataFrame(candidate_rows)
display(candidate_table)
assert candidate_table.groupby(["split", "seed"])["SHA256 parameter awal"].nunique().eq(1).all()
selection_rows = []
for split in results["primary_selection"]:
    assert split["selection_only"]
    for arm in split["arms"]:
        selection_rows.append(
            {
                "split": split["split_seed"],
                "kelompok": arm["arm"],
                "model utama": arm["primary_candidate"],
                "jumlah seed": arm["n_seeds"],
                "selection NDCG mean": arm["selection_ndcg_at_10_mean"],
                "selection NDCG SD": arm["selection_ndcg_at_10_std"],
                "cosine terpilih": split["baseline_candidate"],
            }
        )
display(pd.DataFrame(selection_rows))
display(
    Markdown(
        "Hash bobot awal cocok pada kedua objective untuk setiap pasangan split/seed. "
        "RNG urutan contoh dipisahkan dari RNG negatif BPR sehingga urutan contoh "
        "per epoch dapat dipasangkan. Patience dapat menghasilkan jumlah epoch "
        "akhir berbeda. Checkpoint terbaik tepat pada batas epoch belum membuktikan konvergensi."
    )
)

# %% [markdown]
# ## Semua epoch, checkpoint, dan perubahan bobot
#
# Checkpoint terbaik ditentukan oleh NDCG selection; bukan loss train terkecil.
# Penurunan loss hanya menyatakan target latihan membaik. Gradien dan perubahan
# parameter memperlihatkan embedding, forward propagation, loss, backpropagation,
# dan Adam bekerja; manfaat rekomendasi tetap diukur dengan metrik holdout.

# %%
epoch_rows = []
for candidate in candidates:
    assert len(candidate["history"]) == candidate["epochs_completed"]
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
arms = ["history_bpr", "history_softmax"]
split_seeds = sorted({candidate["split_seed"] for candidate in candidates})
training_seeds = [42, 2026, 7]
seed_colors = {42: "tab:blue", 2026: "tab:orange", 7: "tab:green"}
fig, axes = plt.subplots(len(split_seeds), len(arms), figsize=(12, 8), squeeze=False)
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
                s=100,
                color=seed_colors[training_seed],
            )
        ax.set(title=f"Split {split_seed} — {arm}", xlabel="Epoch", ylabel="Selection NDCG@10")
        ax.legend(fontsize=8)
fig.suptitle("Semua epoch; bintang = checkpoint yang dipilih selection")
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(len(split_seeds), len(arms), figsize=(12, 8), squeeze=False)
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
fig.suptitle("Loss menurut objective masing-masing; skala BPR dan softmax berbeda")
plt.tight_layout()
plt.show()

# %%
trace_rows, update_rows, example_rows, vector_rows = [], [], [], []
for candidate in candidates:
    trace = candidate["first_step_trace"]
    trace_rows.append(
        {
            "kandidat": candidate["name"],
            "objective": trace["objective"],
            "batch": trace["batch_size"],
            "iterasi Adam setelah update": trace["optimizer_iterations"],
            **trace["loss_before_update"],
        }
    )
    for variable, value in trace["variable_update_norms"].items():
        update_rows.append(
            {
                "kandidat": candidate["name"],
                "parameter": variable,
                "norma perubahan sesudah Adam": value,
            }
        )
    for index, (before, after, count, vector) in enumerate(
        zip(
            trace["raw_positive_scores_before"],
            trace["raw_positive_scores_after"],
            trace["context_positive_counts_after_target_removal"],
            trace["context_pooled_vectors_before"],
        ),
        start=1,
    ):
        assert count >= 1, "Target training harus menyisakan riwayat konteks."
        negative = trace["raw_negative_scores_before"]
        example_rows.append(
            {
                "kandidat": candidate["name"],
                "contoh": index,
                "karya konteks sesudah target dikeluarkan": count,
                "norma vektor konteks": sum(value * value for value in vector) ** 0.5,
                "skor target sebelum": before,
                "skor target sesudah": after,
                "skor negatif BPR sebelum": negative[index - 1] if negative else None,
            }
        )
        vector_rows.append(
            {
                "kandidat": candidate["name"],
                "contoh": index,
                **{
                    f"dimensi {dimension}": value for dimension, value in enumerate(vector, start=1)
                },
            }
        )
display(pd.DataFrame(trace_rows))
display(pd.DataFrame(update_rows))
display(pd.DataFrame(example_rows).fillna("softmax memakai seluruh kandidat yang diizinkan"))
display(pd.DataFrame(vector_rows))
display(
    Markdown(
        "Vektor adalah hasil pooling embedding riwayat **sebelum Dense**, setelah target "
        "dikeluarkan. Norma gradien tercatat sebelum Adam; norma perubahan parameter "
        "tercatat sesudah Adam dan bukan gradien per parameter. Bukti mekanisme ini "
        "harus dibaca bersama tes pengecualian target dan kualitas selection/assessment."
    )
)

# %% [markdown]
# ## Assessment pada lima cohort
#
# Semua model dinilai pada pengguna dan karya target yang sama. Lima cohort
# adalah seluruh pengguna relevan, pengguna dengan satu atau lebih karya
# positif di fit, tanpa karya positif, tepat satu karya positif, dan sedikitnya
# dua karya positif. Tiga kelompok terakhir saling terpisah; “seluruh” dan
# “positif” merupakan gabungan, sehingga angkanya tidak boleh dijumlahkan
# sebagai pengamatan independen.
#
# Recall@10 sama dengan Hit Rate@10 karena ada satu target relevan per pengguna.
# NDCG@10 dan MRR@10 juga memberi bobot pada posisi target. Hanya target holdout
# yang teramati dinilai: buku lain tanpa rating bukan otomatis rekomendasi buruk.
# Cakupan katalog dan entropy menggambarkan sebaran paparan, bukan kepuasan atau
# keberagaman semantik secara langsung.
# Benchmark historis mempertahankan hasil serta kebijakan fallback aslinya.
# Model baru, cosine baru, dan BPR ID run ketujuh memakai fallback popularitas
# yang sama untuk pengguna tanpa positif. Hasil cohort positif membantu
# membedakan pengaruh neural dari fallback pada pembanding historis lainnya.

# %%
ranking_results = results["ranking_results"]
ranking_rows, cohort_rows = [], []
for ranking in ranking_results:
    assert ranking["status"] == "completed"
    cohort_rows.append(
        {
            "split": ranking["split_seed"],
            "karya katalog": ranking["catalogue_works"],
            **ranking["cohort_counts"],
        }
    )
    primaries = {
        *ranking["primary_models"].values(),
        ranking["baseline_name"],
        ranking["previous_primary_models"]["bpr_id"],
        "popularity",
    }
    for name, cohorts in ranking["metrics"].items():
        family = (
            candidate_by_name[name]["arm"] if name in candidate_by_name else "benchmark historis"
        )
        if name in {baseline["name"] for baseline in baselines}:
            family = "item cosine positif"
        if name == "popularity":
            family = "popularitas fit"
        for cohort, metrics in cohorts.items():
            assert metrics["users"] == ranking["cohort_counts"][cohort]
            ranking_rows.append(
                {
                    "split": ranking["split_seed"],
                    "model": name,
                    "kelompok": family,
                    "cohort": cohort,
                    "model utama": name in primaries,
                    **metrics,
                }
            )
ranking_table = pd.DataFrame(ranking_rows)
display(pd.DataFrame(cohort_rows))
for cohort in COHORTS:
    display(Markdown(f"**Seluruh model pada cohort `{cohort}`:**"))
    display(ranking_table[ranking_table["cohort"] == cohort].reset_index(drop=True))

# %%
primary_ranking = ranking_table[ranking_table["model utama"]]
fig, axes = plt.subplots(len(split_seeds), 3, figsize=(18, 9), squeeze=False)
for row_index, split_seed in enumerate(split_seeds):
    for column_index, cohort in enumerate(COMPARISON_COHORTS):
        ax = axes[row_index, column_index]
        group = primary_ranking[
            (primary_ranking["split"] == split_seed) & (primary_ranking["cohort"] == cohort)
        ]
        group.set_index("model")["ndcg_at_10"].plot.barh(ax=ax, color="tab:blue")
        ax.set(title=f"Split {split_seed} — {cohort}", xlabel="Assessment NDCG@10", ylabel="")
        ax.tick_params(axis="y", labelsize=7)
fig.suptitle("Model utama dipilih melalui selection sebelum assessment run ini")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Diagnosis dengan riwayat pengguna lain
#
# Setelah model utama dipilih dan dikunci, riwayat positif diganti dengan
# riwayat donor melalui pemetaan siklik tetap dari pengguna fit yang mempunyai
# riwayat positif. Bobot model tetap; ini **bukan training baru**. Karya yang
# dikeluarkan dari kandidat tetap karya fit milik penerima rekomendasi. Pengguna
# tanpa positif tetap memakai fallback popularitas.
#
# Diagnosis ini memeriksa sensitivitas model terhadap riwayat yang diberikan.
# Jika kualitas turun ketika riwayat ditukar, hasil itu mendukung kegunaan
# riwayat asli dalam protokol ini. Pertukaran dapat mengubah **isi dan panjang**
# riwayat sekaligus; hasilnya bukan identifikasi efek kausal isi buku saja.
# Daftar berubah tanpa peningkatan relevansi juga belum membuktikan personalisasi
# yang bermanfaat. Penurunan ataupun ketiadaan penurunan perlu dibaca bersama
# ukuran cohort dan interval ketidakpastian.

# %%
shuffle_metric_rows, shuffle_comparison_rows = [], []
for diagnostic in results["history_shuffle_results"]:
    assert diagnostic["diagnostic_only"] and not diagnostic["used_for_selection"]
    for cohort in COHORTS:
        normal, shuffled = (
            diagnostic[key][cohort] for key in ("normal_metrics", "shuffled_metrics")
        )
        assert normal["users"] == shuffled["users"]
        shuffle_metric_rows.append(
            {
                "split": diagnostic["split_seed"],
                "model": diagnostic["candidate"],
                "cohort": cohort,
                "pengguna": normal["users"],
                "hit riwayat asli": normal["hits_at_10"],
                "hit riwayat donor": shuffled["hits_at_10"],
                "NDCG asli": normal["ndcg_at_10"],
                "NDCG donor": shuffled["ndcg_at_10"],
                "Recall asli": normal["recall_at_10"],
                "Recall donor": shuffled["recall_at_10"],
                "karya unik asli": normal["recommended_unique_works"],
                "karya unik donor": shuffled["recommended_unique_works"],
            }
        )
    for comparison in diagnostic["paired_normal_minus_shuffled"]:
        shuffle_comparison_rows.append(
            {
                "split": diagnostic["split_seed"],
                "model": diagnostic["candidate"],
                "cohort": comparison["cohort"],
                "metrik": comparison["metric"],
                "selisih asli - donor": comparison["difference"],
                "batas bawah 95%": comparison["ci95"][0],
                "batas atas 95%": comparison["ci95"][1],
                "pengguna": comparison["users"],
                "draws": comparison["draws"],
                "seed": comparison["seed"],
            }
        )
shuffle_table = pd.DataFrame(shuffle_metric_rows)
shuffle_bootstrap = pd.DataFrame(shuffle_comparison_rows)
display(shuffle_table)
display(shuffle_bootstrap)
display(
    Markdown("Selisih **asli − donor** positif berarti kualitas lebih baik dengan riwayat asli.")
)

# %% [markdown]
# ## Variasi seed dan ketidakpastian perbandingan
#
# Tiga seed mengukur variasi inisialisasi dan mini-batch pada data yang sama;
# mereka bukan tiga dataset independen. Dua split dianalisis terpisah karena
# data/pengguna dapat tumpang tindih. Bootstrap meresampling pengguna secara
# berpasangan; selisih positif kandidat − referensi menguntungkan kandidat
# untuk Recall/NDCG/MRR. Interval bersyarat pada assessment yang dipakai ulang
# dan tidak mengoreksi banyak perbandingan.
# Interval yang ditetapkan untuk run ini mencakup cohort seluruh pengguna,
# riwayat positif, dan minimal dua positif. Cohort tanpa positif dan tepat satu
# positif tetap mempunyai metrik lengkap, tetapi tidak ditambah bootstrap
# secara post-hoc di notebook.

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
for ranking in ranking_results:
    for comparison in ranking["paired_bootstrap"]:
        bootstrap_rows.append(
            {
                "split": ranking["split_seed"],
                "kandidat": comparison["model"],
                "referensi": comparison["reference"],
                "cohort": comparison["cohort"],
                "metrik": comparison["metric"],
                "selisih kandidat - referensi": comparison["difference"],
                "batas bawah 95%": comparison["ci95"][0],
                "batas atas 95%": comparison["ci95"][1],
                "pengguna": comparison["users"],
                "draws": comparison["draws"],
                "seed": comparison["seed"],
            }
        )
bootstrap_table = pd.DataFrame(bootstrap_rows)
for cohort in COMPARISON_COHORTS:
    for metric in ("recall_at_10", "ndcg_at_10", "mrr_at_10"):
        display(Markdown(f"**Semua perbandingan `{metric}` pada `{cohort}`:**"))
        display(
            bootstrap_table[
                (bootstrap_table["cohort"] == cohort) & (bootstrap_table["metrik"] == metric)
            ].reset_index(drop=True)
        )

# %% [markdown]
# ## Keputusan yang dapat diambil dari hasil ini
#
# Peningkatan terhadap BPR ID lama, terhadap pembanding cosine riwayat, dan
# terhadap popularitas menjawab pertanyaan berbeda. Pertukaran riwayat membantu
# menguji ketergantungan masukan, bukan menggantikan evaluasi relevansi. Klaim
# umum tetap membutuhkan data konfirmasi yang tidak dipakai menyetel rancangan.
# Tidak ada model yang otomatis dipromosikan ke aplikasi berdasarkan assessment
# pengembangan ini.
#
# **Kesimpulan run ini:** memakai riwayat adalah arah yang layak, dan pembanding
# cosine menunjukkan hasil yang lebih kuat pada dua split pengembangan. Neural
# softmax berhasil memperbaiki satu split tetapi tidak mengulang keberhasilan
# itu pada split lainnya. Menambahkan arsitektur history saja belum cukup untuk
# menyatakan neural unggul. Tidak ada dasar untuk menyalahkan data secara
# keseluruhan atau menganggap tambahan epoch pasti menyelesaikan masalah;
# variasi seed, pemilihan checkpoint, cohort, dan pembanding tetap perlu dinilai.

# %%
for ranking in ranking_results:
    display(Markdown(f"**Split {ranking['split_seed']} — NDCG model utama pada riwayat positif:**"))
    comparisons = bootstrap_table[
        (bootstrap_table["split"] == ranking["split_seed"])
        & (bootstrap_table["cohort"] == "fit_positive_history")
        & (bootstrap_table["metrik"] == "ndcg_at_10")
    ]
    display(
        comparisons[
            comparisons["kandidat"].isin(ranking["primary_models"].values())
            & comparisons["referensi"].isin(
                [
                    "popularity",
                    ranking["baseline_name"],
                    ranking["previous_primary_models"]["bpr_id"],
                ]
            )
        ].reset_index(drop=True)
    )
    paired = comparisons[
        comparisons["kandidat"].str.startswith(f"split{ranking['split_seed']}_history_softmax_seed")
        & comparisons["referensi"].str.startswith(f"split{ranking['split_seed']}_history_bpr_seed")
    ]
    assert len(paired) == len(training_seeds)
    display(
        Markdown(
            f"Pada pasangan seed yang sama, softmax memiliki NDCG lebih tinggi dari BPR pada "
            f"**{int((paired['selisih kandidat - referensi'] > 0).sum())}/{len(paired)}** pasangan. "
            f"**{int((paired['batas bawah 95%'] > 0).sum())}** interval berada seluruhnya di atas nol; "
            f"**{int((paired['batas atas 95%'] < 0).sum())}** seluruhnya di bawah nol. "
            "Ini ringkasan deskriptif interval bersyarat, bukan replikasi dataset independen."
        )
    )
    display(paired.reset_index(drop=True))
display(
    Markdown(
        f"Run ini mencatat **{len(candidates)} training neural**, **{len(baselines)} konfigurasi "
        f"klasik**, dan **{len(epochs)} epoch neural**. "
        f"**{len(results['history_shuffle_results'])}** evaluasi pertukaran riwayat tidak "
        "menambah jumlah training. Diagnosis personalisasi dari run terdahulu juga "
        "analisis bukti tersimpan, bukan model yang dilatih ulang."
    )
)

# %% [markdown]
# ## Penyimpanan bukti dan reproduksi
#
# Notebook membaca JSON di repository, sedangkan model, state optimizer/RNG,
# log, serta rekomendasi rinci disimpan di workspace lokal. Manifest mencatat
# verifikasi hash, pengujian perangkat lunak, dan pemeriksaan akhir sesi Colab.
# Status runtime adalah catatan audit tersimpan, bukan pemeriksaan langsung
# ketika notebook dibuka kembali.
#
# Untuk memeriksa notebook, jalankan `scripts/notebook_quality.py check-pair`
# pada pasangan `.ipynb`/`.py` dan `check-saved` dengan tiga JSON pada
# `EVIDENCE_FILES` sebagai `--dependency`. Restart & Run All hanya membaca bukti;
# ia tidak menjalankan kembali eksperimen atau membuka assessment baru.

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
            {"catatan": "Arsip terverifikasi", "nilai": len(manifest["backups"])},
        ]
    )
)
display(pd.DataFrame(manifest["backups"]))
display(
    pd.DataFrame(
        [{"catatan akhir sesi": key, "nilai": value} for key, value in manifest["shutdown"].items()]
    )
)
display(pd.DataFrame({"batas interpretasi dari hasil run": results["limitations"]}))
assert not manifest["paid_resources_authorized"]
assert manifest["verification"]["all_backups_verified"]
