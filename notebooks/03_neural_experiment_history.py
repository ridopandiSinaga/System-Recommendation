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
# # Riwayat Eksperimen Neural Book Recommender
#
# **Neural network sudah bekerja; manfaat tambahan dan batas tuning dinilai
# dari bukti setiap run.** Notebook ini membuka seluruh konfigurasi, epoch,
# hasil dua benchmark awal, ablation regularisasi, tindak lanjut warm-up,
# evaluasi pengembangan internal dengan dua split dan tiga seed training,
# serta pembanding bias/MF yang dituning dan evaluasi ranking katalog penuh.
# Run ketujuh menambahkan objective ranking dan metadata buku; rincian seluruh
# kandidat/epoch run tersebut tersedia di [notebook 04](04_neural_ranking_experiments.ipynb).
# Run kedelapan menguji encoder riwayat positif dan pembanding cosine, disertai
# diagnosis pertukaran riwayat; lihat [notebook 05](05_neural_history_experiments.ipynb).
# Semua kandidat, termasuk kontrol dan kandidat yang tidak terpilih, ditampilkan.
# Angka dibaca dari JSON di repository; menjalankan
# notebook ini tidak melakukan training atau evaluasi test tambahan.
#
# Notebook [02_neural_book_recommender.ipynb](02_neural_book_recommender.ipynb)
# memperagakan embedding, forward propagation, loss, backpropagation, dan Adam.
# Output tersimpannya memakai subset smoke test yang berbeda dari benchmark utama
# di sini. Hasil smoke test tidak digabung sebagai skor benchmark.
#
# Jalankan `uv sync --dev`, buka notebook dari checkout proyek, lalu
# **Restart & Run All**. Tidak memerlukan TensorFlow, GPU, CSV dataset asli,
# akses Colab, atau folder checkpoint privat `artifacts/colab`.

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
        if (path / "docs/evidence/neural_colab_t4_results.json").is_file()
    ),
    None,
)
if ROOT is None:
    raise FileNotFoundError("Buka dari checkout proyek atau set BOOK_RECOMMENDER_PROJECT_ROOT.")
EVIDENCE = ROOT / "docs/evidence"
# Daftar eksplisit membuat sumber angka dan dependency stamp dapat diaudit.
EVIDENCE_FILES = {
    "original": "neural_colab_t4_results.json",
    "original_trace": "neural_colab_t4_first_step_trace.json",
    "original_manifest": "neural_colab_t4_run_manifest.json",
    "residual": "neural_colab_residual_results.json",
    "residual_manifest": "neural_colab_residual_run_manifest.json",
    "audit": "neural_followup_data_audit.json",
    "residual_signal": "neural_residual_signal_audit.json",
    "ablation": "neural_colab_ablation_results.json",
    "ablation_manifest": "neural_colab_ablation_run_manifest.json",
    "warmup": "neural_colab_warmup_results.json",
    "warmup_manifest": "neural_colab_warmup_run_manifest.json",
    "multiseed": "neural_colab_multiseed_results.json",
    "multiseed_manifest": "neural_colab_multiseed_run_manifest.json",
    "multiseed_posthoc": "neural_colab_multiseed_additional_comparisons.json",
    "comparator": "neural_colab_comparator_results.json",
    "comparator_manifest": "neural_colab_comparator_run_manifest.json",
    "ranking": "neural_colab_ranking_results.json",
    "ranking_manifest": "neural_colab_ranking_run_manifest.json",
    "history": "neural_colab_history_results.json",
    "history_manifest": "neural_colab_history_run_manifest.json",
    "personalization_diagnosis": "neural_personalization_diagnosis.json",
}
evidence = {
    key: json.loads((EVIDENCE / filename).read_text(encoding="utf-8"))
    for key, filename in EVIDENCE_FILES.items()
}
original, residual, audit = (evidence[key] for key in ("original", "residual", "audit"))
original_config, residual_config = original["configuration"], residual["configuration"]
candidates = residual["candidate_validation_results"]
selected_name = residual["selection"]["best_neural_candidate"]
pd.set_option("display.max_columns", 24)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 90)
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
# ## Dua run benchmark awal dan keamanan hasil
#
# Run residual memakai CSV input yang sama persis dengan run MLP. Karena test
# sudah dilihat pada run pertama, run kedua merupakan eksperimen pengembangan
# pada test yang dipakai ulang, bukan konfirmasi pada test baru yang tersembunyi.
# Nama `serving_candidate` di JSON berarti kandidat pilihan dalam eksperimen;
# model tersebut **belum dipromosikan ke aplikasi**.

# %%
run_rows = []
for label, result, manifest in (
    ("MLP awal", original, evidence["original_manifest"]),
    ("Bias + residual", residual, evidence["residual_manifest"]),
):
    completed = pd.Timestamp(result["completed_utc"]).tz_convert("Asia/Jakarta")
    run_rows.append(
        {
            "run": label,
            "session": manifest["session_name"],
            "selesai (Asia/Jakarta)": completed.isoformat(),
            "GPU": result["gpu"]["gpu_details"][0]["device_name"],
            "TensorFlow": result["gpu"]["tensorflow_version"],
            "durasi eksperimen (detik)": result["total_seconds"],
            "arsip terverifikasi": len(manifest["backups"]),
            "tes perangkat lunak lulus": manifest["runtime_tests"]["passed"],
            "runtime dihentikan": manifest["shutdown"]["stopped"],
            "sumber daya berbayar diizinkan": manifest.get("paid_resources_authorized", False),
        }
    )
display(pd.DataFrame(run_rows))
assert residual["dataset"]["input_bytes_match_original_experiment"]
assert residual["previous_published_original_nn"]["same_input_sha256_verified"]
assert not residual["selection"]["test_evaluated_at_selection"]
assert not evidence["residual_manifest"]["model_promotion_to_application"]
display(
    Markdown(
        "Verifikasi input: **CSV identik**. Seleksi kandidat residual dilakukan **sebelum "
        "evaluasi test pada run tersebut**. Status backup/shutdown di atas adalah catatan "
        "tersimpan, bukan pemeriksaan runtime secara langsung. Tes perangkat lunak "
        "menilai kebenaran implementasi; jumlah tes bukan ukuran kualitas rekomendasi."
    )
)

# %% [markdown]
# ## Data mendukung pembelajaran, tetapi jarang dan cakupannya terbatas
#
# Rating eksplisit 1–10 dinormalisasi menjadi `(rating - 1) / 9`. Nilai 0 pada
# sumber data tidak dipakai sebagai rating negatif. Split dilakukan pada karya
# kanonik per pengguna untuk menghindari kebocoran antar-edisi; katalog dan
# vocabulary dibentuk dari train. Evaluasi rating neural hanya berlaku ketika
# ID pengguna **dan** ISBN tersedia dalam train. Cold-start memakai fallback.

# %%
dataset = original["dataset"]
display(
    pd.DataFrame(
        [
            ("Rating mentah", dataset["raw_ratings_rows"]),
            (
                "Interaksi implicit/rating 0 yang dikeluarkan",
                dataset["implicit_interactions_excluded"],
            ),
            ("Rating eksplisit mentah", dataset["raw_explicit_rating_rows"]),
            (
                "Rating eksplisit bersih dengan metadata",
                dataset["clean_explicit_interactions_with_metadata"],
            ),
            ("Pasangan training", audit["train_pairs"]),
            ("Pengguna training", audit["train_users"]),
            ("ISBN training", audit["train_books"]),
            ("Kepadatan matriks (%)", 100 * audit["train_density"]),
            ("Median rating per pengguna", audit["median_ratings_per_user"]),
            (
                "Pengguna dengan paling banyak 3 rating (%)",
                100 * audit["users_at_most_3_ratings_fraction"],
            ),
            (
                "Pengguna dengan paling banyak 5 rating (%)",
                100 * audit["users_at_most_5_ratings_fraction"],
            ),
        ],
        columns=["ukuran", "nilai"],
    )
)
availability_names = {
    "both_known": "Pengguna dan ISBN dikenal",
    "known_user_unknown_book": "Pengguna dikenal, ISBN baru",
    "unknown_user_known_book": "Pengguna baru, ISBN dikenal",
    "both_unknown": "Pengguna dan ISBN baru",
}
coverage = pd.DataFrame(
    {
        split: {label: audit[split][key] for key, label in availability_names.items()}
        for split in ("validation", "test")
    }
)
assert (coverage.sum() == [audit["validation"]["pairs"], audit["test"]["pairs"]]).all()
display(coverage)
ax = coverage.T.plot.bar(stacked=True, figsize=(9, 4), rot=0)
ax.set(ylabel="Jumlah pasangan rating", title="Cakupan ID dari vocabulary training")
ax.legend(title="Kelompok", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Konfigurasi dua run benchmark awal
#
# MLP awal: `concat(user_embedding, book_embedding, user * book)` → Dense ReLU
# 32 → Dense ReLU 16 → sigmoid. Residual: baseline global + bias pengguna + bias
# buku, ditambah `scale × MLP` dengan hidden LeakyReLU dan output linear yang
# dimulai dari nol. Bias dihitung dari train dan dibekukan.
#
# Semua neural memakai Adam, MSE rating ternormalisasi + L2, embedding 16,
# hidden 32/16, learning rate 0.001, batch 1024, patience 3, dan seed 42.
# Bias-only adalah kontrol tanpa training neural. L2 dan bentuk hidden tidak
# berpengaruh pada prediksinya karena residual scale = 0.
#
# Tabel ini menampilkan **4 konfigurasi neural yang dilatih + 1 kontrol**.
# Baris “tidak dievaluasi” bukan skor nol: dua kandidat residual yang tidak
# terpilih memang tidak dievaluasi pada test.

# %%
experiment_rows = [
    {
        "percobaan": "mlp_original",
        "keluarga": "MLP ReLU + sigmoid",
        "embedding": original_config["embedding_size"],
        "hidden": str(original_config["hidden_units"]),
        "learning_rate": original_config["learning_rate"],
        "batch": original_config["batch_size"],
        "L2": original_config["l2"],
        "residual_scale": None,
        "user/item shrinkage": "tidak berlaku",
        "seed": original_config["seed"],
        "epoch maksimum": original_config["requested_epochs"],
        "epoch selesai": original["training"]["epochs_completed"],
        "epoch terbaik": original["training"]["best_epoch"],
        "validation RMSE": min(row["val_rmse"] for row in original["training"]["history"]),
        "test RMSE (135)": original["rating_test"]["models"]["neural"]["rmse_rating_scale"],
        "status test": "dievaluasi pada run awal",
    }
]
for candidate in candidates:
    is_bias = candidate["kind"] == "bias_only"
    test_model = (
        "bias_only" if is_bias else "best_neural" if candidate["name"] == selected_name else None
    )
    experiment_rows.append(
        {
            "percobaan": candidate["name"],
            "keluarga": "kontrol bias-only" if is_bias else "bias + MLP LeakyReLU",
            "embedding": None if is_bias else residual_config["embedding_size"],
            "hidden": "tidak digunakan" if is_bias else str(residual_config["hidden_units"]),
            "learning_rate": None if is_bias else residual_config["learning_rate"],
            "batch": None if is_bias else residual_config["batch_size"],
            "L2": None if is_bias else candidate["l2"],
            "residual_scale": candidate["residual_scale"],
            "user/item shrinkage": f"{candidate['user_shrinkage']:g}/{candidate['item_shrinkage']:g}",
            "seed": residual_config["seed"],
            "epoch maksimum": 0 if is_bias else residual_config["max_neural_epochs_each"],
            "epoch selesai": candidate["epochs_completed"],
            "epoch terbaik": candidate["best_epoch"],
            "validation RMSE": candidate["validation"]["rmse_rating_scale"],
            "test RMSE (135)": (
                residual["rating_test"]["models"][test_model]["known_pairs"]["rmse_rating_scale"]
                if test_model
                else None
            ),
            "status test": "kontrol dievaluasi"
            if is_bias
            else "terpilih; dievaluasi"
            if test_model
            else "tidak dievaluasi",
        }
    )
experiments = pd.DataFrame(experiment_rows)
display(experiments.fillna("—"))
display(
    Markdown(
        f"Kandidat terpilih di antara empat kandidat run residual: **`{selected_name}`**. "
        "Pemilihan ini tidak memasukkan MLP dari run pertama sebagai kandidat baru "
        "dan tidak berarti residual mengalahkan MLP pada validation."
    )
)

# %% [markdown]
# ## Semua epoch, termasuk yang bukan checkpoint terbaik
#
# RMSE dalam skala rating 1–10; MSE dalam skala normalisasi 0–1 dan tidak
# termasuk penalti L2. Pada residual, objective optimisasi mini-batch memakai
# output linear mentah + L2. Metrik train per epoch pada history, validation,
# dan test memotong prediksi ke rentang rating valid sebelum dihitung.
# Checkpoint terbaik dipulihkan setelah early stopping. Grafik setiap kandidat
# memakai sumbu sendiri agar perubahan kecil terlihat; bandingkan angka tabel
# untuk besaran absolut.

# %%
histories = {"mlp_original": original["training"]["history"]}
histories.update({c["name"]: c["history"] for c in candidates if c["history"]})
best_epochs = experiments.set_index("percobaan")["epoch terbaik"].to_dict()
epochs = pd.DataFrame(
    [
        {"percobaan": name, **record, "checkpoint terbaik": record["epoch"] == best_epochs[name]}
        for name, history in histories.items()
        for record in history
    ]
)
display(epochs)
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (name, history) in zip(axes.flat, histories.items(), strict=True):
    frame = pd.DataFrame(history)
    ax.plot(frame["epoch"], frame["train_rmse"], "o-", label="Train RMSE")
    ax.plot(frame["epoch"], frame["val_rmse"], "o-", label="Validation RMSE")
    ax.axvline(best_epochs[name], color="black", linestyle=":", label="Checkpoint terbaik")
    ax.set(title=name, xlabel="Epoch", ylabel="RMSE rating 1–10", xticks=frame["epoch"])
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
assert (
    len(epochs)
    == original["training"]["epochs_completed"] + residual["training"]["total_neural_epochs"]
)

# %%
residual_validation = experiments[experiments["percobaan"] != "mlp_original"]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.barh(residual_validation["percobaan"], residual_validation["validation RMSE"], color="#3876a5")
for index, value in enumerate(residual_validation["validation RMSE"]):
    ax.text(value, index, f"  {value:.9f}", va="center", fontsize=9)
ax.set(xlabel="Validation RMSE (140 pasangan; lebih kecil lebih baik)", xlim=(0, 2.25))
plt.tight_layout()
plt.show()
neural_candidates = sorted(
    (c for c in candidates if c["kind"] == "neural"),
    key=lambda c: c["validation"]["rmse_rating_scale"],
)
candidate_gap = (
    neural_candidates[1]["validation"]["rmse_rating_scale"]
    - neural_candidates[0]["validation"]["rmse_rating_scale"]
)
display(
    Markdown(
        f"Dua kandidat residual teratas hanya berbeda **{candidate_gap:.8f} RMSE** "
        "pada 140 pasangan validation. Ini belum membuktikan nilai L2 terpilih optimal. "
        "MLP awal berhenti membaik setelah epoch 2; residual terbaik berada di epoch 1. "
        "Menambah epoch saja belum didukung hasil ini."
    )
)

# %% [markdown]
# ## Rating: cohort yang sama dan seluruh pengguna dengan fallback
#
# Tabel pertama membandingkan 135 pasangan dengan kedua ID dikenal. Angka MLP
# berasal dari run awal, dengan input identik; model itu tidak dilatih ulang
# pada run residual. Tabel berikutnya mencakup 500 pasangan. Untuk bias/residual,
# 365 pasangan cold memakai fallback tetap: mean pengguna → mean ISBN → mean
# global dari train. Jadi angka 500 pasangan tidak berarti embedding melayani
# semua pengguna/buku. Empat prediksi warm bias/residual dipotong ke rentang 1–10.
#
# Baseline tersedia lengkap, termasuk mean item. `selected_serving` dan
# `best_neural` pada JSON adalah model yang sama dalam run ini; tidak dihitung
# sebagai dua percobaan berbeda.

# %%
model_labels = {
    "train_global_mean": "Mean global",
    "train_user_mean_global_fallback": "Mean user → global",
    "train_item_mean_global_fallback": "Mean ISBN → global",
    "train_user_item_global_fallback": "Mean user → ISBN → global",
    "bias_only": "Bias-only",
    "best_neural": "Residual terpilih",
}
warm_rows = [
    {
        "model": "MLP awal",
        "pasangan": original["rating_test"]["known_pairs"],
        "RMSE": original["rating_test"]["models"]["neural"]["rmse_rating_scale"],
        "MAE": original["rating_test"]["models"]["neural"]["mae_rating_scale"],
    }
]
full_rows, group_rows = [], []
for key, label in model_labels.items():
    model_result = residual["rating_test"]["models"][key]
    for scope, target in (("known_pairs", warm_rows), ("all_pairs", full_rows)):
        metric = model_result[scope]
        target.append(
            {
                "model": label,
                "pasangan": metric["pairs"],
                "RMSE": metric["rmse_rating_scale"],
                "MAE": metric["mae_rating_scale"],
            }
        )
    for group, metric in model_result["by_availability"].items():
        group_rows.append(
            {
                "model": label,
                "kelompok": group,
                "pasangan": metric["pairs"],
                "RMSE": metric["rmse_rating_scale"],
                "MAE": metric["mae_rating_scale"],
            }
        )
display(Markdown("**135 pasangan warm: seluruh hasil rating yang dievaluasi**"))
display(pd.DataFrame(warm_rows))
display(Markdown("**500 pasangan dengan fallback: run residual**"))
display(pd.DataFrame(full_rows))
display(Markdown("**Rincian menurut ketersediaan ID**"))
display(pd.DataFrame(group_rows))

# %% [markdown]
# ## Ketidakpastian: urutan validation dan test dapat berbalik
#
# Bootstrap berikut sudah tersimpan dari run residual: 1.000 pengambilan ulang
# pasangan yang sama, seed 123. Interval ini bersyarat pada satu split test yang
# telah dipakai ulang; ia tidak memperbaiki bias akibat proses pengembangan.
# Interval melintasi nol, sehingga peningkatan residual atas mean pengguna
# belum meyakinkan. Mengalahkan mean pengguna juga berbeda dari mengalahkan
# kontrol bias-only, yang justru memiliki RMSE test sedikit lebih rendah.

# %%
bootstrap = residual["rating_test"]["best_neural_vs_user_mean_paired_bootstrap"]
display(
    pd.DataFrame(
        [
            {
                "perbandingan": "Residual RMSE − mean user RMSE",
                "pasangan": bootstrap["pairs"],
                "bootstrap draws": bootstrap["draws"],
                "selisih": bootstrap["point_estimate"],
                "batas bawah 95%": bootstrap["interval_95_percent"][0],
                "batas atas 95%": bootstrap["interval_95_percent"][1],
            }
        ]
    )
)
ordering = pd.DataFrame(
    [
        ("MLP awal", experiments.iloc[0]["validation RMSE"], warm_rows[0]["RMSE"]),
        (
            "Mean user",
            residual["selection"]["validation_user_mean"]["rmse_rating_scale"],
            residual["rating_test"]["models"]["train_user_mean_global_fallback"]["known_pairs"][
                "rmse_rating_scale"
            ],
        ),
        (
            "Bias-only",
            candidates[0]["validation"]["rmse_rating_scale"],
            residual["rating_test"]["models"]["bias_only"]["known_pairs"]["rmse_rating_scale"],
        ),
        (
            "Residual terpilih",
            residual["selection"]["best_neural_validation"]["rmse_rating_scale"],
            residual["rating_test"]["models"]["best_neural"]["known_pairs"]["rmse_rating_scale"],
        ),
    ],
    columns=["model", "validation RMSE (140)", "test RMSE (135)"],
)
display(ordering)

# %% [markdown]
# ## Ranking: tujuan rekomendasi belum membaik
#
# Evaluasi ranking memakai 272 holdout dengan rating ≥8 dan katalog kandidat
# yang sama (5.000 ISBN / 4.364 karya). Target yang tidak ada dalam katalog
# tetap dihitung sebagai miss. Hanya 93 holdout memiliki target kandidat dan
# riwayat pengguna; cohort bersyarat ini ditampilkan terpisah.
#
# Satu holdout relevan per pengguna membuat Recall@10 = Hit Rate@10 pada
# protokol ini. Objective pemilihan model adalah RMSE rating, bukan NDCG/Recall.
# Hasil MLP dan residual yang sama di sini tidak membuktikan skor prediksinya
# identik; hanya metrik ranking agregatnya yang sama.
# Kedua hit pada jalur neural berasal dari fallback popularity untuk pengguna
# baru; pada 93 kasus dengan target dan riwayat tersedia, neural memperoleh 0 hit.

# %%
display(pd.DataFrame([residual["ranking_test"]["availability"]]))
ranking_rows = []
for run_label, result in (("MLP awal", original), ("Residual terpilih", residual)):
    for model_name, model_result in result["ranking_test"]["models"].items():
        for scope, count_key, metric_key in (
            ("seluruh target relevan", "count", "metrics"),
            ("target + history tersedia", "conditional_count", "conditional_metrics"),
        ):
            count, metrics = model_result[count_key], model_result[metric_key]
            ranking_rows.append(
                {
                    "run": run_label,
                    "model": model_name,
                    "cohort": scope,
                    "jumlah": count,
                    "hits@10": round(metrics["hit_rate_at_10"] * count),
                    **metrics,
                }
            )
ranking = pd.DataFrame(ranking_rows)
display(ranking)
fig, ax = plt.subplots(figsize=(9, 4))
ranking_plot = ranking[
    (ranking["run"] == "Residual terpilih") & (ranking["cohort"] == "seluruh target relevan")
]
ax.bar(
    ["Residual + popularity\nfallback", "Item-based + popularity\nfallback", "Popularity"],
    ranking_plot["hit_rate_at_10"] * 100,
)
ax.set(ylabel="Hit Rate@10 (%)", title="272 target relevan; kandidat yang hilang tetap miss")
for index, (_, row) in enumerate(ranking_plot.iterrows()):
    ax.text(
        index,
        row["hit_rate_at_10"] * 100,
        f"{row['hits@10']} / {row['jumlah']}",
        ha="center",
        va="bottom",
    )
ax.set_ylim(0, max(ranking_plot["hit_rate_at_10"]) * 125)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bukti satu langkah: embedding → forward → loss → gradient → Adam
#
# Trace pertama setiap neural menyimpan delapan pasangan encoded ID, target,
# prediksi sebelum/sesudah pembaruan, loss, norma gradient per variabel, dan
# satu vektor embedding yang benar-benar berubah. Ini membuktikan mekanisme
# training; delapan contoh ini bukan hasil evaluasi generalisasi.
#
# **Perhatian pada residual:** output head dimulai dari nol. Pada langkah
# pertama, gradient dari data menuju embedding adalah nol; perubahan embedding
# saat itu berasal dari regularisasi. Setelah head berubah, gradient data dapat
# mengalir ke embedding. Trace langkah pertama tidak cukup untuk menyimpulkan
# besarnya kontribusi neural setelah training.

# %%
traces = {"mlp_original": evidence["original_trace"]}
traces.update({c["name"]: c["first_step_trace"] for c in candidates if c["first_step_trace"]})
trace_rows, prediction_rows, gradient_rows, embedding_rows = [], [], [], []
for name, trace in traces.items():
    trace_rows.append(
        {
            "percobaan": name,
            "MSE sebelum": trace["mse_before"],
            "L2 loss": trace["regularization_loss"],
            "L2 / MSE": trace["regularization_loss"] / trace["mse_before"],
            "total loss sebelum": trace["total_loss_before"],
            "MSE setelah": trace["mse_after"],
            "norma perubahan embedding": trace["embedding_delta_norm"],
            "gradient data embedding sebelum": trace.get(
                "embedding_data_gradient_norm_before", "tidak dipisah dalam trace"
            ),
            "optimizer": trace["optimizer"],
            "iterasi optimizer": trace["optimizer_iterations"],
        }
    )
    for index, pair in enumerate(trace["encoded_pairs"]):
        prediction_rows.append(
            {
                "percobaan": name,
                "user_idx": pair[0],
                "book_idx": pair[1],
                "target [0,1]": trace["targets_normalized"][index],
                "prediksi sebelum": trace["predictions_before"][index],
                "prediksi setelah": trace["predictions_after"][index],
            }
        )
    gradient_rows.extend(
        {"percobaan": name, "variabel": variable, "norma gradient total": norm}
        for variable, norm in trace["gradient_norms"].items()
    )
    for index, (before, gradient, after) in enumerate(
        zip(
            trace["user_embedding_before"],
            trace["user_embedding_gradient"],
            trace["user_embedding_after"],
            strict=True,
        )
    ):
        embedding_rows.append(
            {
                "percobaan": name,
                "dimensi": index,
                "sebelum": before,
                "gradient total": gradient,
                "setelah": after,
                "delta": after - before,
            }
        )
display(pd.DataFrame(trace_rows))
display(Markdown("**Semua prediksi dari mini-batch trace**"))
display(pd.DataFrame(prediction_rows))
display(Markdown("**Norma gradient semua variabel**"))
display(pd.DataFrame(gradient_rows))
display(Markdown("**Seluruh 16 dimensi embedding pengguna yang dicatat per model**"))
display(pd.DataFrame(embedding_rows))

# %% [markdown]
# ## Diagnosis bobot tersimpan: seberapa personal koreksi residual?
#
# Audit tambahan membaca bobot checkpoint dengan NumPy, melakukan forward pass
# pada train dan validation yang sama, dan memeriksa kecocokan RMSE validation
# terhadap hasil TensorFlow tersimpan. Tidak ada training atau evaluasi test
# tambahan. Script reproduksinya:
# [`scripts/audit_residual_signal.py`](../scripts/audit_residual_signal.py).
#
# Koreksi berikut dinyatakan pada skala rating 1–10, **sebelum clipping**, yaitu
# `9 × residual_scale × output_MLP`. Standar deviasi yang sangat kecil berarti
# hampir semua pasangan mendapat koreksi yang sama; nilai rata-rata koreksi
# saja bukan bukti personalisasi. Rasio norma embedding dibandingkan kontrol
# tanpa training dengan inisialisasi seed yang sama.

# %%
signal_audit = evidence["residual_signal"]
assert signal_audit["selected_candidate"] == selected_name
signal_rows, norm_rows, agreement_rows = [], [], []
for candidate in signal_audit["candidates"]:
    for split in ("train", "validation"):
        metric = candidate[f"{split}_residual_rating"]
        signal_rows.append(
            {
                "percobaan": candidate["name"],
                "cohort": split,
                "pasangan": metric["pairs"],
                "koreksi mean": metric["mean"],
                "koreksi std": metric["std"],
                "koreksi min": metric["min"],
                "koreksi max": metric["max"],
            }
        )
    for layer, norm in candidate["embedding_norms"].items():
        ratio = candidate["embedding_norm_ratio_to_untrained_control"][layer]
        norm_rows.append(
            {
                "percobaan": candidate["name"],
                "embedding": layer,
                "norma tersimpan": norm,
                "rasio terhadap kontrol awal": ratio,
                "penurunan norma (%)": (1 - ratio) * 100,
            }
        )
    agreement_rows.append({"percobaan": candidate["name"], **candidate["validation_check"]})
    assert candidate["validation_check"]["passed"]
display(Markdown("**Koreksi residual untuk setiap pasangan train dan validation warm**"))
display(pd.DataFrame(signal_rows))
display(Markdown("**Norma embedding checkpoint terhadap kontrol awal**"))
display(pd.DataFrame(norm_rows))
display(Markdown("**Pemeriksaan kesesuaian forward NumPy dengan metrik TensorFlow tersimpan**"))
display(pd.DataFrame(agreement_rows))
selected_signal = next(c for c in signal_audit["candidates"] if c["name"] == selected_name)
display(
    Markdown(
        f"Koreksi residual terpilih pada **{signal_audit['train_pairs']:,}** pasangan "
        f"train: mean **{selected_signal['train_residual_rating']['mean']:.8f}** rating, "
        f"standar deviasi **{selected_signal['train_residual_rating']['std']:.9f}**. "
        "Ini bukti bahwa koreksi yang dipelajari hampir konstan, bukan personalisasi "
        "yang kuat. Penyusutan embedding terukur; apakah L2 menyebabkan hilangnya "
        "sinyal harus diuji melalui ablation terkontrol."
    )
)

# %% [markdown]
# ## Ablation lanjutan: apakah regularisasi menekan kontribusi neural?
#
# Run ketiga menguji diagnosis koreksi hampir konstan, **hanya pada train dan
# validation yang sama**. Worker tidak menerima atau membaca CSV test, dan
# tidak menghasilkan metrik test/ranking baru. Validation ini sudah dipakai
# untuk pengembangan sebelumnya; perbaikan di sini perlu dikonfirmasi melalui
# evaluasi independen setelah rancangan model dibekukan.
#
# Dua kontrol membedakan kemampuan neural dari kalibrasi sederhana:
# **bias-only** dan **bias + satu konstanta yang dihitung dari residual train**.
# Lima neural memakai head nol dengan L2 1e-4, 1e-5, 1e-6, 1e-7, atau 0;
# satu neural mengubah hanya inisialisasi head menjadi acak kecil pada L2 1e-6.
# Faktor lain tetap. Setiap epoch, checkpoint terbaik, gradient data/L2,
# norma embedding, dan variasi koreksi disimpan agar mekanisme dapat diperiksa.
# L2 1e-4 dan 1e-5 dengan head nol sengaja mengulang konfigurasi residual
# terdahulu; delapan baris run ini bukan delapan konfigurasi yang seluruhnya baru.
#
# Kontrol skalar memakai rata-rata `(target_normalized - baseline_raw)` pada
# train. Tidak ada konstanta yang dioptimalkan menggunakan validation.

# %%
ablation = evidence["ablation"]
ablation_manifest = evidence["ablation_manifest"]
assert ablation["schema_version"] == 1
assert ablation["test_used"] is False
assert ablation["dataset"]["input_bytes_match_original_experiment"]
assert ablation["dataset"]["canonical_work_leakage_checks_passed"]
assert ablation["dataset"]["train_pairs"] == audit["train_pairs"]
assert ablation["dataset"]["validation_known_pairs"] == audit["validation"]["both_known"]
ablation_config = ablation["configuration"]
ablation_candidates = ablation["candidate_validation_results"]
ablation_by_name = {candidate["name"]: candidate for candidate in ablation_candidates}
ablation_neural = [candidate for candidate in ablation_candidates if candidate["kind"] == "neural"]
display(
    pd.DataFrame(
        [
            {
                "run": "Ablation regularisasi dan inisialisasi head",
                "session": ablation_manifest["session_name"],
                "selesai (Asia/Jakarta)": pd.Timestamp(ablation["completed_utc"])
                .tz_convert("Asia/Jakarta")
                .isoformat(),
                "GPU": ablation["gpu"]["gpu_details"][0]["device_name"],
                "TensorFlow": ablation["gpu"]["tensorflow_version"],
                "durasi (detik)": ablation["total_seconds"],
                "status": ablation["status"],
                "semua kandidat selesai": ablation["all_candidates_completed"],
                "pasangan train": ablation["dataset"]["train_pairs"],
                "validation tersedia": ablation["dataset"]["validation_pairs"],
                "validation kedua ID dikenal": ablation["dataset"]["validation_known_pairs"],
                "test digunakan": ablation["test_used"],
                "arsip terverifikasi": len(ablation_manifest["backups"]),
                "runtime dihentikan": ablation_manifest["shutdown"]["stopped"],
                "sumber daya berbayar diizinkan": ablation_manifest.get(
                    "paid_resources_authorized", False
                ),
            }
        ]
    )
)
ablation_rows = []
for candidate in ablation_candidates:
    metric = candidate.get("validation") or {}
    is_neural = candidate["kind"] == "neural"
    ablation_rows.append(
        {
            "percobaan": candidate["name"],
            "jenis": candidate["kind"],
            "status": candidate["status"],
            "catatan kegagalan": candidate.get("message", "—"),
            "L2": candidate.get("l2") if is_neural else None,
            "head": candidate.get("head_initialization") if is_neural else "tidak digunakan",
            "embedding": ablation_config["embedding_size"] if is_neural else None,
            "hidden": str(ablation_config["hidden_units"]) if is_neural else "—",
            "learning rate": ablation_config["learning_rate"] if is_neural else None,
            "batch": ablation_config["batch_size"] if is_neural else None,
            "seed": ablation_config["seed"],
            "epoch maksimum": ablation_config["max_neural_epochs_each"] if is_neural else 0,
            "epoch selesai": candidate["epochs_completed"],
            "epoch terbaik": candidate["best_epoch"],
            "validation pasangan": metric.get("pairs"),
            "validation RMSE": metric.get("rmse_rating_scale"),
            "validation MAE": metric.get("mae_rating_scale"),
            "konstanta skala rating": (
                9 * candidate["scalar_intercept_normalized"]
                if "scalar_intercept_normalized" in candidate
                else None
            ),
            "durasi (detik)": candidate.get("seconds"),
            "status test": "tidak dijalankan",
        }
    )
ablation_experiments = pd.DataFrame(ablation_rows)
display(ablation_experiments.fillna("—"))
fig, ax = plt.subplots(figsize=(11, 4))
comparison = ablation_experiments.sort_values("validation RMSE", ascending=False)
ax.barh(comparison["percobaan"], comparison["validation RMSE"], color="#3876a5")
for index, value in enumerate(comparison["validation RMSE"]):
    if pd.notna(value):
        ax.text(value, index, f"  {value:.6f}", va="center", fontsize=8)
ax.axvline(
    experiments.iloc[0]["validation RMSE"],
    color="#ae512d",
    linestyle="--",
    label="MLP awal: referensi historis, CSV validation sama",
)
ax.set(
    xlabel="RMSE validation, 140 pasangan (lebih kecil lebih baik)",
    xlim=(0, max(2.25, comparison["validation RMSE"].max() + 0.3)),
)
ax.legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.show()
display(Markdown("**Konfigurasi lengkap run; termasuk batas waktu dan spesifikasi kandidat**"))
display(pd.json_normalize(ablation_config).T.rename(columns={0: "nilai"}))
display(Markdown("**Kriteria dan hasil seleksi berdasarkan validation**"))
display(pd.json_normalize(ablation["selection"]).T.rename(columns={0: "nilai"}))

# %% [markdown]
# ### Semua epoch ablation dan checkpoint terbaik
#
# Checkpoint untuk evaluasi akhir setiap kandidat dipilih memakai validation,
# lalu bobot terbaik dipulihkan. Epoch terakhir tidak otomatis menjadi hasil
# yang dipakai. Kontrol tidak memiliki epoch training neural. Kolom durasi,
# status, dan jumlah epoch di atas mencatat apa yang benar-benar selesai.

# %%
ablation_epochs = pd.DataFrame(
    [
        {
            "percobaan": candidate["name"],
            **record,
            "checkpoint terbaik": record["epoch"] == candidate["best_epoch"],
        }
        for candidate in ablation_neural
        for record in candidate["history"]
    ]
)
with pd.option_context("display.max_rows", None):
    display(ablation_epochs)
assert len(ablation_epochs) == ablation["training"]["total_neural_epochs"]
plot_candidates = [candidate for candidate in ablation_neural if candidate["history"]]
fig, axes = plt.subplots(3, 2, figsize=(12, 10), squeeze=False)
for ax, candidate in zip(axes.flat, plot_candidates):
    frame = pd.DataFrame(candidate["history"])
    ax.plot(frame["epoch"], frame["train_rmse"], "o-", label="Train")
    ax.plot(frame["epoch"], frame["val_rmse"], "o-", label="Validation")
    ax.axvline(candidate["best_epoch"], color="black", linestyle=":", label="Terbaik")
    ax.set(title=candidate["name"], xlabel="Epoch", ylabel="RMSE", xticks=frame["epoch"])
    ax.legend(fontsize=8)
for ax in list(axes.flat)[len(plot_candidates) :]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Gradient dan koreksi selama training, bukan hanya satu langkah pertama
#
# Diagnosis dihitung pada sampel train tetap dan seluruh pasangan validation
# dengan kedua ID dikenal. Norma gradient **data** dan **L2** dipisahkan untuk
# melihat sumber pembaruan bobot. Besarnya norma saja tidak menyatakan arah
# gradient atau membuktikan generalisasi. Koreksi adalah tambahan MLP terhadap
# baseline bias pada skala rating, sebelum clipping. Standar deviasi koreksi
# mengukur variasinya antar-pasangan; variasi besar belum berarti prediksi benar.
# Koreksi dihitung sebagai selisih output dan baseline float32, sehingga variasi
# sangat kecil mendekati presisi numerik tidak dianggap sinyal personalisasi.
# Gradient dan loss diagnosis memakai output mentah pada probe train tetap,
# bukan semua rating training. Gradient embedding menjumlahkan indeks berulang
# sebelum normanya dihitung; jangan membandingkan angkanya langsung dengan
# norma trace historis yang memakai representasi sparse berbeda. Metrik RMSE
# per epoch dan hasil validation memakai prediksi yang dipotong ke rentang 1–10.
# Head acak kecil hanya mengubah kernel output; backbone dan urutan batch sama.
#
# `initial` adalah kondisi awal, `after_first_update` sesudah satu update,
# `epoch_end` sebelum pemulihan bobot, dan `restored_best` checkpoint yang benar
# dipakai untuk hasil validation. Tabel lengkap mempertahankan semua tahap.
# Pemulihan terbaik hanya mengganti bobot untuk inferensi; counter Adam tetap
# menunjukkan epoch terakhir. Checkpoint pemulihan training disimpan terpisah
# dengan pasangan bobot dan state optimizer yang sesuai.

# %%
ablation_diagnostic_rows, ablation_gradient_rows, ablation_correction_rows = [], [], []
for candidate in ablation_candidates:
    for diagnostic in candidate.get("diagnostics", []):
        identity = {
            "percobaan": candidate["name"],
            "tahap": diagnostic["stage"],
            "epoch": diagnostic["epoch"],
            "iterasi optimizer": diagnostic.get("optimizer_iterations"),
        }
        gradients = diagnostic.get("gradient_norm_totals", {})
        embedding_norms = diagnostic.get("embedding_norms", {})
        ablation_diagnostic_rows.append(
            {
                **identity,
                "MSE data sampel": diagnostic.get("data_mse"),
                "jumlah pasangan probe gradient": diagnostic.get("gradient_probe_pairs"),
                "L2 loss": diagnostic.get("regularization_loss"),
                "norma gradient data": gradients.get("data"),
                "norma gradient L2": gradients.get("l2"),
                "norma embedding user": embedding_norms.get("user_l2"),
                "norma embedding ISBN": embedding_norms.get("book_l2"),
                "epoch state optimizer": diagnostic.get("optimizer_state_epoch"),
            }
        )
        for variable, gradient in diagnostic.get("gradient_norms", {}).items():
            ablation_gradient_rows.append(
                {
                    **identity,
                    "variabel": variable,
                    "norma gradient data": gradient["data"],
                    "norma gradient L2": gradient["l2"],
                }
            )
        for cohort, correction in diagnostic.get("corrections", {}).items():
            ablation_correction_rows.append(
                {
                    **identity,
                    "cohort": cohort,
                    "pasangan": correction["pairs"],
                    "koreksi mean": correction["mean_rating"],
                    "koreksi std": correction["std_rating"],
                    "koreksi min": correction["min_rating"],
                    "koreksi max": correction["max_rating"],
                }
            )

ablation_diagnostics = pd.DataFrame(ablation_diagnostic_rows)
ablation_corrections = pd.DataFrame(ablation_correction_rows)
display(Markdown("**Variasi koreksi pada checkpoint terbaik setiap neural**"))
display(
    ablation_corrections[
        (ablation_corrections["tahap"] == "restored_best")
        & (ablation_corrections["cohort"] == "validation")
    ].merge(
        ablation_experiments[["percobaan", "validation RMSE", "L2", "head"]],
        on="percobaan",
        validate="one_to_one",
    )
)
with pd.option_context("display.max_rows", None):
    display(Markdown("**Semua tahap: loss, norma gradient, dan norma embedding**"))
    display(ablation_diagnostics)
    display(Markdown("**Sebaran koreksi: sampel train dan validation**"))
    display(ablation_corrections)
    display(Markdown("**Gradient setiap variabel pada setiap tahap**"))
    display(pd.DataFrame(ablation_gradient_rows))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, frame in ablation_corrections[
    (ablation_corrections["tahap"] == "epoch_end")
    & (ablation_corrections["cohort"] == "validation")
].groupby("percobaan", sort=False):
    axes[0].plot(frame["epoch"], frame["koreksi mean"], "o-", label=name)
    axes[1].plot(frame["epoch"], frame["koreksi std"], "o-", label=name)
axes[0].set(title="Rata-rata koreksi pada validation", xlabel="Epoch", ylabel="Poin rating")
axes[1].set(title="Variasi koreksi antar-pasangan validation", xlabel="Epoch", ylabel="Std rating")
axes[1].set_yscale("symlog", linthresh=1e-7)
axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Overfit mini-batch nyata: diagnosis kapasitas, bukan skor benchmark
#
# Percobaan terpisah mempelajari berulang kali satu mini-batch train berukuran
# 128 pasangan tanpa L2. Ia menguji apakah jalur pembelajaran mampu menurunkan
# loss pada data yang dilihat berulang. Tidak memakai validation untuk seleksi,
# tidak memakai test, dan tidak dapat dipakai sebagai ukuran rekomendasi pada
# pengguna baru. Ukuran model dan optimizer mengikuti konfigurasi diagnosis
# yang disimpan; hasilnya dipisah dari delapan kandidat benchmark di atas.

# %%
overfit = ablation["small_batch_overfit"]
display(
    pd.json_normalize({key: value for key, value in overfit.items() if key != "history"}).T.rename(
        columns={0: "nilai"}
    )
)
overfit_history = pd.DataFrame(overfit.get("history", []))
with pd.option_context("display.max_rows", None):
    display(overfit_history)
if not overfit_history.empty:
    ax = overfit_history.plot(x="update", y="mse", figsize=(8, 3.5), legend=False)
    ax.set(
        xlabel="Update pada mini-batch yang sama",
        ylabel="MSE normalisasi",
        title="Diagnosis train-only",
    )
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Tindak lanjut adaptif: memberi waktu lebih lama pada residual tanpa L2 kuat
#
# Run keempat diputuskan **setelah** ablation memperlihatkan sinyal mulai tumbuh
# pada epoch 3–4, tetapi **sebelum** training tindak lanjut dimulai. Dua kandidat
# L2 0 dan 1e-7 diberi 12 epoch tetap tanpa early stopping; faktor lain mengikuti
# ablation. Ini perubahan protokol yang disengaja dan dicatat, bukan kandidat
# yang sejak awal sudah ditentukan bersama run ketiga.
#
# Tujuannya menguji apakah patience 3 terlalu cepat menghentikan residual.
# Tetap memakai validation pengembangan yang sama, tanpa membuka test, tanpa
# evaluasi ranking baru, dan tanpa mengulang diagnosis overfit mini-batch.
# Kandidat terbaik dipilih dari seluruh 12 epoch menurut validation. Pemilihan
# adaptif ini menambah penggunaan validation; hasilnya bukan konfirmasi independen.

# %%
warmup = evidence["warmup"]
warmup_manifest = evidence["warmup_manifest"]
assert warmup["schema_version"] == 1
assert warmup["test_used"] is False
assert warmup["dataset"]["input_bytes_match_original_experiment"]
assert warmup["dataset"]["train_pairs"] == ablation["dataset"]["train_pairs"]
assert warmup["dataset"]["validation_known_pairs"] == ablation["dataset"]["validation_known_pairs"]
assert warmup["configuration"]["early_stopping_enabled"] is False
warmup_candidates = warmup["candidate_validation_results"]
warmup_by_name = {candidate["name"]: candidate for candidate in warmup_candidates}
display(
    pd.DataFrame(
        [
            {
                "run": "Warm-up: 12 epoch tetap",
                "session": warmup_manifest["session_name"],
                "selesai (Asia/Jakarta)": pd.Timestamp(warmup["completed_utc"])
                .tz_convert("Asia/Jakarta")
                .isoformat(),
                "status": warmup["status"],
                "semua kandidat selesai": warmup["all_candidates_completed"],
                "durasi (detik)": warmup["total_seconds"],
                "test digunakan": warmup["test_used"],
                "arsip terverifikasi": len(warmup_manifest["backups"]),
                "runtime dihentikan": warmup_manifest["shutdown"]["stopped"],
            }
        ]
    )
)
warmup_rows, trajectory_comparisons = [], []
for candidate in warmup_candidates:
    previous = next(
        c
        for c in ablation_neural
        if c["l2"] == candidate["l2"]
        and c["head_initialization"] == candidate["head_initialization"]
    )
    warmup_rows.append(
        {
            "percobaan": candidate["name"],
            "status": candidate["status"],
            "L2": candidate["l2"],
            "head": candidate["head_initialization"],
            "epoch selesai": candidate["epochs_completed"],
            "epoch terbaik": candidate["best_epoch"],
            "validation RMSE": candidate["validation"]["rmse_rating_scale"],
            "validation MAE": candidate["validation"]["mae_rating_scale"],
            "train RMSE epoch terakhir": candidate["history"][-1]["train_rmse"],
            "validation RMSE epoch terakhir": candidate["history"][-1]["val_rmse"],
            "RMSE terbaik ablation 4 epoch": previous["validation"]["rmse_rating_scale"],
            "selisih dari run 4 epoch": candidate["validation"]["rmse_rating_scale"]
            - previous["validation"]["rmse_rating_scale"],
            "status test": "tidak dijalankan",
        }
    )
    for old_record, new_record in zip(previous["history"], candidate["history"]):
        assert old_record["epoch"] == new_record["epoch"]
        trajectory_comparisons.append(
            {
                "percobaan": candidate["name"],
                "epoch": new_record["epoch"],
                "selisih train RMSE": new_record["train_rmse"] - old_record["train_rmse"],
                "selisih validation RMSE": new_record["val_rmse"] - old_record["val_rmse"],
            }
        )
warmup_experiments = pd.DataFrame(warmup_rows)
display(warmup_experiments)
display(Markdown("**Pemeriksaan replikasi lintasan epoch 1–4 terhadap ablation**"))
display(pd.DataFrame(trajectory_comparisons))
display(
    Markdown(
        "Konfigurasi, seed, data, dan urutan batch dipertahankan. Lintasan tanpa L2 "
        "direplikasi tepat pada empat epoch awal; kandidat L2 1e-7 memiliki "
        "perbedaan numerik kecil sesudah epoch 1 seperti terlihat di tabel. "
        "Operasi GPU tidak dipaksa deterministik secara ketat; penyebab detail "
        "selisih belum diisolasi. Jangan menyebut kedua lintasan identik bit demi bit."
    )
)
display(Markdown("**Konfigurasi lengkap dan kriteria seleksi warm-up**"))
display(pd.json_normalize(warmup["configuration"]).T.rename(columns={0: "nilai"}))
display(pd.json_normalize(warmup["selection"]).T.rename(columns={0: "nilai"}))

# %% [markdown]
# ### Seluruh 12 epoch, termasuk yang kalah dari checkpoint terbaik
#
# Garis vertikal abu-abu menandai batas epoch 4 pada run ablation sebelumnya.
# Garis hitam menandai checkpoint terbaik dalam run 12 epoch. Menggunakan
# checkpoint terbaik menghindari menyamakan bobot terakhir dengan hasil terbaik.

# %%
warmup_epochs = pd.DataFrame(
    [
        {
            "percobaan": candidate["name"],
            **record,
            "checkpoint terbaik": record["epoch"] == candidate["best_epoch"],
        }
        for candidate in warmup_candidates
        for record in candidate["history"]
    ]
)
display(warmup_epochs)
assert len(warmup_epochs) == warmup["training"]["total_neural_epochs"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
for ax, candidate in zip(axes.flat, warmup_candidates, strict=True):
    frame = pd.DataFrame(candidate["history"])
    ax.plot(frame["epoch"], frame["train_rmse"], "o-", label="Train")
    ax.plot(frame["epoch"], frame["val_rmse"], "o-", label="Validation")
    ax.axvline(4, color="gray", linestyle="--", label="Batas ablation lama")
    ax.axvline(candidate["best_epoch"], color="black", linestyle=":", label="Terbaik")
    ax.set(title=candidate["name"], xlabel="Epoch", ylabel="RMSE", xticks=frame["epoch"])
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Mekanisme pada warm-up: gradient, embedding, dan variasi koreksi
#
# Definisi diagnosis sama dengan run ablation: gradient dari probe train tetap,
# koreksi sebelum clipping, serta checkpoint terbaik untuk inferensi. Seluruh
# tahap disimpan agar perubahan setelah epoch 4 tetap dapat diperiksa.

# %%
warmup_diagnostic_rows, warmup_correction_rows, warmup_gradient_rows = [], [], []
for candidate in warmup_candidates:
    for diagnostic in candidate["diagnostics"]:
        identity = {
            "percobaan": candidate["name"],
            "tahap": diagnostic["stage"],
            "epoch": diagnostic["epoch"],
            "iterasi optimizer": diagnostic["optimizer_iterations"],
        }
        warmup_diagnostic_rows.append(
            {
                **identity,
                "MSE data sampel": diagnostic["data_mse"],
                "L2 loss": diagnostic["regularization_loss"],
                "norma gradient data": diagnostic["gradient_norm_totals"]["data"],
                "norma gradient L2": diagnostic["gradient_norm_totals"]["l2"],
                "norma embedding user": diagnostic["embedding_norms"]["user_l2"],
                "norma embedding ISBN": diagnostic["embedding_norms"]["book_l2"],
                "epoch state optimizer": diagnostic.get("optimizer_state_epoch"),
            }
        )
        for cohort, correction in diagnostic["corrections"].items():
            warmup_correction_rows.append(
                {
                    **identity,
                    "cohort": cohort,
                    "pasangan": correction["pairs"],
                    "koreksi mean": correction["mean_rating"],
                    "koreksi std": correction["std_rating"],
                    "koreksi min": correction["min_rating"],
                    "koreksi max": correction["max_rating"],
                }
            )
        for variable, gradient in diagnostic["gradient_norms"].items():
            warmup_gradient_rows.append(
                {
                    **identity,
                    "variabel": variable,
                    "norma gradient data": gradient["data"],
                    "norma gradient L2": gradient["l2"],
                }
            )
warmup_corrections = pd.DataFrame(warmup_correction_rows)
display(Markdown("**Koreksi pada checkpoint terbaik**"))
display(
    warmup_corrections[
        (warmup_corrections["tahap"] == "restored_best")
        & (warmup_corrections["cohort"] == "validation")
    ].merge(
        warmup_experiments[["percobaan", "validation RMSE", "L2"]],
        on="percobaan",
        validate="one_to_one",
    )
)
with pd.option_context("display.max_rows", None):
    display(Markdown("**Loss, norma gradient, dan embedding: semua tahap**"))
    display(pd.DataFrame(warmup_diagnostic_rows))
    display(Markdown("**Sebaran koreksi: semua tahap**"))
    display(warmup_corrections)
    display(Markdown("**Gradient setiap variabel: semua tahap**"))
    display(pd.DataFrame(warmup_gradient_rows))
fig, ax = plt.subplots(figsize=(9, 4))
for name, frame in warmup_corrections[
    (warmup_corrections["tahap"] == "epoch_end") & (warmup_corrections["cohort"] == "validation")
].groupby("percobaan", sort=False):
    ax.plot(frame["epoch"], frame["koreksi std"], "o-", label=name)
ax.axvline(4, color="gray", linestyle="--", label="Batas ablation sebelumnya")
ax.set(xlabel="Epoch", ylabel="Std koreksi, poin rating", title="Variasi koreksi selama warm-up")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Run kelima: evaluasi pengembangan internal dengan dua split dan tiga seed
#
# Empat run sebelumnya memperlihatkan batas validation yang hanya memiliki
# 140 pasangan warm. Run kelima memperluas evaluasi dengan **membagi ulang
# train lama**, tanpa membaca CSV validation atau test dari benchmark lama.
# Karena sumber train ini sudah pernah dipakai pada eksperimen sebelumnya,
# hasilnya adalah **evaluasi pengembangan internal**, bukan test independen baru.
# RMSE-nya tidak dibandingkan langsung sebagai perbaikan atas angka benchmark
# lama, karena populasi fit dan holdout berbeda.
#
# Pada setiap split seed 17/29, sebanyak 6.000 pengguna yang mempunyai minimal
# dua karya dipilih, lalu dibagi menjadi 3.000 pengguna selection dan 3.000
# pengguna assessment yang saling terpisah. Satu karya per pengguna ditahan;
# **seluruh baris rating untuk semua edisi karya itu** ikut dipindahkan ke
# partisi yang sesuai. Setiap pengguna tetap memiliki minimal satu karya dalam
# fit. Karena itu satu partisi memiliki tepat 3.000 pengguna, tetapi dapat
# memiliki lebih dari 3.000 pasangan rating.
#
# MLP awal dan residual tanpa L2 masing-masing dilatih dengan seed training
# 42/2026/7 pada kedua split: 12 percobaan neural. Checkpoint memakai selection
# saja, maksimal 12 epoch dengan patience 3. **Semua bobot harus dikunci sebelum
# assessment pertama dihitung.** Assessment hanya melaporkan hasil seluruh
# kandidat; tidak memilih pemenang untuk dipromosikan ke aplikasi.
# Seed training mengubah bobot awal **sekaligus urutan mini-batch**. Walaupun
# field JSON bernama `initialization_seed`, pengaruhnya bukan inisialisasi saja.
#
# Semua pengguna holdout sudah dikenal dari fit secara sengaja, dan katalog
# berasal dari train lama. Cakupan ID yang lebih tinggi pada desain ini tidak
# membuktikan masalah cold-start pada benchmark lama sudah teratasi.

# %%
multiseed = evidence["multiseed"]
multiseed_manifest = evidence["multiseed_manifest"]
assert multiseed["schema_version"] == 1
assert multiseed["experiment_kind"] == "multiseed_internal_development"
assert multiseed["original_test_used"] is False
assert multiseed["original_validation_used"] is False
assert multiseed["dataset"]["source_pairs"] == audit["train_pairs"]
assert multiseed["selection"]["assessment_used_for_selection"] is False
if multiseed["assessment_performed"]:
    assert multiseed["selection"]["all_weights_locked"]
multiseed_candidates = multiseed["candidate_validation_results"]
multiseed_controls = multiseed["controls"]
multiseed_all_records = [*multiseed_candidates, *multiseed_controls]
multiseed_by_name = {record["name"]: record for record in multiseed_all_records}
selection_by_split = {record["split_seed"]: record for record in multiseed["selection"]["by_split"]}
display(
    pd.DataFrame(
        [
            {
                "run": "Pengembangan internal: 2 split × 3 seed × 2 arsitektur",
                "session": multiseed_manifest["session_name"],
                "selesai (Asia/Jakarta)": pd.Timestamp(multiseed["completed_utc"])
                .tz_convert("Asia/Jakarta")
                .isoformat(),
                "GPU": multiseed["gpu"]["gpu_details"][0]["device_name"],
                "TensorFlow": multiseed["gpu"]["tensorflow_version"],
                "durasi (detik)": multiseed["total_seconds"],
                "status": multiseed["status"],
                "assessment dihitung": multiseed["assessment_performed"],
                "validation lama digunakan": multiseed["original_validation_used"],
                "test lama digunakan": multiseed["original_test_used"],
                "arsip terverifikasi": len(multiseed_manifest["backups"]),
                "runtime dihentikan": multiseed_manifest["shutdown"]["stopped"],
            }
        ]
    )
)
display(Markdown("**Seluruh konfigurasi dan aturan penguncian bobot**"))
display(pd.json_normalize(multiseed["configuration"]).T.rename(columns={0: "nilai"}))
display(pd.json_normalize(multiseed["selection"]).T.rename(columns={0: "nilai"}))
display(Markdown("**Ukuran partisi dan jejak hash pembagian data**"))
display(pd.json_normalize(multiseed["dataset"]["splits"]))
for split in multiseed["dataset"]["splits"]:
    assert split["selection_assessment_users_disjoint"]
    assert split["all_heldout_editions_retained"]
    assert split["all_users_keep_fit_history"]
    assert (
        split["fit_pairs"] + split["selection_pairs"] + split["assessment_pairs"]
        == split["source_pairs"]
    )
split_coverage_rows = []
for split_seed in multiseed["configuration"]["split_seeds"]:
    representative = next(c for c in multiseed_candidates if c["split_seed"] == split_seed)
    for partition in ("selection", "assessment"):
        score = representative.get(partition)
        if not score:
            continue
        known_pairs = score["known_pairs"]["pairs"]
        all_pairs = score["all_pairs"]["pairs"]
        split_coverage_rows.append(
            {
                "split seed": split_seed,
                "partisi": partition,
                "pengguna yang direncanakan": multiseed["configuration"]["users_per_partition"],
                "pasangan seluruhnya": all_pairs,
                "pasangan kedua ID dikenal": known_pairs,
                "cakupan ID (%)": 100 * known_pairs / all_pairs,
                **{f"availability.{key}": value for key, value in score["availability"].items()},
            }
        )
display(Markdown("**Cakupan aktual: jumlah pengguna dan jumlah rating tidak disamakan**"))
multiseed_coverage = pd.DataFrame(split_coverage_rows)
display(multiseed_coverage)
if (
    multiseed_coverage["pasangan kedua ID dikenal"] == multiseed_coverage["pasangan seluruhnya"]
).all():
    display(
        Markdown(
            "**Cakupan aktual 100% warm pada kedua partisi/kedua split.** "
            "Fallback tidak digunakan pada run ini; metrik known/full sama karena "
            "baris evaluasinya sama. Ini akibat desain holdout serta data yang tersisa "
            "di fit, bukan bukti solusi cold-start."
        )
    )

# %% [markdown]
# ### Semua model, kontrol, dan hasil selection/assessment
#
# Kontrol mean global, mean pengguna, dan bias pengguna + buku dihitung ulang
# dari fit masing-masing split. Kolom `known_pairs` memakai pasangan dengan
# kedua ID dikenal. Kolom `all_pairs` memakai fallback bersama yang ditentukan
# oleh protokol jika terdapat pasangan cold, sehingga cakupan model dapat dibandingkan
# pada baris yang sama. Metrik dibobot menurut pasangan rating, bukan rata-rata
# dari skor tiap pengguna. Kolom pemenang selection tidak berasal dari assessment.
# Fallback bersama adalah mean pengguna fit → mean ISBN fit → mean global fit;
# aturan ini juga berlaku pada kontrol untuk baris dengan ID yang tidak dikenal.

# %%
multiseed_record_rows, multiseed_metric_rows = [], []
for record in multiseed_all_records:
    is_neural = record["kind"] == "neural"
    split_selection = selection_by_split[record["split_seed"]]
    multiseed_record_rows.append(
        {
            "percobaan": record["name"],
            "jenis": record["kind"],
            "split seed": record["split_seed"],
            "arsitektur": record.get("architecture", "kontrol"),
            "seed training": record.get("initialization_seed"),
            "L2": record.get("l2") if is_neural else None,
            "status": record.get("status", "kontrol tersedia"),
            "epoch selesai": record.get("epochs_completed", 0),
            "epoch terbaik selection": record.get("best_epoch"),
            "terpilih menurut selection": record["name"] == split_selection["best_candidate"],
            "save/load terverifikasi": record.get("model_roundtrip_verified"),
            "durasi (detik)": record.get("seconds"),
            "status test independen": "tidak dijalankan",
        }
    )
    for partition in ("selection", "assessment"):
        score = record.get(partition)
        if not score:
            continue
        for scope in ("known_pairs", "all_pairs"):
            metric = score[scope]
            multiseed_metric_rows.append(
                {
                    "percobaan": record["name"],
                    "split seed": record["split_seed"],
                    "arsitektur": record.get("architecture", "kontrol"),
                    "seed training": record.get("initialization_seed"),
                    "partisi": partition,
                    "cohort": scope,
                    "pasangan": metric["pairs"],
                    "RMSE": metric["rmse_rating_scale"],
                    "MAE": metric["mae_rating_scale"],
                }
            )
multiseed_experiments = pd.DataFrame(multiseed_record_rows)
multiseed_metrics = pd.DataFrame(multiseed_metric_rows)
display(multiseed_experiments.fillna("—"))
display(Markdown("**Identitas checkpoint terpilih yang dikunci sebelum assessment**"))
display(
    pd.DataFrame(
        [
            {"percobaan": candidate["name"], "file": filename, "SHA256": digest}
            for candidate in multiseed_candidates
            for filename, digest in candidate.get("selected_model_fingerprints", {}).items()
        ]
    )
)
with pd.option_context("display.max_rows", None):
    display(multiseed_metrics)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, split_seed in zip(axes, multiseed["configuration"]["split_seeds"], strict=True):
    frame = multiseed_metrics[
        (multiseed_metrics["split seed"] == split_seed)
        & (multiseed_metrics["partisi"] == "assessment")
        & (multiseed_metrics["cohort"] == "known_pairs")
    ]
    labels = frame["percobaan"].str.replace(f"split{split_seed}_", "", regex=False)
    ax.barh(labels, frame["RMSE"], color="#3876a5")
    for index, value in enumerate(frame["RMSE"]):
        ax.text(value, index, f"  {value:.4f}", va="center", fontsize=8)
    ax.set(title=f"Assessment internal, split {split_seed}", xlabel="RMSE (lebih kecil lebih baik)")
    if not frame.empty:
        ax.set_xlim(0, frame["RMSE"].max() + 0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Seluruh epoch pada semua 12 training, tanpa menyembunyikan variasi seed
#
# Validation pada history adalah **selection**, bukan assessment. Tidak ada
# kurva assessment per epoch karena assessment baru dibuka sesudah seluruh
# bobot kandidat terkunci. Garis putus-putus menunjukkan checkpoint selection
# terbaik yang dipakai saat penilaian akhir.

# %%
multiseed_epochs = pd.DataFrame(
    [
        {
            "percobaan": candidate["name"],
            "split seed": candidate["split_seed"],
            "arsitektur": candidate["architecture"],
            "seed training": candidate["initialization_seed"],
            **record,
            "checkpoint terbaik selection": record["epoch"] == candidate["best_epoch"],
        }
        for candidate in multiseed_candidates
        for record in candidate["history"]
    ]
)
assert len(multiseed_epochs) == multiseed["training"]["total_neural_epochs"]
with pd.option_context("display.max_rows", None):
    display(multiseed_epochs)
fig, axes = plt.subplots(4, 3, figsize=(15, 13))
for ax, candidate in zip(axes.flat, multiseed_candidates, strict=True):
    frame = pd.DataFrame(candidate["history"])
    ax.plot(frame["epoch"], frame["train_rmse"], "o-", label="Train")
    ax.plot(frame["epoch"], frame["val_rmse"], "o-", label="Selection")
    ax.axvline(candidate["best_epoch"], color="black", linestyle=":", label="Terbaik selection")
    ax.set(title=candidate["name"], xlabel="Epoch", ylabel="RMSE", xticks=frame["epoch"])
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bukti langkah pertama pada setiap training baru
#
# Trace berikut berasal dari update Adam pertama masing-masing dari 12 model.
# MSE, penalti L2, dan perubahan embedding menunjukkan bahwa training dijalankan;
# nilainya bukan skor assessment. Pada residual head nol, gradient data menuju
# embedding pada langkah pertama masih nol. Head dan pembelajaran berikutnya
# perlu dibedakan dari efek regularisasi, sebagaimana diagnosis run sebelumnya.

# %%
multiseed_trace_rows, multiseed_first_gradients = [], []
for candidate in multiseed_candidates:
    trace = candidate.get("first_step_trace")
    if not trace:
        continue
    multiseed_trace_rows.append(
        {
            "percobaan": candidate["name"],
            "MSE sebelum": trace["mse_before"],
            "L2 loss": trace["regularization_loss"],
            "total loss sebelum": trace["total_loss_before"],
            "MSE setelah": trace["mse_after"],
            "norma perubahan embedding": trace["embedding_delta_norm"],
            "gradient data embedding sebelum": trace.get(
                "embedding_data_gradient_norm_before", "tidak dipisah dalam trace"
            ),
            "iterasi optimizer": trace["optimizer_iterations"],
        }
    )
    multiseed_first_gradients.extend(
        {"percobaan": candidate["name"], "variabel": variable, "norma gradient total": norm}
        for variable, norm in trace["gradient_norms"].items()
    )
display(pd.DataFrame(multiseed_trace_rows))
with pd.option_context("display.max_rows", None):
    display(pd.DataFrame(multiseed_first_gradients))

# %% [markdown]
# ### Ketidakpastian: bootstrap kelompok pengguna pada assessment internal
#
# Interval berikut menggunakan 1.000 resampling **pengguna**, mempertahankan
# seluruh baris edisi/rating milik pengguna terpilih. Dua model dibandingkan
# pada pasangan yang sama. Selisih adalah RMSE kandidat − RMSE referensi:
# negatif menguntungkan kandidat. Interval yang melintasi nol belum mendukung
# perbedaan yang meyakinkan menurut prosedur ini.
#
# Interval bersyarat pada split dan model yang sudah dilatih; ia bukan interval
# atas semua kemungkinan dataset. Banyak kandidat/perbandingan juga diperiksa,
# sehingga satu interval tidak boleh dipakai untuk memilih pemenang assessment.
# Hasil dua split disajikan terpisah karena sumber data dan pengguna dapat
# tumpang tindih; dua split tersebut bukan dua dataset independen.

# %%
multiseed_bootstrap_rows = []
for split_result in multiseed["assessment_results"]:
    for comparison in split_result["paired_comparisons"]:
        bootstrap = comparison["bootstrap"]
        low, high = bootstrap["interval_95_percent"]
        multiseed_bootstrap_rows.append(
            {
                "split seed": split_result["split_seed"],
                "kandidat": comparison["candidate"],
                "referensi": comparison["reference"],
                "cohort": comparison["scope"],
                "pasangan": bootstrap["pairs"],
                "kelompok pengguna": bootstrap["clusters"],
                "bootstrap draws": bootstrap["draws"],
                "selisih RMSE": bootstrap["point_estimate"],
                "batas bawah 95%": low,
                "batas atas 95%": high,
                "interval melintasi nol": low <= 0 <= high,
                "metode": bootstrap["method"],
            }
        )
multiseed_bootstrap = pd.DataFrame(multiseed_bootstrap_rows)
with pd.option_context("display.max_rows", None):
    display(multiseed_bootstrap)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for row, split_seed in enumerate(multiseed["configuration"]["split_seeds"]):
    for column, scope in enumerate(("known_pairs", "all_pairs")):
        ax = axes[row, column]
        frame = multiseed_bootstrap[
            (multiseed_bootstrap["split seed"] == split_seed)
            & (multiseed_bootstrap["cohort"] == scope)
        ]
        labels = (
            frame["kandidat"].str.replace(f"split{split_seed}_", "", regex=False)
            + " − "
            + frame["referensi"].str.replace(f"split{split_seed}_", "", regex=False)
        )
        positions = range(len(frame))
        ax.hlines(positions, frame["batas bawah 95%"], frame["batas atas 95%"], color="#3876a5")
        ax.plot(frame["selisih RMSE"], positions, "o", color="#3876a5")
        ax.axvline(0, color="black", linestyle=":")
        ax.set(
            yticks=list(positions),
            yticklabels=labels,
            title=f"Split {split_seed}: {scope}",
            xlabel="Δ RMSE kandidat − referensi, interval bootstrap 95%",
        )
        ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analisis tambahan sesudah melihat hasil: semua neural melawan mean global
#
# Perbandingan yang ditentukan sebelum training memakai kontrol bias dan
# pasangan residual/MLP pada seed yang sama. Sesudah assessment terlihat,
# mean global ternyata merupakan kontrol non-neural terkuat dari tiga kontrol
# pada kedua split. Karena itu analisis **post-hoc** berikut menambahkan
# bootstrap setiap neural terhadap mean global.
#
# Analisis ini hanya membaca prediksi yang sudah tersimpan dan terverifikasi:
# tidak ada training, prediksi baru, seleksi ulang, pembukaan test lama, atau
# perubahan model serving. Interval bersifat eksploratif, tidak dikoreksi untuk
# banyak perbandingan, dan tidak digabung lintas split. Keduanya tetap dicatat:
# analisis yang telah ditentukan sebelum training dan tambahan sesudah melihat
# hasil. Tambahan ini tidak dihitung sebagai model atau epoch baru.

# %%
multiseed_posthoc = evidence["multiseed_posthoc"]
assert multiseed_posthoc["analysis_kind"] == "posthoc_neural_vs_global_control"
assert multiseed_posthoc["posthoc"] is True
assert multiseed_posthoc["predeclared"] is False
assert multiseed_posthoc["no_new_training"]
assert multiseed_posthoc["no_new_predictions"]
assert multiseed_posthoc["original_test_used"] is False
assert multiseed_posthoc["serving_changed"] is False
assert multiseed_posthoc["verification"]["all_prediction_rows_match_source"]
assert multiseed_posthoc["verification"]["all_stored_rmse_reproduced"]
display(
    pd.json_normalize(
        {
            key: multiseed_posthoc[key]
            for key in (
                "status",
                "analysis_kind",
                "created_utc",
                "predeclared",
                "posthoc",
                "no_new_training",
                "no_new_predictions",
                "source_results",
                "prediction_csv_fingerprints",
                "script_fingerprint",
                "verification",
            )
        }
    ).T.rename(columns={0: "nilai"})
)
posthoc_rows = []
for comparison in multiseed_posthoc["comparisons"]:
    bootstrap = comparison["bootstrap"]
    low, high = bootstrap["interval_95_percent"]
    posthoc_rows.append(
        {
            "split seed": comparison["split_seed"],
            "kandidat": comparison["candidate"],
            "referensi": comparison["reference"],
            "cohort": comparison["scope"],
            "dipilih pada selection": comparison["candidate"]
            == selection_by_split[comparison["split_seed"]]["best_neural_candidate"],
            "pasangan": bootstrap["pairs"],
            "kelompok pengguna": bootstrap["clusters"],
            "bootstrap draws": bootstrap["draws"],
            "RMSE kandidat": comparison["candidate_rmse"],
            "RMSE mean global": comparison["reference_rmse"],
            "selisih RMSE": bootstrap["point_estimate"],
            "batas bawah 95%": low,
            "batas atas 95%": high,
            "interval melintasi nol": low <= 0 <= high,
        }
    )
posthoc_comparisons = pd.DataFrame(posthoc_rows)
display(posthoc_comparisons)
posthoc_known = posthoc_comparisons[posthoc_comparisons["cohort"] == "known_pairs"]
mlp_intervals = posthoc_known[
    posthoc_known["kandidat"].map(lambda name: multiseed_by_name[name]["architecture"] == "mlp")
]
residual_intervals = posthoc_known[
    posthoc_known["kandidat"].map(
        lambda name: multiseed_by_name[name]["architecture"] == "residual"
    )
]
display(
    Markdown(
        f"Pada **{len(mlp_intervals)}** model MLP, **{int((mlp_intervals['batas atas 95%'] < 0).sum())}** "
        "interval berada seluruhnya di bawah nol dibanding mean global. "
        f"Pada **{len(residual_intervals)}** model residual, "
        f"**{int((residual_intervals['batas bawah 95%'] > 0).sum())}** interval berada seluruhnya di atas nol. "
        "Ini mendukung manfaat MLP pada assessment internal warm yang dipakai, "
        "dengan batas analisis post-hoc di atas. Baris known/full yang sama tidak "
        "dihitung sebagai percobaan independen tambahan."
    )
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, split_seed in zip(axes, multiseed["configuration"]["split_seeds"], strict=True):
    frame = posthoc_comparisons[
        (posthoc_comparisons["split seed"] == split_seed)
        & (posthoc_comparisons["cohort"] == "known_pairs")
    ]
    labels = frame["kandidat"].str.replace(f"split{split_seed}_", "", regex=False)
    positions = range(len(frame))
    ax.hlines(positions, frame["batas bawah 95%"], frame["batas atas 95%"], color="#3876a5")
    ax.plot(frame["selisih RMSE"], positions, "o", color="#3876a5")
    ax.axvline(0, color="black", linestyle=":")
    ax.set(
        yticks=list(positions),
        yticklabels=labels,
        xlabel="Δ RMSE neural − mean global, interval bootstrap 95%",
        title=f"Post-hoc: assessment internal split {split_seed}",
    )
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variasi seed training: ringkasan deskriptif, bukan tiga split independen
#
# Mean dan standar deviasi di bawah berasal dari tiga seed training pada
# **data yang sama** untuk setiap arsitektur/split. Angka ini mengukur variasi
# inisialisasi dan urutan mini-batch yang diamati, bukan standard error terhadap populasi pengguna.
# Tidak ada pooling seluruh enam model sebagai enam replikasi dataset, dan
# tidak ada promosi kandidat berdasarkan assessment.

# %%
multiseed_seed_summary = pd.DataFrame(multiseed["seed_summary"])
display(multiseed_seed_summary)
display(
    Markdown(
        "**Batas interpretasi run kelima:** assessment pengembangan ini lebih besar, "
        "tetapi tetap dibentuk dari train yang sudah pernah dilihat dan pengguna "
        "yang sengaja dipertahankan warm. Hasilnya menguji konsistensi pada desain "
        "ini. Bukti tentang pengguna/buku baru, ranking top-10, dan test independen "
        "masih memerlukan evaluasi tersendiri."
    )
)

# %% [markdown]
# ## Run keenam: pembanding bias/MF yang dituning dan ranking katalog penuh
#
# Run kelima menemukan MLP lebih baik daripada tiga kontrol sederhana pada
# assessment internal. Run keenam menguji pembanding yang lebih kuat dan
# mengukur kualitas daftar rekomendasi, memakai **split pengembangan internal
# yang sama, yang assessment-nya sudah pernah dilihat**. Selection tetap dipakai
# untuk tuning, tetapi assessment ini kini digunakan ulang secara eksplisit.
# Tidak ada klaim konfirmasi pada test independen baru.
#
# Pada setiap split, 16 kombinasi shrinkage pengguna/buku dari `{0, 5, 20, 100}`
# dipakai untuk bias yang dioptimalkan bersama. Biased matrix factorization
# memakai tiga nilai regularisasi `{1e-4, 1e-3, 1e-2}` × tiga seed training
# `{42, 2026, 7}`: total 18 training MF pada dua split. Maksimal 30 epoch dengan
# patience 5. Lambda dipilih menurut **mean RMSE selection tiga seed**, lalu
# kandidat utama ditetapkan memakai selection. Semua model dikunci sebelum
# assessment dan ranking pada run ini dihitung.
#
# **MF merupakan model embedding dangkal dengan dot product dan bias, bukan
# deep MLP.** Delapan belas training MF dicatat terpisah dari 24 training deep
# MLP/residual pada lima run sebelumnya. Enam MLP dari run kelima hanya dimuat
# ulang dari checkpoint tersimpan; tidak dihitung sebagai training baru.
#
# Ranking memakai seluruh katalog 5.000 ISBN, digabung ke karya kanonik dengan
# maksimum skor ISBN mentah. Seluruh karya yang sudah ada di fit pengguna
# dikeluarkan. Satu karya holdout relevan bila salah satu rating edisinya ≥8;
# target yang tidak tersedia tetap dihitung miss. Tiga MLP, tiga MF pada lambda
# terpilih, bias terpilih, dan popularity dari fit dinilai pada cohort yang sama.
# Tidak ada tuning berdasarkan ranking.

# %%
comparator = evidence["comparator"]
comparator_manifest = evidence["comparator_manifest"]
assert comparator["experiment_kind"] == "tuned_bias_mf_ranking_reused_development"
assert comparator["original_test_used"] is False
assert comparator["original_validation_used"] is False
assert comparator["dataset"]["reused_prior_assessment"] is True
assert comparator["model_promotion_to_application"] is False
comparator_config = comparator["configuration"]
bias_grid = comparator["bias_grid_results"]
mf_candidates = comparator["candidate_validation_results"]
reused_mlps = comparator["reused_mlp_results"]
comparator_selection = {row["split_seed"]: row for row in comparator["lambda_selection"]}
mf_by_name = {row["name"]: row for row in mf_candidates}
assert all(row["selection_only"] for row in comparator_selection.values())
assert all(not row["trained_in_this_run"] for row in reused_mlps)
display(
    pd.DataFrame(
        [
            {
                "run": "Bias dituning, MF dangkal, dan ranking katalog penuh",
                "session": comparator_manifest["session_name"],
                "selesai (Asia/Jakarta)": pd.Timestamp(comparator["completed_utc"])
                .tz_convert("Asia/Jakarta")
                .isoformat(),
                "status": comparator["status"],
                "durasi (detik)": comparator["total_seconds"],
                "konfigurasi bias": len(bias_grid),
                "training MF": len(mf_candidates),
                "MLP hanya dimuat ulang": len(reused_mlps),
                "assessment dihitung ulang": comparator["assessment_performed"],
                "assessment sudah pernah dilihat": comparator["dataset"]["reused_prior_assessment"],
                "arsip terverifikasi": len(comparator_manifest["backups"]),
                "runtime dihentikan": comparator_manifest["shutdown"]["stopped"],
            }
        ]
    )
)
display(Markdown("**Konfigurasi lengkap dan aturan pemilihan sebelum assessment/ranking**"))
display(pd.json_normalize(comparator_config).T.rename(columns={0: "nilai"}))
display(pd.json_normalize(comparator["dataset"]["splits"]))

# %% [markdown]
# ### Seluruh 32 konfigurasi bias yang dioptimalkan bersama
#
# Target dinormalisasi `(rating - 1) / 9`, dengan mean global fit tetap. Bias
# pengguna dan buku diperbarui bergantian untuk meminimalkan:
#
# `sum((target - mean_global - bias_user - bias_book)^2)`
# `+ alpha_user * sum(bias_user^2) + alpha_item * sum(bias_book^2)`.
#
# Maksimal 1.000 iterasi dengan toleransi perubahan parameter 1e-10. Flag
# konvergensi, iterasi, objective fit, dan delta parameter dicatat untuk semua
# konfigurasi. Ini pembanding yang dituning; kontrol bias awal memakai prosedur
# dan shrinkage tetap yang berbeda. Hanya bias terpilih menurut selection yang
# dinilai pada assessment/ranking, sehingga kandidat lain tidak memiliki skor
# assessment dan tidak boleh ditafsirkan mendapat skor nol.

# %%
bias_grid_rows = []
for row in bias_grid:
    score = row.get("selection", {}).get("known_pairs", {})
    bias_grid_rows.append(
        {
            "percobaan": row["name"],
            "split seed": row["split_seed"],
            "alpha user": row["user_shrinkage"],
            "alpha buku": row["item_shrinkage"],
            "status": row["status"],
            "konvergen": row.get("converged"),
            "iterasi": row.get("iterations"),
            "delta parameter terakhir": row.get("final_max_parameter_delta"),
            "objective fit": row.get("fit_objective"),
            "selection RMSE": score.get("rmse_rating_scale"),
            "selection MAE": score.get("mae_rating_scale"),
            "dipilih pada selection": row["selected"],
            "status assessment": "dinilai"
            if row["selected"] and comparator["assessment_performed"]
            else "tidak dinilai",
        }
    )
bias_grid_frame = pd.DataFrame(bias_grid_rows)
display(bias_grid_frame)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, split_seed in zip(axes, comparator_config["split_seeds"], strict=True):
    table = bias_grid_frame[bias_grid_frame["split seed"] == split_seed].pivot(
        index="alpha user", columns="alpha buku", values="selection RMSE"
    )
    picture = ax.imshow(table.to_numpy(), cmap="viridis_r", aspect="auto")
    for row_index in range(len(table.index)):
        for column_index in range(len(table.columns)):
            ax.text(
                column_index,
                row_index,
                f"{table.iloc[row_index, column_index]:.4f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    ax.set(
        xticks=range(len(table.columns)),
        xticklabels=table.columns,
        yticks=range(len(table.index)),
        yticklabels=table.index,
        xlabel="Alpha buku",
        ylabel="Alpha pengguna",
        title=f"Selection RMSE, split {split_seed}",
    )
    fig.colorbar(picture, ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### MF dangkal: seluruh lambda, seed, checkpoint, dan epoch
#
# Prediksi mentah MF adalah `mean_global + bias_user + bias_book + dot(p_user, q_book)`.
# Embedding 16 dimensi dimulai dari normal stddev 0,05. Bias dimulai dari bias
# terpilih lalu **tetap boleh berubah** selama Adam, learning rate 0,005. Karena
# bias ikut dilatih, hasil MF yang memburuk tidak membuktikan faktor laten tidak
# bermanfaat pada semua rancangan.
#
# Objective mini-batch adalah MSE rata-rata ditambah
# `lambda * mean(||p_user||² + ||q_book||² + bias_user² + bias_book²)` pada pasangan
# yang diamati. Penalti ini berbobot frekuensi interaksi dan berbeda dari L2
# Keras atas seluruh matriks pada run sebelumnya; angka lambda tidak dibandingkan
# langsung dengan L2 lama. Penalti bias MF juga tidak mempertahankan shrinkage
# joint bias yang dipilih sebelumnya, sehingga efek faktor laten dan perubahan
# bias tidak dapat dipisahkan dari perbandingan ini saja.
# Train RMSE berasal dari prediksi mentah; RMSE selection
# dan assessment memotong prediksi ke rating 1–10. `reg_loss` adalah rata-rata
# penalti dari mini-batch selama epoch, bukan penalti checkpoint akhir saja.

# %%
mf_rows = []
for row in mf_candidates:
    selected = comparator_selection.get(row["split_seed"], {})
    mf_rows.append(
        {
            "percobaan": row["name"],
            "split seed": row["split_seed"],
            "model": "MF embedding dangkal",
            "seed training": row["initialization_seed"],
            "lambda sampled-pair": row["regularization_lambda"],
            "status": row["status"],
            "epoch selesai": row["epochs_completed"],
            "epoch terbaik selection": row["best_epoch"],
            "terbaik tepat di batas epoch": row.get("best_at_epoch_cap"),
            "berhenti di batas epoch": row.get("stopped_at_epoch_cap"),
            "selection RMSE": row.get("validation", {}).get("rmse_rating_scale"),
            "lambda dipilih untuk ranking": row["name"]
            in selected.get("selected_mf_candidates", []),
            "MF utama dari selection": row["name"] == selected.get("primary_mf_candidate"),
            "save/load terverifikasi": row.get("model_roundtrip_verified"),
        }
    )
mf_experiments = pd.DataFrame(mf_rows)
display(mf_experiments)
lambda_rows = [
    {
        "split seed": row["split_seed"],
        **choice,
        "dipilih menurut mean selection": choice["regularization_lambda"] == row["selected_lambda"],
    }
    for row in comparator["lambda_selection"]
    for choice in row["grid"]
]
display(Markdown("**Pemilihan lambda menurut mean tiga seed training pada selection**"))
display(pd.DataFrame(lambda_rows))
display(pd.json_normalize(comparator["lambda_selection"]))
mf_epochs = pd.DataFrame(
    [
        {
            "percobaan": row["name"],
            "split seed": row["split_seed"],
            "lambda": row["regularization_lambda"],
            "seed training": row["initialization_seed"],
            **epoch,
            "checkpoint terbaik selection": epoch["epoch"] == row["best_epoch"],
        }
        for row in mf_candidates
        for epoch in row["history"]
    ]
)
assert len(mf_epochs) == comparator["training"]["total_neural_epochs"]
with pd.option_context("display.max_rows", None):
    display(mf_epochs)
fig, axes = plt.subplots(6, 3, figsize=(15, 18))
for ax, row in zip(axes.flat, mf_candidates, strict=True):
    frame = pd.DataFrame(row["history"])
    ax.plot(frame["epoch"], frame["train_rmse"], label="Train mentah")
    ax.plot(frame["epoch"], frame["val_rmse"], label="Selection clipped")
    ax.axvline(row["best_epoch"], color="black", linestyle=":", label="Terbaik selection")
    ax.set(title=row["name"], xlabel="Epoch", ylabel="RMSE")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
display(
    Markdown(
        "Checkpoint terbaik yang berada tepat di batas epoch belum membuktikan konvergensi. "
        "Flag tersebut dilaporkan terpisah dari early stopping dan konvergensi bias."
    )
)
display(Markdown("**Bukti update pertama MF: loss, perubahan embedding, dan gradient**"))
mf_trace_rows, mf_gradient_rows = [], []
for row in mf_candidates:
    trace = row.get("first_step_trace")
    if not trace:
        continue
    mf_trace_rows.append(
        {
            "percobaan": row["name"],
            "MSE sebelum": trace["mse_before"],
            "penalti sampled-pair": trace["regularization_loss"],
            "total loss sebelum": trace["total_loss_before"],
            "norma perubahan embedding": trace["embedding_delta_norm"],
            "iterasi optimizer": trace["optimizer_iterations"],
        }
    )
    mf_gradient_rows.extend(
        {"percobaan": row["name"], "variabel": variable, "norma gradient total": norm}
        for variable, norm in trace["gradient_norms"].items()
    )
display(pd.DataFrame(mf_trace_rows))
display(pd.DataFrame(mf_gradient_rows))

# %% [markdown]
# ### Assessment rating yang dipakai ulang: semua MF, bias terpilih, dan MLP tersimpan
#
# Semua 18 MF ditampilkan untuk transparansi, termasuk lambda yang tidak dipilih.
# Kolom pemilihan berasal dari selection, bukan hasil assessment. Enam MLP berikut
# adalah checkpoint dari run kelima; identitas file dan status tanpa training ulang
# ditampilkan. Angka assessment ini tidak menambah replikasi dataset independen.

# %%
display(
    pd.DataFrame(
        [
            {
                "MLP tersimpan": row["name"],
                "split seed": row["split_seed"],
                "seed training": row["initialization_seed"],
                "dilatih ulang": row["trained_in_this_run"],
                "selection RMSE": row["selection"]["known_pairs"]["rmse_rating_scale"],
                "assessment RMSE": row["assessment"]["known_pairs"]["rmse_rating_scale"],
                "hash checkpoint sumber": row["source_model_fingerprints"],
            }
            for row in reused_mlps
        ]
    )
)
comparator_rating_rows, comparator_rating_bootstrap_rows = [], []
for split_result in comparator["assessment_results"]:
    selection = comparator_selection[split_result["split_seed"]]
    for name, score in split_result["models"].items():
        for scope in ("known_pairs", "all_pairs"):
            metric = score[scope]
            comparator_rating_rows.append(
                {
                    "split seed": split_result["split_seed"],
                    "model": name,
                    "cohort": scope,
                    "pasangan": metric["pairs"],
                    "RMSE": metric["rmse_rating_scale"],
                    "MAE": metric["mae_rating_scale"],
                    "model utama dari selection": name
                    in (
                        selection["primary_mlp_candidate"],
                        selection["primary_mf_candidate"],
                        selection["bias_candidate"],
                    ),
                }
            )
    for comparison in split_result["paired_comparisons"]:
        bootstrap = comparison["bootstrap"]
        comparator_rating_bootstrap_rows.append(
            {
                "split seed": split_result["split_seed"],
                "kandidat": comparison["candidate"],
                "referensi": comparison["reference"],
                "cohort": comparison["scope"],
                "pasangan": bootstrap["pairs"],
                "kelompok pengguna": bootstrap["clusters"],
                "selisih RMSE": bootstrap["point_estimate"],
                "batas bawah 95%": bootstrap["interval_95_percent"][0],
                "batas atas 95%": bootstrap["interval_95_percent"][1],
            }
        )
comparator_ratings = pd.DataFrame(comparator_rating_rows)
comparator_rating_bootstrap = pd.DataFrame(comparator_rating_bootstrap_rows)
with pd.option_context("display.max_rows", None):
    display(comparator_ratings)
    display(Markdown("**Bootstrap pengguna untuk RMSE: negatif menguntungkan kandidat**"))
    display(comparator_rating_bootstrap)

# %% [markdown]
# ### Ranking: seluruh model yang ditetapkan sebelum assessment
#
# Ranking memakai skor mentah, tanpa clipping yang dapat membuat banyak skor
# tertinggi menjadi sama. Agregasi edisi menggunakan maksimum skor; tie dipecahkan
# menurut urutan leksikografis kunci karya. Popularity menghitung jumlah pengguna
# fit berbeda yang memberi rating ≥8 per karya. Semua model mengecualikan karya
# yang telah dibaca pengguna dengan aturan yang sama.
#
# Hanya pengguna dengan karya holdout relevan yang masuk metrik ranking. Karena
# setiap pengguna mempunyai satu karya relevan, Recall@10 sama dengan Hit@10.
# Karya katalog lain yang tidak diberi rating bukan negatif yang diketahui.
# Hit rendah berarti model jarang menemukan target holdout yang dicatat; ini
# tidak membuktikan seluruh rekomendasi lain tidak disukai pengguna.
# NDCG dan MRR membedakan posisi hit. **Cakupan katalog dan entropy frekuensi
# rekomendasi mengukur penyebaran paparan, bukan keragaman semantik buku.** Entropy
# memakai log natural; versi ternormalisasi membaginya dengan log jumlah karya
# katalog. Angka yang lebih tersebar belum tentu menghasilkan rekomendasi relevan.

# %%
ranking_comparator_rows, ranking_cohorts, ranking_comparator_bootstrap_rows = [], [], []
for result in comparator["ranking_results"]:
    ranking_cohorts.append(
        {
            "split seed": result["split_seed"],
            "ISBN katalog": result["catalogue_isbns"],
            "karya katalog": result["catalogue_works"],
            "pengguna assessment": result["assessment_users"],
            "pengguna dengan target relevan": result["relevant_users"],
            "target tersedia": result["available_targets"],
            "cakupan target": result["target_coverage"],
            "pengguna dikenal": result["known_users"],
            "MLP utama": result["primary_mlp"],
            "MF utama": result["primary_mf"],
            "bias terpilih": result["bias_name"],
        }
    )
    for name, metric in result["metrics"].items():
        ranking_comparator_rows.append(
            {
                "split seed": result["split_seed"],
                "model": name,
                **metric,
                "model utama dari selection": name
                in (result["primary_mlp"], result["primary_mf"], result["bias_name"]),
            }
        )
    for comparison in result["paired_bootstrap"]:
        ranking_comparator_bootstrap_rows.append(
            {
                "split seed": result["split_seed"],
                "kandidat": comparison["model"],
                "referensi": comparison["reference"],
                "metrik": comparison["metric"],
                "pengguna": comparison["users"],
                "selisih metrik": comparison["difference"],
                "batas bawah 95%": comparison["ci95"][0],
                "batas atas 95%": comparison["ci95"][1],
                "bootstrap draws": comparison["draws"],
            }
        )
ranking_comparator = pd.DataFrame(ranking_comparator_rows)
ranking_comparator_bootstrap = pd.DataFrame(ranking_comparator_bootstrap_rows)
display(pd.DataFrame(ranking_cohorts))
display(ranking_comparator)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for row_index, split_seed in enumerate(comparator_config["split_seeds"]):
    frame = ranking_comparator[ranking_comparator["split seed"] == split_seed]
    labels = frame["model"].str.replace(f"split{split_seed}_", "", regex=False)
    axes[row_index, 0].barh(labels, frame["recall_at_10"] * 100)
    axes[row_index, 0].set(title=f"Split {split_seed}: relevansi", xlabel="Recall@10 (%)")
    axes[row_index, 1].barh(labels, frame["catalogue_coverage"] * 100, color="#b48138")
    axes[row_index, 1].set(
        title=f"Split {split_seed}: penyebaran paparan", xlabel="Cakupan karya katalog (%)"
    )
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Ketidakpastian ranking: arah selisih berbeda dari RMSE
#
# Bootstrap berpasangan memakai pengguna yang sama pada kedua model. Untuk
# Recall/NDCG/MRR, **selisih positif menguntungkan kandidat**, sedangkan pada
# tabel RMSE selisih negatif menguntungkan kandidat. Semua interval berikut
# bersyarat pada assessment internal yang sudah digunakan ulang, tanpa koreksi
# banyak perbandingan; tidak digunakan untuk menyeleksi ulang model.

# %%
with pd.option_context("display.max_rows", None):
    display(ranking_comparator_bootstrap)
primary_ranking_comparisons = ranking_comparator_bootstrap[
    ranking_comparator_bootstrap.apply(
        lambda row: (
            row["kandidat"]
            in (
                comparator_selection[row["split seed"]]["primary_mf_candidate"],
                comparator_selection[row["split seed"]]["primary_mlp_candidate"],
            )
            and row["referensi"]
            in (
                comparator_selection[row["split seed"]]["bias_candidate"],
                "popularity",
                comparator_selection[row["split seed"]]["primary_mlp_candidate"],
            )
        ),
        axis=1,
    )
]
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for row_index, split_seed in enumerate(comparator_config["split_seeds"]):
    for column_index, metric in enumerate(("recall_at_10", "ndcg_at_10", "mrr_at_10")):
        ax = axes[row_index, column_index]
        frame = primary_ranking_comparisons[
            (primary_ranking_comparisons["split seed"] == split_seed)
            & (primary_ranking_comparisons["metrik"] == metric)
        ]
        labels = (frame["kandidat"] + " − " + frame["referensi"]).str.replace(
            f"split{split_seed}_", "", regex=False
        )
        positions = range(len(frame))
        ax.hlines(positions, frame["batas bawah 95%"], frame["batas atas 95%"])
        ax.plot(frame["selisih metrik"], positions, "o")
        ax.axvline(0, color="black", linestyle=":")
        ax.set(
            yticks=list(positions),
            yticklabels=labels,
            title=f"Split {split_seed}: {metric}",
            xlabel="Selisih kandidat − referensi",
        )
        ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ruang eksplorasi sampai run keenam (catatan historis)
#
# Bagian ini mempertahankan diagnosis sebelum run ketujuh. Objective ranking,
# embedding 32 pada arsitektur baru, dan metadata yang masih disebut usulan di
# bawah kemudian diuji pada run ketujuh; lihat ringkasan terbaru di akhir dan
# [notebook 04](04_neural_ranking_experiments.ipynb). Ini belum menjadi sweep
# ukuran embedding pada arsitektur yang sama atau evaluasi cold-start.
#
# Ruang pencarian masih terbatas. MLP/residual kini dibandingkan dengan MF
# embedding dangkal dan bias yang dituning, tetapi ukuran embedding,
# lebar jaringan, learning rate per arsitektur, dan batch size belum dituning. Run kelima
# menambahkan variasi seed training dan split internal; itu belum menggantikan
# konfirmasi pada data independen atau pengujian cold-start.
# Run awal menemukan koreksi hampir konstan. Ablation berikutnya menguji L2 dan
# inisialisasi head, lalu warm-up menguji training 12 epoch; tabel serta diagnosis di atas menyatakan hasilnya. Pencarian
# tersebut belum menguji ukuran embedding atau data independen. Run keenam
# menambahkan tuning bias/MF dan ranking pada assessment internal yang dipakai ulang.
#
# L2 Keras dihitung sebagai `lambda * sum(square(weights))`; besarnya perlu
# dinilai bersama MSE, jumlah bobot, dan gradient, bukan hanya angka lambda.
# Hasil ablation perlu dibaca bersama perubahan variasi koreksi dan kualitas
# validation. Mengurangi penyusutan bobot saja belum membuktikan neural lebih baik.
# Sumber: [Keras regularizers](https://keras.io/api/layers/regularizers/).

# %%
display(
    pd.DataFrame(
        [
            ("Embedding", "16", "Belum membandingkan 8/32/64"),
            (
                "Hidden units",
                "MLP/residual 32,16; MF dot product tanpa hidden MLP",
                "Belum sweep kapasitas MLP",
            ),
            (
                "Learning rate Adam",
                "MLP/residual 0.001; MF 0.005",
                "Belum sweep learning rate pada arsitektur yang sama",
            ),
            ("Batch size", "1024", "Belum dibandingkan batch lain"),
            (
                "L2 neural",
                "; ".join(f"{value:g}" for value in sorted({c["l2"] for c in ablation_neural})),
                "Satu seed/split; kekuatan regularisasi belum terkonfirmasi lintas split",
            ),
            (
                "Inisialisasi head residual",
                "Nol; acak kecil pada L2 1e-6",
                "Satu pasangan ablation terkontrol, bukan pencarian semua kombinasi",
            ),
            (
                "Durasi training residual",
                "Patience 3; L2 0/1e-7 juga 12 epoch tetap tanpa early stopping",
                "Warm-up adaptif pada validation lama; belum replikasi independen",
            ),
            (
                "Residual scale",
                "0.25 dan 1.0; kontrol 0",
                "Scale tetap pada run ablation; belum interaksi scale × learning rate",
            ),
            (
                "Bias shrinkage",
                "Joint bias: user × buku dari {0,5,20,100}, 16 kombinasi/split",
                "Grid terbatas dan assessment pengembangan telah dipakai ulang",
            ),
            (
                "Regularisasi MF",
                "Lambda sampled-pair 1e-4/1e-3/1e-2, 3 seed/split",
                "Tidak sebanding langsung dengan L2 atas seluruh matriks embedding",
            ),
            (
                "Objective",
                "MSE rating dengan regularisasi; ranking kini dievaluasi",
                "Belum melatih objective ranking",
            ),
            (
                "Seed / split",
                "Run 1–4: seed 42, split lama; run 5–6: training 42/2026/7 × split 17/29",
                "Dua split berasal dari train lama; bukan dataset independen",
            ),
            ("Fitur neural", "ID user dan ISBN", "Belum ada metadata untuk cold-start"),
        ],
        columns=["aspek", "sudah dicoba", "batas eksplorasi"],
    )
)

# %% [markdown]
# ## Rencana setelah run keenam (sebagian dijalankan pada run ketujuh)
#
# Urutan ini menguji hipotesis sebelum menambah kompleksitas. Angka di kolom
# konfigurasi adalah usulan saat run keenam selesai, bukan hasil atau janji
# peningkatan. Run ketujuh menguji BPR, selection berbasis NDCG, serta metadata;
# usulan tuning lainnya dan test independen tetap belum dijalankan.
#
# Diagnosis L2/head, kontrol kalibrasi skalar, dan overfit mini-batch sudah
# dicatat di run ketiga; warm-up 12 epoch tetap dicatat di run keempat.
# Run kelima memperbesar selection/assessment internal dan mengulang training
# pada tiga seed training; run keenam menambahkan bias/MF yang dituning dan
# ranking katalog penuh. Langkah berikutnya perlu mengikuti hasilnya dan
# keterbatasan populasi warm yang dipakai.
#
# 1. **Sesuaikan training dengan daftar rekomendasi:** perbandingan katalog
#    penuh masih dimenangkan popularity pada Recall/NDCG/MRR. Coba pasangan
#    positif/negatif dengan BPR atau BCE dan negative sampling yang ditetapkan
#    sebelum training. Gunakan metrik ranking pada selection untuk memilih
#    checkpoint, lalu evaluasi assessment sekali. Pertahankan popularity dan
#    bias yang dituning sebagai pembanding. Interaksi tidak teramati adalah
#    kandidat negatif sampling yang tidak pasti; rating 0 bukan bukti tidak suka.
#    BPR dirancang untuk personalized ranking pada implicit feedback
#    ([Rendle dkk., BPR](https://arxiv.org/abs/1205.2618)).
# 2. **Perluas tuning secara terukur:** lambda MF terpilih 1e-2 berada di ujung
#    atas grid dan seluruh MF memilih epoch 1; ini belum menunjukkan MF optimal.
#    Uji regularisasi lebih kuat, learning rate lebih kecil, atau bias yang
#    dibekukan sebagai ablation terpisah. Untuk MLP, embedding 8/16/32 dan
#    learning rate 3e-4/1e-3/3e-3 masih usulan. Tetapkan anggaran dan pilihan
#    berdasarkan selection, tanpa membuka assessment untuk memilih ulang.
# 3. **Konfirmasi dan tangani cakupan:** siapkan test independen yang tidak
#    dibuka selama pengembangan. Pertahankan protokol karya kanonik serta
#    pemisahan pengguna selection/assessment. Coba fitur judul/penulis/
#    penerbit atau representasi karya untuk cold-start dan bandingkan dengan
#    content/hybrid yang sudah ada. Metrik cold-start dilaporkan sendiri;
#    menaikkan ukuran embedding ID saja tidak memberi representasi bagi ID
#    yang tidak pernah dipelajari.
#
# Setiap run baru perlu mencatat hipotesis, perubahan tunggal yang diuji,
# konfigurasi, hash data/source, seed, durasi, semua epoch, kegagalan bila ada,
# kriteria seleksi, dan hasil pada cohort yang jelas. Jangan membuka test untuk
# setiap kandidat tuning. Log juga run yang gagal atau tidak mengungguli kontrol.

# %% [markdown]
# ## Kesimpulan enam run pertama (catatan historis)
#
# - **Tujuan pembelajaran tercapai:** implementasi neural dan bukti langkah
#   training memperlihatkan embedding, forward propagation, loss,
#   backpropagation, dan optimisasi pada proyek nyata.
# - **Belum optimal:** ada ablation regularisasi, head, durasi training,
#   tiga seed training, dua split internal, serta pembanding bias/MF yang
#   dituning. Ukuran embedding, learning
#   rate, objective ranking, serta fitur cold-start belum dituning. Run kelima
#   menambah bukti pengembangan internal; tidak menyediakan test independen baru.
#   Run keenam mengukur ranking pada assessment internal yang sudah digunakan ulang.
# - **Keunggulan MLP bergantung pada kekuatan pembanding:** enam MLP pada run
#   kelima mengungguli tiga kontrol awal. Setelah bias dituning pada run keenam,
#   keunggulan itu tidak konsisten: selisih RMSE MLP utama terhadap bias pada
#   split 17 kecil dan intervalnya melintasi nol; pada split 29 MLP lebih buruk.
#   MF utama juga belum memperbaiki RMSE bias secara konsisten. Temuan lama
#   tetap benar untuk kontrol lama, tetapi tidak cukup untuk menyatakan deep
#   learning lebih unggul daripada pembanding yang dituning.
# - **Prediksi rating belum menghasilkan ranking terbaik:** keenam MLP dan
#   keenam MF yang dinilai untuk ranking memiliki Recall/NDCG/MRR@10 lebih
#   rendah daripada popularity pada kedua split. MLP utama mendapatkan
#   33/1.776 dan 21/1.806 hit, sedangkan popularity 75/1.776 dan 66/1.806.
#   Interval Recall MLP utama dikurangi popularity seluruhnya negatif.
#   Bias yang kuat untuk RMSE juga belum menjadi ranking terbaik; tujuan
#   pemilihan checkpoint berikutnya harus sesuai dengan keluaran aplikasi.
# - **L2 memengaruhi penyusutan embedding**, tetapi menghapus L2 belum memberi
#   peningkatan validation yang berarti. Neural mampu menghafal mini-batch
#   nyata; kemampuan itu berbeda dari generalisasi pada seluruh benchmark.
# - **Warm-up pada run keempat sampai 12 epoch memperjelas overfitting:** train RMSE turun,
#   validation memburuk, dan kedua checkpoint terbaik tetap epoch 1. Dalam
#   batas percobaan ini, penghentian pada epoch 4 bukan penyebab utama kegagalan
#   memperbaiki validation. Menambah durasi saja tidak didukung hasil run ini.
# - **Hasil test historis tetap:** MLP terbaik pada validation belum terbaik
#   pada test; residual awal mendekati bias-only dan kalah tipis dari kontrol
#   itu pada test. Ranking neural awal belum mengalahkan item-based/popularity.
# - **Data dapat dipakai, tetapi desain evaluasi dan cold-start penting:**
#   mayoritas pengguna hanya punya sedikit interaksi, dan hanya 27% pasangan
#   test memiliki kedua ID dikenal. Jangan menyamakan hasil fallback dengan
#   keberhasilan embedding atau menganggap data tidak mendukung neural sama
#   sekali.
# - **Aplikasi tetap memakai baseline yang ada** sampai tambahan neural
#   memberi manfaat yang terukur dan dapat diulang.

# %%
completed_neural_count = sum(candidate["epochs_completed"] > 0 for candidate in ablation_neural)
warmup_trained_count = sum(candidate["epochs_completed"] > 0 for candidate in warmup_candidates)
multiseed_trained_count = sum(
    candidate["epochs_completed"] > 0 for candidate in multiseed_candidates
)
historical_record_count = (
    len(experiments)
    + len(ablation_experiments)
    + len(warmup_experiments)
    + len(multiseed_experiments)
)
historical_deep_training_count = (
    4 + completed_neural_count + warmup_trained_count + multiseed_trained_count
)
historical_epoch_count = (
    len(epochs) + len(ablation_epochs) + len(warmup_epochs) + len(multiseed_epochs)
)
mf_trained_count = sum(candidate["epochs_completed"] > 0 for candidate in mf_candidates)
best_ablation_name = ablation["selection"]["best_neural_candidate"]
best_ablation = ablation_by_name[best_ablation_name]
best_ablation_metric = best_ablation["validation"]["rmse_rating_scale"]
control_validation = ablation_experiments[ablation_experiments["jenis"] != "neural"]
best_control_metric = control_validation["validation RMSE"].min()
best_restored_correction = ablation_corrections[
    (ablation_corrections["percobaan"] == best_ablation_name)
    & (ablation_corrections["tahap"] == "restored_best")
    & (ablation_corrections["cohort"] == "validation")
].iloc[0]
display(
    Markdown(
        f"**Total riwayat:** {historical_record_count + len(bias_grid) + len(mf_candidates)} catatan kandidat "
        f"dalam enam run, termasuk **{historical_deep_training_count} training deep MLP/residual** "
        f"dan **{mf_trained_count} training MF embedding dangkal**, dengan "
        f"**{historical_epoch_count + len(mf_epochs)} epoch** training embedding. "
        f"Lima run awal tetap menyimpan {historical_record_count} catatan dan {historical_epoch_count} epoch. "
        f"Run keenam menambah {len(bias_grid)} konfigurasi bias dan {len(mf_candidates)} percobaan MF; "
        f"{len(reused_mlps)} MLP yang dimuat ulang tidak dihitung sebagai training baru. "
        "Diagnosis mini-batch dicatat terpisah. "
        "Pengulangan konfigurasi pada run berbeda tetap dicatat sebagai percobaan, "
        "bukan dianggap konfigurasi unik.\n\n"
        f"**Hasil ablation:** neural terpilih **`{best_ablation_name}`**, "
        f"validation RMSE **{best_ablation_metric:.6f}**; "
        f"kontrol terbaik **{best_control_metric:.6f}**. "
        f"Selisih neural − kontrol **{best_ablation_metric - best_control_metric:+.6f}** "
        "(negatif berarti RMSE neural lebih kecil). "
        f"Std koreksi validation pada checkpoint terpilih "
        f"**{best_restored_correction['koreksi std']:.8f}** poin rating. "
        "Angka ini menilai variasi koreksi; manfaat prediksi dinilai terpisah lewat RMSE.\n\n"
        f"**Diagnosis kapasitas:** MSE mini-batch **{overfit['initial_mse']:.6f} → "
        f"{overfit['final_mse']:.6f}**, setelah **{overfit['updates_completed']}** update. "
        "Hasil ini bukan metrik generalisasi. **Run ablation tidak menjalankan test.**"
    )
)
strong_l2 = ablation_by_name["residual_l2_1e4_zero_head"]
without_l2 = ablation_by_name["residual_l2_0_zero_head"]
strong_restored = next(d for d in strong_l2["diagnostics"] if d["stage"] == "restored_best")
unregularized_restored = next(d for d in without_l2["diagnostics"] if d["stage"] == "restored_best")
unregularized_last = next(
    d for d in reversed(without_l2["diagnostics"]) if d["stage"] == "epoch_end"
)
display(
    Markdown(
        "**Interpretasi ablation yang terkontrol:** menghapus L2 dibanding 1e-4 "
        f"mengubah norma embedding pengguna checkpoint terbaik dari "
        f"**{strong_restored['embedding_norms']['user_l2']:.4f}** menjadi "
        f"**{unregularized_restored['embedding_norms']['user_l2']:.4f}**. "
        "Jadi regularisasi memang menekan bobot dalam konfigurasi ini. Namun "
        f"RMSE validation hanya membaik sebesar **"
        f"{strong_l2['validation']['rmse_rating_scale'] - without_l2['validation']['rmse_rating_scale']:.8f}**. "
        "Koreksi pada checkpoint terbaik masih hampir berupa offset bersama. "
        "Peningkatan variasi output belum memberi peningkatan rekomendasi yang meyakinkan. "
        f"Diagnosis mini-batch menurunkan MSE **{100 * overfit['relative_mse_reduction']:.4f}%**, "
        "sehingga jaringan dan optimizer terbukti mampu mempelajari data nyata berulang; "
        "desain evaluasi, objective, dan generalisasi masih perlu diperbaiki."
    )
)
display(
    Markdown(
        "**Sinyal yang mulai muncul terlambat:** tanpa L2, std koreksi validation "
        f"naik dari **{unregularized_restored['corrections']['validation']['std_rating']:.6f}** "
        f"pada checkpoint terbaik menjadi **{unregularized_last['corrections']['validation']['std_rating']:.6f}** "
        f"pada epoch **{unregularized_last['epoch']}**. "
        f"Train RMSE terakhir **{without_l2['history'][-1]['train_rmse']:.6f}**, "
        f"validation RMSE terakhir **{without_l2['history'][-1]['val_rmse']:.6f}**. "
        "Training berhenti karena validation belum mengalahkan epoch 1 selama tiga epoch. "
        "Temuan ini menjadi alasan tindak lanjut 12 epoch yang dicatat sebagai run keempat."
    )
)
best_warmup_name = warmup["selection"]["best_neural_candidate"]
best_warmup = warmup_by_name[best_warmup_name]
best_warmup_rmse = best_warmup["validation"]["rmse_rating_scale"]
display(
    Markdown(
        f"**Hasil warm-up 12 epoch:** **`{best_warmup_name}`**, checkpoint epoch "
        f"**{best_warmup['best_epoch']}**, validation RMSE **{best_warmup_rmse:.6f}**. "
        f"Selisih terhadap neural terbaik run ablation: "
        f"**{best_warmup_rmse - best_ablation_metric:+.6f}**; "
        f"terhadap MLP awal: **{best_warmup_rmse - experiments.iloc[0]['validation RMSE']:+.6f}**. "
        "Negatif berarti RMSE lebih kecil pada validation pengembangan yang sama. "
        "Semua kandidat dan seluruh epoch tetap ditampilkan, termasuk hasil yang memburuk. "
        "**Belum ada test atau ranking baru untuk model hasil warm-up.**"
    )
)
display(
    Markdown(
        "**Jawaban terhadap hipotesis early stopping:** kedua kandidat tetap memilih "
        "epoch 1 walaupun dilatih sampai epoch 12. Tanpa L2, train RMSE akhir "
        f"**{warmup_by_name['residual_l2_0_zero_head_fixed12']['history'][-1]['train_rmse']:.6f}**, "
        "sedangkan validation RMSE akhir "
        f"**{warmup_by_name['residual_l2_0_zero_head_fixed12']['history'][-1]['val_rmse']:.6f}**. "
        "Jaringan benar-benar mempelajari train, tetapi pembelajaran tambahan itu "
        "tidak berpindah menjadi peningkatan validation. Ini mendukung diagnosis "
        "overfitting pada rancangan saat ini, bukan kesimpulan bahwa data sama sekali "
        "tidak dapat digunakan untuk neural."
    )
)
display(Markdown("**Temuan run kelima pada assessment pengembangan internal:**"))
for split_seed in multiseed["configuration"]["split_seeds"]:
    summaries = {
        row["architecture"]: row
        for row in multiseed["seed_summary"]
        if row["split_seed"] == split_seed
    }
    bias_result = multiseed_by_name[f"split{split_seed}_bias"]["assessment"]["known_pairs"]
    global_result = multiseed_by_name[f"split{split_seed}_global_mean"]["assessment"]["known_pairs"]
    chosen_name = selection_by_split[split_seed]["best_neural_candidate"]
    chosen = multiseed_by_name[chosen_name]
    display(
        Markdown(
            f"**Split {split_seed}:** mean RMSE dari tiga seed training pada pasangan dikenal: "
            f"MLP **{summaries['mlp']['known_rmse_mean']:.6f} ± "
            f"{summaries['mlp']['known_rmse_std']:.6f} SD**, residual "
            f"**{summaries['residual']['known_rmse_mean']:.6f} ± "
            f"{summaries['residual']['known_rmse_std']:.6f} SD**; "
            f"kontrol mean global **{global_result['rmse_rating_scale']:.6f}** dan "
            f"bias **{bias_result['rmse_rating_scale']:.6f}**. "
            f"Neural yang dipilih **sebelum assessment**, `{chosen_name}`, "
            f"mendapat assessment RMSE **{chosen['assessment']['known_pairs']['rmse_rating_scale']:.6f}**. "
            "SD ini variasi bobot awal dan urutan mini-batch pada split yang sama, bukan interval kepercayaan."
        )
    )
display(
    Markdown(
        "Bandingkan besaran selisih dan interval bootstrap pengguna pada tabel run kelima, "
        "bukan hanya urutan rata-rata. Kedua split tetap dianalisis terpisah; tidak ada "
        "kandidat yang dipromosikan berdasarkan assessment ini. Cakupan warm yang tinggi "
        "berasal dari aturan pembagian data, bukan perbaikan penanganan cold-start."
    )
)
display(Markdown("**Temuan run keenam: model utama yang ditentukan sebelum assessment/ranking**"))
primary_comparator_rows = []
for ranking_result in comparator["ranking_results"]:
    split_seed = ranking_result["split_seed"]
    chosen = comparator_selection[split_seed]
    rating_result = next(
        row for row in comparator["assessment_results"] if row["split_seed"] == split_seed
    )
    for name, role in (
        (chosen["bias_candidate"], "Joint bias terpilih"),
        (chosen["primary_mf_candidate"], "MF utama dari selection"),
        (chosen["primary_mlp_candidate"], "MLP utama dari selection"),
        ("popularity", "Popularity dari fit"),
    ):
        ranking_metric = ranking_result["metrics"][name]
        rating_metric = rating_result["models"].get(name, {}).get("known_pairs", {})
        primary_comparator_rows.append(
            {
                "split seed": split_seed,
                "peran": role,
                "model": name,
                "assessment RMSE": rating_metric.get("rmse_rating_scale"),
                "ranking pengguna relevan": ranking_metric["users"],
                "hits@10": ranking_metric["hits_at_10"],
                "Recall@10": ranking_metric["recall_at_10"],
                "NDCG@10": ranking_metric["ndcg_at_10"],
                "MRR@10": ranking_metric["mrr_at_10"],
                "cakupan karya katalog": ranking_metric["catalogue_coverage"],
                "entropy frekuensi ternormalisasi": ranking_metric["normalized_frequency_entropy"],
            }
        )
display(pd.DataFrame(primary_comparator_rows).fillna("tidak dinilai"))
for split_seed, chosen in comparator_selection.items():
    rating_ci = comparator_rating_bootstrap[
        (comparator_rating_bootstrap["split seed"] == split_seed)
        & (comparator_rating_bootstrap["cohort"] == "known_pairs")
        & (comparator_rating_bootstrap["kandidat"] == chosen["primary_mlp_candidate"])
        & (comparator_rating_bootstrap["referensi"] == chosen["bias_candidate"])
    ].iloc[0]
    ranking_ci = ranking_comparator_bootstrap[
        (ranking_comparator_bootstrap["split seed"] == split_seed)
        & (ranking_comparator_bootstrap["metrik"] == "recall_at_10")
        & (ranking_comparator_bootstrap["kandidat"] == chosen["primary_mlp_candidate"])
        & (ranking_comparator_bootstrap["referensi"] == "popularity")
    ].iloc[0]
    display(
        Markdown(
            f"**Split {split_seed}: MLP utama dibanding pembanding kuat.** "
            f"Selisih RMSE terhadap `{chosen['bias_candidate']}` "
            f"**{rating_ci['selisih RMSE']:+.6f}**, interval bootstrap pengguna 95% "
            f"**[{rating_ci['batas bawah 95%']:+.6f}, {rating_ci['batas atas 95%']:+.6f}]**. "
            "Negatif menguntungkan MLP. "
            f"Selisih Recall@10 terhadap popularity "
            f"**{100 * ranking_ci['selisih metrik']:+.3f} poin persentase**, interval 95% "
            f"**[{100 * ranking_ci['batas bawah 95%']:+.3f}, "
            f"{100 * ranking_ci['batas atas 95%']:+.3f}] poin persentase**. "
            "Untuk Recall, positif menguntungkan MLP. Interval bersyarat pada "
            "assessment yang dipakai ulang, tanpa koreksi banyak perbandingan."
        )
    )
display(
    Markdown(
        f"**Diagnosis run keenam:** {sum(row['converged'] for row in bias_grid)}/{len(bias_grid)} "
        "konfigurasi bias mencapai toleransi konvergensi. "
        f"Seluruh {len(mf_candidates)} MF selesai dengan best epoch "
        f"{sorted({row['best_epoch'] for row in mf_candidates})} dan jumlah epoch "
        f"{sorted({row['epochs_completed'] for row in mf_candidates})}; "
        "lambda terpilih 1e-2 adalah batas atas grid. Variasi faktor dan bias yang "
        "dapat dilatih belum menunjukkan perbaikan RMSE dan ranking yang konsisten "
        "terhadap bias yang dituning pada rancangan ini. Cakupan paparan MF yang lebih besar juga tidak menaikkan "
        "relevansi top-10. Ini membatasi klaim hasil run ini, bukan seluruh kemungkinan MF."
    )
)
display(
    Markdown(
        "Tabel ringkas ini mempertahankan pilihan dari selection; tidak memilih model menurut "
        "hasil ranking atau assessment. Seluruh seed, lambda, dan interval tetap tersedia di "
        "bagian run keenam. RMSE yang lebih kecil dan ranking yang lebih baik adalah dua "
        "hasil berbeda yang perlu dinilai masing-masing. Ranking hanya mengamati satu "
        "target holdout relevan per pengguna; karya lain yang tidak dinilai bukan negatif "
        "yang diketahui. Assessment yang dipakai ulang, "
        "pengguna warm, serta katalog tetap membatasi generalisasi; tidak ada promosi ke aplikasi."
    )
)

# %% [markdown]
# ## Run ketujuh: objective ranking dan metadata
#
# Run ini menindaklanjuti masalah ranking melalui tiga kelompok neural dua
# menara: MSE dengan ID, BPR dengan ID, dan BPR dengan ID + metadata. Seluruhnya
# dipilih memakai NDCG@10 selection atas katalog penuh. Dua split internal dan
# tiga seed per kelompok menghasilkan 18 rencana training baru. Semua hasil,
# epoch, loss, bukti gradien, variasi seed, cohort, dan bootstrap tersedia di
# **[notebook 04 — Neural Ranking Experiments](04_neural_ranking_experiments.ipynb)**.
#
# MSE dan BPR mengubah objective sekaligus supervisi; perbandingan keduanya
# tidak mengisolasi efek loss saja. Tambahan metadata dibandingkan pada BPR
# yang sama. Unit buku kini karya kanonik; MLP historis berbasis ISBN tetap
# benchmark, bukan kontrol arsitektur identik. Assessment internal dipakai
# ulang dari run terdahulu dan tidak menjadi test independen.
#
# **Hasil run ketujuh:** BPR ID utama menemukan target pada 72/1.776 dan 72/1.806
# pengguna, meningkat dari MLP lama 33/1.776 dan 21/1.806. Popularitas mendapat
# 75/1.776 dan 66/1.806. Neural kini mendekati popularitas, tetapi interval
# selisih Recall/NDCG BPR utama terhadap popularitas melintasi nol pada kedua
# split; belum ada keunggulan yang konsisten atau bukti kesetaraan.
#
# Pada cohort riwayat positif bersama, kenaikan NDCG BPR terhadap MLP lama tetap
# terlihat, sehingga perbaikan bukan hanya fallback. Arsitektur/unit karya/seleksi
# checkpoint juga berubah; efek masing-masing belum dipisahkan pada perbandingan lama/baru. BPR ID
# mengungguli MSE ID pada NDCG di semua enam pasangan seed yang dicocokkan, tetapi
# hanya dua interval seluruhnya di atas nol. Tambahan metadata belum konsisten:
# dua selisih NDCG positif dan empat negatif dibandingkan BPR ID yang dipasangkan;
# model metadata utama menemukan 67 dan 70 target. Seluruh interval dan batas
# interpretasi ditampilkan dalam notebook 04.

# %%
ranking_run = evidence["ranking"]
ranking_run_manifest = evidence["ranking_manifest"]
ranking_new_candidates = ranking_run["candidate_validation_results"]
ranking_new_epochs = sum(candidate["epochs_completed"] for candidate in ranking_new_candidates)
ranking_trained_count = sum(
    candidate["epochs_completed"] > 0 for candidate in ranking_new_candidates
)
ranking_primary_rows = []
for ranking_result in ranking_run["ranking_results"]:
    primary_models = [
        *ranking_result["primary_models"].items(),
        ("MLP lama", ranking_result["old_primary_mlp"]),
        ("Popularitas", "popularity"),
    ]
    for arm, name in primary_models:
        overall = ranking_result["metrics"][name]["all_relevant"]
        positive = ranking_result["metrics"][name]["fit_positive_history"]
        ranking_primary_rows.append(
            {
                "split": ranking_result["split_seed"],
                "model utama": arm,
                "kandidat": name,
                "target ditemukan": f"{overall['hits_at_10']}/{overall['users']}",
                "Recall@10": overall["recall_at_10"],
                "NDCG@10": overall["ndcg_at_10"],
                "target ditemukan (riwayat positif)": f"{positive['hits_at_10']}/{positive['users']}",
                "NDCG@10 riwayat positif": positive["ndcg_at_10"],
            }
        )
display(pd.DataFrame(ranking_primary_rows))
display(
    pd.DataFrame(
        [
            {
                "kandidat": candidate["name"],
                "kelompok": candidate["arm"],
                "split": candidate["split_seed"],
                "seed training": candidate["initialization_seed"],
                "status": candidate["status"],
                "epoch selesai": candidate["epochs_completed"],
                "epoch pilihan selection": candidate["best_epoch"],
                "selection NDCG@10": candidate["selection"]["all_relevant"]["ndcg_at_10"],
                "selection Recall@10": candidate["selection"]["all_relevant"]["recall_at_10"],
            }
            for candidate in ranking_new_candidates
        ]
    )
)
display(
    Markdown(
        f"**Total kumulatif tujuh run:** "
        f"**{historical_record_count + len(bias_grid) + len(mf_candidates) + len(ranking_new_candidates)} "
        f"catatan kandidat**, "
        f"**{historical_deep_training_count + ranking_trained_count} training deep neural** "
        f"dan **{mf_trained_count} training MF embedding dangkal**, "
        f"dengan **{historical_epoch_count + len(mf_epochs) + ranking_new_epochs} epoch**. "
        f"Run ketujuh menambah {len(ranking_new_candidates)} catatan, "
        f"{ranking_trained_count} training dengan sedikitnya satu epoch, "
        f"dan {ranking_new_epochs} epoch. MLP dan popularitas yang dipakai ulang "
        "sebagai benchmark tidak dihitung sebagai training baru. "
        "Diagnosis mini-batch tetap dicatat terpisah.\n\n"
        "Hasil assessment terbaru dan batas kesimpulannya disajikan di notebook 04. "
        "Kesimpulan enam run pertama di atas dipertahankan sebagai catatan "
        "historis sehingga alasan setiap tindak lanjut tetap dapat ditelusuri."
    )
)

# %% [markdown]
# ## Run kedelapan: membaca riwayat positif secara langsung
#
# Diagnosis sesudah run ketujuh menemukan tumpang tindih tinggi antara BPR ID
# dan popularitas. Ini menjadi alasan menguji representasi pengguna dari
# himpunan karya positif fit, disertai pembanding klasik cosine pada riwayat
# yang sama. Target training dikeluarkan dari riwayat sebelum pooling.
# Dua objective, BPR dan full softmax, memakai arsitektur dan target yang sama
# pada dua split dan tiga seed: 12 rencana training neural. Grid tetangga
# 50/200 × shrinkage 0/10 menambah delapan catatan konfigurasi klasik.
#
# Checkpoint dipilih melalui NDCG selection, lalu model dikunci sebelum
# assessment run ini. Diagnosis tambahan menukar riwayat pada model utama
# tanpa melatih ulang dan mempertahankan exclusion milik penerima rekomendasi.
# Pertukaran memengaruhi isi sekaligus panjang riwayat; ia memeriksa sensitivitas
# masukan, bukan mengisolasi efek kausal isi. Assessment tetap data pengembangan
# yang dipakai ulang. Seluruh kandidat, epoch, lima cohort, ketidakpastian,
# diagnosis, serta hasil ringkas tersedia di
# **[notebook 05 — Neural History Experiments](05_neural_history_experiments.ipynb)**.
#
# **Hasil run kedelapan:** cosine riwayat terpilih menemukan 191/1.776 dan
# 198/1.806 target, dibanding popularitas 75/1.776 dan 66/1.806. Ini menunjukkan
# sinyal rekomendasi yang berguna dalam data pengembangan. Neural softmax utama
# menemukan 109 dan 66 target; history-BPR utama 70 dan 67; BPR ID historis 72
# pada kedua split. Keunggulan neural tidak konsisten, dan semua neural utama
# masih kalah dari cosine terpilih pada NDCG cohort positif.
#
# Pertukaran riwayat mengubah hit softmax utama split 17 dari 109 menjadi 48,
# tetapi split 29 tetap 66. Kegunaan riwayat asli tampak pada checkpoint split 17;
# hasil itu tidak boleh digeneralisasikan ke seluruh model. Pemilihan tetap
# melalui selection, assessment dipakai ulang, dan tidak ada promosi aplikasi
# atau klaim test independen. Rincian ketidakpastian ada di notebook 05.

# %%
history_run = evidence["history"]
history_manifest = evidence["history_manifest"]
history_diagnosis = evidence["personalization_diagnosis"]
history_candidates = history_run["candidate_validation_results"]
history_baselines = history_run["baseline_grid_results"]
history_epoch_count = sum(candidate["epochs_completed"] for candidate in history_candidates)
history_deep_count = sum(candidate["epochs_completed"] > 0 for candidate in history_candidates)
display(
    pd.DataFrame(
        [
            {
                "split": split["split_seed"],
                "BPR ID historis": split["primary_bpr"],
                "cohort": "riwayat positif",
                "top-1 sama dengan popularitas (%)": 100 * split["overlap"]["positive"]["same_top1"]
                / split["overlap"]["positive"]["users"],
                "rata-rata karya sama di top-10": split["overlap"]["positive"]["mean_common_top10"],
            }
            for split in history_diagnosis["splits"]
        ]
    )
)
history_primary_rows = []
for ranking_result in history_run["ranking_results"]:
    primary_models = [
        *ranking_result["primary_models"].items(),
        ("Cosine riwayat", ranking_result["baseline_name"]),
        ("BPR ID lama", ranking_result["previous_primary_models"]["bpr_id"]),
        ("Popularitas", "popularity"),
    ]
    for label, name in primary_models:
        overall = ranking_result["metrics"][name]["all_relevant"]
        positive = ranking_result["metrics"][name]["fit_positive_history"]
        multiple = ranking_result["metrics"][name]["multiple_fit_positive"]
        history_primary_rows.append(
            {
                "split": ranking_result["split_seed"], "model utama": label, "kandidat": name,
                "target ditemukan": f"{overall['hits_at_10']}/{overall['users']}",
                "Recall@10": overall["recall_at_10"], "NDCG@10": overall["ndcg_at_10"],
                "NDCG@10 riwayat positif": positive["ndcg_at_10"],
                "NDCG@10 minimal 2 positif": multiple["ndcg_at_10"],
            }
        )
display(pd.DataFrame(history_primary_rows))
display(
    pd.DataFrame(
        [
            {
                "kandidat": candidate["name"], "split": candidate["split_seed"],
                "jenis": candidate["kind"], "status": candidate["status"],
                "epoch selesai": candidate.get("epochs_completed"),
                "epoch terbaik": candidate.get("best_epoch"),
                "selection NDCG@10": candidate["selection"]["all_relevant"]["ndcg_at_10"],
                "selection Recall@10": candidate["selection"]["all_relevant"]["recall_at_10"],
            }
            for candidate in [*history_candidates, *history_baselines]
        ]
    ).fillna("tidak berlaku untuk cosine")
)
display(
    Markdown(
        f"**Total kumulatif delapan run:** "
        f"**{historical_record_count + len(bias_grid) + len(mf_candidates) + len(ranking_new_candidates) + len(history_candidates) + len(history_baselines)} "
        f"catatan kandidat**, "
        f"**{historical_deep_training_count + ranking_trained_count + history_deep_count} training deep neural** "
        f"dan **{mf_trained_count} training MF embedding dangkal**, "
        f"dengan **{historical_epoch_count + len(mf_epochs) + ranking_new_epochs + history_epoch_count} epoch**. "
        f"Run kedelapan menambah {len(history_candidates)} catatan neural dan "
        f"{len(history_baselines)} konfigurasi cosine, dengan {history_epoch_count} epoch neural. "
        f"{len(history_run['history_shuffle_results'])} diagnosis pertukaran riwayat dan "
        "analisis personalisasi dari rekomendasi tersimpan tidak dihitung sebagai training baru. "
        "Seluruh seed, cohort, interval bootstrap, dan hasil diagnosis tersedia di notebook 05."
    )
)

# %% [markdown]
# ## Menambahkan eksperimen tanpa kehilangan riwayat
#
# Notebook ini adalah laporan bukti tersimpan; notebook 02 adalah demonstrasi
# training. Untuk run berikutnya:
#
# 1. Simpan JSON hasil + manifest baru dalam `docs/evidence` dengan nama run
#    unik, tanpa menimpa run lama. Simpan seluruh kandidat dan semua epoch;
#    isikan status gagal/terhenti beserta alasannya bila percobaan tidak selesai.
#    Checkpoint dan log besar tetap di workspace; manifest menyimpan hash.
# 2. Tambahkan file ke `EVIDENCE_FILES`, tambahkan adapter kecil ke tabel
#    konfigurasi/history/metrics sesuai schema run, dan tampilkan label cohort
#    serta status evaluasinya. Kolom test kandidat yang tidak diuji tetap kosong.
#    Perubahan schema baru harus eksplisit agar skor tidak diam-diam digabung
#    dengan protokol yang berbeda.
# 3. Jalankan ulang notebook dari awal, sinkronkan pasangan Jupytext, dan
#    gunakan `scripts/notebook_quality.py stamp` dengan **setiap JSON dalam
#    `EVIDENCE_FILES`** sebagai `--dependency`. Perbarui daftar dependency pada
#    CI. `check-pair` dan `check-saved` harus lulus sebelum hasil dibagikan.
#
# Bukti yang dipakai notebook ini terbatas pada run yang sudah diekspor ke
# repository. Log debugging lokal dan percobaan pada sesi percakapan lain tidak
# dianggap benchmark baru tanpa konfigurasi serta hasil yang dapat diverifikasi.
