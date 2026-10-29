"""Build the eight-run portfolio ledger from saved evidence, without training.

Only the Python standard library is needed. --check verifies generated documents
and exact source pointers instead of writing them. Counts follow notebook 03:
recorded candidates/controls, not unique architectures or every evaluation call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE_KEYS = (
    "t4",
    "residual",
    "ablation",
    "warmup",
    "multiseed",
    "comparator",
    "ranking",
    "history",
)
EXPECTED_RECORDS = (1, 4, 8, 2, 18, 50, 18, 20)
EXPECTED_EPOCHS = (5, 12, 24, 24, 75, 108, 106, 136)
NARRATIVES = [
    {
        "title": "MLP awal: membuktikan mekanisme belajar",
        "question": "Bisakah embedding pengguna dan ISBN, forward propagation, MSE, backpropagation, dan Adam bekerja dalam proyek rekomendasi nyata?",
        "change": "Embedding 16 dimensi pengguna/buku, concatenation beserta interaksi perkalian, Dense ReLU 32/16, lalu sigmoid untuk rating ternormalisasi. Training memakai MSE + L2, bukan objective ranking.",
        "result": "Lima epoch selesai; validation memilih epoch 2. Pada 135 pasangan test dengan kedua ID dikenal, RMSE neural 1,504167, sedangkan rata-rata rating pengguna 1,422509. Ranking neural dengan fallback menemukan 2 dari 272 target, pembanding item-based dan popularitas masing-masing 5.",
        "limitation": "Hanya 135/500 pasangan test memiliki kedua ID dikenal. Ranking 272 target dan RMSE 135 pasangan adalah kelompok berbeda; angka tidak boleh digabung. Berhasil melatih jaringan belum berarti rekomendasinya lebih baik.",
        "next_step": "Pisahkan pola rating umum melalui bias pengguna/buku, lalu minta jaringan mempelajari koreksinya.",
        "scope": "Benchmark awal; checkpoint dipilih dari validation. Test dilihat pada run ini, sehingga pemakaiannya pada run berikutnya bukan test baru.",
        "common": "MSE rating; embedding 16; hidden 32/16 ReLU; Adam lr 0,001; batch 1.024; L2 1e-5; seed 42; maksimal 8 epoch; patience 3.",
    },
    {
        "title": "Bias + residual: apakah neural memberi koreksi berguna?",
        "question": "Apakah jaringan lebih mudah belajar setelah kecenderungan rating pengguna dan buku dijelaskan oleh baseline?",
        "change": "Bias dari train dibekukan. MLP LeakyReLU menambahkan koreksi dengan skala 0,25 atau 1; L2 1e-5 dibandingkan 1e-4. Bias-only menjadi kontrol tanpa training neural.",
        "result": "Residual skala 0,25 dan L2 1e-4 dipilih: validation RMSE 1,746604. Pada 135 pasangan test dikenal, RMSE 1,404086, sedikit lebih buruk dari bias-only 1,400129. Ranking neural masih 2/272 target.",
        "limitation": "Validation hanya 140 pasangan dikenal. Test yang sama sudah dilihat pada fase 1. Dua residual yang tidak dipilih tidak diberi skor test; kosong berarti tidak dievaluasi, bukan nol. Label serving_candidate dalam bukti bukan deployment aplikasi.",
        "next_step": "Periksa apakah L2 atau inisialisasi output terlalu menekan koreksi neural, dengan kontrol analitik sederhana.",
        "scope": "Pengembangan pada validation lama dan test yang dipakai ulang; bukan konfirmasi independen.",
        "common": "MSE rating raw + L2; embedding 16; hidden 32/16 LeakyReLU; Adam lr 0,001; batch 1.024; shrinkage pengguna/buku 0/20; seed 42; maksimal 10 epoch; patience 3.",
    },
    {
        "title": "Ablation L2 dan output: mencari penyebab koreksi kecil",
        "question": "Apakah koreksi kecil terjadi karena regularisasi, output awal nol, atau jaringan memang tidak mampu menyesuaikan data?",
        "change": "L2 diuji dari 1e-4 sampai nol, disertai output acak kecil pada L2 1e-6. Kontrol bias-only dan bias + satu scalar yang dihitung dari train memeriksa apakah perbaikan sekadar pergeseran rata-rata.",
        "result": "Keenam neural memilih epoch 1 dan berhenti setelah 4 epoch. Tanpa L2 menjadi kandidat terbaik: validation RMSE 1,746581; kontrol scalar 1,747000. Diagnosis terpisah pada 128 pasangan train menurunkan MSE 0,026764 menjadi 0,00004253 setelah 150 update.",
        "limitation": "Perbaikan validation sangat kecil pada 140 pasangan. Menghafal mini-batch membuktikan kapasitas optimisasi, bukan generalisasi. Diagnosis 150 update tidak masuk 24 epoch benchmark atau hitungan kandidat. Beberapa konfigurasi sengaja mengulang fase 2.",
        "next_step": "Periksa kemungkinan sinyal baru muncul terlambat dengan menjalankan dua kandidat selama 12 epoch penuh.",
        "scope": "Diagnosis validation yang sama; tidak membuka test dan tidak mengganti model aplikasi.",
        "common": "Residual skala 0,25; MSE + L2 per baris; embedding 16; hidden 32/16 LeakyReLU; Adam lr 0,001; batch 1.024; shrinkage 0/20; seed 42; maksimal 10 epoch; patience 3.",
    },
    {
        "title": "Warm-up 12 epoch: apakah berhenti terlalu cepat?",
        "question": "Apakah early stopping sebelumnya menghentikan pembelajaran sebelum koreksi neural menjadi berguna?",
        "change": "Residual L2 nol dan 1e-7 dilatih tepat 12 epoch, early stopping dimatikan. Checkpoint tetap dipilih dari validation; kontrol fase 3 tidak dilatih ulang.",
        "result": "Kedua model menyelesaikan 12 epoch, tetapi checkpoint terbaik tetap epoch 1. Train membaik sementara validation memburuk; kandidat tanpa L2 mempertahankan validation RMSE 1,746581.",
        "limitation": "Hasil ini menolak penjelasan 'kurang lama' untuk dua konfigurasi dalam batas percobaan ini; bukan bukti bahwa semua arsitektur pasti gagal jika dilatih lebih lama. Tidak ada evaluasi test baru.",
        "next_step": "Perluas pemeriksaan ke dua pembagian pengembangan dan tiga seed agar kesimpulan tidak bergantung pada satu pembagian kecil.",
        "scope": "Tindak lanjut adaptif setelah melihat ablation, memakai validation lama; tidak membuka test.",
        "common": "Residual skala 0,25; embedding 16; hidden 32/16; Adam lr 0,001; batch 1.024; shrinkage 0/20; seed 42; tepat 12 epoch; output awal nol.",
    },
    {
        "title": "Dua split × tiga seed: menilai kestabilan prediksi rating",
        "question": "Apakah MLP dan residual tetap bekerja ketika pembagian pengembangan dan inisialisasi diubah?",
        "change": "Train lama dibagi menjadi fit, selection, dan assessment internal; split 17/29, seed 42/2026/7. MLP L2 1e-5 dibanding residual tanpa L2, serta kontrol global mean, user mean, dan bias per split.",
        "result": "Dua belas neural selesai dalam 75 epoch. Rata-rata assessment RMSE MLP antarseed 1,633008/1,597189 pada split 17/29, dibanding residual 1,920037/1,837842. MLP utama dipilih seed 42/7 berdasarkan selection.",
        "limitation": "Ini pembagian ulang data train yang sudah digunakan, bukan test independen. Seed mengubah inisialisasi dan urutan mini-batch, bukan memberi tiga dataset independen. Kemenangan terhadap kontrol bias awal belum membuktikan keunggulan terhadap bias yang dituning.",
        "next_step": "Berikan pembanding klasik kesempatan tuning yang jelas, lalu ukur ranking katalog penuh selain RMSE.",
        "scope": "Assessment pengembangan internal; seluruh 12 jaringan dikunci sebelum assessment run ini. Original validation/test tidak digunakan.",
        "common": "MSE rating; embedding 16; hidden 32/16; Adam lr 0,001; batch 1.024; maksimal 12 epoch; patience 3; split 17/29; seed 42/2026/7; fallback rating sama.",
    },
    {
        "title": "Bias dituning dan MF: apakah benchmark cukup kuat?",
        "question": "Apakah keuntungan MLP bertahan terhadap bias yang dioptimalkan bersama dan matrix factorization, dan apakah RMSE rendah berarti ranking baik?",
        "change": "Grid shrinkage bias pengguna/buku {0,5,20,100} menghasilkan 32 konfigurasi. MF 16 dimensi memakai tiga lambda × tiga seed × dua split, total 18 training. Enam MLP fase 5 dimuat ulang, tidak dilatih lagi.",
        "result": "Bias terpilih 5/20 pada split 17 dan 5/5 pada split 29; lambda MF 0,01 pada keduanya. Assessment RMSE bias 1,634786/1,579396, MLP utama 1,627100/1,601035, MF utama 1,636338/1,595686. MLP utama hanya menemukan 33/1.776 dan 21/1.806 target; popularitas 75/1.776 dan 66/1.806.",
        "limitation": "Ranking tidak dipakai memilih checkpoint pada fase ini: pemilihannya masih RMSE. MF adalah embedding dangkal, bukan deep MLP. Iterasi solver bias bukan epoch neural. Assessment internal sudah dilihat dan tidak memberi konfirmasi independen.",
        "next_step": "Selaraskan target latihan dan pemilihan checkpoint dengan ranking; uji BPR serta tambahan metadata.",
        "scope": "Pengembangan adaptif pada assessment fase 5 yang dipakai ulang; original validation/test tidak digunakan.",
        "common": "Bias: squared error + ridge, maksimum 1.000 iterasi, toleransi 1e-10. MF: MSE rating + penalti sampled-pair; embedding 16; Adam lr 0,005; batch 1.024; maksimum 30 epoch; patience 5. Lambda dipilih dari mean RMSE selection tiga seed.",
    },
    {
        "title": "Objective ranking dan metadata: mengejar posisi buku relevan",
        "question": "Apakah BPR memperbaiki ranking dibanding MSE, dan apakah judul/penulis/penerbit menambah informasi?",
        "change": "Dua menara MLP dengan embedding 32 membandingkan MSE ID, BPR ID, dan BPR ID + TF-IDF metadata 1.024 fitur. Unit kandidat menjadi karya kanonik. Checkpoint dan seed utama dipilih memakai NDCG@10 selection.",
        "result": "BPR ID utama menemukan 72/1.776 dan 72/1.806 target, mendekati popularitas 75/1.776 dan 66/1.806. Keenam pasangan seed BPR ID memiliki NDCG lebih tinggi dari MSE ID pada cohort positif, tetapi hanya dua interval seluruhnya di atas nol. Metadata memberi dua selisih positif dan empat negatif terhadap BPR ID.",
        "limitation": "MSE memakai rerata rating per karya; BPR memakai karya dengan salah satu rating >=8. Supervisi dan jumlah update ikut berubah, sehingga ini bukan isolasi loss saja. Dibanding MLP lama, arsitektur, unit ISBN/karya, fallback dan aturan checkpoint juga berbeda. Interval BPR utama versus popularitas melintasi nol pada kedua split; itu bukan bukti kesetaraan.",
        "next_step": "Daftar BPR ID masih sangat mirip popularitas. Uji representasi yang membaca riwayat positif secara langsung, dengan cosine pada riwayat sama sebagai pembanding.",
        "scope": "Assessment pengembangan yang dipakai ulang; semua 18 fit dan hash checkpoint dikunci sebelum assessment run ini. Tidak ada promosi aplikasi.",
        "common": "Embedding 32; user tower Dense64 ReLU→32; item tower ID+proyeksi metadata→Dense64 ReLU→32; dot/sqrt32 + bias karya + global logit tetap. Adam lr 0,001; batch 1.024; lambda 1e-5; maksimum 12 epoch; patience 3. Metadata nol pada kedua arm ID; bobot awal berpasangan.",
    },
    {
        "title": "Encoder riwayat dan cosine: menguji personalisasi yang berguna",
        "question": "Apakah membaca kumpulan buku yang disukai meningkatkan kualitas, dan apakah jaringan neural lebih berguna dari kemiripan klasik pada riwayat sama?",
        "change": "History-BPR dan full-softmax memakai embedding masukan/keluaran terpisah, pooling sum/sqrt(count), Dense64 tanh→32, serta bias karya. Target dikeluarkan dari riwayat latihan. Pengguna training perlu >=2 positif; inferensi boleh satu positif. Grid cosine k50/200 × shrink0/10 menjadi kontrol.",
        "result": "Cosine terpilih k50/shrink10 pada kedua split menemukan 191/1.776 dan 198/1.806 target. Softmax utama 109/1.776 dan 66/1.806; history-BPR 70/1.776 dan 67/1.806. Semua neural utama masih lebih rendah dari cosine terpilih pada NDCG cohort positif. Pertukaran riwayat membuat hit softmax utama split 17 turun 109→48, sedangkan split 29 tetap 66.",
        "limitation": "Dibanding ID-BPR lama, representasi, populasi target training, dan regularisasi berubah. Perbandingan BPR/softmax dalam fase ini lebih terkontrol. Donor mengubah isi dan panjang riwayat sekaligus; diagnosis bukan bukti kausal isi saja. Lima cohort saling tumpang tindih. Assessment tetap pengembangan yang dipakai ulang.",
        "next_step": "Jadikan cosine riwayat sebagai benchmark kuat untuk pengujian berikutnya; perbaikan neural harus konsisten dan diperiksa pada konfirmasi yang belum dipakai merancang eksperimen. Ini arah tindak lanjut, bukan hasil training tambahan atau deployment.",
        "scope": "Dua belas neural dan delapan cosine dikunci sebelum assessment. Empat diagnosis pertukaran riwayat tidak melatih ulang. Tidak ada promosi otomatis ke aplikasi.",
        "common": "Embedding 32; encoder tanpa parameter ID pengguna; Adam lr 0,001; batch 1.024; reg 1e-5 × mean semua parameter trainable²; maksimum 20 epoch; patience 4. BPR/softmax memakai target dan bobot awal sama. Cosine: Cij/sqrt(ci*cj) × Cij/(Cij+shrink), diagonal nol, top-k per baris sumber; seri skor dipecah popularitas fit lalu leksikal, neural leksikal.",
    },
]


def source_path(key):
    return f"docs/evidence/neural_colab_{key}_results.json"


def ref(key, pointer):
    return {"path": source_path(key), "json_pointer": pointer}


def resolve(document, pointer):
    for part in pointer.strip("/").split("/") if pointer else []:
        part = part.replace("~1", "/").replace("~0", "~")
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def notebook_for(phase):
    name = {7: "04_neural_ranking_experiments", 8: "05_neural_history_experiments"}.get(
        phase, "03_neural_experiment_history"
    )
    return f"notebooks/{name}.ipynb"


def rating_payload(metrics):
    known = metrics.get("known_pairs")
    return known if isinstance(known, dict) else metrics


def build_ledger(root=ROOT):
    documents = {
        key: json.loads((root / source_path(key)).read_text(encoding="utf-8"))
        for key in PHASE_KEYS
    }
    phases, records = [], []
    for phase, key in enumerate(PHASE_KEYS, 1):
        result, narrative = documents[key], NARRATIVES[phase - 1]
        config = result["configuration"]
        phase_rows = []
        if phase == 1:
            candidate = {
                "name": "mlp_original",
                "kind": "neural",
                "architecture": "mlp",
                **result["training"],
                "status": result["status"],
            }
            rows = [("", candidate)]
        else:
            arrays = ["candidate_validation_results"]
            if phase == 5:
                arrays += ["controls"]
            if phase == 6:
                arrays = ["bias_grid_results", "candidate_validation_results"]
            if phase == 8:
                arrays += ["baseline_grid_results"]
            rows = [
                (f"/{array}/{index}", row)
                for array in arrays
                for index, row in enumerate(result[array])
            ]
        for pointer, row in rows:
            name = row["name"]
            kind = (
                "mf"
                if phase == 6 and row["kind"] == "neural"
                else "deep_neural"
                if row["kind"] == "neural"
                else "classic_or_control"
            )
            trained = kind in ("mf", "deep_neural")
            params = {
                field: row[field]
                for field in (
                    "split_seed",
                    "initialization_seed",
                    "architecture",
                    "arm",
                    "objective",
                    "l2",
                    "regularization_lambda",
                    "residual_scale",
                    "user_shrinkage",
                    "item_shrinkage",
                    "head_initialization",
                    "control_kind",
                    "k",
                    "shrink",
                    "scalar_intercept_normalized",
                    "scalar_intercept_rating",
                )
                if field in row
            }
            objective = (
                "MSE rating ternormalisasi + regularisasi"
                if trained and phase <= 6
                else "MSE sigmoid(raw score), target rerata rating karya + regularisasi"
                if row.get("objective") == "mse"
                else "Mean softplus(score negatif − score target) + regularisasi"
                if row.get("objective") == "bpr"
                else "Full-catalogue cross-entropy, mask fit selain target + regularisasi"
                if row.get("objective") == "softmax"
                else config["bias_objective"]
                if phase == 6
                else "Kemiripan cosine berbobot co-occurrence; tanpa backpropagation"
                if phase == 8
                else "Kontrol statistik dari fit/train; tanpa training neural"
            )
            if phase == 1:
                selection = (
                    {
                        "pairs": result["dataset"]["validation_known_pairs"],
                        "rmse_rating_scale": min(e["val_rmse"] for e in row["history"]),
                    }
                    if "validation_known_pairs" in result["dataset"]
                    else {"rmse_rating_scale": min(e["val_rmse"] for e in row["history"])}
                )
                selection_ref = ref(key, "/training/history")
                selection_derivation = "minimum val_rmse across saved epochs"
            else:
                field = "selection" if "selection" in row else "validation"
                selection, selection_ref = row[field], ref(key, pointer + "/" + field)
                selection_derivation = None
            selected = phase == 1
            selection_role = (
                "kandidat tunggal; checkpoint terbaik validation"
                if selected
                else "kandidat pembanding"
            )
            if phase in (2, 3, 4):
                selected = name == result["selection"]["best_neural_candidate"]
                selection_role = (
                    "neural terpilih melalui validation"
                    if selected
                    else "kontrol"
                    if not trained
                    else "neural tidak terpilih"
                )
            elif phase == 5:
                selected = any(
                    name == choice["best_candidate"] for choice in result["selection"]["by_split"]
                )
                selection_role = (
                    "utama split melalui selection"
                    if selected
                    else "kontrol"
                    if not trained
                    else "seed/arsitektur pembanding"
                )
            elif phase == 6:
                choice = next(
                    s for s in result["lambda_selection"] if s["split_seed"] == row["split_seed"]
                )
                selected = name in (choice["bias_candidate"], choice["primary_mf_candidate"])
                selection_role = (
                    ("bias terpilih" if not trained else "MF utama")
                    if selected
                    else (
                        "lambda terpilih; seed pembanding"
                        if name in choice["selected_mf_candidates"]
                        else "grid tidak terpilih"
                    )
                )
            elif phase in (7, 8):
                choice = next(
                    s for s in result["primary_selection"] if s["split_seed"] == row["split_seed"]
                )
                selected = name in [
                    s["primary_candidate"] for s in choice["arms"]
                ] or name == choice.get("baseline_candidate")
                selection_role = (
                    ("neural utama per arm" if trained else "cosine terpilih")
                    if selected
                    else (
                        "checkpoint terpilih; seed pembanding"
                        if trained
                        else "cosine grid pembanding"
                    )
                )
            record = {
                "id": f"run{phase}:{name}",
                "phase": phase,
                "phase_key": key,
                "name": name,
                "kind": kind,
                "source_kind": row["kind"],
                "status": row.get("status", result["status"]),
                "new_embedding_training": trained,
                "objective": objective,
                "parameters": params,
                "common_configuration_ref": f"/phases/{phase - 1}/configuration",
                "epochs_completed": int(row.get("epochs_completed", 0)),
                "best_epoch": row.get("best_epoch"),
                "solver_iterations": row.get("iterations"),
                "selection": selection,
                "selection_source": selection_ref,
                "selection_derivation": selection_derivation,
                "selection_metric": "ndcg_at_10" if phase >= 7 else "rmse_rating_scale",
                "selected_primary": selected,
                "decision": selection_role,
                "assessment": [],
                "later_evaluations": [],
                "scope": narrative["scope"],
                "source": ref(key, pointer),
                "source_name_is_display_alias": phase == 1,
                "notebook": notebook_for(phase),
            }
            if phase == 1:
                for kind_label, path in (
                    ("rating_test", "/rating_test/models/neural"),
                    ("ranking_test", "/ranking_test/models/neural_with_popularity_fallback"),
                ):
                    record["assessment"].append(
                        {
                            "kind": kind_label,
                            "phase": phase,
                            "metrics": resolve(result, path),
                            "source": ref(key, path),
                        }
                    )
            elif phase == 2:
                alias = (
                    "bias_only"
                    if row["kind"] == "bias_only"
                    else "best_neural"
                    if selected
                    else None
                )
                if alias:
                    path = f"/rating_test/models/{alias}"
                    record["assessment"].append(
                        {
                            "kind": "rating_test_reused",
                            "phase": phase,
                            "metrics": resolve(result, path),
                            "source": ref(key, path),
                        }
                    )
                if selected:
                    path = "/ranking_test/models/neural_with_popularity_fallback"
                    record["assessment"].append(
                        {
                            "kind": "ranking_test_reused",
                            "phase": phase,
                            "metrics": resolve(result, path),
                            "source": ref(key, path),
                        }
                    )
            elif "assessment" in row:
                record["assessment"].append(
                    {
                        "kind": "ranking_assessment" if phase >= 7 else "rating_assessment",
                        "phase": phase,
                        "metrics": row["assessment"],
                        "source": ref(key, pointer + "/assessment"),
                    }
                )
            record["assessment_absence_reason"] = None
            if not record["assessment"]:
                record["assessment_absence_reason"] = (
                    "Test sengaja tidak dijalankan pada fase diagnosis."
                    if phase in (3, 4)
                    else "Kandidat tidak terpilih untuk assessment/test pada fase asal."
                )
            phase_rows.append(record)
        count = Counter(row["kind"] for row in phase_rows)
        counts = {
            "records": len(phase_rows),
            "deep_neural_fits": count["deep_neural"],
            "mf_fits": count["mf"],
            "classic_or_control_records": count["classic_or_control"],
            "embedding_epochs": sum(r["epochs_completed"] for r in phase_rows),
        }
        assert counts["records"] == EXPECTED_RECORDS[phase - 1]
        assert counts["embedding_epochs"] == EXPECTED_EPOCHS[phase - 1]
        phases.append(
            {
                "phase": phase,
                "key": key,
                **narrative,
                "counts": counts,
                "source": source_path(key),
                "source_sha256": hashlib.sha256((root / source_path(key)).read_bytes()).hexdigest(),
                "completed_utc": result["completed_utc"],
                "notebook": notebook_for(phase),
                "configuration": {
                    k: v
                    for k, v in config.items()
                    if k not in ("candidates", "data_dir", "input_dir", "output")
                },
                "selection_decision_source": ref(
                    key,
                    "/primary_selection"
                    if phase >= 7
                    else "/lambda_selection"
                    if phase == 6
                    else "/training"
                    if phase == 1
                    else "/selection",
                ),
            }
        )
        records.extend(phase_rows)

    # Add later uses to the original record, never create another training row.
    by_name = {row["name"]: row for row in records}
    assert len(by_name) == len(records)
    for phase, key in enumerate(PHASE_KEYS, 1):
        result = documents[key]
        for field, metrics_field, label in (
            ("assessment_results", "models", "rating_assessment"),
            ("ranking_results", "metrics", "ranking_assessment"),
        ):
            for index, assessment in enumerate(result.get(field, [])):
                for name, metrics in assessment[metrics_field].items():
                    record = by_name.get(name)
                    if record is None:
                        continue  # popularity is an evaluation reference, not an extra candidate fit
                    target = "later_evaluations" if phase > record["phase"] else "assessment"
                    if phase < record["phase"]:
                        raise AssertionError("An evaluation cannot precede the recorded fit.")
                    if any(
                        item["phase"] == phase and item["kind"] == label for item in record[target]
                    ):
                        continue
                    record[target].append(
                        {
                            "kind": label,
                            "phase": phase,
                            "metrics": metrics,
                            "source": ref(key, f"/{field}/{index}/{metrics_field}/{name}"),
                        }
                    )
                    if target == "assessment":
                        record["assessment_absence_reason"] = None
    totals = {key: sum(p["counts"][key] for p in phases) for key in phases[0]["counts"]}
    totals["embedding_fits"] = totals["deep_neural_fits"] + totals["mf_fits"]
    assert totals == {
        "records": 121,
        "deep_neural_fits": 54,
        "mf_fits": 18,
        "classic_or_control_records": 49,
        "embedding_epochs": 490,
        "embedding_fits": 72,
    }
    # Verify every copied metric against its precise original JSON location.
    document_by_path = {source_path(key): value for key, value in documents.items()}
    for record in records:
        source = record["source"]
        original = resolve(document_by_path[source["path"]], source["json_pointer"])
        if not record["source_name_is_display_alias"]:
            assert original["name"] == record["name"]
            assert original.get("epochs_completed", 0) == record["epochs_completed"]
        source = record["selection_source"]
        if record["selection_derivation"] is None:
            assert (
                resolve(document_by_path[source["path"]], source["json_pointer"])
                == record["selection"]
            )
        for observation in record["assessment"] + record["later_evaluations"]:
            source = observation["source"]
            assert (
                resolve(document_by_path[source["path"]], source["json_pointer"])
                == observation["metrics"]
            )
    diagnosis = documents["ablation"]["small_batch_overfit"]
    return {
        "schema_version": 1,
        "title": "Register lengkap delapan fase eksperimen rekomendasi buku",
        "counting_reference": "notebooks/03_neural_experiment_history.py",
        "counting_policy": "Records follow the eight-run notebook registry. Repeated configurations count as separate run records; reloaded models, evaluations, and auxiliary diagnostics are not additional candidate fits.",
        "totals": totals,
        "phases": phases,
        "records": records,
        "excluded_from_candidate_counts": {
            "small_batch_diagnostic": {
                "phase": 3,
                "pairs": diagnosis["pairs"],
                "updates": diagnosis["updates_completed"],
                "initial_mse": diagnosis["initial_mse"],
                "final_mse": diagnosis["final_mse"],
                "source": ref("ablation", "/small_batch_overfit"),
                "scope": "Train-only capacity diagnostic; 150 updates are not full-data benchmark epochs.",
            },
            "history_shuffle_evaluations": {
                "phase": 8,
                "count": len(documents["history"]["history_shuffle_results"]),
                "source": ref("history", "/history_shuffle_results"),
                "trained_again": False,
            },
            "other_references": [
                "Reused models are linked in later_evaluations, without another fit record.",
                "Popularity and early rating/item-based reference scores are benchmark evaluations, not additional candidate records in notebook03.",
                "Notebook02 smoke test, GPU preflight and software tests are separate from this benchmark registry.",
            ],
        },
    }


def format_metric(record):
    metric = record["selection"]
    if record["phase"] >= 7:
        metric = metric["all_relevant"]
        return f"N {metric['ndcg_at_10']:.6f}; R {metric['recall_at_10']:.6f}"
    return f"RMSE {rating_payload(metric)['rmse_rating_scale']:.6f}"


def format_assessment(record):
    values = []
    for observation in record["assessment"]:
        metrics = observation["metrics"]
        if observation["kind"].startswith("rating"):
            values.append(f"RMSE {rating_payload(metrics)['rmse_rating_scale']:.6f}")
        else:
            overall = metrics.get("all_relevant", metrics.get("metrics", metrics))
            count = overall.get("users", metrics.get("count"))
            hits = overall.get("hits_at_10", round(overall["recall_at_10"] * count))
            values.append(f"{hits}/{count} hit; N {overall['ndcg_at_10']:.6f}")
    return "; ".join(values) if values else "tidak dievaluasi"


def format_parameters(record):
    p, phase = record["parameters"], record["phase"]
    values = []
    if phase in (2, 3) and not record["new_embedding_training"]:
        return (
            "bias + scalar fit dari train; shrink u/i 0/20"
            if record["source_kind"] == "bias_plus_scalar"
            else "bias-only; shrink u/i 0/20; residual tidak aktif"
        )
    if "split_seed" in p:
        values.append(f"split {p['split_seed']}")
    if "initialization_seed" in p:
        values.append(f"seed {p['initialization_seed']}")
    if phase == 1:
        values.append("MLP; seed 42; L2 1e-5")
    if "control_kind" in p:
        values.append(p["control_kind"])
    if phase in (2, 3, 4, 5) and record["kind"] == "deep_neural":
        values.append(p.get("architecture", "residual"))
    for field, label in (
        ("l2", "L2"),
        ("regularization_lambda", "λ"),
        ("residual_scale", "skala"),
        ("k", "k"),
        ("shrink", "shrink"),
    ):
        if field in p:
            values.append(f"{label} {p[field]:g}")
    if "user_shrinkage" in p:
        values.append(f"shrink u/i {p['user_shrinkage']:g}/{p['item_shrinkage']:g}")
    if "head_initialization" in p:
        values.append("head nol" if p["head_initialization"] == "zeros" else "head acak kecil")
    if "arm" in p:
        values.append(p["arm"])
    if not values:
        values.append(record["source_kind"])
    return "; ".join(values)


def markdown(ledger):
    text = [
        "# Register lengkap eksperimen",
        "",
        "Delapan fase ini membentuk satu perjalanan penelitian: membuat jaringan dapat belajar, menguji penyebab hasil lemah, memperkuat pembanding, lalu menyelaraskan model dengan tujuan rekomendasi. Semua angka di halaman ini diturunkan dari delapan JSON bukti; model tidak dilatih ulang ketika register dibuat.",
        "",
        "**121 catatan konfigurasi: 54 training deep neural, 18 training MF embedding dangkal, dan 49 kontrol/konfigurasi klasik; total 490 epoch training embedding.** Ini bukan 121 arsitektur unik. Konfigurasi yang sengaja diulang pada run berbeda tetap dicatat. Iterasi solver bias, diagnosis mini-batch, model yang dimuat ulang, dan evaluasi pertukaran riwayat tidak ditambahkan sebagai epoch atau fit benchmark baru.",
        "",
        "Versi terstruktur lengkap: [experiment-ledger.json](experiment-ledger.json). Sumber penghitungan: [notebook 03](../../notebooks/03_neural_experiment_history.ipynb). Mekanisme belajar diperagakan di [notebook 02](../../notebooks/02_neural_book_recommender.ipynb); smoke test notebook 02 memakai subset berbeda dan tidak masuk benchmark ini.",
        "",
        "| Fase | Pertanyaan | Catatan | Deep | MF | Kontrol/klasik | Epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in ledger["phases"]:
        c = p["counts"]
        text.append(
            f"| [{p['phase']}](#fase-{p['phase']}) | {p['title']} | {c['records']} | {c['deep_neural_fits']} | {c['mf_fits']} | {c['classic_or_control_records']} | {c['embedding_epochs']} |"
        )
    text += [
        "| **Total** | | **121** | **54** | **18** | **49** | **490** |",
        "",
        "## Cara membaca angka",
        "",
        "RMSE mengukur kesalahan menebak rating; lebih kecil lebih baik. N adalah NDCG@10 dan R adalah Recall@10; lebih besar lebih baik. Hit berarti satu karya target relevan muncul di sepuluh rekomendasi. Angka ranking di tabel memakai cohort semua pengguna relevan. Semua cohort, MAE/MRR, coverage dan metrik lain yang tersedia tetap disimpan tanpa pembulatan dalam JSON ledger.",
        "",
        "Kolom epoch berbentuk **terbaik/selesai**, bukan jumlah pembaruan Adam. Tanda — berarti tidak ada training embedding. Selection memilih checkpoint atau konfigurasi; assessment/test melaporkan hasil sesudah pemilihan. Nama kandidat dapat diklik untuk membuka sumber bukti. JSON Pointer dalam ledger menunjukkan letak baris dan metrik secara tepat di file tersebut.",
        "",
        "Skor antarprotokol tidak boleh dianggap satu papan peringkat. Fase 1–4 memakai validation lama yang kecil; fase 5 membuat pembagian internal dari train lama; fase 6–8 memakai ulang assessment internal. Rating memakai pasangan pengguna/ISBN yang dikenal, sedangkan ranking memakai target karya dan fallback yang dijelaskan per fase. Tabel ini tidak menyatakan test independen atau promosi model ke aplikasi.",
        "",
        "Assessment pada tabel adalah evaluasi pada **fase asal** kandidat. Hasil model yang dipakai ulang di fase berikutnya tersimpan pada `later_evaluations`, sehingga riwayat lengkap dapat ditelusuri tanpa menggandakan jumlah fit. `selected_primary` berarti pilihan dalam protokol fase itu, bukan model terbaik universal atau model produksi.",
        "",
    ]
    for p in ledger["phases"]:
        phase = p["phase"]
        clean_title = p["title"]
        for before, after in (
            ("Warm-up", "Latihan"),
            ("+", "dan"),
            ("×", "kali"),
            (":", ""),
            ("?", ""),
            ("—", " "),
        ):
            clean_title = clean_title.replace(before, after)
        clean_title = " ".join(clean_title.split())
        text += [f'<a id="fase-{phase}"></a>', "", f"## Fase {phase} {clean_title}", ""]
        for label, key in (
            ("Pertanyaan", "question"),
            ("Perubahan", "change"),
            ("Hasil", "result"),
            ("Batas kesimpulan", "limitation"),
            ("Langkah berikutnya", "next_step"),
        ):
            text += [f"**{label}.** {p[key]}", ""]
        text += [
            f"**Pengaturan bersama.** {p['common']}",
            "",
            f"**Sumber dan ruang lingkup.** [JSON fase {phase}](../evidence/{Path(p['source']).name}); [notebook penjelasan](../../{p['notebook']}). {p['scope']}",
            "",
            "| Kandidat | Pengaturan pembeda | Epoch terbaik/selesai | Selection | Assessment/test fase asal | Keputusan |",
            "|---|---|---:|---|---|---|",
        ]
        for r in (r for r in ledger["records"] if r["phase"] == phase):
            epoch = (
                f"{r['best_epoch']}/{r['epochs_completed']}" if r["new_embedding_training"] else "—"
            )
            decision = r["decision"]
            if r["solver_iterations"] is not None:
                decision += f"; {r['solver_iterations']} iterasi solver"
            text.append(
                f"| [{r['name']}](../evidence/{Path(r['source']['path']).name}) | {format_parameters(r)} | {epoch} | {format_metric(r)} | {format_assessment(r)} | {decision} |"
            )
        text.append("")
    text += [
        "## Diagnosis dan pembanding yang tidak menambah jumlah kandidat",
        "",
        "Diagnosis kapasitas fase 3 memakai 128 pasangan train selama 150 update, dengan MSE 0,026764486 → 0,000042529. Ini latihan tambahan khusus diagnosis pada batch kecil, dicatat terpisah dari 72 fit embedding benchmark. Fase 4 hanya merujuk hasil diagnosis tersebut. Empat evaluasi donor riwayat fase 8 mempertahankan bobot model dan tidak melakukan training lagi.",
        "",
        "Skor popularitas, rata-rata rating, dan item-based pada benchmark awal tetap penting sebagai referensi; aturan hitungan notebook 03 tidak menjadikannya tambahan baris kandidat terdaftar. Enam MLP yang dimuat ulang pada fase 6, serta model historis yang dinilai ulang pada fase 7–8, tetap merujuk fit asal. Tes perangkat lunak dan preflight GPU membuktikan aspek operasional, bukan meningkatkan jumlah eksperimen kualitas model.",
        "",
        "## Mereproduksi register",
        "",
        "Dari root repository, jalankan perintah berikut. Hanya Python standard library yang diperlukan; tidak mengakses Colab, dataset privat, atau TensorFlow.",
        "",
        "```bash",
        "python scripts/build_portfolio_ledger.py",
        "python scripts/build_portfolio_ledger.py --check",
        "```",
        "",
        "Pemeriksaan memastikan keanggotaan dan jumlah per fase, total epoch, nama kandidat unik dalam register, kecocokan metrik dengan JSON Pointer sumber, dan kesamaan dokumen yang dihasilkan. SHA-256 setiap sumber tercatat dalam JSON ledger. Tabel Markdown membulatkan enam desimal; JSON mempertahankan presisi bukti sumber.",
        "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ledger = build_ledger()
    documents = {
        "experiment-ledger.json": json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        "experiment-ledger.md": markdown(ledger),
    }
    folder = ROOT / "docs/portfolio"
    for filename, content in documents.items():
        target = folder / filename
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Ledger is missing or stale: {target}")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "checked" if args.check else "written", **ledger["totals"]}))


if __name__ == "__main__":
    main()
