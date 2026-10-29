"""Post-hoc paired comparisons using verified, already saved assessment predictions.

The global-mean control was identified after inspecting the completed run. This
supplement is exploratory: no retraining, new predictions, checkpoint selection,
original test evaluation, ranking evaluation, or serving change is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .colab_neural_experiment import ROOT, atomic_json, fingerprint_files, sha256_file, utc_now
    from .colab_neural_multiseed import CANDIDATES, FROZEN_INPUT_SHA256, SPLIT_SEEDS, user_cluster_bootstrap
    from .colab_residual_experiment import regression_metrics
except ImportError:
    from colab_neural_experiment import ROOT, atomic_json, fingerprint_files, sha256_file, utc_now
    from colab_neural_multiseed import CANDIDATES, FROZEN_INPUT_SHA256, SPLIT_SEEDS, user_cluster_bootstrap
    from colab_residual_experiment import regression_metrics


def audit(artifact_dir, output):
    import numpy as np
    import pandas as pd

    artifact_dir = Path(artifact_dir).resolve()
    verification_path = artifact_dir.parent / "backup_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    final = verification.get("final_backup.tar.gz", {})
    if final.get("kind") != "completed_experiment" or not isinstance(final.get("files"), dict):
        raise RuntimeError("A locally verified final experiment backup is required.")
    if Path(final["extracted_directory"]).resolve() != artifact_dir:
        raise RuntimeError("Artifact directory differs from the verified final extraction.")
    verified_files = {}

    def verified_file(relative):
        path = artifact_dir / relative
        expected = final["files"].get(relative)
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Unverified or modified assessment input: {relative}")
        verified_files[relative] = expected
        return path

    results_path = verified_file("results.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if (results.get("experiment_kind") != "multiseed_internal_development"
            or not results.get("assessment_performed") or not results.get("all_candidates_completed")
            or results.get("original_test_used") or results.get("original_validation_used")):
        raise RuntimeError("Expected a completed development-only assessment of all declared candidates.")
    candidates = results["candidate_validation_results"]
    if ({row["name"] for row in candidates} != {row["name"] for row in CANDIDATES}
            or len(candidates) != len(CANDIDATES)
            or any(row["status"] != "completed" or not row.get("assessment_scored") for row in candidates)
            or results["configuration"]["split_seeds"] != list(SPLIT_SEEDS)):
        raise RuntimeError("Candidate membership differs from the twelve declared comparisons.")
    source_path = verified_file("inputs/train_ratings.csv")
    if sha256_file(source_path) != FROZEN_INPUT_SHA256["train_ratings.csv"]:
        raise RuntimeError("Source ratings do not match the frozen training input.")
    source = pd.read_csv(source_path, dtype={"isbn": str})
    comparisons = []
    prediction_fingerprints = {}

    def reported_path(path):
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    for split_seed in results["configuration"]["split_seeds"]:
        assessment_path = verified_file(f"splits/split_{split_seed}/assessment_ratings.csv")
        prediction_path = verified_file(f"splits/split_{split_seed}/assessment_predictions.csv")
        assessment = pd.read_csv(assessment_path, dtype={"isbn": str})
        prediction = pd.read_csv(prediction_path, dtype={"isbn": str})
        alignment_columns = ["source_row", "user_id", "isbn", "book_rating", "work_key"]
        pd.testing.assert_frame_equal(assessment[alignment_columns], prediction[alignment_columns],
                                      check_dtype=False, check_exact=True)
        indices = prediction.source_row.to_numpy()
        if (not np.issubdtype(indices.dtype, np.integer) or len(set(indices)) != len(indices)
                or (indices < 0).any() or (indices >= len(source)).any()):
            raise RuntimeError("Assessment source-row indices are invalid or duplicated.")
        pd.testing.assert_frame_equal(
            prediction[["user_id", "isbn", "book_rating"]].reset_index(drop=True),
            source.iloc[indices][["user_id", "isbn", "book_rating"]].reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
        if prediction.both_ids_known.dtype != bool:
            raise RuntimeError("Known-pair flag must be an explicit boolean column.")
        known = prediction.both_ids_known.to_numpy()
        summary = next(row for row in results["dataset"]["splits"] if row["split_seed"] == split_seed)
        if (len(prediction) != summary["assessment_pairs"]
                or prediction.user_id.nunique() != summary["assessment_users"]):
            raise RuntimeError("Assessment row/user counts differ from the stored split summary.")
        split_results = next(row for row in results["assessment_results"] if row["split_seed"] == split_seed)
        reference = f"split{split_seed}_global_mean"
        split_candidates = [row for row in candidates if row["split_seed"] == split_seed]
        for scope, mask in (("known_pairs", known), ("all_pairs", np.ones(len(prediction), dtype=bool))):
            targets = prediction.book_rating.to_numpy(float)[mask]
            reproduced = {}
            for name in [reference, *[row["name"] for row in split_candidates]]:
                metrics = regression_metrics(targets, prediction[name].to_numpy(float)[mask], np)
                stored = split_results["models"][name][scope]
                if metrics["pairs"] != stored["pairs"]:
                    raise RuntimeError(f"Stored pair count differs: {name}/{scope}")
                for metric in ("rmse_rating_scale", "mae_rating_scale"):
                    np.testing.assert_allclose(metrics[metric], stored[metric], rtol=1e-12, atol=1e-12)
                reproduced[name] = metrics
            for candidate in split_candidates:
                name = candidate["name"]
                bootstrap = user_cluster_bootstrap(
                    prediction.user_id.to_numpy()[mask], targets,
                    prediction[name].to_numpy(float)[mask], prediction[reference].to_numpy(float)[mask],
                    np, draws=1000, seed=123,
                )
                comparisons.append({
                    "split_seed": split_seed, "candidate": name, "reference": reference,
                    "scope": scope, "candidate_rmse": reproduced[name]["rmse_rating_scale"],
                    "reference_rmse": reproduced[reference]["rmse_rating_scale"], "bootstrap": bootstrap,
                })
        prediction_fingerprints[f"split_{split_seed}"] = {
            "path": reported_path(prediction_path),
            **fingerprint_files(prediction_path.parent, [prediction_path.name])[prediction_path.name],
        }
    script_path = Path(__file__).resolve()
    report = {
        "schema_version": 1, "status": "completed", "analysis_kind": "posthoc_neural_vs_global_control",
        "created_utc": utc_now(), "predeclared": False, "posthoc": True,
        "rationale": "The global-mean control was strongest among non-neural controls after viewing "
                     "the completed run; compare every declared neural candidate without selecting a subset.",
        "no_new_training": True, "no_new_predictions": True, "original_test_used": False,
        "ranking_evaluated": False, "serving_changed": False,
        "source_results": {"path": reported_path(results_path),
                           **fingerprint_files(results_path.parent, [results_path.name])[results_path.name]},
        "prediction_csv_fingerprints": prediction_fingerprints,
        "script_fingerprint": {"path": reported_path(script_path),
                               **fingerprint_files(script_path.parent, [script_path.name])[script_path.name]},
        "comparisons": comparisons,
        "verification": {
            "backup_manifest_sha256": sha256_file(verification_path),
            "backup_verification_path": reported_path(verification_path),
            "verified_final_archive_sha256": final["sha256"],
            "verified_input_files": verified_files, "all_prediction_rows_match_source": True,
            "all_stored_rmse_reproduced": True, "comparison_count": len(comparisons),
            "candidate_count": len(candidates),
        },
        "limitations": [
            "Post-hoc exploratory comparisons were not part of the predeclared assessment protocol.",
            "Intervals are conditional paired user-cluster percentiles and are not adjusted for multiple comparisons.",
            "The two development splits overlap in their source observations and are never pooled.",
            "Assessment data came from previously used training data, not fresh independent test data.",
            "Global mean uses the same fit-derived fallback on cold pairs; saved availability masks are preserved.",
        ],
    }
    atomic_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path,
                        default=ROOT / "artifacts/colab/run-20261014-multiseed/final")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/evidence/neural_colab_multiseed_additional_comparisons.json")
    args = parser.parse_args(argv)
    report = audit(args.artifact_dir, args.output)
    print(json.dumps({"status": report["status"], "comparisons": len(report["comparisons"]),
                      "predeclared": False, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
