# Experiments and reproduction

The app serves the hybrid bundle in `models/`. Neural models, rating baselines, and history-based cosine are experiments. Their saved results support the comparisons in the README; they have not been promoted to the app.

There are three useful paths through the project:

| Goal | Path | What is needed |
|---|---|---|
| Inspect or regenerate the reported tables | Notebooks 03–05 | Committed JSON evidence and development dependencies |
| Train a model on a fresh checkout | Notebook 02 or `book-rec train-neural` | Raw Book-Crossing CSVs and the `neural` extra |
| Repeat a specific recorded experiment | Its worker and original frozen inputs | Verified input archives and, for later phases, outputs from the preceding run |

Raw datasets, per-user prediction CSVs, and model checkpoints from the Colab runs are not included. A clean clone can regenerate the evidence notebooks, but cannot independently retrain every historical run from the compact JSON alone. Creating new splits is a new experiment, not an exact reproduction of a recorded run.

## Read the notebooks

Start with [03 — Experiment history](../notebooks/03_neural_experiment_history.ipynb) for the decisions and results. Continue with [04 — Ranking objectives](../notebooks/04_neural_ranking_experiments.ipynb) and [05 — Reading-history models](../notebooks/05_neural_history_experiments.ipynb) for the later studies. These notebooks read files from [`docs/evidence/`](evidence/); executing them does not train a model or open a new test set.

[01 — End-to-end baseline](../notebooks/01_end_to_end_book_recommender.ipynb) explains the app pipeline and catalog-size benchmark. [02 — Neural training](../notebooks/02_neural_book_recommender.ipynb) demonstrates embeddings, forward propagation, loss, gradients, and Adam. Its saved smoke-run results use a different subset from the main experiments.

From the repository root:

```bash
uv sync --python 3.12 --frozen --dev
uv run --frozen jupyter nbconvert --to notebook --execute notebooks/03_neural_experiment_history.ipynb --output-dir artifacts/experiment-history
uv run --frozen jupyter nbconvert --to notebook --execute notebooks/04_neural_ranking_experiments.ipynb --output-dir artifacts/experiment-history
uv run --frozen jupyter nbconvert --to notebook --execute notebooks/05_neural_history_experiments.ipynb --output-dir artifacts/experiment-history
```

The [configuration ledger](portfolio/experiment-ledger.md) and its [JSON version](portfolio/experiment-ledger.json) provide source pointers for all 121 recorded configurations: 54 deep-neural fits, 18 matrix-factorization fits, and 49 controls or classical configurations. Repeated configurations are counted as separate records; this is not a count of unique architectures.

## Decisions and evidence

| Study | Question and outcome | Saved evidence | Worker |
|---|---|---|---|
| Initial MLP | Can the embedding network learn? Training worked, but it did not beat the rating baseline. | [Results](evidence/neural_colab_t4_results.json), [first-step trace](evidence/neural_colab_t4_first_step_trace.json), [manifest](evidence/neural_colab_t4_run_manifest.json) | [Initial experiment](../scripts/colab_neural_experiment.py) |
| Residual model | Does a neural correction improve a user/item bias predictor? The controls motivated a closer look at the learned correction. | [Results](evidence/neural_colab_residual_results.json), [manifest](evidence/neural_colab_residual_run_manifest.json), [signal audit](evidence/neural_residual_signal_audit.json) | [Residual experiment](../scripts/colab_residual_experiment.py) |
| Regularization ablation | Is regularization or initialization suppressing useful corrections? Validation-only checks did not establish a ranking improvement. | [Results](evidence/neural_colab_ablation_results.json), [manifest](evidence/neural_colab_ablation_run_manifest.json) | [Ablation worker](../scripts/colab_neural_ablation.py) |
| Fixed 12-epoch follow-up | Does more training resolve the weak correction? This adaptive follow-up added no fresh test or ranking evidence. | [Results](evidence/neural_colab_warmup_results.json), [manifest](evidence/neural_colab_warmup_run_manifest.json) | [Ablation worker, `warmup` preset](../scripts/colab_neural_ablation.py) |
| Two splits, three seeds | Is the rating result stable across splits and initialization? Compare MLP and residual models against controls. | [Results](evidence/neural_colab_multiseed_results.json), [manifest](evidence/neural_colab_multiseed_run_manifest.json), [additional comparisons](evidence/neural_colab_multiseed_additional_comparisons.json) | [Multiseed worker](../scripts/colab_neural_multiseed.py) |
| Stronger rating baselines | Does the neural result survive comparison with tuned joint bias and matrix factorization? Evaluate rating and ranking separately. | [Results](evidence/neural_colab_comparator_results.json), [manifest](evidence/neural_colab_comparator_run_manifest.json) | [Comparator worker](../scripts/colab_comparator_experiment.py) |
| Ranking objectives and metadata | Does training directly for ranking help? Compare BPR with MSE controls and metadata variants. | [Results](evidence/neural_colab_ranking_results.json), [manifest](evidence/neural_colab_ranking_run_manifest.json) | [Ranking worker](../scripts/colab_ranking_experiment.py) |
| Positive reading history | Does using the user's liked works improve personalization? Cosine beats the selected neural models; softmax gains are inconsistent across splits. | [Results](evidence/neural_colab_history_results.json), [manifest](evidence/neural_colab_history_run_manifest.json), [personalization audit](evidence/neural_personalization_diagnosis.json) | [History worker](../scripts/colab_history_experiment.py) |

## How to interpret the results

Rating RMSE and MAE use the original 1–10 rating scale and known user/item pairs; lower is better. Ranking asks whether a held-out relevant work appears near the top of the recommendation list; higher Recall@10 and NDCG@10 are better. A lower rating RMSE does not establish a better recommendation list.

The initial MLP uses its recorded train/validation/test split; the residual evaluation reuses that split. Ablation and warmup use training and validation only. The multiseed study makes internal fit/selection/assessment splits from the earlier training data. Later comparator, ranking, and history studies reuse those assessment splits for development. Selection still determines checkpoints and primary models, but repeated assessment is no longer an independent test.

The history study uses split seeds 17 and 29 and training seeds 42, 2026, and 7. Its all-relevant cohorts have 1,776 and 1,806 users, respectively, with one target work per user. Users overlap between splits; do not add the two denominators as independent samples. Every evaluated user has a fit history, so this is not a cold-start evaluation.

The neural history encoder removes the target work from its input before training. It aggregates embeddings of liked works, then uses Dense layers with 64 and 32 units. BPR and softmax are compared using the same architecture, target pairs, and initial weights. Selection NDCG@10 chooses checkpoints and primary models, with Recall@10 and earlier checkpoints as tie-breakers. The cosine comparison uses positive user/work co-occurrences with `k ∈ {50, 200}` and shrinkage in `{0, 10}`.

This experimental cosine is different from the serving app's adjusted cosine over explicit ratings. The serving hybrid was not included in the history study. Its results should not be inserted into that table from a different protocol.

Work identity is based on normalized title and author, with selected edition qualifiers removed. Subtitle variants can escape this grouping. Book-Crossing has no interaction timestamps, so these splits are not temporal backtests.

## Prepare data for a new run

Obtain the [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Put `Books.csv`, `Ratings.csv`, and `Users.csv` under `data/raw/`, or use authenticated Kaggle CLI access:

```bash
uv run --frozen kaggle datasets download -d arashnic/book-recommendation-dataset -p data/raw --unzip
uv run --frozen book-rec check-data --data-dir data/raw
```

Keep credentials outside the repository. Cleaning rejects invalid identifiers and out-of-range ratings, excludes implicit rating 0, averages duplicate user/ISBN ratings, and removes ratings without book metadata. Iterative k-core filtering removes sparse users and books until the minimum interaction rules hold. See [`data.py`](../src/book_recommender/data.py).

Install the optional neural dependencies and train a small new run:

```bash
uv sync --python 3.12 --frozen --dev --extra neural
uv run --frozen --extra neural book-rec train-neural --data-dir data/raw --max-books 5000 --sample-users 100 --epochs 5 --output models/neural
uv run --frozen --extra neural book-rec recommend-neural --model-dir models/neural --user-id 5709 --top-n 10
```

Choose a user ID present in the prepared data. TensorFlow can run on CPU; a GPU is optional for this CLI path. The historical bounded Colab workers also accept `--require-gpu`, which explicitly fails when a GPU is unavailable.

To train the bias-plus-residual architecture, add `--architecture residual --user-shrinkage 0 --item-shrinkage 20 --residual-scale 0.25` and choose a separate output directory. `recommend-neural` reads the architecture from the artifact metadata. These neural artifacts do not change what Streamlit or FastAPI serves.

Notebook 02 supports `BOOK_RECOMMENDER_DATA_DIR`, `BOOK_RECOMMENDER_NOTEBOOK_MAX_BOOKS`, `BOOK_RECOMMENDER_NOTEBOOK_EVAL_USERS`, and `BOOK_RECOMMENDER_NEURAL_EPOCHS`. Set `BOOK_RECOMMENDER_PROJECT_ROOT` when executing from outside the checkout. [`tests/fixtures/notebook_data/`](../tests/fixtures/notebook_data/) is synthetic and must not be used as Book-Crossing performance evidence.

## Repeat a recorded experiment

The workers verify frozen input hashes. Use the original verified archives and a new output directory for each repeat. Do not change expected hashes to make different data pass. Run settings, source fingerprints, dependency versions, device details, and archive metadata are in the linked run manifests.

| Worker | Required input lineage | Recorded limits |
|---|---|---|
| `colab_neural_experiment.py` | Original three raw CSVs | 8-epoch limit, patience 3; 5 epochs completed; configuration in the initial results |
| `colab_residual_experiment.py` | Initial run's frozen books, train, validation, and test CSVs | 10 epochs; 600 s training budget |
| `colab_neural_ablation.py --preset ablation` | Frozen books, train, and validation only; test CSV prohibited | 10 epochs |
| `colab_neural_ablation.py --preset warmup` | Same validation-only input | 12 epochs; 600 s training budget |
| `colab_neural_multiseed.py` | Frozen books and train CSV only | 12 epochs; 900 s training budget |
| `colab_comparator_experiment.py` | Multiseed splits, results, six selected MLP models, and input manifest | 30 epochs; 900 s training, 300 s ranking |
| `colab_ranking_experiment.py` | Comparator splits, results, ranking CSVs, and input manifest | 12 epochs; 1,200 s training, 300 s ranking |
| `colab_history_experiment.py` | Ranking study's splits, results, ranking CSVs, and input manifest | 20 epochs; 1,200 s training, 300 s ranking |

For the comparator input, preserve `train_books.csv`, `previous_results.json`, the six `splits/split_{17,29}/{fit,selection,assessment}_ratings.csv` files, and the six selected MLP model directories. The input manifest records the size and SHA-256 of every copied file.

Ranking and history inputs replace the MLP checkpoints with four prior ranking CSVs: `previous_ranking/split_{17,29}/{ranking_user_outcomes,ranking_top10}.csv`. Their manifest includes `files`, `previous_results_sha256`, and `previous_ranking_fingerprints`. The history worker requires prior results of type `ranking_objective_metadata_reused_development`. See the workers' `verify_inputs` functions for the exact checks.

For example, after preparing and verifying the history input directory:

```bash
uv run --frozen --extra neural python scripts/colab_history_experiment.py --input-dir artifacts/history-repeat-inputs --output artifacts/history-repeat --epochs 20 --batch-size 1024 --max-training-seconds 1200 --max-ranking-seconds 300 --require-gpu
```

The workers save per-epoch model, optimizer, and RNG state. Best inference weights and the last training checkpoint are distinct outputs. Time-budget stops and incomplete candidates remain explicit in the results. Never treat a partial run as a completed comparison.

Colab authentication and file transfer are separate from these workers. [`colab_gpu_probe.py`](../scripts/colab_gpu_probe.py) checks the device; [`colab_command.py`](../scripts/colab_command.py) and [`monitor_colab_experiment.py`](../scripts/monitor_colab_experiment.py) support bounded commands and verified downloads. Use a new session name beginning with `book-recommendation-neural-`, verify downloaded archives and their extracted files, and stop that runtime after retrieval. These helpers require the external Colab CLI used by the original workflow; they are not needed to read the committed evidence.

The TensorFlow-free audit scripts reproduce metrics and diagnostics from downloaded prediction files: [multiseed](../scripts/audit_multiseed_assessment.py), [ranking](../scripts/audit_ranking_assessment.py), and [history](../scripts/audit_history_assessment.py). They require the verified local backups; the compact evidence JSON is not a substitute for those predictions.

## Catalog benchmark and serving

The [catalog-size benchmark](CATALOG_SIZE_BENCHMARK.md) has a separate protocol: five validation repeats of 200 users and a fixed test cohort of 200. It compares interaction/content catalog sizes while holding algorithm settings constant. No challenger passed the quality and resource gates, so only the retained baseline was evaluated on the fixed test. This does not mean challengers lost a test comparison that was never run.

That document contains the full `benchmark-catalog` command and summary-generation steps. The [compact summary](evidence/catalog_size_benchmark_summary.json) is committed; large per-user and timing outputs are excluded. Latency values are measurements from that run, not a current hosted-service guarantee.

The serving artifact remains `hybrid-5b00e40e8959`. Base weights are 0.40 / 0.45 / 0.15, with dynamic adjustments enabled. Its manifest and the shared service validate compatibility before loading. New training results should be evaluated under a fixed protocol before replacing that artifact.

## Maintain the evidence

Keep configuration, seeds, split identities, epoch history, selection rules, failures, and source hashes with every run. Add results and a run manifest to `docs/evidence/`, update the relevant notebook, execute it, and refresh its dependency stamp using [`notebook_quality.py`](../scripts/notebook_quality.py). Keep each `.ipynb` paired with its `.py` source and update the notebook dependencies in [CI](../.github/workflows/ci.yml).

Optional TensorFlow tests are enabled with `BOOK_REC_RUN_TF_TESTS=1`; the CI neural job lists the maintained test command. Standard API/data tests and notebook evidence checks do not require a full training run. Changes to conclusions must follow the evidence, including unsuccessful candidates and the distinction between development assessment and independent test data.
