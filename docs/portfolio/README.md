# Portfolio and experiment index

- [Case study PDF](book-recommender.pdf): the project as a sequence of decisions, experiments, and results. The document is in Indonesian.
- [Project overview and local demo](../../README.md): what the app serves and how to run it.
- [Experiment guide](../EXPERIMENTS.md): notebook order, evaluation protocols, and reproduction requirements.
- [Configuration ledger](experiment-ledger.md): the full experiment record, with links to source evidence. A [JSON version](experiment-ledger.json) is available for inspection.
- [Experiment history notebook](../../notebooks/03_neural_experiment_history.ipynb): saved tables and plots across the studies.

The PDF and ledger document completed work and its limitations. Neural experiments remain separate from the hybrid model served by the app.

## PDF source links

The PDF is preserved as originally exported. Use this index to open its evidence on GitHub; links inside the PDF refer to the original local source package.

| PDF page | Topic | Repository sources |
|---|---|---|
| 1 | Problem and dataset | [Initial experiment dataset](../evidence/neural_colab_t4_results.json) |
| 2 | Ratings and book editions | [History experiment context](../evidence/neural_colab_history_results.json), [data preparation](../../src/book_recommender/data.py), [work identity](../../src/book_recommender/content_based.py) |
| 3 | Catalog-size decision | [Benchmark summary](../evidence/catalog_size_benchmark_summary.json), [protocol](../CATALOG_SIZE_BENCHMARK.md) |
| 4 | Initial neural and residual models | [MLP results](../evidence/neural_colab_t4_results.json), [residual results](../evidence/neural_colab_residual_results.json) |
| 5 | Optimization checks | [Ablation](../evidence/neural_colab_ablation_results.json), [12-epoch follow-up](../evidence/neural_colab_warmup_results.json) |
| 6 | Stronger comparators | [Multiseed results](../evidence/neural_colab_multiseed_results.json), [bias and MF comparison](../evidence/neural_colab_comparator_results.json) |
| 7 | Ranking objectives | [Ranking results](../evidence/neural_colab_ranking_results.json), [history experiment settings](../evidence/neural_colab_history_results.json) |
| 8 | History-based comparison | [Ranking metrics and selected models](../evidence/neural_colab_history_results.json) |
| 9 | App and serving model | [Artifact manifest](../../models/hybrid.joblib.manifest.json), [shared service](../../src/book_recommender/service.py), [Streamlit](../../app/streamlit_app.py), [API](../../app/api.py) |
| 10 | Outcome and remaining work | [Configuration ledger](experiment-ledger.json), [evaluation scope](../EXPERIMENTS.md#how-to-interpret-the-results) |
