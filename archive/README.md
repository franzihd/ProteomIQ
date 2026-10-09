# Archive

Code and data from the exploratory phase of the project. It is **superseded** and not part
of the reported thesis results. It is kept only for transparency and traceability.

- `notebooks/`: early exploration of the LUAD data from the patient-level phase.
- `src/model/train.py`, `evaluate*.py`, `generate_test_splits.py`, `recall_at_k.py`,
  `run_experiments.py`, `recall_curve.py`, `src/text_generation.py`,
  `src/plotting/plot_scaling.py`, `plot_recall_vs_cohorts.py`: the original **patient-level**
  architecture (abundance concatenated to ESM2, clinical text appended to the UniProt text,
  trained per CPTAC cohort). This was replaced by the protein-level architecture in `src/`.
- `src/evaluation/pathway/`: KEGG pathway AUROC / recall@K evaluations. They were replaced by
  the zero-shot classification and GO-term retrieval evaluations.
- `src/model/recall_at_k_protein_level.py`, `sweep_lr_epochs.py`: an early self-identification
  diagnostic and a hand-rolled grid search, replaced by `cross_modal_eval.py` and the W&B sweeps.
- `src/plotting/plot_{cross_modal_eval,go_retrieval_results,knn_embedding_quality,zero_shot_classification_results}.py`:
  per-checkpoint plots, replaced by `src/plotting/plot_per_eval.py`.
- `src/embeddings/textual/{fuse_protein_text_llm,generate_function_paraphrases}.py` and
  `data/full_proteome/protein_function_paraphrases_sample*.csv`: LLM text-augmentation
  experiments. They never went beyond a sample run because of API budget limits.
- `data/`: pathway results, old figures, old splits (9-cohort random, patient-level, random control),
  and two superseded checkpoints (`final_v2_dims_masked_best.pt` is M0 before the
  data-order fix).

Relative paths inside these scripts point to their original location under `src/`. They are
not expected to run from here without adjustment.
