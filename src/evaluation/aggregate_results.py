"""
Walks every eval output on disk -- the 5 topic-specific evals (cross-modal,
GO-term retrieval, DeepLoc/EC-top-level/GO-Slim-MF zero-shot classification)
plus the 3 task-agnostic checks (k-NN, linear probe, alignment/uniformity)
-- and writes ONE long-format CSV, data/evaluation/all_results.csv, so a
thesis table or plot never has to be built by hand-grepping a dozen
scattered JSON files across experiment subfolders again.

Coverage differs by eval, and this file does NOT paper over that -- it
reflects whatever is actually saved on disk:
- The 5 topic evals + linear probe: validation-pool numbers exist for all
  10 ablation checkpoints (unfreeze N=0-3, the 4 text variants, multi-
  positive, multiview); linear probe itself was only ever run for 3
  (baseline, multi-positive, multiview) -- narrower by original design.
- k-NN: saved for 7 checkpoints (baseline + the 4 text variants + multi-
  positive + multiview) -- NOT the unfreeze checkpoints. This is broader
  than the "3 checkpoints only" scope decided for the task-agnostic checks
  in CLAUDE.md's text -- the actual saved files disagree with that
  description; not reconciled further here, just reported as found.
- Alignment/uniformity: only ever run for 3 checkpoints (baseline, multi-
  positive, multiview), matching linear probe's scope.
Every row's absence for a given (eval, checkpoint) pair means it was never
run, not that it was filtered out.

Scope, deliberately narrowed rather than exhaustive:
- Only the 10 checkpoints that make up the reported ablation study (unfreeze
  N=0-3, the 4 text-concatenation variants, multi-positive, multiview) plus
  final_v2_dims_masked_best (the pre-11y-fix M0, kept for reference, labeled
  distinctly -- see CLAUDE.md's note on two separate M0 checkpoints).
- Only the "ensemble" prompt condition for the 3 classification evals (the
  condition used throughout the ablation comparisons) -- --engineered/
  --label-only files exist on disk but are out of scope here.
- Only min_positives=10 for GO-term retrieval (the value used everywhere).
- DeepLoc: only the 9-way (--drop-chloroplast) variant, the one actually
  used in every reported comparison.

One row per (eval, checkpoint, pool, variant, metric). "pool" unifies what
each eval calls its held-out-protein grouping: cross-modal/GO-retrieval have
two literal separate runs (validation/test); the 3 classification evals
report several subsets from ONE run (all/seen/unseen/unseen_validation/
unseen_test) -- these are renamed here to seen/all/unseen_combined/
validation/test so "pool" means the same thing across every eval: which
proteins (by the model's own train split) underlie this number, and whether
it's the repeat-comparison-safe pool (validation) or the once-only pool
(test).

Also writes two compact views of the same numbers, kept in SEPARATE files so
validation and test never share a table (validation = choosing, test = the
one final report on multiview):
- data/evaluation/results_summary.csv -- validation only, one row per
  ablation checkpoint, headline metric per topic eval.
- data/evaluation/results_final_test.csv -- test only, multiview only, every
  headline metric plus its baselines. The DeepLoc/EC/GO-MF scripts compute a
  test row for every checkpoint as a by-product; those are deliberately NOT
  included here and must not be cited.

Usage:
    uv run python src/evaluation/aggregate_results.py
"""
import json
from pathlib import Path

import pandas as pd

CROSS_MODAL_DIR = Path("data/evaluation/cross_modal_retrieval")
GO_RETRIEVAL_DIR = Path("data/evaluation/go_retrieval/results")
CLASSIFICATION_DIR = Path("data/evaluation/zero_shot_classification")
KNN_DIR = Path("data/evaluation/knn_embedding_quality")
LINEAR_PROBE_DIR = Path("data/evaluation/linear_probe")
ALIGNMENT_UNIFORMITY_DIR = Path("data/evaluation/alignment_uniformity")
OUTPUT_PATH = Path("data/evaluation/all_results.csv")
SUMMARY_PATH = Path("data/evaluation/results_summary.csv")
FINAL_TEST_PATH = Path("data/evaluation/results_final_test.csv")
FINAL_CHECKPOINT = "multiview_best"

# checkpoint filename stem -> (group, human-readable label)
CHECKPOINT_INFO = {
    "biobert_unfreeze_0_m0_best": ("unfreeze", "N=0 (M0)"),
    "biobert_unfreeze_1_m0_best": ("unfreeze", "N=1"),
    "biobert_unfreeze_2_m0_best": ("unfreeze", "N=2"),
    "biobert_unfreeze_3_m0_best": ("unfreeze", "N=3"),
    "location_only_masked_best": ("text", "Function+Location"),
    "m1_family_masked_best": ("text", "M1 (+Family)"),
    "m2_family_location_masked_best": ("text", "M2 (+Family+Location)"),
    "m3_name_family_location_best": ("text", "M3 (+unmasked name)"),
    "multi_positive_best": ("multi_positive", "multi-positive"),
    "multiview_best": ("multiview", "multiview (winner)"),
    "final_v2_dims_masked_best": ("baseline_prefix_fix", "M0 (pre-11y-fix, superseded)"),
}


def ckpt_group_label(stem: str) -> tuple[str, str]:
    return CHECKPOINT_INFO.get(stem, ("other", stem))


def rows_from_cross_modal(path: Path) -> list[dict]:
    d = json.load(open(path))
    stem = path.stem.replace("summary_", "").removesuffix("_test")
    group, label = ckpt_group_label(stem)
    pool = d["split_key"]
    rows = []
    for direction_key, direction_name in [("text_to_protein", "text_to_protein"),
                                           ("protein_to_text", "protein_to_text")]:
        for metric, value in d[direction_key].items():
            rows.append(dict(eval="cross_modal", checkpoint=stem, checkpoint_group=group,
                              checkpoint_label=label, pool=pool, variant=direction_name,
                              metric=metric, value=value))
    return rows


def rows_from_go_retrieval(path: Path) -> list[dict]:
    d = json.load(open(path))
    checkpoint_stem = Path(d["checkpoint"]).stem
    group, label = ckpt_group_label(checkpoint_stem)
    pool = "test" if path.stem.endswith("_test") else "validation"
    branch = d["branch"]
    rows = []
    for query_type, qd in d["query_types"].items():
        variant = f"{branch}_{query_type}"
        rows.append(dict(eval="go_retrieval", checkpoint=checkpoint_stem, checkpoint_group=group,
                          checkpoint_label=label, pool=pool, variant=variant,
                          metric="map", value=qd["mAP"]))
        rows.append(dict(eval="go_retrieval", checkpoint=checkpoint_stem, checkpoint_group=group,
                          checkpoint_label=label, pool=pool, variant=variant,
                          metric="map_random_baseline", value=qd["random_baseline_mAP"]))
        for k, v in qd["mean_P_at_K"].items():
            rows.append(dict(eval="go_retrieval", checkpoint=checkpoint_stem, checkpoint_group=group,
                              checkpoint_label=label, pool=pool, variant=variant,
                              metric=f"p{k}", value=v))
        for k, v in qd["mean_R_at_K"].items():
            rows.append(dict(eval="go_retrieval", checkpoint=checkpoint_stem, checkpoint_group=group,
                              checkpoint_label=label, pool=pool, variant=variant,
                              metric=f"r{k}", value=v))
        for k, v in qd["mean_R_at_K_normalized"].items():
            rows.append(dict(eval="go_retrieval", checkpoint=checkpoint_stem, checkpoint_group=group,
                              checkpoint_label=label, pool=pool, variant=variant,
                              metric=f"r{k}_normalized", value=v))
    return rows


CLASSIFICATION_POOL_MAP = {
    "all": "all",
    "seen": "seen",
    "unseen": "unseen_combined",
    "unseen_validation": "validation",
    "unseen_test": "test",
}


def rows_from_classification(path: Path, eval_name: str) -> list[dict]:
    d = json.load(open(path))
    checkpoint_stem = Path(d["checkpoint"]).stem
    group, label = ckpt_group_label(checkpoint_stem)
    rows = []
    for subset_key, pool in CLASSIFICATION_POOL_MAP.items():
        subset = d.get(subset_key)
        if not subset:
            continue
        for metric in ("accuracy", "macro_f1", "majority_baseline_accuracy", "random_baseline_accuracy", "n"):
            if metric in subset:
                rows.append(dict(eval=eval_name, checkpoint=checkpoint_stem, checkpoint_group=group,
                                  checkpoint_label=label, pool=pool, variant="ensemble",
                                  metric=metric, value=subset[metric]))
    return rows


def rows_from_knn(path: Path) -> list[dict]:
    """knn_embedding_quality_eval.py output: task-agnostic (no query text) --
    always validation pool per its own design (CLAUDE.md 11ag/11ah), and only
    ever run for a subset of checkpoints (see aggregate_results.py's own
    docstring)."""
    d = json.load(open(path))
    stem = Path(d["checkpoint"]).stem
    group, label = ckpt_group_label(stem)
    rows = []
    for eval_set, r in d["results"].items():
        for k, v in r["agreement_at_k"].items():
            rows.append(dict(eval="knn", checkpoint=stem, checkpoint_group=group,
                              checkpoint_label=label, pool="validation", variant=eval_set,
                              metric=f"agreement_at_{k}", value=v))
        for k, v in r["raw_esm2_agreement_at_k"].items():
            rows.append(dict(eval="knn", checkpoint=stem, checkpoint_group=group,
                              checkpoint_label=label, pool="validation", variant=eval_set,
                              metric=f"raw_esm2_agreement_at_{k}", value=v))
        rows.append(dict(eval="knn", checkpoint=stem, checkpoint_group=group,
                          checkpoint_label=label, pool="validation", variant=eval_set,
                          metric="random_baseline", value=r["random_baseline"]))
    return rows


def rows_from_linear_probe(path: Path) -> list[dict]:
    """linear_probe_eval.py output: trained on the 'seen' (training) split,
    scored on unseen_validation -- test reserved, same standing rule as
    everything else (CLAUDE.md 11ah)."""
    d = json.load(open(path))
    stem = Path(d["checkpoint"]).stem
    group, label = ckpt_group_label(stem)
    rows = []
    for eval_set, r in d["results"].items():
        for metric in ("probe_accuracy", "raw_esm2_probe_accuracy", "random_baseline", "majority_baseline"):
            rows.append(dict(eval="linear_probe", checkpoint=stem, checkpoint_group=group,
                              checkpoint_label=label, pool="validation", variant=eval_set,
                              metric=metric, value=r[metric]))
    return rows


def rows_from_alignment_uniformity(path: Path) -> list[dict]:
    """alignment_uniformity_eval.py output: no downstream task at all, no
    'variant' axis -- always validation pool (CLAUDE.md 11ag)."""
    d = json.load(open(path))
    stem = Path(d["checkpoint"]).stem
    group, label = ckpt_group_label(stem)
    rows = []
    for metric in ("alignment", "alignment_mismatched_reference", "uniformity_protein",
                   "uniformity_text", "uniformity_pooled", "uniformity_random_reference"):
        rows.append(dict(eval="alignment_uniformity", checkpoint=stem, checkpoint_group=group,
                          checkpoint_label=label, pool="validation", variant="n/a",
                          metric=metric, value=d[metric]))
    return rows


def main():
    all_rows = []

    for path in CROSS_MODAL_DIR.rglob("summary_*.json"):
        all_rows.extend(rows_from_cross_modal(path))

    for path in GO_RETRIEVAL_DIR.rglob("summary_*_min10.json"):
        all_rows.extend(rows_from_go_retrieval(path))
    for path in GO_RETRIEVAL_DIR.rglob("summary_*_min10_test.json"):
        all_rows.extend(rows_from_go_retrieval(path))

    # ensemble condition only: filenames with no _engineered/_label_only suffix.
    # deeploc: ensemble 9-way-human file is "summary_<stem>_9way_human.json" exactly.
    for path in (CLASSIFICATION_DIR / "deeploc").rglob("summary_*_9way_human.json"):
        if path.stem.endswith(("_engineered", "_label_only")):
            continue
        all_rows.extend(rows_from_classification(path, "deeploc"))

    for task in ["ec_top_level", "go_mf"]:
        for path in (CLASSIFICATION_DIR / task).rglob("summary_*.json"):
            if path.stem.endswith(("_engineered", "_label_only")):
                continue
            all_rows.extend(rows_from_classification(path, task))

    for path in KNN_DIR.rglob("summary_*.json"):
        all_rows.extend(rows_from_knn(path))

    for path in LINEAR_PROBE_DIR.rglob("summary_*.json"):
        all_rows.extend(rows_from_linear_probe(path))

    for path in ALIGNMENT_UNIFORMITY_DIR.rglob("summary_*.json"):
        all_rows.extend(rows_from_alignment_uniformity(path))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["eval", "checkpoint_group", "checkpoint", "pool", "variant", "metric"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH}")
    print(f"\nEvals: {sorted(df['eval'].unique())}")
    print(f"Checkpoints: {sorted(df['checkpoint'].unique())}")
    print(f"Pools: {sorted(df['pool'].unique())}")

    write_validation_summary(df)
    write_final_test_summary(df)


def pick(df: pd.DataFrame, **conditions) -> pd.DataFrame:
    for col, val in conditions.items():
        df = df[df[col] == val]
    return df


def write_validation_summary(df: pd.DataFrame):
    """One row per ablation checkpoint, headline metric per topic eval, validation pool only."""
    val = df[df.pool == "validation"]
    rows = []
    for stem, (group, label) in CHECKPOINT_INFO.items():
        if group == "baseline_prefix_fix":
            continue  # superseded pre-fix M0, never cited

        def v(**c):
            hit = pick(val, checkpoint=stem, **c)
            return float(hit.value.iloc[0]) if len(hit) else float("nan")

        go_ret = pick(val, eval="go_retrieval", checkpoint=stem, metric="map")
        rows.append({
            "checkpoint": label,
            "group": group,
            "cross_modal_T2P_R1%(val)": round(v(eval="cross_modal", variant="text_to_protein", metric="r1"), 4),
            "cross_modal_T2P_R50%(val)": round(v(eval="cross_modal", variant="text_to_protein", metric="r50"), 4),
            "deeploc_acc%(val)": round(100 * v(eval="deeploc", metric="accuracy"), 2),
            "ec_acc%(val)": round(100 * v(eval="ec_top_level", metric="accuracy"), 2),
            "go_mf_acc%(val)": round(100 * v(eval="go_mf", metric="accuracy"), 2),
            "go_retrieval_mean_mAP(val)": round(go_ret.value.mean(), 4) if len(go_ret) else float("nan"),
        })
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    print(f"Wrote {SUMMARY_PATH} (validation only, {len(rows)} checkpoints)")


def write_final_test_summary(df: pd.DataFrame):
    """Multiview on the reserved test pool only -- every headline number plus its baselines."""
    test = pick(df, pool="test", checkpoint=FINAL_CHECKPOINT)
    rows = []

    def add(eval_name, variant, metric, value):
        rows.append(dict(eval=eval_name, variant=variant, metric=metric, value=value))

    for direction in ("text_to_protein", "protein_to_text"):
        for metric in ("r1", "r5", "r10", "r50", "median_rank"):
            hit = pick(test, eval="cross_modal", variant=direction, metric=metric)
            if len(hit):
                add("cross_modal", direction, metric, float(hit.value.iloc[0]))

    for eval_name in ("deeploc", "ec_top_level", "go_mf"):
        for metric in ("accuracy", "macro_f1", "random_baseline_accuracy", "majority_baseline_accuracy", "n"):
            hit = pick(test, eval=eval_name, metric=metric)
            if len(hit):
                value = float(hit.value.iloc[0])
                add(eval_name, "ensemble", metric, value if metric == "n" else round(100 * value, 2)
                    if metric != "macro_f1" else round(value, 4))

    go = test[test["eval"] == "go_retrieval"]
    for variant in sorted(go.variant.unique()):
        for metric in ("map", "map_random_baseline", "p10", "r100"):
            hit = pick(go, variant=variant, metric=metric)
            if len(hit):
                add("go_retrieval", variant, metric, round(float(hit.value.iloc[0]), 4))
    for query_type in ("name", "definition"):
        maps = go[(go.metric == "map") & go.variant.str.endswith(f"_{query_type}")]
        if len(maps):
            add("go_retrieval", f"mean_{query_type}", "map", round(maps.value.mean(), 4))

    pd.DataFrame(rows).to_csv(FINAL_TEST_PATH, index=False)
    print(f"Wrote {FINAL_TEST_PATH} (test only, {FINAL_CHECKPOINT} only, {len(rows)} rows)")


if __name__ == "__main__":
    main()
