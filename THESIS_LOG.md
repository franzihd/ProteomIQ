# Thesis log

Plain-language, date-ordered notes for writing — not a technical reference
(that's `CLAUDE.md`, which has exact numbers, tables, and code paths for
everything below). Each entry follows the same shape:

- **What we did**
- **Why** (the motivation/question going in)
- **Result** (what actually happened)
- **Takeaway** — written close to how it could go straight into a thesis
  sentence, so writing is closer to transcribing than composing from
  scratch. Go to `CLAUDE.md`'s bracketed section number for the exact
  numbers/tables when you need them.

**Tags:**
- **[IN SCOPE]** — section 11 onward. This is what the thesis narrates.
- **[BACKGROUND]** — sections 1–10. Motivated in-scope decisions but isn't
  itself narrated (see the scope decision at the bottom).

---

## BACKGROUND ERA (sections 1–10) — compressed

These entries are here so you remember *why* certain in-scope choices were
made, not to write up in the thesis itself.

### 2026-06 — Patient-level baseline built
**What:** CPTAC data for 9 cohorts; protein abundance + ESM2 → protein
input, UniProt annotation + clinical description → text input; InfoNCE
training. LUAD-only proof of concept: loss 4.79→0.86.
**Takeaway:** the basic CLIP-style contrastive setup works mechanically —
this justified continuing to invest in the approach, but everything about
*what to put in the text/protein inputs* changed later. `[§1–§7]`

### 2026-06-28 — First pathway AUROC
**What:** built pathway-based evaluation (KEGG pathway names as queries,
AUROC over ranking).
**Result:** mean AUROC ~0.72–0.75, but the best-scoring pathways were ones
whose members share obvious naming vocabulary (e.g. glycosyltransferase).
**Takeaway:** first hint that some of the model's apparent skill is
lexical pattern-matching rather than biological understanding — a concern
that recurs and gets progressively better evidence throughout the project.
`[§8c]`

### 2026-07-06 — Pathway selection had to become objective
**What:** professor rejected a hand-picked ~20-pathway shortlist as
selection bias. Replaced with pre-specified coverage/power filters applied
to all 320 KEGG pathways before any scores were seen → 245 pathways passed.
**Takeaway:** this 245-pathway set is the one piece of background-era
machinery that *does* carry into the thesis — describe it plainly in
Methods as "the evaluation set," no need to narrate the rejected hand-pick
attempt. `[§8d]`

### 2026-07-07 — Decision: drop clinical description from training text
**What/why:** training text was UniProt annotation + clinical description,
but real queries are never clinical-description-shaped — a train/inference
mismatch. Decided to retrain on UniProt annotation only.
**Takeaway:** fixing a mismatch between what a model is trained on and how
it's actually queried is a legitimate, reportable methodological fix in
its own right, independent of whether it improves the headline metric.
`[§8e, "Next steps as of 2026-07-07"]`

### 2026-07-11 — Bigger pivot: drop abundance, go protein-level
**What/why:** dropping clinical description alone didn't close the
generalization gap. Decided to drop abundance from training entirely and
handle patient-specificity at inference time instead of trying to train it
into the embedding.
**Result:** doing this exposed a real bug-in-waiting — without
patient-varying text, the same protein across many patients becomes a
literally duplicate training row, which would inject contradictory
gradients into InfoNCE if not deduplicated.
**Takeaway:** a change made for one reason (patient-specificity) forced a
second, unrelated fix (global deduplication + protein-level, not
patient-level, splitting) — worth narrating as "one decision cascaded into
a redesign of the evaluation split," since that redesign is what directly
enables the homology-aware split later. `[§10a–§10d]`

### 2026-07-12 — Caught an invalid evaluation design before running it
**What:** worked through what a valid test for the planned
`activity_score` (cosine similarity × abundance rank) would look like.
**Result:** the obvious approach (AUROC against static pathway membership)
would unfairly penalize the model for correctly demoting a real pathway
member that just has low abundance in one patient — exactly what
`activity_score` is supposed to do.
**Takeaway:** redesigned into a two-stage "retrieve, then annotate"
pipeline instead of one blended score. Good example for the thesis of
catching a flawed evaluation *before* spending compute on it, not after.
`[§10j, §10k]`

### 2026-07-13 — Three ablations in one day
- **Gene-name masking** (strip "Gene: " prefix from training text):
  **every generalization metric improved** even though training loss got
  *worse*. **Takeaway:** the old checkpoint's lower loss was partly
  shortcut-driven (matching a protein to its own name), not deeper
  learning — removing the shortcut hurt the easy metric and helped the
  real one. `[§10m]`
- **GO-term augmentation** (add GO Biological Process/Cellular Component
  terms to training text): **negative result**, tried two variants
  (BP+CC, BP-only), both hurt pathway AUROC/recall@K despite helping
  training loss and self-ID recall. **Takeaway:** more text ≠ better
  signal — added tokens can function as a *new* self-identification
  shortcut rather than genuine biological content, mirroring the masking
  finding from the opposite direction. `[§10n]`
- **LLM-fusion augmentation** (Claude-rewritten, coherent per-protein
  paragraphs instead of raw GO concatenation): built, spot-checked, quality
  looked good — **paused on budget**, not abandoned (~$25–55 needed, $4
  available). **Takeaway:** an open thread, not a dead end — flag as future
  work with a clear resume path, not as "we didn't get to it." `[§10o]`
- **Cross-modal closed-set retrieval eval** built (standard CLIP protocol).
  **Result:** clearly shows the contrastive objective works, but
  *disagreed* with pathway AUROC on which checkpoint is best.
  **Takeaway:** different evaluations measure genuinely different things —
  this becomes a recurring, load-bearing theme (repeated again in-scope at
  §11d/§11h). `[§10p]`

---

## IN SCOPE (section 11) — this is the thesis

### 2026-07-15 — Why we rebuilt the protein universe and the split
**What/why:** two problems with everything before this point: (1) the
protein universe was arbitrarily limited to whatever CPTAC's mass-spec
pipeline happened to detect — a technical artifact, not a principled
boundary; (2) the train/val/test split had no homology control, so a
near-identical paralog could leak across splits and inflate apparent
generalization.
**Result:** fixed both *together*, deliberately, as one clean redesign
rather than staging a two-tier system — an earlier draft that would have
kept both problems separately was explicitly rejected mid-planning because
it would have made it impossible to attribute a result to either fix.
**Takeaway:** this is the thesis's opening methodological argument —
motivate Chapter 3 (Methods) with these two named, concrete problems, not
a vague "we improved the model." `[§11a]`

### 2026-07-15 — Fetching the full human proteome
**What:** queried UniProt's search API directly for all reviewed human
entries (`organism_id:9606 AND reviewed:true`) — a different, faster
approach than the earlier per-cohort ID-mapping method (no job/poll step
needed).
**Result:** 20,431 entries fetched → 16,893 usable after dropping
unannotated/no-sequence entries. Found and fixed a real bug: a UniProt
entry can map to several Ensembl protein IDs (one per isoform); the first
version picked one arbitrarily and could silently mismatch which ID a
CPTAC cohort's file actually used — fixed by picking the ID with the
widest cohort coverage. Also stripped inline citation clutter
("(PubMed:23056470)") from annotation text, which was disproportionately
bloating the best-studied proteins' text past BioBERT's token limit.
**Takeaway:** two small, easy-to-miss correctness bugs (isoform ID choice,
citation noise) were caught and fixed *before* they could quietly bias
results — both are worth a sentence each in Methods as evidence of
careful data engineering, not just "we downloaded UniProt." `[§11b]`

### 2026-07-15 — ESM2 embeddings for the full proteome
**What:** ran ESM2 (`esm2_t30_150M_UR50D`, frozen) over all 16,893
proteins' sequences → `data/full_proteome/esm2_embeddings.h5`, one 640-dim
vector per protein. Same script/model as the cohort-level version, just
pointed at the new unified protein list instead of 9 separate per-cohort
files.
**Takeaway:** this is the "why frozen, pre-computed embeddings make this
feasible" point for Methods — restate the original design rationale (ESM2
is too large to fine-tune, and doesn't need to change per patient/cohort,
so computing it once for the canonical protein list is strictly better
than the old per-cohort redundancy). `[§4a for the original rationale;
file itself confirmed to exist at `data/full_proteome/esm2_embeddings.h5`]`

### 2026-07-15/16 — BioBERT embeddings for the full proteome (masked)
**What:** ran BioBERT over the (gene-name-masked) UniProt annotation for
each of the 16,893 proteins → `data/full_proteome/biobert_embeddings_masked.h5`,
one 768-dim vector per protein.
**Takeaway:** masking (§10m/§10n's finding, re-confirmed at this new scale
in §11g below) was folded in as the default from the start here, not
re-litigated — worth one sentence noting the decision was carried forward
rather than re-tested from scratch, with the re-verification (§11g) as the
actual proof it still holds. `[referenced throughout §11]`

### 2026-07-15/16 — CD-HIT clustering + the homology-aware split
**What:** clustered all 16,890 sequenced+annotated proteins with CD-HIT at
90% identity.
**Result:** 16,400 clusters, only 320 with more than one member (mostly
isoforms/near-paralogs) — the large majority of the human proteome (95%)
has no near-duplicate at all at this threshold. Split at the *cluster*
level (not individual protein) into train/val/test →
`data/protein_splits_full_proteome.json` (13,515 / 1,689 / 1,689).
**Takeaway:** a useful, citable number for Methods — "only 4.8% of the
proteome is redundant at 90% identity, so CD-HIT does real, targeted work
for a small minority rather than reorganizing everything into a few big
family buckets." Also cite the ProtET paper comparison here: even a
published protein-CLIP model has this same homology-leakage gap, which is
a caution (it's an overlooked issue in the field), not a reason to skip
fixing it. `[§11c]`

### ~2026-07-15/16 — First sanity check: does the new split actually change anything?
**What:** ran an unmasked baseline training pass, then cross-modal
evaluation restricted to the new homology-aware test pool.
**Result:** R@1 nearly halved vs. the old (non-homology-split) baseline,
while R@50 barely moved.
**Takeaway:** this specific pattern (top-1 crashes, top-50 barely changes)
is *exactly* what you'd expect if the old split's inflated top-1 score was
coming from near-duplicate paralogs sitting in both train and "held-out"
test — a same-signature confirmation repeated more starkly later
(§11h/§11i). Good evidence to lead with in Results: "before we even tuned
anything, the split alone produced this signature." `[§11d]`

### ~2026-07-16 — Hyperparameter sweep #1: lr/batch_size/epochs, and a real confound caught
**What:** ran a 36-config grid (lr × batch_size × epochs).
**Result:** ranking by raw val_loss was misleading — InfoNCE's
random-guessing floor is `ln(batch_size)`, so smaller batches look
artificially better on raw loss regardless of true quality. Corrected with
`gap_below_random = ln(batch_size) − val_loss`, which **completely
reversed** the ranking (everything at the top became batch_size=512,
previously at the bottom).
**Takeaway:** a clean, quantified example of "the naive metric ranks
models in the opposite order from the metric that actually matters" — use
this as a standalone methodological point in Methods/Discussion, it's a
strong, self-contained argument for careful evaluation design. `[§11e]`

### ~2026-07-16 — Hyperparameter sweep #2: hidden_dim/output_dim, and a rank-capping constraint
**What:** searched projection-head width (hidden_dim) and shared embedding
size (output_dim).
**Result:** found a real mathematical constraint first (a linear layer
can't output more independent directions than its input width, so
output_dim > hidden_dim is structurally invalid, not just untested) — this
ruled out some combinations before wasting compute on them. Winning
config: lr=1e-4, batch_size=512, hidden_dim=1024, output_dim=1024 (raising
hidden_dim *alone*, without also raising output_dim, made things *worse*
— confirms it's the shared embedding space size that matters, not raw MLP
capacity). Also tested 2048/2048 to confirm 1024 wasn't just "we stopped
too early" — it scored worse, confirming 1024 is a genuine peak.
**Takeaway:** derive-a-constraint-before-searching is a good methodology
story — "we didn't just grid-search blindly, we used a known property of
linear layers to prune the search space first." `[§11f]`

### ~2026-07-16 — Masking re-verified at the final architecture
**What:** trained one more checkpoint identical to the final architecture
in every way except masked vs. unmasked text — isolating masking as the
only variable, at the new scale/split/hyperparameters.
**Result:** masking won clearly on pathway AUROC (+0.019 overall, +0.018
test-only) — and this gain is **almost numerically identical** to the
original §10m finding on a completely different setup (protein universe,
split, architecture).
**Takeaway:** this is strong, thesis-worthy evidence that masking is a
real, general property of the training text, not a fluke of one
particular configuration — lead with "we re-confirmed this independently
at the final architecture, and got the same effect size" in Discussion.
Cross-modal retrieval, notably, barely reacted to masking at all in this
comparison — another instance of "different evals measure different
things," worth pairing with the §10p/§11d version of the same point.
`[§11g]`

### ~2026-07-16 — Pathway recall@K and self-ID recall@K on the final checkpoint
**Result:** pathway recall@K improved genuinely over every prior
checkpoint (beat the old 9-cohort baseline at every K) — real, ~5-9x
random chance for an open-vocabulary task. Self-ID recall@K, in contrast,
**fell sharply** (R@1: 18.5%→2.0%), shrinking gradually from R@1 to R@100.
**Takeaway:** this is the cleanest, most complete version of the
"leakage vs. genuine improvement" story in the whole project — as the
model improved on every shortcut-*resistant* metric, a shortcut-*flavored*
metric fell, because its old high value was itself an artifact of a less
rigorous split. Also worth stating explicitly: cross-modal's closed,
all-unseen pool is a stress test, not representative of real deployment
(~80% of a real upload would be already-trained-on proteins) — the mixed
pools (self-ID recall@K, pathway AUROC/recall@K) are what actually
approximate real use. `[§11h]`

### ~2026-07-16 — Split-control diagnostic: is the homology split's benefit real or a hyperparameter artifact?
**What/why:** the §11h comparison changed three things at once (protein
universe, split method, hyperparameters), so its results couldn't be
cleanly attributed to the split alone. Built a `--random-control` variant
(same protein universe, but split by individual protein instead of
cluster) to isolate the variable.
**Result:** both changes help independently and additively (~+0.018 from
hyperparameter tuning, ~+0.012–0.018 from the homology split on top). The
homology split isn't just a stricter/more honest correction — it
genuinely scores *better* than random splitting even at matched
hyperparameters.
**Takeaway:** a textbook controlled experiment — good to describe
explicitly as "we ran a controlled ablation to separate two confounded
variables" in Methods/Results, since it's a methodologically clean,
self-contained argument regardless of the specific numbers. `[§11i]`

### ~2026-07-16 — Found and fixed test-set leakage in our own model-selection process
**What:** re-examined exactly how the §11e/§11f winners had been chosen.
**Result:** the lr/batch_size/epochs shortlist was chosen cleanly (only
touched validation loss), but the *final* pick among the shortlist, and
the whole hidden_dim/output_dim search, had used `test_only_auroc` to
compare candidates — i.e., the test split was used for model selection,
not just a single final report. A real, if mild, mistake.
**Fix:** added `validation_only_auroc` (same logic, but validation-split
positives) as the metric for all future selection; `test_only_auroc`
reserved for exactly one final, untouched check.
**Takeaway:** a genuinely valuable thing to be transparent about in the
thesis — catching your own methodology mistake and fixing it *before* it
compounds is a stronger story than never having made it, and it's the
kind of rigor a thesis committee should reward. Cite the original CLIP
paper's precedent here (tuned cheaply on an in-distribution proxy, never
touched their ~30 zero-shot benchmarks during tuning). `[§11j]`

### 2026-07-24 — v1 hyperparameter sweep (the one actually used)
**What/why:** rather than running the planned two-stage W&B sweep (§11k,
built but never run — superseded, files later deleted), built a simpler,
prioritized "v1" sweep: joint lr/batch_size/epochs search via W&B,
selected purely by validation-split InfoNCE loss (via the same
`gap_below_random` correction from §11e) — deliberately never touching
pathway AUROC or the test split, to avoid repeating the §11j mistake by
construction rather than by discipline.
**Also happened:** caught a wrong citation while setting the search range
— had been citing ProteinCLIP's shared embedding dimension as 512;
verified against the actual paper text and found it's really 128, and
corrected it before finalizing the search range. Cross-validated the
search range against ProteinCLIP's real training budget by converting to
total optimizer steps (the only unit comparable across such different
dataset sizes) — the independently-derived range and ProteinCLIP's real
numbers lined up well.
**Takeaway:** two good methodology points for the thesis: (1) tuning
infrastructure was deliberately kept decoupled from evaluation, as a
structural guarantee, not just a promise to be careful; (2) always verify
a citation against primary-source text before it shapes a design decision
— the ProtET/ProteinCLIP mixup earlier in this same effort (similar names,
overlapping hyperparameter ranges) is worth a one-line "lesson learned"
in the thesis if there's room. `[§11l]`

### 2026-07-25 (morning) — Sweep continuation, checkpoints not yet written up
**What:** two further checkpoints appeared beyond what §11l documents —
`final_v1_sweep_masked_best.pt` (hidden_dim=512, output_dim=128) and
`final_v2_dims_masked_best.pt` (hidden_dim=2048, output_dim=1024, used
below for the DeepLoc eval).
**Open item:** the decision process behind these (presumably a
hidden_dim/output_dim search on top of the v1 lr/batch/epoch winner)
isn't documented in CLAUDE.md yet — **go back and write this up before
drafting the Methods/hyperparameter-tuning section**, so there's a clean,
complete record of how the actually-final architecture was chosen (same
level of detail as §11e/§11f).

### 2026-07-25 — DeepLoc zero-shot subcellular-localization classification
**What/why:** built a structurally different evaluation from everything
else in the thesis — not a ranking metric (pathway AUROC), but a forced
10-way classification task, closer to standard zero-shot image
classification. Built a human-only DeepLoc-1.0 test set (2,645 proteins,
filtered by intersecting DeepLoc's UniProt accessions with our own already
human-only proteome — guarantees correctness "for free" rather than a
fresh taxonomy lookup; seen/unseen assigned via *our* protein-level train
split, not DeepLoc's own irrelevant train/test partition, since nothing is
trained on DeepLoc labels at all — this is genuinely zero-shot).
**Method:** standard CLIP zero-shot recipe — 3 independently-phrased
prompts per class, each encoded/normalized, averaged, then re-normalized
("prompt ensembling").
**Result:**
- 10-way: 17.73% accuracy — *below* the trivial majority-class baseline
  (36.79%, "always guess Nucleus").
- 9-way, human-only (dropped "Chloroplast," which has zero human members
  and is a guaranteed miss whenever predicted): 19.43% — helped a little,
  not the main problem.
- Diagnosed *why*: Nucleus (37% of true labels, the single biggest class)
  was predicted only ~5% of the time; rare classes (Lysosome/Vacuole,
  Peroxisome) were over-predicted 10–13x.
- Tried class-wise mean-centering calibration (subtract each class's
  average similarity across all proteins before argmax) as a cheap,
  no-retrain fix. **This failed** (19.43%→19.58%, effectively no change) —
  and *why* it failed is itself a finding: per-class mean similarities
  were nearly identical, and the most over-predicted class didn't even
  have an above-average mean, which rules out a simple additive bias as
  the mechanism.
- Seen/unseen generalization gap stayed small (~1–2pp) throughout.
**Takeaway (Discussion-ready):** this is a clean, complete negative-result
story — a documented hypothesis (class-wise bias), a fair test, a
specific, diagnosable reason for the failure (it's likely a variance
effect, not a mean effect — untested, flagged as future work), and a
plausible root-cause hypothesis for the underlying weak performance
(training text is UniProt *function* text, not dedicated location text,
so location is only incidentally present — unlike pathway/function
concepts, which the model handles considerably better). Frame this in the
thesis as "we identified a real limitation, tested the obvious fix,
reported honestly that it didn't work, and explained why" — not as an
unexplained bad number. `[§11m]`

### 2026-07-25 (later) — Thesis scope decision + outline
**What:** decided the thesis narrates section 11 onward only; updated
`CLAUDE.md`'s "Proposed thesis structure" to a 6-chapter outline
(Introduction → Related Work → Methods → Experiments/Results → Discussion
→ Conclusion & Future Work), each bullet mapped to the exact §11a–§11m
subsection it draws from. Built this log file as a separate, skimmable,
date-ordered companion to that outline.
**Takeaway:** you now have two documents that do different jobs — the
`CLAUDE.md` outline says *where each piece of content goes in the thesis*,
this log says *what happened and what we learned, in the order it
happened*. Write chapter-by-chapter from the outline, and pull specific
motivations/numbers/interpretations from this log as you go.

---

## Open threads for future entries in this log
- Document the sweep that produced `final_v1_sweep_masked_best.pt` /
  `final_v2_dims_masked_best.pt` properly (not yet in CLAUDE.md) — do this
  before writing the hyperparameter-tuning part of Methods.
- z-score-standardization follow-up to the DeepLoc calibration negative
  result, if pursued.
- `activity_score` / Stage 2 (per-patient abundance annotation) validation
  on a larger sample of patients — planned, not yet done at scale.
- Stage 3 (Claude API interpretation layer) — proof-of-concept, not
  started.

## Scope decision (why sections 1–10 are BACKGROUND, not thesis content)
Section 11 of `CLAUDE.md` opens by explaining this directly: sections 1–10
were "a sequence of exploratory pivots needed to arrive at a defensible
design, not the design itself." On 2026-07-25 this was made an explicit
writing decision: start the thesis at section 11, don't narrate the
patient-level baseline or the ablation-by-ablation history. The one
exception is machinery still actively *used* in-scope (the 245-pathway
selection criteria, §8d) — describe that plainly in Methods, as-is, not as
a discovery narrative.
