# Midterm talk outline — ProteomIQ

Target: ~20 minutes, small committee (a few faculty members — assume no
shared context, explain acronyms once). Roughly one slide per numbered
point below (~12-15 slides). Pull exact numbers/tables from the bracketed
`CLAUDE.md` section references; pull ready-to-say phrasing from the
matching dated entry in `THESIS_LOG.md`.

**Framing to open with:** "ProteomIQ lets a researcher ask a plain-English
question about their own patient's protein sample and get back a ranked
list of relevant proteins — something no existing proteomics tool
(GSEA included) can do, because they all require a fixed, predefined gene
set instead of free text."

---

## 1. Motivation (~2 min)
- The problem: a researcher with a new proteomics sample has no way to ask
  "what's relevant to X" in plain language — only fixed pathway/gene-set
  tools (GSEA).
- The idea: CLIP-style contrastive training — pair each protein's sequence
  with its functional text description, learn a shared embedding space,
  then rank any patient's measured proteins against a free-text query.
- **Why protein-level, not patient-level** (the actual thesis
  differentiator): querying at patient granularity returns *other
  patients* — not actionable. Querying at protein granularity returns
  *EGFR, MET, KRAS* — directly actionable. State this contrast explicitly,
  it's the cleanest one-liner of the whole motivation.
- CellWhisperer is the direct inspiration for this project, not just the
  closest prior work found afterward — my professor's idea for ProteomIQ
  came from seeing CellWhisperer do this for single-cell transcriptomics;
  ProteomIQ asks the same question for mass-spec proteomics, at protein
  (not sample) granularity.

## 2. Related work (~2 min, brief — don't over-invest here)
- CLIP (contrastive image-text pretraining) — the general recipe being
  adapted.
- ESM2 — frozen protein language model, produces the protein-side input.
- BioBERT — frozen biomedical text encoder, produces the text-side input.
- CellWhisperer — closest prior work, different modality/granularity (see
  above).
- ProteinCLIP — a real published protein-text CLIP model; mention that
  this project's architecture *converged* onto a similar design
  independently (single hidden-layer projection heads, shared embedding
  dimension in a similar range) — a citation-verified comparison, not a
  copied design `[§11l]`.
- GSEA — the practical baseline/alternative; name explicitly what it
  *can't* do (free-text queries against one specific patient's measured
  proteins) that this tool can.

## 3. Methods — this is the bulk of the talk (~7 min)

### 3a. Data (~1.5 min)
- Full reviewed human UniProt proteome — 16,893 proteins, not just
  whatever one measurement pipeline happened to detect `[§11b]`.
- One concrete "we found and fixed a real bug" story: a UniProt entry can
  map to several Ensembl protein IDs (one per isoform); picking the wrong
  one would silently mismatch which measurement a protein's abundance
  value corresponds to. Fixed by picking the ID with the widest cohort
  coverage. Good, concrete evidence of careful data engineering.

### 3b. Homology-aware split (~1.5 min)
- The problem: a naive random train/test split lets near-identical
  paralogs leak across the split, inflating apparent generalization.
- The fix: CD-HIT clustering at 90% identity, split at the *cluster*
  level, not the individual-protein level `[§11c]`.
- One good number: only ~4.8% of the human proteome is redundant at this
  threshold — CD-HIT does real, targeted work for a small minority, not a
  blanket reorganization.
- Name the comparison to ProtET (a published protein-CLIP paper) explicitly
  lacking this control — this is a genuine, citable methodological
  improvement over existing practice, not just "best practice."

### 3c. Architecture (~1.5 min)
- Frozen ESM2 (protein) + frozen BioBERT (text) → small trainable
  projection heads → shared embedding space, InfoNCE loss.
- Gene-name masking: strip the gene name from training text so the model
  can't just pattern-match a protein to its own name. Counterintuitively,
  masking made training loss *worse* but every downstream generalization
  metric *better* — evidence the un-masked model's apparent skill was
  partly a shortcut, not deeper learning `[§10m, re-verified at final
  architecture in §11g]`.

### 3d. Model selection, done cleanly (~1.5 min)
- Both the hyperparameters (lr/batch size/epochs) and the architecture
  (hidden/output dimension) were chosen via a W&B sweep, selected purely
  on **held-out validation loss** — never touching any of the reported
  evaluation metrics during selection `[§11l, §11p]`.
- This is a genuinely strong, self-contained methodology point: "the
  checkpoint we report results for was chosen with zero involvement from
  the metrics we're about to show you" — say this explicitly, it
  preempts an obvious committee question about cherry-picking.

### 3e. Evaluation protocol (~1 min)
- Two families: (1) standard closed-set bidirectional retrieval (cross-modal,
  CLIP-style R@K); (2) zero-shot forced-choice classification (DeepLoc,
  EC, GO-MF), same recipe each time — prompt ensembling, seen/unseen split
  relative to the model's own training split, random + majority-class
  baselines.

## 4. Results (~5 min)

- **Cross-modal retrieval — works clearly.** Far above random baseline in
  both directions (protein→text and text→protein) `[§11d]`.
- **DeepLoc (subcellular localization, 10-way forced choice) — a genuine
  weak spot.** Below the trivial majority-class baseline. Tried an
  obvious fix (calibration) — it didn't help, and diagnosed *why* it
  didn't (rules out a simple additive per-class bias) `[§11m]`.
- **EC (enzyme class, 7-way forced choice) — also below majority
  baseline.** Tested explanatory vs. bare-keyword prompts specifically to
  rule out a lexical-shortcut explanation — the small, ambiguous-direction
  gap between the two rules that out `[§11n]`.
- **GO-Slim-MF (molecular function) — the positive result.** Clears both
  random *and* majority-class baselines, under a properly-redesigned
  multi-label scoring rule (forced by the biology — a protein genuinely
  can have more than one molecular function) `[§11o]`.
- **The honest overall story:** the contrastive objective clearly produces
  a *real, usable* embedding space (cross-modal + GO-MF prove this) — but
  forced-choice classification is weak specifically where the label
  vocabulary (subcellular location, catalytic mechanism) is only
  incidentally present in the UniProt function text the model actually
  trained on. GO-MF's success is the natural control case: its vocabulary
  *is* close to what the training text describes. This is a stated
  hypothesis, not yet directly tested — say that honestly if asked.

## 5. Honest self-assessment: doubts about performance, scale as the
   leading suspect (~1-2 min)
- State this plainly, before the committee asks: the weak forced-choice
  numbers (DeepLoc, EC) are more likely explained by dataset scale than by
  a flaw in the architecture or training recipe — the architecture side
  was already checked carefully (masking ablation, lr/batch-size/epochs
  and hidden/output-dim swept on held-out validation loss) and each of
  those levers behaved as expected, which points *away* from "the
  architecture is wrong" and *toward* "there isn't enough data to learn
  more from."
- The concrete number: after deduping to one row per protein (patient-
  invariant text), training has only ~13,515 unique protein-text pairs
  `[§11c, §11l]` — small by contrastive-learning standards.
- Verified scale comparisons already on record, not guessed: this is
  roughly **30,000x smaller** than CLIP's original ~400M image-text pairs,
  and roughly **34x smaller** than ProteinCLIP's 465,770 sequence-
  annotation pairs `[§11l]`. Both numbers were checked against primary
  sources, following the same discipline as the ProteinCLIP/ProtET
  citation correction already made in this project.
- CellWhisperer specifically: cite the batch-size/hyperparameter
  comparisons already made (`[§11l]` — this project's `lr`/`batch_size`
  search range was set to bracket CellWhisperer's own defaults), but be
  upfront that an exact training-pair-count comparison to CellWhisperer
  was not independently verified before this talk — say "I'd want to
  confirm that number before putting it in the thesis" rather than citing
  an unverified figure, consistent with how every other cross-paper number
  in this project was fact-checked before use.
- Why this framing matters for the thesis: it turns "why are two of four
  evals weak" from an open question into a stated, reasoned hypothesis —
  the same honesty already applied to the DeepLoc/EC/GO-MF result pattern
  in section 4 — and it's the direct motivation for the "more positive
  pairs" future-work item in section 7 (LLM-fusion / paraphrase
  augmentation) rather than that item appearing out of nowhere.

## 6. Working prototype — live demo or screenshots (~2-3 min)
- Show the three-stage pipeline end to end: upload a sample → Stage 1
  (semantic retrieval, restricted to that sample's own measured proteins)
  → Stage 2 (real abundance percentile attached per protein) → Stage 3
  (Claude API interpretation in plain language) `[§11r]`.
- Good concrete example to show live: same query on two different
  patients producing genuinely different, correctly-grounded
  interpretations — demonstrates the tool is patient-specific, not
  generic protein lookup.
- Be upfront if asked: this is a validated proof-of-concept (spot-checked
  on real patients/queries, fact-checked against the underlying numbers),
  deliberately *not* part of the four formally reported evaluations above.

## 7. Remaining work + timeline (~2 min)
- Possible follow-up: a linear-probe check on the frozen protein
  embeddings, to separate "bad protein representation" from "bad
  text-prompt alignment" for the two weak evals (DeepLoc, EC) — named as
  future work, not yet done.
- New training methods to try for raw performance improvement, distinct
  from the diagnostic/eval work above: the deduped training set is small
  (12,574–13,515 unique protein-text pairs, one per protein once
  patient-invariant), so the main lever left is increasing the number/
  diversity of *positive pairs* per protein rather than more architecture
  tuning. Concretely: resuming the paused LLM-fusion text augmentation
  (built and spot-checked, blocked only on API budget) and reconsidering
  LLM-paraphrase augmentation (multiple differently-worded positives per
  protein) — flagged in project notes as worth a fresh look now that
  GO-term concatenation was ruled out as a fix. Framed honestly as
  "possibilities to explore," not a committed plan — time-boxed by the
  2026-10-10 hand-in.
- Hand-in: 2026-10-10. State concretely what's left: writing up the
  thesis text itself (structure already drafted), any follow-up
  experiments time allows.
- Close by restating the contribution independent of the mixed
  eval numbers: a defensible, homology-aware, evaluation-decoupled
  methodology, honestly reported wins *and* losses, and a working
  end-to-end prototype — that combination is the actual deliverable,
  not any single metric.

---

## Anticipated committee questions (prepare short answers)
- *"Why didn't you just use raw pathway/GO term overlap instead of
  training a model?"* → GSEA-style tools can't take free-text queries;
  that's the whole point of this project.
- *"Don't you also have a fixed set, via your pre-embedded ESM2
  proteins?"* → Yes, but on a different axis than GSEA's limitation, and
  that's the part that matters: GSEA is fixed on the *concept* — a gene
  set represents one specific named pathway, and a query outside the
  curated database has no gene set at all. ProteomIQ is fixed only on
  *which proteins exist* (16,893 proteins — essentially the entire
  reviewed human proteome, not a small curated subset), while the actual
  query is fully open free text, encoded fresh at query time. A real,
  smaller limitation worth naming honestly: a protein missing from
  UniProt or genuinely novel has no ESM2 embedding and can never be
  retrieved — true, but a much narrower gap than GSEA's.
- *"Is the low DeepLoc/EC score a problem for the thesis?"* → No — it's a
  reported, diagnosed limitation, not a hidden one; the thesis's
  contribution is the methodology and honest evaluation, not a single
  headline number.
- *"How do you know the checkpoint wasn't cherry-picked to look good on
  these evals?"* → It wasn't — selection was on held-out validation loss
  only, decoupled from every reported metric (§3d above).
- *"Does the tool actually work for a real researcher?"* → Promising,
  spot-checked proof-of-concept; not yet formally/at-scale validated —
  say this plainly, don't oversell it.
- *"Why not just get more training data?"* → CPTAC cohort count is
  already maxed out (9/10 usable, COAD structurally broken), and the
  full-proteome move already uses every reviewed human protein UniProt
  has — the ceiling isn't cohort count, it's unique protein-text pairs
  (~13,515). The real lever left is more/better *positive pairs* per
  protein (LLM-fusion/paraphrase augmentation, §5 and §7), not more raw
  proteomics data.
