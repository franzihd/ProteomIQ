"""
Generates the frozen pool of 5 generic natural-language prompt templates per
zero-shot classification task (DeepLoc, EC top-level, GO-MF), used by
zero_shot_deeploc_eval.py / zero_shot_ec_eval.py / zero_shot_go_mf_eval.py
for the "Engineered" (N=1) and "Ensemble" (N=5) prompt conditions.

*** FROZEN as of revision 2026-08-15d -- do not run this script again to
regenerate data/evaluation/zero_shot_classification/zero_shot_prompt_templates.json. *** The three
tasks' final templates include small manual semantic/grammatical
corrections (2026-08-15d, see each task's "revisions" list in the JSON)
made directly in the JSON file, on top of what this script's own
2026-08-15b API call produced -- re-running main() would silently discard
those hand-edits by overwriting "templates" with a fresh generation, while
leaving the stale META_PROMPTS text in this file's "meta_prompt" field
technically still accurate (it documents what was actually sent to the
API), just no longer a full description of the final template text. If a
future change is genuinely needed, edit data/evaluation/zero_shot_prompt_
templates.json directly and add a new dated revisions entry, the same way
2026-08-15d was done -- don't call main() again for these three tasks.

Methodology note: one fixed meta-prompt per task, applied once, before any
evaluation results exist -- same discipline as the pathway NL query rewrites
(pathway_nl_queries.py, CLAUDE.md Timeline 8e) and the pathway selection
criteria (pathway_final_choice.py, 8d). Templates are not hand-tuned or
re-generated after seeing accuracy numbers.

Revision (2026-08-13): the first generated set leaned into experimental-
methodology-flavored phrasing for a few templates per task (e.g. "Immuno-
fluorescence staining revealed...", "Microscopy analysis shows..."), a third,
unintended register beyond the intended "generic natural language vs.
database jargon" axis -- caught on manual review, before any evaluation was
run, so re-generating is a design fix, not a performance-based cherry-pick.
The meta-prompt now explicitly asks for plain, simple, mutually similar
sentences and excludes experimental-method framing.

Revision (2026-08-15a): after generation, filling all 7 EC classes into all 5
EC templates (see zero_shot_ec_eval.py) surfaced a grammar bug specific to
EC template #5 -- {label} there sat adjectivally before "class"
("...grouped within the {label} class"), which needs the singular enzyme
name ("the hydrolase class"), while {label} is always filled with the
plural canonical class name ("Hydrolases", same value used for the
Label-only condition) -- producing "the Hydrolases class", which is
ungrammatical. Hand-patched template #5 only, directly in
data/evaluation/zero_shot_classification/zero_shot_prompt_templates.json -- a one-line, targeted
grammar fix, not a full regeneration. Superseded by the 2026-08-15b full
regeneration below, kept here for the audit trail.

Revision (2026-08-15b): all 3 tasks' templates regenerated from scratch
(exactly one new Claude API call per task), for two reasons surfaced during
manual review of the 2026-08-15a templates -- not from any evaluation
result, and no evaluation has been run against these templates as of this
revision:
1. **Semantic drift between templates was never actually forbidden.** The
   previous meta-prompt only asked for "diverse" templates, with no
   constraint that the 5 make the same underlying biological claim --
   nothing prevented Claude from generating 5 templates that each imply a
   subtly different claim (different certainty, causality, or scope) rather
   than 5 paraphrases of one fixed claim. Concretely: GO-MF template #4 read
   "This protein contributes to {label}." -- "contributes to" implies
   partial/indirect causal involvement in a function, a different, weaker
   claim than "has {label}" or "is capable of {label}" (direct possession),
   which the other 4 templates stated. Averaging embeddings of 5 sentences
   that do not even agree on what they are claiming is not a coherent
   ensemble -- this went uncaught by the 2026-08-15a review (which checked
   grammar, not semantic consistency). The new meta-prompt makes semantic
   equivalence an explicit, checked constraint (SEMANTIC_CONSISTENCY_
   INSTRUCTION below), with an explicit "contributes to" ban for GO-MF.
2. **Claude never saw the actual label strings.** The previous meta-prompt
   described the label *category* in the abstract (e.g. "a subcellular
   compartment name") but never showed Claude the real label strings
   {label} gets substituted with at evaluation time -- grammatical fit
   against unusual labels (e.g. "Lysosome/Vacuole" -> "lysosome or
   vacuole", "Golgi apparatus" keeping its capital G) could therefore only
   be caught after the fact by a human, as happened in 2026-08-15a. The new
   meta-prompt instead gives Claude the exact label strings each task
   actually substitutes into {label} at eval time (DEEPLOC_LABELS /
   EC_LABELS below, both duplicated from -- and must stay in sync with --
   DEEPLOC_SENTENCE_LABELS.values() / [c.lower() for c in CLASSES] in the
   respective eval scripts; duplicated rather than imported, to keep this
   script free of the eval scripts' torch/transformers dependency.
   GO_MF_LABELS is loaded live from data/evaluation/zero_shot_classification/go_mf/class_selection.json
   instead of duplicated, since that file is already the single source of
   truth for GO-MF's data-derived label list and needs no separate sync)
   and explicitly instructs it to check every one before finalizing.

Revision (2026-08-15c): the 2026-08-15b regeneration's own EC template #3
("This enzyme is a member of the {label} class.") reproduced the identical
adjectival-{label}-before-"class" grammar bug from 2026-08-15a, just in a
different template slot -- "the hydrolases class" is ungrammatical for the
same reason as before (wants the singular "hydrolase"). The new meta-prompt's
explicit per-label grammar-check instruction did not prevent recurrence.
Hand-patched by dropping "class" ("This enzyme is a member of the {label}.",
matching the plural-object pattern already used by 3 of the other 4 EC
templates) -- caught by re-running the same fill-every-label verification
used throughout this revision history, before any evaluation was run. Not
sent back to the API (a one-word deletion, not a regeneration) -- "exactly
one new API call per task" was already satisfied by 2026-08-15b.

Also encountered while running 2026-08-15b, worth recording as an
implementation note rather than a template-content revision: claude-sonnet-5
runs adaptive thinking by default even when the `thinking` parameter is
omitted, and the first attempt at this regeneration crashed on the go_mf
call (the longest, most constraint-heavy meta-prompt of the three) --
max_tokens=500 was entirely consumed by an internal ThinkingBlock, leaving
zero tokens for the actual JSON response. Fixed by passing
`thinking={"type": "disabled"}` explicitly in generate_templates() (this
task is straightforward templated-text generation with nothing to reason
through) and raising max_tokens to 1024 as a safety margin. DeepLoc and EC
had already completed successfully before the crash and were not
regenerated a second time, to honor "exactly one new API call per task".

A fourth "UniProt-style" condition (fixed field-format templates, e.g.
"Subcellular location: {label}.") was considered and deliberately dropped:
verified live against UniProt's REST API (EGFR/P00533) that this field format
is NOT the actual training text -- Stage 1 trains only on the FUNCTION
comment (long prose), never on SUBCELLULAR LOCATION / CATALYTIC ACTIVITY /
GO-term fields. A condition calling itself "UniProt-style" would therefore
test something other than what its name claims, and the only defensible
remaining interpretation ("terse jargon vs. natural sentence") was judged too
weak/redundant with the Label-only condition to keep. See CLAUDE.md Timeline
section 11 (zero-shot prompt engineering discussion, 2026-08-13/15) for the
full reasoning.

Usage:
    uv run python src/evaluation/generate_zero_shot_prompt_templates.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anthropic

OUT_PATH = Path("data/evaluation/zero_shot_classification/zero_shot_prompt_templates.json")
GO_MF_CLASS_SELECTION_PATH = Path("data/evaluation/zero_shot_classification/go_mf/class_selection.json")

MODEL = "claude-sonnet-5"
N_TEMPLATES = 5
# Note: `temperature` is not accepted on claude-sonnet-5 (400 on any value, not just
# non-default) -- Anthropic's replacement for determinism is prompt design, not a
# sampling parameter. We don't rely on temperature=0 here anyway: all 5 templates are
# generated in a single response (one call, JSON array output), not via repeated calls
# that would need to agree with each other.

# The exact strings {label} gets substituted with at evaluation time for
# DeepLoc/EC. Duplicated from zero_shot_deeploc_eval.py's
# DEEPLOC_SENTENCE_LABELS.values() / zero_shot_ec_eval.py's
# [c.lower() for c in CLASSES] -- keep in sync if either changes. Given to
# the meta-prompt directly so Claude checks grammatical fit against the
# real substitution values, not an abstract description of the label
# category.
DEEPLOC_LABELS = [
    "nucleus", "cytoplasm", "extracellular space", "mitochondrion",
    "cell membrane", "endoplasmic reticulum", "chloroplast",
    "Golgi apparatus", "lysosome or vacuole", "peroxisome",
]
EC_LABELS = [
    "oxidoreductases", "transferases", "hydrolases", "lyases",
    "isomerases", "ligases", "translocases",
]


def load_go_mf_labels() -> list[str]:
    """GO-MF's labels are data-derived (see build_go_mf_eval_set.py), not a
    fixed taxonomy like DeepLoc/EC -- loaded live from the class-selection
    file rather than duplicated, since that file is already the single
    source of truth and this keeps the two impossible to drift apart."""
    with open(GO_MF_CLASS_SELECTION_PATH) as f:
        return json.load(f)["selected_classes"]


def format_labels(labels: list[str]) -> str:
    return ", ".join(f"'{l}'" for l in labels)


STYLE_INSTRUCTION = (
    "Keep the sentences plain and simple, similar in style and structure to "
    "each other -- straightforward statements, not elaborate or abstract "
    "phrasing. Do not reference specific experimental methods (e.g. "
    "microscopy, immunofluorescence staining, biochemical assays) or phrase "
    "it as reporting an experimental finding -- state the fact directly."
)

# New in the 2026-08-15b revision (see docstring reason 1): the previous
# meta-prompt only asked for "diverse" templates, which left room for the 5
# to differ in what they actually claim (certainty, causality, scope), not
# just in wording. STYLE_INSTRUCTION (above) governs tone/register; this
# governs meaning, as an explicit, separately-stated constraint.
SEMANTIC_CONSISTENCY_INSTRUCTION = (
    "Critically, all 5 templates must express exactly the same underlying "
    "biological claim -- vary ONLY the wording and sentence structure, "
    "never the meaning. Do not let any one template imply a different "
    "degree of certainty, causality, directness, or scope than the others "
    "(for example, a phrase implying indirect or partial involvement, like "
    "'contributes to', is a different, weaker claim than direct possession, "
    "like 'has' or 'is capable of' -- do not mix claim types like this "
    "across the 5 templates). The 5 templates must be interchangeable "
    "paraphrases of ONE fixed statement, not 5 different statements."
)


def label_format_instruction(labels: list[str]) -> str:
    """New in the 2026-08-15b revision (see docstring reason 2): gives
    Claude the exact label strings {label} will be substituted with at
    evaluation time, not just an abstract description of the label
    category, so grammatical fit can be checked directly instead of
    discovered after the fact."""
    return (
        "Each template must contain the placeholder {label} exactly once "
        "-- not zero times, not two or more times. {label} will be "
        "substituted, one at a time, with each of these exact labels: "
        + format_labels(labels) + ". Every one of your 5 templates must "
        "read as a grammatically correct, natural sentence for every "
        "single one of these labels -- check each one before finalizing "
        "your answer."
    )


META_PROMPTS = {
    "deeploc": (
        "Generate exactly 5 diverse, natural-language sentence templates "
        "that a biologist would use to describe where a protein is "
        "located within a cell.\n\n"
        + label_format_instruction(DEEPLOC_LABELS) + "\n\n"
        + SEMANTIC_CONSISTENCY_INSTRUCTION + "\n\n"
        + STYLE_INSTRUCTION + " Avoid database-specific jargon or fixed "
        "field-style formatting (e.g. do not write 'Subcellular location: "
        "{label}.'). Return your answer as a JSON array of exactly 5 "
        "strings, nothing else -- no markdown, no numbering, no "
        "explanation."
    ),
    "ec": (
        "Generate exactly 5 diverse, natural-language sentence templates "
        "that a biologist would use to describe which broad catalytic "
        "class an enzyme belongs to.\n\n"
        + label_format_instruction(EC_LABELS) + "\n\n"
        + SEMANTIC_CONSISTENCY_INSTRUCTION + "\n\n"
        + STYLE_INSTRUCTION + " Avoid database-specific jargon or fixed "
        "field-style formatting. Return your answer as a JSON array of "
        "exactly 5 strings, nothing else -- no markdown, no numbering, no "
        "explanation."
    ),
    "go_mf": (
        "Generate exactly 5 diverse, natural-language sentence templates "
        "that a biologist would use to describe a protein's molecular "
        "function. Note that a protein can have more than one molecular "
        "function, so avoid templates implying this is its only or "
        "defining function (e.g. avoid \"This protein's sole function is "
        "{label}.\").\n\n"
        + label_format_instruction(load_go_mf_labels()) + "\n\n"
        + SEMANTIC_CONSISTENCY_INSTRUCTION + " In particular, do NOT use "
        "'contributes to {label}' or any similar phrasing implying "
        "indirect or partial involvement -- every template must state "
        "that the protein directly has or performs the molecular "
        "function, with the same directness throughout.\n\n"
        + STYLE_INSTRUCTION + " Avoid database-specific jargon or fixed "
        "field-style formatting. Return your answer as a JSON array of "
        "exactly 5 strings, nothing else -- no markdown, no numbering, no "
        "explanation."
    ),
}

REVISION_NOTE_2026_08_15B = (
    "2026-08-15b: regenerated from scratch (exactly one new Claude API "
    "call per task) with a stricter meta-prompt requiring all 5 templates "
    "to express the same underlying biological claim (not just varied "
    "wording), showing Claude the exact label strings {label} gets "
    "substituted with at evaluation time, and (GO-MF only) explicitly "
    "forbidding 'contributes to'-style phrasing that implies "
    "indirect/partial involvement rather than direct possession of the "
    "function. Done before any evaluation was run against these "
    "templates -- not based on classification results. See "
    "generate_zero_shot_prompt_templates.py's module docstring for the "
    "full rationale."
)


def generate_templates(client, task, meta_prompt):
    # claude-sonnet-5 runs adaptive thinking by default even when `thinking`
    # is omitted -- confirmed the hard way: the go_mf call (the longest,
    # most constraint-heavy meta-prompt of the three) spent its entire
    # max_tokens=500 budget on an internal ThinkingBlock and returned zero
    # text, so json.loads() failed on an empty string. This task is a
    # straightforward templated-text generation with nothing to reason
    # through, so thinking is disabled outright rather than just raising
    # max_tokens (which would still be at risk on a future, longer
    # meta-prompt) -- and max_tokens is bumped too, as a safety margin.
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": meta_prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    text = text.strip()
    # Strip accidental markdown code fences, just in case.
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()

    templates = json.loads(text)
    if not isinstance(templates, list) or len(templates) != N_TEMPLATES:
        raise ValueError(
            f"[{task}] expected a JSON array of {N_TEMPLATES} strings, got: {templates!r}"
        )
    for t in templates:
        count = t.count("{label}")
        if count != 1:
            raise ValueError(
                f"[{task}] template must contain '{{label}}' exactly once, "
                f"found {count} occurrence(s): {t!r}"
            )
    return templates


def main():
    client = anthropic.Anthropic()

    # Carry forward any existing revision history (e.g. the 2026-08-15a EC
    # template #5 hand-patch) rather than silently dropping it on this full
    # regeneration -- the audit trail of what changed, when, and why is
    # kept across every past revision, not just the latest one.
    existing = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            existing = json.load(f)

    result = {}
    for task, meta_prompt in META_PROMPTS.items():
        print(f"Generating templates for '{task}' ...")
        templates = generate_templates(client, task, meta_prompt)
        prior_revisions = existing.get(task, {}).get("revisions", [])
        result[task] = {
            "templates": templates,
            "model": MODEL,
            "meta_prompt": meta_prompt,
            "revisions": prior_revisions + [REVISION_NOTE_2026_08_15B],
        }
        for i, t in enumerate(templates):
            print(f"  [{i}] {t}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved frozen prompt templates to {OUT_PATH}")


if __name__ == "__main__":
    main()
