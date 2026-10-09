"""
Generates the 5 prompt templates per zero-shot task (frozen, not meant to be re-run).

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
