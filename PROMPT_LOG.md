# Prompt log

Record of prompts given to Claude (Claude Code) during AI-assisted work
sessions on this thesis, for AI-usage disclosure/tracking purposes. Each
entry is a session, dated, with the representative prompts that drove the
session's work (minor back-and-forth clarifications are folded into the
prompt they belong to, not listed as separate entries).

---

## 2026-07-27 — Stage 3 (LLM interpretation) + Streamlit app rewrite

1. *"Let's continue working on the thesis. I can't decide between building
   the Streamlit app integration or the Stage 3 LLM interpretation layer
   next — what would you recommend?"*

2. *"Sounds good, let's do everything step by step together, explain me
   what you do each step so that I can follow the process."*

3. *"I just want the proteins embedded that are part of the sample"* /
   *"we decided to not do the activity score"* — corrections to the
   retrieval design: restrict retrieval to only the proteins measured in
   the patient/uploaded sample rather than the whole proteome, and rank
   purely by cosine similarity rather than blending in abundance.

4. *"Rewrite app.py to the current architecture"* — with follow-up
   decisions to reuse the already-built retrieval/interpretation functions
   rather than reimplementing them, and to put the LLM interpretation step
   behind an explicit button rather than calling it automatically.

5. *"Okay, update CLAUDE.md up to now."*

**What this session produced:** `src/model/interpret.py` (Stage 3, plus
sample-restricted Stage 1 retrieval for the interactive tool), a full
rewrite of `src/app.py` to the current main-line architecture, both
validated against real patient data; `CLAUDE.md` section 11r documents the
technical details and design decisions in full.
