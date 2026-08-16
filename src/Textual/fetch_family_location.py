"""
Adds two new columns, `Family` and `Location`, to the existing
data/full_proteome/protein_annotations.csv -- built for the ProtST-inspired
M0-M3 training-text ablation (CLAUDE.md "Next steps as of 2026-08-16"):
    M0 = Function only (existing baseline)
    M1 = M0 + Family
    M2 = M1 + Location
    M3 = M2 + unmasked Gene Name

Does NOT touch any existing column (ENSP_ID, UniProt_Accession, Gene,
Annotation, Sequence, GO_Biological_Process, GO_Cellular_Component) --
additive only, same precedent as the _masked/_go embedding variants
(never overwrite, always add alongside).

Family = UniProt's SIMILARITY comment (protein family/superfamily
membership), e.g. "Belongs to the small GTPase superfamily, Arf family."
Citation-stripped with the exact same regex fetch_full_human_proteome.py
already uses for FUNCTION text.

Location = UniProt's SUBCELLULAR LOCATION comment, TOP-LEVEL COMPARTMENT
ONLY, deduplicated -- deliberately NOT the full hierarchical path UniProt
returns. Each `subcellularLocations[].location.value` is often already a
full comma-joined hierarchy string on its own (e.g. "Cytoplasm,
cytoskeleton, spindle pole" is ONE location entry, not three). A
multi-localizing protein can have several such entries sharing the same
top-level prefix -- naively joining them all produces a repetitive,
term-list-like sentence (worst case found during validation: RGS14, 11
raw entries -> 298 chars with "Cytoplasm" repeated 3 times), structurally
close to the GO-term-list format that was diagnosed as a self-ID shortcut
and abandoned in section 10n. Taking only the first comma-segment of each
entry and deduplicating fixes this (RGS14 -> 6 distinct compartments, 94
chars, zero repetition) and is arguably the more query-relevant
granularity anyway (a natural-language query is more likely to ask "is
this a nuclear protein" than "is this at the spindle pole specifically").
Verified empirically (2026-08-16, live UniProt sample, n=500): worst-case
length drops from 404->153 chars project-wide under this rule.

Both fields were validated for length/coverage before this script was
written (CLAUDE.md "Next steps as of 2026-08-16"): over the actual
16,893-protein set, Family covers 75.4% (mean 48 chars), Location covers
92.6% (mean 49 chars, using the full-hierarchy length estimate --
top-level-only is shorter still). Token-budget impact is small: adding
both fields to the masked Function baseline raises mean BioBERT token
count from ~155 to ~180 (+16%), and the >512-truncation rate barely moves
(3.7% -> 3.9%) -- confirmed with the real BioBERT tokenizer, not a rough
character estimate.

To run:
    uv run python src/Textual/fetch_family_location.py
"""
import argparse
import re

import pandas as pd
import requests

INPUT_PATH = "data/full_proteome/protein_annotations.csv"
OUTPUT_PATH = "data/full_proteome/protein_annotations.csv"

# Identical to fetch_full_human_proteome.py's CITATION_PATTERN -- same
# provenance-tag noise (PubMed refs, "By similarity", etc.) shows up in
# SIMILARITY text too, not just FUNCTION.
CITATION_PATTERN = re.compile(
    r'\s*\((?:PubMed:\d+(?:,\s*PubMed:\d+)*|By similarity|Probable|Potential)\)'
)


def strip_citations(text):
    return CITATION_PATTERN.sub('', text).strip()


def get_family(entry):
    for c in entry.get('comments', []):
        if c.get('commentType') == 'SIMILARITY':
            val = strip_citations(c['texts'][0].get('value', ''))
            if not val:
                return ''
            return val if val.endswith('.') else val + '.'
    return ''


def get_location(entry):
    """Top-level compartment only, deduplicated, first-seen order. See
    module docstring for why the full hierarchical path is deliberately
    not used."""
    top_levels = []
    for c in entry.get('comments', []):
        if c.get('commentType') != 'SUBCELLULAR LOCATION':
            continue
        for sl in c.get('subcellularLocations', []):
            v = sl.get('location', {}).get('value')
            if not v:
                continue
            top = v.split(',')[0].strip()
            if top and top not in top_levels:
                top_levels.append(top)
    if not top_levels:
        return ''
    return 'Located in ' + ', '.join(top_levels) + '.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pages', type=int, default=None,
                        help='debug: stop after N pages instead of fetching all ~41 pages')
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)
    our_accessions = set(df['UniProt_Accession'])
    print(f'Loaded {INPUT_PATH}: {len(df)} proteins, {len(our_accessions)} unique accessions')

    url = ('https://rest.uniprot.org/uniprotkb/search'
           '?query=organism_id:9606+AND+reviewed:true'
           '&fields=accession,cc_similarity,cc_subcellular_location'
           '&format=json&size=500')

    family_by_acc = {}
    location_by_acc = {}
    n_fetched = 0
    page = 0

    while url:
        response = requests.get(url)
        response.raise_for_status()
        page_data = response.json()
        results = page_data.get('results', [])
        n_fetched += len(results)
        page += 1

        for entry in results:
            acc = entry.get('primaryAccession', '')
            if acc not in our_accessions:
                continue
            family_by_acc[acc] = get_family(entry)
            location_by_acc[acc] = get_location(entry)

        print(f'Page {page}: fetched {n_fetched} so far, matched {len(family_by_acc)} of ours')

        if args.max_pages is not None and page >= args.max_pages:
            print(f'Stopping after {page} pages (--max-pages)')
            break

        link = response.headers.get('Link', '')
        if 'rel="next"' in link:
            url = link.split('<')[1].split('>')[0]
        else:
            url = None

    df['Family'] = df['UniProt_Accession'].map(family_by_acc).fillna('')
    df['Location'] = df['UniProt_Accession'].map(location_by_acc).fillna('')

    n_matched = sum(1 for acc in our_accessions if acc in family_by_acc)
    n_missing = len(our_accessions) - n_matched
    n_family = (df['Family'] != '').sum()
    n_location = (df['Location'] != '').sum()

    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f'Total fetched from UniProt: {n_fetched}')
    print(f'Matched to our protein set: {n_matched} / {len(our_accessions)} ({n_missing} not found in this crawl)')
    print(f'Family coverage: {n_family} / {len(df)} = {n_family / len(df) * 100:.1f}%')
    print(f'Location coverage: {n_location} / {len(df)} = {n_location / len(df) * 100:.1f}%')
    print(f'Saved -> {OUTPUT_PATH} (added columns: Family, Location; all existing columns unchanged)')


if __name__ == '__main__':
    main()
