"""
Adds the UniProt Family and Subcellular Location fields to data/full_proteome/protein_annotations.csv.

Usage:
    uv run python src/data_acquisition/fetch_family_location.py
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
