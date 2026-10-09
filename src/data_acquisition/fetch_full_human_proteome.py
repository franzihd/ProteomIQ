import argparse
import re

import pandas as pd
import requests

COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]

# Same cap and rationale as fetch_uniprot.py's MAX_GO_TERMS_PER_CATEGORY.
MAX_GO_TERMS_PER_CATEGORY = 10

# UniProt FUNCTION comments embed evidence-provenance tags inline, e.g.
# "...leading to its degradation (By similarity)." or "(PubMed:23056470)".
# These carry no biological content for a text encoder -- just noise tokens
# that eat into BioBERT's 512-token budget, and disproportionately so for
# well-studied proteins (more citations = more research attention = longer,
# more truncation-prone annotations).
CITATION_PATTERN = re.compile(
    r'\s*\((?:PubMed:\d+(?:,\s*PubMed:\d+)*|By similarity|Probable|Potential)\)'
)


def strip_citations(text):
    return CITATION_PATTERN.sub('', text)


def extract_go_terms(uniprot_entry, category_letter):
    """Identical logic to fetch_uniprot.py's extract_go_terms -- duplicated
    rather than imported because fetch_uniprot.py runs argparse at module
    level (no __main__ guard), so importing it would execute its own CLI
    parsing and crash."""
    terms = []
    seen = set()
    for xref in uniprot_entry.get('uniProtKBCrossReferences', []):
        if xref.get('database') != 'GO':
            continue
        for prop in xref.get('properties', []):
            if prop.get('key') != 'GoTerm':
                continue
            value = prop['value']
            if not value.startswith(f'{category_letter}:'):
                continue
            label = value[2:]
            if label not in seen:
                seen.add(label)
                terms.append(label)
    return terms[:MAX_GO_TERMS_PER_CATEGORY]


def extract_ensembl_protein_ids(uniprot_entry, accession):
    """Returns (all_ensp_ids, canonical_ensp_id). A UniProt entry can map to
    multiple Ensembl transcripts/proteins (one per isoform); the canonical
    isoform is conventionally f"{accession}-1"."""
    ensp_by_isoform = {}
    first_seen = None
    for xref in uniprot_entry.get('uniProtKBCrossReferences', []):
        if xref.get('database') != 'Ensembl':
            continue
        protein_id = None
        for prop in xref.get('properties', []):
            if prop.get('key') == 'ProteinId':
                protein_id = prop['value']
                break
        if not protein_id:
            continue
        if first_seen is None:
            first_seen = protein_id
        isoform_id = xref.get('isoformId')
        if isoform_id:
            ensp_by_isoform[isoform_id] = protein_id

    all_ensp_ids = set(ensp_by_isoform.values())
    if first_seen:
        all_ensp_ids.add(first_seen)
    canonical = ensp_by_isoform.get(f'{accession}-1', first_seen)
    return all_ensp_ids, canonical


def load_abundance_ensp_ids():
    """Maps each ENSP ID used as a column header in any cohort's
    rank_normalized_proteomics.csv to the number of cohorts that use it --
    the file Stage 2 abundance lookups actually query. A protein's chosen
    ENSP_ID must match one of these (when it has one) or those lookups
    will silently miss it. Some proteins have multiple isoform-specific
    Ensembl IDs and different cohorts inconsistently use different ones
    for "the same" gene -- counting cohort coverage per ID lets the
    matching step deterministically prefer whichever ID is actually
    shared by the most cohorts, instead of picking arbitrarily."""
    counts = {}
    for cohort in COHORTS:
        path = f'data/cptac/{cohort}/rank_normalized_proteomics.csv'
        try:
            df = pd.read_csv(path, header=[0, 1], nrows=0)
        except FileNotFoundError:
            continue
        for col in df.columns[1:]:
            ensp = str(col[1])
            counts[ensp] = counts.get(ensp, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pages', type=int, default=None,
                        help='debug: stop after N pages instead of fetching all ~41 pages')
    args = parser.parse_args()

    abundance_ensp_counts = load_abundance_ensp_ids()
    print(f'ENSP IDs already used as abundance-matrix columns across {len(COHORTS)} cohorts: {len(abundance_ensp_counts)}')

    url = ('https://rest.uniprot.org/uniprotkb/search'
           '?query=organism_id:9606+AND+reviewed:true'
           '&fields=accession,gene_names,cc_function,sequence,go_p,go_c,xref_ensembl'
           '&format=json&size=500')

    rows = []
    n_fetched = 0
    n_no_function = 0
    n_no_ensembl = 0
    n_matched_to_abundance_id = 0
    page = 0

    while url:
        response = requests.get(url)
        response.raise_for_status()
        page_data = response.json()
        results = page_data.get('results', [])
        n_fetched += len(results)
        page += 1

        for entry in results:
            accession = entry.get('primaryAccession', '')

            gene = ''
            if entry.get('genes'):
                gene = entry['genes'][0].get('geneName', {}).get('value', '')

            annotation = ''
            for comment in entry.get('comments', []):
                if comment.get('commentType') == 'FUNCTION':
                    annotation = gene + ': ' + strip_citations(comment['texts'][0].get('value', ''))
                    break
            if not annotation:
                n_no_function += 1
                continue

            all_ensp_ids, canonical_ensp = extract_ensembl_protein_ids(entry, accession)
            if not canonical_ensp:
                n_no_ensembl += 1
                continue

            # Prefer whichever isoform ID the MOST CPTAC cohorts' abundance
            # matrices already use (not just any matching ID -- different
            # cohorts can use different isoform IDs for the same gene, and
            # picking arbitrarily among them would make Stage 2 lookups
            # silently miss this protein in whichever cohorts use a
            # different one). Falls back to our own canonical-isoform pick
            # only if this protein was never in any cohort's matrix at all.
            candidates = [(eid, abundance_ensp_counts[eid]) for eid in all_ensp_ids
                          if eid in abundance_ensp_counts]
            if candidates:
                candidates.sort(key=lambda x: (-x[1], x[0]))
                ensp_id = candidates[0][0]
                n_matched_to_abundance_id += 1
            else:
                ensp_id = canonical_ensp

            sequence = entry.get('sequence', {}).get('value', '')
            go_bp = extract_go_terms(entry, 'P')
            go_cc = extract_go_terms(entry, 'C')

            rows.append({
                'ENSP_ID': ensp_id,
                'UniProt_Accession': accession,
                'Gene': gene,
                'Annotation': annotation,
                'Sequence': sequence,
                'GO_Biological_Process': '; '.join(go_bp),
                'GO_Cellular_Component': '; '.join(go_cc),
            })

        print(f'Page {page}: fetched {n_fetched} so far, {len(rows)} kept')

        if args.max_pages is not None and page >= args.max_pages:
            print(f'Stopping after {page} pages (--max-pages)')
            break

        link = response.headers.get('Link', '')
        if 'rel="next"' in link:
            url = link.split('<')[1].split('>')[0]
        else:
            url = None

    df_proteins = pd.DataFrame(rows)
    df_proteins = df_proteins.drop_duplicates(subset=['ENSP_ID'])

    import os
    os.makedirs('data/full_proteome', exist_ok=True)
    output_path = 'data/full_proteome/protein_annotations.csv'
    df_proteins.to_csv(output_path, index=False)

    print()
    print(f'Total fetched from UniProt: {n_fetched}')
    print(f'Dropped (no FUNCTION annotation): {n_no_function}')
    print(f'Dropped (no Ensembl Protein xref): {n_no_ensembl}')
    print(f'Matched to an existing cohort abundance-matrix ENSP ID: {n_matched_to_abundance_id}')
    print(f'Total proteins saved: {len(df_proteins)} -> {output_path}')


if __name__ == '__main__':
    main()
