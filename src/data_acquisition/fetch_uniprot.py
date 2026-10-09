import pandas as pd
import requests
import time
import argparse


# Data-driven choice: the typical
# protein has ~4 GO terms per category and is untouched by this cap; it only
# bounds the long tail of heavily-studied proteins (up to 90+ terms) so they
# don't dominate the token budget purely from research attention. Cap=10 is
# the point where, on a real 299-protein sample, raising it further starts
# costing much more added BioBERT-truncation risk than the GO signal gained.
MAX_GO_TERMS_PER_CATEGORY = 10


def extract_go_terms(uniprot_entry, category_letter):
    """category_letter: 'P' (Biological Process) or 'C' (Cellular Component).
    GoTerm property values look like 'P:apoptotic process' -- the category
    letter prefix is stripped from the returned labels."""
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


parser = argparse.ArgumentParser()
parser.add_argument("--cohort", required=True)
args = parser.parse_args()

proteomics = pd.read_csv(f'data/cptac/{args.cohort}/proteomics_raw.csv', header=[0,1])
database_ID = proteomics.columns.get_level_values(1)[1:].astype(str).tolist()

print(database_ID[:10])

response = requests.post(
    'https://rest.uniprot.org/idmapping/run',
    data={
        'from': 'Ensembl_Protein',
        'to': 'UniProtKB',
        'ids': ','.join(database_ID)
    }
)

if not response.text:
    print(f"UniProt API returned empty response (status {response.status_code}). Wait a moment and retry.")
    exit(1)
job_id = response.json()['jobId']
print(f'Job ID: {job_id}')

all_results = []

url = f'https://rest.uniprot.org/idmapping/status/{job_id}'

while url:
    response = requests.get(url)
    page_data = response.json()

    if 'jobStatus' in page_data:
        print(f'Job status: {page_data["jobStatus"]}')
        time.sleep(5)
        continue

    all_results.extend(page_data.get('results', []))
    print(f'Fetched {len(all_results)} so far...')

    link = response.headers.get('Link', '')
    if 'rel="next"' in link:
        url = link.split('<')[1].split('>')[0]
    else:
        url = None


data = {'results': all_results}
print(f'Total: {len(data["results"])}')



results = requests.get(
    f'https://rest.uniprot.org/idmapping/uniprotkb/{job_id}',
    params={'fields': 'accession,gene_names,cc_function,sequence', 'format': 'json', 'size': 500}
)


rows = []
for entry in data['results']:
    ensp_id = entry ['from']
    uniprot = entry['to']

    accession = uniprot.get('primaryAccession', '')

    gene = ''
    if uniprot.get('genes'):
        gene = uniprot['genes'][0].get('geneName', {}).get('value', '')

    annotation = ''
    for comment in uniprot.get('comments', []):
        if comment.get('commentType') == 'FUNCTION':
            annotation = gene + ': ' + comment['texts'][0].get('value', '')
            break

    sequence = uniprot.get('sequence', {}).get('value', '')

    go_bp = extract_go_terms(uniprot, 'P')
    go_cc = extract_go_terms(uniprot, 'C')

    rows.append({
        'ENSP_ID': ensp_id,
        'UniProt_Accession': accession,
        'Gene': gene,
        'Annotation': annotation,
        'Sequence': sequence,
        'GO_Biological_Process': '; '.join(go_bp),
        'GO_Cellular_Component': '; '.join(go_cc),
    })

df_proteins = pd.DataFrame(rows)
df_proteins.to_csv(f"data/cptac/{args.cohort}/protein_annotations.csv", index=False)
print(f"Saved annotations for {len(df_proteins)} proteins to data/cptac/{args.cohort}/protein_annotations.csv")