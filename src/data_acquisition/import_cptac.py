import cptac
import os
import pandas

# This script imports the proteomics and clinical data for each of the 10 CPTAC cohorts and saves them as csv files in the data directory. 
# If the files already exist, it skips the download.

# Define the CPTAC cohorts and their corresponding classes
COHORTS = {
    'luad':  cptac.Luad,
    'brca':  cptac.Brca,
    'ccrcc': cptac.Ccrcc,
    'coad':  cptac.Coad,
    'hnscc': cptac.Hnscc,
    'lscc':  cptac.Lscc,
    'ucec':  cptac.Ucec,
    'gbm':   cptac.Gbm,
    'pdac':  cptac.Pdac,
    'ov':    cptac.Ov,
}

for cancer_name, cptac_class in COHORTS.items():
    # define the paths to the proteomics and clinical data
    proteomics_path = f'data/{cancer_name}/proteomics_raw.csv'
    clinical_path = f'data/{cancer_name}/clinical.csv'

    # check, whether the files already exist, if not, download the data and save it as csv
    proteomics_exists = os.path.exists(proteomics_path)
    clinical_exists = os.path.exists(clinical_path)

    if proteomics_exists and clinical_exists:
        print(f'{cancer_name}: Data already exists, skipping download')
        continue

    try:
        # instantiate the cptac class for this cohort
        ds = cptac_class()
        
        # list the data sources available for this cohort
        sources_df = ds.list_data_sources()


        proteomics = None
        # try each proteomics source until one works
        proteomics_rows = sources_df[sources_df['Data type'] == 'proteomics']
        for source in proteomics_rows['Available sources'].values[0]:
            try:
                proteomics = ds.get_proteomics(source=source)
                proteomics_source = source
                break
            except Exception:
                print(f'  source {source} failed, trying next...')

        if proteomics is None:
            print(f'{cancer_name.upper()}: No working proteomics source found, skipping')
            continue


        os.makedirs(os.path.dirname(proteomics_path), exist_ok=True)
        proteomics.to_csv(proteomics_path)
        print(f'{proteomics.shape[0]} samples, {proteomics.shape[1]} proteins -> saved to {proteomics_path}')

        # select the clinical source, which is the one that contains the clinical data
        clinical_rows = sources_df[sources_df['Data type'] == 'clinical']
        clinical_source = clinical_rows['Available sources'].values[0][0]

        print(f'{cancer_name.upper()}: Downloading clinical data from source {clinical_source}')
        # download the clinical data
        clinical = ds.get_clinical(source=clinical_source)
        os.makedirs(os.path.dirname(clinical_path), exist_ok=True)
        clinical.to_csv(clinical_path)
        print(f'{clinical.shape[0]} samples, {clinical.shape[1]} clinical variables -> saved to {clinical_path}')

    except Exception as e:
        print(f'Error downloading data for {cancer_name}: {e}')
        