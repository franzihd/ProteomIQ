import cptac
import os

lung = cptac.Luad()
proteomics = lung.get_proteomics('umich')
clinical = lung.get_clinical('mssm')

os.makedirs('data/lung_cancer/proteomics', exist_ok=True)
os.makedirs('data/lung_cancer/clinical',  exist_ok=True)

proteomics.to_csv('data/lung_cancer/proteomics/proteomics_raw.csv')
clinical.to_csv(  'data/lung_cancer/clinical/clinical_raw.csv')

print("Done!")
print("Proteomics shape:", proteomics.shape)
print("Clinical shape:  ", clinical.shape)
