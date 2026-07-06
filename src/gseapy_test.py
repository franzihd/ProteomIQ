import gseapy as gp
import csv

names = gp.get_library_name()
#print(names)

# Alle Pathways + ihre Protein-Sets laden
kegg = gp.get_library(name='KEGG_2021_Human')


# Wie viele Pathways?
print(kegg.keys())

with open('data/evaluation/kegg_pathways.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Pathway Name', 'Number of Proteins'])
    for pathway_name, proteins in kegg.items():
        writer.writerow([pathway_name, len(proteins)])

proteins_pancreatic_cancer = kegg['Pancreatic cancer']
#print("member proteins (gene name format) in pancreatic cancer pathway")
#print(proteins_pancreatic_cancer)


# Alle Namen 
#print(list(kegg.keys()))