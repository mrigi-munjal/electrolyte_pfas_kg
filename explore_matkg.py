import pandas as pd

# Load the triples CSV file
df = pd.read_csv('SUBRELOBJ.csv')

print(df.head())
relation_counts = df['Rel'].value_counts().reset_index()
relation_counts.columns = ['Relation_Type', 'Count']

# Split into components for clarity
relation_counts[['Tag1', 'Tag2']] = relation_counts['Relation_Type'].str.split('-', expand=True)

# Display the unique relations with counts

print("Unique Relation Types with Counts:")
print(relation_counts)

sample_10_rows = df.sample(10, random_state=42)
print("Sample of 10 rows from the dataset:")
print(sample_10_rows)

# Identify unique CHM entries in the Subject column (Rel startswith 'CHM-')
mask_subject_chm = df['Rel'].str.startswith('CHM-')
unique_subjects = sorted(df.loc[mask_subject_chm, 'Subject'].unique())

# Identify unique CHM entries in the Object column (Rel endswith '-CHM')
mask_object_chm = df['Rel'].str.endswith('-CHM')
unique_objects = sorted(df.loc[mask_object_chm, 'Object'].unique())

# Save full lists to CSV
subject_csv_path = 'output/unique_CHM_subject.csv'
object_csv_path = 'output/unique_CHM_object.csv'
pd.DataFrame({'CHM_in_subject': unique_subjects}).to_csv(subject_csv_path, index=False)
pd.DataFrame({'CHM_in_object': unique_objects}).to_csv(object_csv_path, index=False)

# Display counts
summary = pd.DataFrame({
    'Role': ['Subject', 'Object'],
    'Unique_CHM_count': [len(unique_subjects), len(unique_objects)]
})
print("Summary of Unique CHM Counts:")
print(summary)

# Display first 10 entries of each as examples
examples = pd.DataFrame({
    'Subject_CHM_First_10': unique_subjects[:10],
    'Object_CHM_First_10': unique_objects[:10]
})
print("First 10 Unique CHM Entries in Subject and Object:")
print(examples)

import re

# Stream through the CSV to gather unique CHM entries without loading entire file
subjects_chm_set = set()
objects_chm_set = set()
terms = {
    'TFMSA': ['TFMSA'],
    'TFSI / LiTFSI': ['TFSI', 'LiTFSI'],
    'FAP / LiFAP': ['FAP', 'LiFAP'],
    'tris(trifluoromethylsulphonyl) methanide (TTM)': ['tris\\(trifluoromethylsulphonyl\\) methanide', 'TTM'],
    'Li-Triflate / LiTF / LiOTF': ['Li-Triflate', 'LiTF', 'LiOTF'],
    'LIBETI': ['LIBETI'],
    'LiTDI': ['LiTDI'],
    'TFEB (Tris(2,2,2-trifluoroethyl)borate)': ['TFEB', 'Tris\\(2,2,2-trifluoroethyl\\)borate'],
    'TFT (Trifluorotoluene)': ['TFT', 'Trifluorotoluene'],
    'NaTFSI': ['NaTFSI']
}

# Read in chunks to reduce memory usage
for chunk in pd.read_csv('SUBRELOBJ.csv', usecols=['Rel','Subject','Object'], chunksize=200000):
    # CHM in subject
    mask_sub = chunk['Rel'].str.startswith('CHM-', na=False)
    subjects_chm_set.update(chunk.loc[mask_sub, 'Subject'].dropna().unique())
    # CHM in object
    mask_obj = chunk['Rel'].str.endswith('-CHM', na=False)
    objects_chm_set.update(chunk.loc[mask_obj, 'Object'].dropna().unique())

# Sort lists
subjects_chm = sorted(subjects_chm_set)
objects_chm = sorted(objects_chm_set)

# Search for matches in these unique lists
rows = []
for term_name, patterns in terms.items():
    subj_matches = [s for s in subjects_chm if any(re.search(p, s, re.IGNORECASE) for p in patterns)]
    obj_matches  = [o for o in objects_chm  if any(re.search(p, o, re.IGNORECASE) for p in patterns)]
    rows.append({
        'Material Term': term_name,
        'Matches in SUBJECT CHM': '; '.join(subj_matches) or '—',
        'Matches in OBJECT CHM' : '; '.join(obj_matches)  or '—'
    })

result_df = pd.DataFrame(rows)

#tools.display_dataframe_to_user("Matches for Requested Materials", result_df)
print("Matches for Requested Materials:")
print(result_df)



import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from math import pi, cos, sin
# import ace_tools as tools

# Load the filtered triples
df = pd.read_csv('cleaned_fluorine_CHM_triples_v3.csv')

# Regex pattern to catch TFSI synonyms
pattern = re.compile(r"trifluoromethylsulfonyl|tfsi", re.IGNORECASE)

# Identify all unique synonym entries in Subject and Object
syn_subjects = df.loc[df['Subject'].str.contains(pattern, na=False), 'Subject'].unique()
syn_objects  = df.loc[df['Object'] .str.contains(pattern, na=False), 'Object' ].unique()
synonyms = sorted(set(list(syn_subjects) + list(syn_objects)))

# Display the synonyms we are mapping to the central node
# tools.display_dataframe_to_user(
#     "Detected TFSI Synonyms (mapped to central node)",
#     pd.DataFrame({'Synonym': synonyms})
# )

print("Detected TFSI Synonyms (mapped to central node):")
print(pd.DataFrame({'Synonym': synonyms}))


# Central node label
central = "Bis(Trifluoromethylsulfonyl)imide"
edges = []

# Build edges: central → neighbor for subject matches
for _, row in df[df['Subject'].str.contains(pattern, na=False)].iterrows():
    target = row['Object']
    # skip if target is also a synonym
    if pattern.search(target):
        continue
    edges.append((central, target, row['Rel']))

# Build edges: central → neighbor for object matches (swap)
for _, row in df[df['Object'].str.contains(pattern, na=False)].iterrows():
    neighbor = row['Subject']
    if pattern.search(neighbor):
        continue
    edges.append((central, neighbor, row['Rel']))

# Create DataFrame, drop duplicates
edges_df = pd.DataFrame(edges, columns=['source', 'target', 'relation']).drop_duplicates()

# Save edges CSV
edges_path = 'output/edges/tfsi_edges_final.csv'
edges_df.to_csv(edges_path, index=False)

# Display a sample of edges
# tools.display_dataframe_to_user("Sample of Final Edges", edges_df.sample(10, random_state=1))

print("Sample of Final Edges:")
print(edges_df.sample(10, random_state=1))

# Build and plot the graph
G = nx.DiGraph()
G.add_node(central)
neighbors = edges_df['target'].unique().tolist()

for nbr in neighbors:
    G.add_node(nbr)
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'])

# Layout: circle around central
pos = {central: (0, 0)}
n = len(neighbors)
r = 5
for i, node in enumerate(neighbors):
    angle = 2 * pi * i / n
    pos[node] = (r * cos(angle), r * sin(angle))

plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=1200, font_size=8, arrowsize=20)
edge_labels = {(u, v): rel for u, v, rel in edges_df.itertuples(index=False)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
plt.title("Neighborhood of Bis(Trifluoromethylsulfonyl)imide")
plt.axis('off')
plt.tight_layout()
plt.savefig('output/figures/tfsi_network_plot.png', dpi=300, bbox_inches='tight')
plt.close()

edges_path = 'output/edges/tfsi_edges_final.csv'

# Download all unique CHMs from SUBRELOBJ.csv (overall file)
df_full = pd.read_csv('SUBRELOBJ.csv', usecols=['Subject', 'Object'])
unique_chms = pd.Series(pd.concat([df_full['Subject'], df_full['Object']]).unique())
unique_chms = unique_chms[unique_chms.str.contains('CHM', na=False)]
unique_chms.to_csv('output/all_unique_CHMs_overall.csv', index=False, header=['Unique_CHM'])
print(f"Saved all unique CHMs from SUBRELOBJ.csv to output/all_unique_CHMs_overall.csv. Total: {len(unique_chms)}")


