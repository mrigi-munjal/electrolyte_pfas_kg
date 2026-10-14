import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
from math import pi, cos, sin

# Load dataset
df = pd.read_csv('cleaned_fluorine_CHM_triples_v3.csv')

def process_chm(label, pattern_str, filename):
    pattern = re.compile(pattern_str, re.IGNORECASE)
    # Detect synonyms in Subject/Object
    subs = df['Subject'][df['Subject'].str.contains(pattern, na=False)].unique().tolist()
    objs = df['Object' ][df['Object' ].str.contains(pattern, na=False)].unique().tolist()
    synonyms = sorted(set(subs + objs))
    print(f"Synonyms mapped to '{label}': {synonyms}")
    
    # Build edges
    edges = []
    # Outgoing: subject matches
    for _, row in df[df['Subject'].str.contains(pattern, na=False)].iterrows():
        tgt = row['Object']
        if pattern.search(tgt):  # skip if also a synonym
            continue
        edges.append((label, tgt, row['Rel']))
    # Incoming: object matches (swap)
    for _, row in df[df['Object'].str.contains(pattern, na=False)].iterrows():
        nbr = row['Subject']
        if pattern.search(nbr):
            continue
        edges.append((label, nbr, row['Rel']))
    
    # Deduplicate and save
    edges_df = pd.DataFrame(edges, columns=['source','target','relation']).drop_duplicates()
    edges_df.to_csv(f'output/edges/{filename}', index=False)
    print(f"Edges for {label}: {len(edges_df)} (saved to output/edges/{filename})")

    # Plot
    G = nx.DiGraph()
    G.add_node(label)
    neighbors = edges_df['target'].unique().tolist()
    for nbr in neighbors:
        G.add_node(nbr)
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # Layout: circle
    pos = {label: (0, 0)}
    n = len(neighbors)
    r = 5
    for i, node in enumerate(neighbors):
        angle = 2 * pi * i / max(n,1)
        pos[node] = (r * cos(angle), r * sin(angle))
    
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, node_size=1000, font_size=7, arrowsize=15)
    edge_labels = {(u, v): rel for u, v, rel in edges_df.itertuples(index=False)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
    plt.title(f"Neighborhood of {label}")
    plt.axis('off')
    # plt.show()
    plt.savefig(f'output/figures/{label.replace(" ", "_")}_neighborhood.png', bbox_inches='tight')
    plt.close()
    print("\n" + "-"*60 + "\n")

# 1) Dimethyl Formamide
process_chm(
    label="Dimethyl Formamide",
    pattern_str=r"dimethyl[\s-]*formamide",
    filename="edges_DimethylFormamide.csv"
)

# Load and filter dataset
df = pd.read_csv('SUBRELOBJ.csv')
rels_to_remove = ['CHM-CMT','CHM-SMT','CHM-SPL','CMT-CHM','SMT-CHM','SPL-CHM']
df = df[~df['Rel'].isin(rels_to_remove)].reset_index(drop=True)

# Function for a single CHM
# def process_chm(label, pattern_str, filename):
#     pattern = re.compile(pattern_str, re.IGNORECASE)
#     # synonyms
#     subs = df['Subject'][df['Subject'].str.contains(pattern, na=False)].unique().tolist()
#     objs = df['Object'][df['Object'].str.contains(pattern, na=False)].unique().tolist()
#     synonyms = sorted(set(subs + objs))
#     print(f"Synonyms mapped to '{label}': {synonyms}")
#     # edges
#     edges = []
#     for _, row in df[df['Subject'].str.contains(pattern, na=False)].iterrows():
#         tgt = row['Object']
#         if pattern.search(tgt): continue
#         edges.append((label, tgt, row['Rel']))
#     for _, row in df[df['Object'].str.contains(pattern, na=False)].iterrows():
#         tgt = row['Subject']
#         if pattern.search(tgt): continue
#         edges.append((label, tgt, row['Rel']))
#     edges_df = pd.DataFrame(edges, columns=['source','target','relation']).drop_duplicates()
#     edges_df.to_csv(f'/mnt/data/{filename}', index=False)
#     print(f"Edges for {label}: {len(edges_df)} saved to /mnt/data/{filename}")
#     # plot
#     G = nx.DiGraph()
#     G.add_node(label)
#     neighbors = edges_df['target'].unique().tolist()
#     for nbr in neighbors: G.add_node(nbr)
#     for _, r in edges_df.iterrows(): G.add_edge(r['source'], r['target'])
#     # layout
#     pos = {label:(0,0)}
#     n = len(neighbors)
#     r = 5
#     for i,nbr in enumerate(neighbors):
#         ang = 2*pi*i/max(n,1)
#         pos[nbr] = (r*cos(ang), r*sin(ang))
#     plt.figure(figsize=(6,6))
#     nx.draw(G,pos,with_labels=True,node_size=1000,font_size=7,arrowsize=15)
#     edge_labels = {(u,v):rel for u,v,rel in edges_df.itertuples(index=False)}
#     nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=5)
#     plt.title(f"Neighborhood of {label}")
#     plt.axis('off')
#     plt.show()

# Process Dimethyl Formamide
# process_chm("Dimethyl Formamide", r"dimethyl[\s-]*formamide", "edges_DimethylFormamide.csv")

def generate_edges_and_preview(label, pattern_str, filename, preview_n=20):
    pattern = re.compile(pattern_str, re.IGNORECASE)
    # Detect synonyms
    subs = df['Subject'][df['Subject'].str.contains(pattern, na=False)].unique().tolist()
    objs = df['Object'][df['Object'].str.contains(pattern, na=False)].unique().tolist()
    synonyms = sorted(set(subs + objs))
    print(f"Synonyms mapped to '{label}': {synonyms}\n")
    
    # Build full edges list
    edges = []
    for _, row in df[df['Subject'].str.contains(pattern, na=False)].iterrows():
        tgt = row['Object']
        if pattern.search(tgt): continue
        edges.append((label, tgt, row['Rel']))
    for _, row in df[df['Object'].str.contains(pattern, na=False)].iterrows():
        tgt = row['Subject']
        if pattern.search(tgt): continue
        edges.append((label, tgt, row['Rel']))
    
    # Deduplicate and save
    edges_df = pd.DataFrame(edges, columns=['source','target','relation']).drop_duplicates()
    edges_df.to_csv(f'output/edges/{filename}', index=False)
    print(f"Saved {len(edges_df)} total edges to output/edges/{filename}\n")

    # Preview sample
    preview = edges_df.sample(min(preview_n, len(edges_df)), random_state=1)
    neighbors = preview['target'].unique().tolist()
    G = nx.DiGraph()
    G.add_node(label)
    for nbr in neighbors: G.add_node(nbr)
    for _, r in preview.iterrows(): G.add_edge(r['source'], r['target'])
    
    # Layout
    pos = {label: (0, 0)}
    n = len(neighbors)
    r = 5
    for i, nbr in enumerate(neighbors):
        angle = 2 * pi * i / max(n,1)
        pos[nbr] = (r * cos(angle), r * sin(angle))
    
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, node_size=800, font_size=7, arrowsize=15)
    edge_labels = {(u, v): rel for u, v, rel in preview.itertuples(index=False)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
    plt.title(f"Preview Neighborhood of {label}")
    plt.axis('off')
    plt.show()
    print("\n" + "-"*60 + "\n")

# Generate for Dimethyl Formamide
generate_edges_and_preview("Dimethyl Formamide", r"dimethyl[\s-]*formamide", "edges_DimethylFormamide.csv")


def plot_edges(csv_path, central_label, sample_size=None):
    # Load edges
    edges = pd.read_csv(csv_path)
    if sample_size:
        edges = edges.sample(sample_size, random_state=1)
    G = nx.DiGraph()
    G.add_node(central_label)
    for _, r in edges.iterrows():
        G.add_edge(r.source, r.target)
    # Circular layout
    neighbors = edges.target.unique().tolist()
    pos = {central_label:(0,0)}
    import math
    for i,n in enumerate(neighbors):
        angle = 2*math.pi*i/max(len(neighbors),1)
        pos[n] = (5*math.cos(angle), 5*math.sin(angle))
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, node_size=800, font_size=7, arrowsize=15)
    plt.title(f"Neighborhood of {central_label}")
    plt.axis('off')
    plt.show()

# Example: plot a 20-node sample for readability
plot_edges("output/edges/edges_DimethylFormamide.csv", "Dimethyl Formamide", sample_size=20)
# plot_edges("edges_Chloroform.csv",          "Chloroform",          sample_size=20)

input_path = 'SUBRELOBJ.csv'
rels_to_remove = {
    'CHM-CMT','CHM-SMT','CHM-SPL',
    'CMT-CHM','SMT-CHM','SPL-CHM'
}

# Patterns for the two CHMs
pattern_df   = re.compile(r"(?i)dimethyl[\s-]*formamide")
pattern_tfsi = re.compile(r"(?i)trifluoromethylsulfonyl|tfsi")

# Sets to accumulate neighbors
targets_df   = set()
targets_tfsi = set()

# Process in chunks
for chunk in pd.read_csv(input_path, chunksize=200000, usecols=['Subject','Object','Rel']):
    # Drop unwanted relation types
    chunk = chunk[~chunk['Rel'].isin(rels_to_remove)]

    # Dimethyl Formamide outgoing
    df_out = chunk[
        chunk['Subject'].str.contains(pattern_df, na=False)
    ]
    for obj in df_out['Object']:
        if not pattern_df.search(str(obj)):
            targets_df.add(obj)

    # Dimethyl Formamide incoming
    df_in = chunk[
        chunk['Object'].str.contains(pattern_df, na=False)
    ]
    for subj in df_in['Subject']:
        if not pattern_df.search(str(subj)):
            targets_df.add(subj)

    # TFSI outgoing
    tf_out = chunk[
        chunk['Subject'].str.contains(pattern_tfsi, na=False)
    ]
    for obj in tf_out['Object']:
        if not pattern_tfsi.search(str(obj)):
            targets_tfsi.add(obj)

    # TFSI incoming
    tf_in = chunk[
        chunk['Object'].str.contains(pattern_tfsi, na=False)
    ]
    for subj in tf_in['Subject']:
        if not pattern_tfsi.search(str(subj)):
            targets_tfsi.add(subj)

# Compute intersection, union, and Jaccard
intersection = targets_df & targets_tfsi
union = targets_df | targets_tfsi
jaccard_score = len(intersection) / len(union) if union else 0.0

print(f"Dimethyl Formamide targets: {len(targets_df)}")
print(f"TFSI targets: {len(targets_tfsi)}")
print(f"Intersection size: {len(intersection)}")
print(f"Union size: {len(union)}")
print(f"Jaccard similarity: {jaccard_score:.4f}")


# 1) Load the cleaned triples (v3)
df = pd.read_csv('cleaned_fluorine_CHM_triples_v3.csv')

# 2) Identify the CHM material in each row
df['CHM_value'] = df.apply(
    lambda r: r['Subject'] if r['Rel'].startswith('CHM-') else r['Object'],
    axis=1
)

# 3) Find the top 25 CHMs by count
top25 = df['CHM_value'].value_counts().head(25).index.tolist()

# 4) Restrict to the allowed relation types (now including CHM-CHM)
allowed = {
    'CHM-CHM',
    'PRO-CHM','CHM-PRO',
    'APL-CHM','CHM-APL',
    'DSC-CHM','CHM-DSC',
}
mask_allowed = df['Rel'].isin(allowed)

# 5) Build neighbor groups in bulk
out = (
    df[mask_allowed & df['Rel'].str.startswith('CHM-')]
      .groupby('CHM_value')['Object']
      .apply(set)
      .to_dict()
)
inc = (
    df[mask_allowed & df['Rel'].str.endswith('-CHM')]
      .groupby('CHM_value')['Subject']
      .apply(set)
      .to_dict()
)

# 6) TFSI neighbor set (now including CHM-CHM relations)
tfsi_re = re.compile(r'(?i)trifluoromethylsulfonyl|tfsi')
tfsi_df = df[df['CHM_value'].str.contains(tfsi_re) & mask_allowed]
tfsi_neigh = set(tfsi_df[tfsi_df['Rel'].str.startswith('CHM-')]['Object']) \
           | set(tfsi_df[tfsi_df['Rel'].str.endswith('-CHM')]['Subject'])

# 7) Compute Jaccard for each of the top 25
results = []
for chm in top25:
    this_neigh = out.get(chm, set()) | inc.get(chm, set())
    inter = this_neigh & tfsi_neigh
    uni   = this_neigh | tfsi_neigh
    jacc  = len(inter)/len(uni) if uni else 0.0
    results.append((chm, round(jacc,4)))

# 8) Print the sorted list
print(f"{'CHM':<40}  Jaccard")
print("-"*52)
for chm, score in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{chm:<40}  {score:>0.4f}")

# 9) WEIGHTED JACCARD CALCULATION
print("\n" + "="*60)
print("WEIGHTED JACCARD SIMILARITY (by relation frequency)")
print("="*60)

# Load the dataset with frequency column
df_weighted = pd.read_csv('cleaned_fluorine_CHM_triples_v3.csv')

# Check if frequency column exists (common names: 'Count', 'Frequency', '# of relations', etc.)
freq_col = None
possible_freq_cols = ['Count', 'Frequency', '# of relations', 'count', 'frequency', 'weight']
for col in possible_freq_cols:
    if col in df_weighted.columns:
        freq_col = col
        break

if freq_col is None:
    print("Warning: No frequency column found. Available columns:")
    print(df_weighted.columns.tolist())
    print("Using frequency = 1 for all relations")
    df_weighted['frequency'] = 1
    freq_col = 'frequency'
else:
    print(f"Using frequency column: '{freq_col}'")

# Identify the CHM material in each row
df_weighted['CHM_value'] = df_weighted.apply(
    lambda r: r['Subject'] if r['Rel'].startswith('CHM-') else r['Object'],
    axis=1
)

# Restrict to allowed relation types
mask_allowed_weighted = df_weighted['Rel'].isin(allowed)
df_weighted = df_weighted[mask_allowed_weighted]

# Build weighted neighbor dictionaries
def build_weighted_neighbors(df, chm_col, neighbor_col, freq_col):
    """Build dictionary of {chm: {neighbor: total_weight}}"""
    result = {}
    grouped = df.groupby(chm_col)
    for chm, group in grouped:
        neighbors = {}
        for _, row in group.iterrows():
            neighbor = row[neighbor_col]
            weight = row[freq_col]
            if neighbor in neighbors:
                neighbors[neighbor] += weight
            else:
                neighbors[neighbor] = weight
        result[chm] = neighbors
    return result

# Build weighted outgoing neighbors (CHM -> Object)
out_weighted = build_weighted_neighbors(
    df_weighted[df_weighted['Rel'].str.startswith('CHM-')],
    'CHM_value', 'Object', freq_col
)

# Build weighted incoming neighbors (Subject -> CHM)
inc_weighted = build_weighted_neighbors(
    df_weighted[df_weighted['Rel'].str.endswith('-CHM')],
    'CHM_value', 'Subject', freq_col
)

# TFSI weighted neighbor set
tfsi_weighted_neighbors = {}
tfsi_df_weighted = df_weighted[df_weighted['CHM_value'].str.contains(tfsi_re)]

# Collect TFSI neighbors with weights
for _, row in tfsi_df_weighted.iterrows():
    if row['Rel'].startswith('CHM-'):
        neighbor = row['Object']
    else:  # ends with '-CHM'
        neighbor = row['Subject']
    
    weight = row[freq_col]
    if neighbor in tfsi_weighted_neighbors:
        tfsi_weighted_neighbors[neighbor] += weight
    else:
        tfsi_weighted_neighbors[neighbor] = weight

def weighted_jaccard(dict1, dict2):
    """Calculate weighted Jaccard similarity between two neighbor dictionaries"""
    all_neighbors = set(dict1.keys()) | set(dict2.keys())
    
    if not all_neighbors:
        return 0.0
    
    intersection_weight = 0.0
    union_weight = 0.0
    
    for neighbor in all_neighbors:
        weight1 = dict1.get(neighbor, 0)
        weight2 = dict2.get(neighbor, 0)
        
        # Intersection: minimum weight
        intersection_weight += min(weight1, weight2)
        # Union: maximum weight
        union_weight += max(weight1, weight2)
    
    return intersection_weight / union_weight if union_weight > 0 else 0.0

# Compute weighted Jaccard for each of the top 25
weighted_results = []
for chm in top25:
    # Combine outgoing and incoming neighbors for this CHM
    chm_neighbors = {}
    
    # Add outgoing neighbors
    if chm in out_weighted:
        for neighbor, weight in out_weighted[chm].items():
            chm_neighbors[neighbor] = chm_neighbors.get(neighbor, 0) + weight
    
    # Add incoming neighbors
    if chm in inc_weighted:
        for neighbor, weight in inc_weighted[chm].items():
            chm_neighbors[neighbor] = chm_neighbors.get(neighbor, 0) + weight
    
    # Calculate weighted Jaccard with TFSI
    weighted_jacc = weighted_jaccard(chm_neighbors, tfsi_weighted_neighbors)
    weighted_results.append((chm, round(weighted_jacc, 4)))

# Print the weighted results
print(f"\n{'CHM':<40}  Weighted Jaccard")
print("-"*52)
for chm, score in sorted(weighted_results, key=lambda x: x[1], reverse=True):
    print(f"{chm:<40}  {score:>0.4f}")

# Compare with unweighted results
print(f"\n{'CHM':<40}  Unweighted  Weighted   Difference")
print("-"*70)
unweighted_dict = {chm: score for chm, score in results}
weighted_dict = {chm: score for chm, score in weighted_results}

for chm in top25:
    unw = unweighted_dict.get(chm, 0.0)
    w = weighted_dict.get(chm, 0.0)
    diff = w - unw
    print(f"{chm:<40}  {unw:>0.4f}      {w:>0.4f}     {diff:>+0.4f}")

# Save results to CSV
results_df = pd.DataFrame({
    'CHM': top25,
    'Unweighted_Jaccard': [unweighted_dict.get(chm, 0.0) for chm in top25],
    'Weighted_Jaccard': [weighted_dict.get(chm, 0.0) for chm in top25]
})
results_df['Difference'] = results_df['Weighted_Jaccard'] - results_df['Unweighted_Jaccard']
results_df = results_df.sort_values('Weighted_Jaccard', ascending=False)
results_df.to_csv('output/jaccard_comparison_results.csv', index=False)
print(f"\nResults saved to output/jaccard_comparison_results.csv")

# Summary statistics
print(f"\nSUMMARY STATISTICS:")
print(f"Average unweighted Jaccard: {results_df['Unweighted_Jaccard'].mean():.4f}")
print(f"Average weighted Jaccard: {results_df['Weighted_Jaccard'].mean():.4f}")
print(f"Average difference: {results_df['Difference'].mean():.4f}")
print(f"Max positive change: {results_df['Difference'].max():.4f}")
print(f"Max negative change: {results_df['Difference'].min():.4f}")


# 10) SYNONYM ANALYSIS FOR TOP 25 CHMs
print("\n" + "="*60)
print("SYNONYM ANALYSIS FOR TOP 25 CHMs")
print("="*60)

def find_synonyms_for_chm(chm_name, df):
    """Find all variations/synonyms of a CHM in the dataset"""
    # Get all CHM values that might be variations of this chemical
    all_chm_values = df['CHM_value'].unique()
    
    # Simple matching strategies
    synonyms = set()
    chm_lower = chm_name.lower()
    
    for candidate in all_chm_values:
        candidate_lower = candidate.lower()
        
        # Exact match
        if candidate_lower == chm_lower:
            synonyms.add(candidate)
            continue
            
        # Check if one contains the other (for abbreviations)
        if len(chm_lower) > 3 and len(candidate_lower) > 3:
            if chm_lower in candidate_lower or candidate_lower in chm_lower:
                synonyms.add(candidate)
                continue
        
        # Check for common chemical name patterns
        # Remove common prefixes/suffixes and spaces/hyphens for comparison
        clean_chm = re.sub(r'[\s\-_]', '', chm_lower)
        clean_candidate = re.sub(r'[\s\-_]', '', candidate_lower)
        
        if clean_chm == clean_candidate and clean_chm != '':
            synonyms.add(candidate)
    
    return sorted(synonyms)

def create_synonym_pattern(synonyms):
    """Create a regex pattern from a list of synonyms"""
    if not synonyms:
        return None
    
    # Escape special regex characters and join with OR
    escaped_synonyms = [re.escape(syn) for syn in synonyms]
    pattern_str = '|'.join(escaped_synonyms)
    return re.compile(f'(?i){pattern_str}')

# Analyze synonyms for each of the top 25
synonym_analysis = {}
for i, chm in enumerate(top25, 1):
    print(f"\n{i:2d}. {chm}")
    print("-" * (len(chm) + 4))
    
    # Find synonyms
    synonyms = find_synonyms_for_chm(chm, df)
    
    # Count occurrences of each synonym
    synonym_counts = {}
    for syn in synonyms:
        count = (df['CHM_value'] == syn).sum()
        synonym_counts[syn] = count
    
    # Sort by count
    sorted_synonyms = sorted(synonym_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Found {len(synonyms)} synonym(s):")
    for syn, count in sorted_synonyms:
        print(f"   - '{syn}' ({count} occurrences)")
    
    # Create regex pattern if multiple synonyms
    if len(synonyms) > 1:
        pattern = create_synonym_pattern(synonyms)
        print(f"   Suggested regex: r'(?i){pattern.pattern[4:]}'")  # Remove (?i) prefix
    else:
        pattern = None
        print(f"   No additional synonyms found (using exact match)")
    
    # Store analysis
    synonym_analysis[chm] = {
        'synonyms': synonyms,
        'counts': synonym_counts,
        'pattern': pattern,
        'total_occurrences': sum(synonym_counts.values())
    }

# Summary of synonym usage
print(f"\n" + "="*60)
print("SYNONYM USAGE SUMMARY")
print("="*60)

chemicals_with_synonyms = []
chemicals_without_synonyms = []

for chm, analysis in synonym_analysis.items():
    if len(analysis['synonyms']) > 1:
        chemicals_with_synonyms.append((chm, len(analysis['synonyms']), analysis['total_occurrences']))
    else:
        chemicals_without_synonyms.append((chm, analysis['total_occurrences']))

print(f"\nChemicals WITH multiple synonyms ({len(chemicals_with_synonyms)}):")
print(f"{'Chemical':<40} {'Synonyms':<10} {'Total Occ.':<12}")
print("-" * 65)
for chm, syn_count, total_occ in sorted(chemicals_with_synonyms, key=lambda x: x[1], reverse=True):
    print(f"{chm:<40} {syn_count:<10} {total_occ:<12}")

print(f"\nChemicals WITHOUT multiple synonyms ({len(chemicals_without_synonyms)}):")
print(f"{'Chemical':<40} {'Occurrences':<12}")
print("-" * 55)
for chm, total_occ in sorted(chemicals_without_synonyms, key=lambda x: x[1], reverse=True):
    print(f"{chm:<40} {total_occ:<12}")

# Save synonym analysis to file
synonym_df_data = []
for chm, analysis in synonym_analysis.items():
    for syn, count in analysis['counts'].items():
        synonym_df_data.append({
            'Primary_CHM': chm,
            'Synonym': syn,
            'Count': count,
            'Has_Multiple_Synonyms': len(analysis['synonyms']) > 1
        })

synonym_df = pd.DataFrame(synonym_df_data)
synonym_df.to_csv('output/top25_synonym_analysis.csv', index=False)
print(f"\nDetailed synonym analysis saved to output/top25_synonym_analysis.csv")

# Create patterns for potential future use
top25_patterns = {}
for chm, analysis in synonym_analysis.items():
    if analysis['pattern']:
        top25_patterns[chm] = analysis['pattern']
    else:
        # Create exact match pattern
        top25_patterns[chm] = re.compile(f'(?i)^{re.escape(chm)}$')

print(f"\nCreated regex patterns for all top 25 chemicals.")
print("These patterns can be used for more accurate neighbor identification in future analyses.")

# 11) TFSI SYNONYM ANALYSIS
print("\n" + "="*60)
print("TFSI SYNONYM ANALYSIS")
print("="*60)

# Use the same synonym analysis logic for TFSI
print("Analyzing TFSI synonyms using the same logic as top 25 CHMs...")

# First, let's see what TFSI-related chemicals are in our dataset
tfsi_candidates = []
all_chm_values = df['CHM_value'].unique()

# Look for anything containing "tfsi" or "trifluoromethylsulfonyl"
tfsi_pattern = re.compile(r'(?i)trifluoromethylsulfonyl|tfsi')
for chm in all_chm_values:
    if tfsi_pattern.search(chm):
        tfsi_candidates.append(chm)

print(f"Found {len(tfsi_candidates)} TFSI-related chemicals in the dataset:")
for i, candidate in enumerate(sorted(tfsi_candidates), 1):
    count = (df['CHM_value'] == candidate).sum()
    print(f"   {i:2d}. '{candidate}' ({count} occurrences)")

# Count total TFSI occurrences
total_tfsi_occurrences = sum((df['CHM_value'] == candidate).sum() for candidate in tfsi_candidates)
print(f"\nTotal TFSI-related occurrences: {total_tfsi_occurrences}")

# Create comprehensive TFSI pattern
if tfsi_candidates:
    escaped_candidates = [re.escape(candidate) for candidate in tfsi_candidates]
    comprehensive_tfsi_pattern = re.compile(f'(?i)({"|".join(escaped_candidates)})')
    print(f"Comprehensive TFSI regex pattern created covering all {len(tfsi_candidates)} variations")
else:
    comprehensive_tfsi_pattern = tfsi_pattern
    print("Using original TFSI pattern as no additional variations found")

# 12) SYNONYM-COLLAPSED JACCARD CALCULATIONS
print("\n" + "="*60)
print("SYNONYM-COLLAPSED JACCARD CALCULATIONS")
print("="*60)
print("Calculating Jaccard similarities where all synonyms are treated as the same entity")

def get_synonym_collapsed_neighbors(chm_patterns, df, allowed_rels):
    """
    Get neighbors for CHMs, collapsing all synonyms into single entities
    Returns dict of {chm_name: set_of_neighbors}
    """
    neighbors_dict = {}
    
    for chm_name, pattern in chm_patterns.items():
        neighbors = set()
        
        # Find all rows where this CHM (or its synonyms) appears
        chm_mask = df['CHM_value'].str.contains(pattern, na=False)
        chm_df = df[chm_mask & df['Rel'].isin(allowed_rels)]
        
        # Get outgoing neighbors (CHM as subject)
        outgoing = chm_df[chm_df['Rel'].str.startswith('CHM-')]
        for _, row in outgoing.iterrows():
            neighbor = row['Object']
            # Don't add if neighbor is also a synonym of any CHM we're analyzing
            is_synonym = False
            for other_pattern in chm_patterns.values():
                if other_pattern.search(str(neighbor)):
                    is_synonym = True
                    break
            if not is_synonym:
                neighbors.add(neighbor)
        
        # Get incoming neighbors (CHM as object)
        incoming = chm_df[chm_df['Rel'].str.endswith('-CHM')]
        for _, row in incoming.iterrows():
            neighbor = row['Subject']
            # Don't add if neighbor is also a synonym of any CHM we're analyzing
            is_synonym = False
            for other_pattern in chm_patterns.values():
                if other_pattern.search(str(neighbor)):
                    is_synonym = True
                    break
            if not is_synonym:
                neighbors.add(neighbor)
        
        neighbors_dict[chm_name] = neighbors
        
    return neighbors_dict

# Create patterns for top 25 CHMs (using existing top25_patterns)
print("Using synonym-aware patterns for top 25 CHMs...")

# Add TFSI to the patterns
all_patterns = top25_patterns.copy()
all_patterns['TFSI'] = comprehensive_tfsi_pattern

# Get synonym-collapsed neighbors for all CHMs
collapsed_neighbors = get_synonym_collapsed_neighbors(all_patterns, df, allowed)

# Calculate synonym-collapsed Jaccard similarities with TFSI
tfsi_collapsed_neighbors = collapsed_neighbors['TFSI']
print(f"TFSI has {len(tfsi_collapsed_neighbors)} unique neighbors when synonyms are collapsed")
print(f"Total unique TFSI targets in the synonym-collapsed TFSI KG: {len(tfsi_collapsed_neighbors)}")
print(f"Total unique TFSI targets in the TFSI knowledge graph: {len(tfsi_collapsed_neighbors)}")

synonym_collapsed_results = []
for chm in top25:
    if chm in collapsed_neighbors:
        chm_neighbors = collapsed_neighbors[chm]
        intersection = chm_neighbors & tfsi_collapsed_neighbors
        union = chm_neighbors | tfsi_collapsed_neighbors
        jaccard = len(intersection) / len(union) if union else 0.0
        synonym_collapsed_results.append((chm, round(jaccard, 4)))
    else:
        synonym_collapsed_results.append((chm, 0.0))

# Print synonym-collapsed results
print(f"\n{'CHM':<40}  Synonym-Collapsed Jaccard")
print("-"*60)
for chm, score in sorted(synonym_collapsed_results, key=lambda x: x[1], reverse=True):
    print(f"{chm:<40}  {score:>0.4f}")

# 13) WEIGHTED + SYNONYM-COLLAPSED JACCARD CALCULATIONS
print("\n" + "="*60)
print("WEIGHTED + SYNONYM-COLLAPSED JACCARD CALCULATIONS")
print("="*60)
print("Calculating Jaccard similarities that are BOTH weighted AND synonym-collapsed")

def get_weighted_synonym_collapsed_neighbors(chm_patterns, df, allowed_rels, freq_col):
    """
    Get weighted neighbors for CHMs, collapsing all synonyms into single entities
    Returns dict of {chm_name: {neighbor: total_weight}}
    """
    neighbors_dict = {}
    
    for chm_name, pattern in chm_patterns.items():
        neighbors = {}
        
        # Find all rows where this CHM (or its synonyms) appears
        chm_mask = df['CHM_value'].str.contains(pattern, na=False)
        chm_df = df[chm_mask & df['Rel'].isin(allowed_rels)]
        
        # Get outgoing neighbors (CHM as subject)
        outgoing = chm_df[chm_df['Rel'].str.startswith('CHM-')]
        for _, row in outgoing.iterrows():
            neighbor = row['Object']
            weight = row[freq_col]
            
            # Don't add if neighbor is also a synonym of any CHM we're analyzing
            is_synonym = False
            for other_pattern in chm_patterns.values():
                if other_pattern.search(str(neighbor)):
                    is_synonym = True
                    break
            if not is_synonym:
                if neighbor in neighbors:
                    neighbors[neighbor] += weight
                else:
                    neighbors[neighbor] = weight
        
        # Get incoming neighbors (CHM as object)
        incoming = chm_df[chm_df['Rel'].str.endswith('-CHM')]
        for _, row in incoming.iterrows():
            neighbor = row['Subject']
            weight = row[freq_col]
            
            # Don't add if neighbor is also a synonym of any CHM we're analyzing
            is_synonym = False
            for other_pattern in chm_patterns.values():
                if other_pattern.search(str(neighbor)):
                    is_synonym = True
                    break
            if not is_synonym:
                if neighbor in neighbors:
                    neighbors[neighbor] += weight
                else:
                    neighbors[neighbor] = weight
        
        neighbors_dict[chm_name] = neighbors
        
    return neighbors_dict

# Get weighted + synonym-collapsed neighbors for all CHMs
weighted_collapsed_neighbors = get_weighted_synonym_collapsed_neighbors(all_patterns, df_weighted, allowed, freq_col)

# Calculate weighted + synonym-collapsed Jaccard similarities with TFSI
tfsi_weighted_collapsed_neighbors = weighted_collapsed_neighbors['TFSI']
print(f"TFSI has {len(tfsi_weighted_collapsed_neighbors)} unique neighbors (weighted + synonym-collapsed)")

weighted_collapsed_results = []
for chm in top25:
    if chm in weighted_collapsed_neighbors:
        chm_neighbors = weighted_collapsed_neighbors[chm]
        # Calculate weighted Jaccard with TFSI
        weighted_collapsed_jacc = weighted_jaccard(chm_neighbors, tfsi_weighted_collapsed_neighbors)
        weighted_collapsed_results.append((chm, round(weighted_collapsed_jacc, 4)))
    else:
        weighted_collapsed_results.append((chm, 0.0))

# Print weighted + synonym-collapsed results
print(f"\n{'CHM':<40}  Weighted+Collapsed Jaccard")
print("-"*60)
for chm, score in sorted(weighted_collapsed_results, key=lambda x: x[1], reverse=True):
    print(f"{chm:<40}  {score:>0.4f}")

# 14) COMPARE ALL FOUR APPROACHES
print("\n" + "="*60)
print("COMPARISON: ORIGINAL vs WEIGHTED vs SYNONYM-COLLAPSED vs WEIGHTED+COLLAPSED")
print("="*60)

# Create comprehensive comparison
comparison_data = []
synonym_collapsed_dict = {chm: score for chm, score in synonym_collapsed_results}
weighted_collapsed_dict = {chm: score for chm, score in weighted_collapsed_results}

for chm in top25:
    original = unweighted_dict.get(chm, 0.0)
    weighted = weighted_dict.get(chm, 0.0)
    collapsed = synonym_collapsed_dict.get(chm, 0.0)
    weighted_collapsed = weighted_collapsed_dict.get(chm, 0.0)
    
    comparison_data.append({
        'CHM': chm,
        'Original_Jaccard': original,
        'Weighted_Jaccard': weighted,
        'Synonym_Collapsed_Jaccard': collapsed,
        'Weighted_Collapsed_Jaccard': weighted_collapsed,
        'Original_vs_Weighted': weighted - original,
        'Original_vs_Collapsed': collapsed - original,
        'Original_vs_WeightedCollapsed': weighted_collapsed - original,
        'Weighted_vs_Collapsed': collapsed - weighted,
        'Weighted_vs_WeightedCollapsed': weighted_collapsed - weighted,
        'Collapsed_vs_WeightedCollapsed': weighted_collapsed - collapsed
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Weighted_Collapsed_Jaccard', ascending=False)

# Print detailed comparison
print(f"\n{'CHM':<20} {'Original':<8} {'Weighted':<8} {'Collapsed':<8} {'Wgt+Col':<8} {'O→W':<6} {'O→C':<6} {'O→WC':<6}")
print("-" * 85)
print("-" * 85)

for _, row in comparison_df.iterrows():
    print(f"{row['CHM']:<20} {row['Original_Jaccard']:>7.4f} {row['Weighted_Jaccard']:>7.4f} "
          f"{row['Synonym_Collapsed_Jaccard']:>7.4f} {row['Weighted_Collapsed_Jaccard']:>7.4f} "
          f"{row['Original_vs_Weighted']:>+5.3f} {row['Original_vs_Collapsed']:>+5.3f} "
          f"{row['Original_vs_WeightedCollapsed']:>+5.3f}")

# Save comprehensive comparison with all four approaches
comparison_df.to_csv('output/comprehensive_jaccard_comparison_all4.csv', index=False)
print(f"\nComprehensive comparison (all 4 approaches) saved to output/comprehensive_jaccard_comparison_all4.csv")

# Summary statistics for all four approaches
print(f"\n" + "="*50)
print("SUMMARY STATISTICS (ALL FOUR APPROACHES)")
print("="*50)
print(f"{'Metric':<15} {'Original':<8} {'Weighted':<8} {'Collapsed':<8} {'Wgt+Col':<8}")
print("-" * 60)
print(f"{'Mean':<15} {comparison_df['Original_Jaccard'].mean():>7.4f} "
      f"{comparison_df['Weighted_Jaccard'].mean():>7.4f} "
      f"{comparison_df['Synonym_Collapsed_Jaccard'].mean():>7.4f} "
      f"{comparison_df['Weighted_Collapsed_Jaccard'].mean():>7.4f}")
print(f"{'Median':<15} {comparison_df['Original_Jaccard'].median():>7.4f} "
      f"{comparison_df['Weighted_Jaccard'].median():>7.4f} "
      f"{comparison_df['Synonym_Collapsed_Jaccard'].median():>7.4f} "
      f"{comparison_df['Weighted_Collapsed_Jaccard'].median():>7.4f}")
print(f"{'Max':<15} {comparison_df['Original_Jaccard'].max():>7.4f} "
      f"{comparison_df['Weighted_Jaccard'].max():>7.4f} "
      f"{comparison_df['Synonym_Collapsed_Jaccard'].max():>7.4f} "
      f"{comparison_df['Weighted_Collapsed_Jaccard'].max():>7.4f}")
print(f"{'Min':<15} {comparison_df['Original_Jaccard'].min():>7.4f} "
      f"{comparison_df['Weighted_Jaccard'].min():>7.4f} "
      f"{comparison_df['Synonym_Collapsed_Jaccard'].min():>7.4f} "
      f"{comparison_df['Weighted_Collapsed_Jaccard'].min():>7.4f}")

# Identify CHMs most affected by each approach
print(f"\n" + "="*50)
print("MOST AFFECTED CHEMICALS (ALL FOUR APPROACHES)")
print("="*50)

# Biggest changes from original to weighted
print("Biggest changes (Original → Weighted):")
biggest_weighted_changes = comparison_df.nlargest(3, 'Original_vs_Weighted')[['CHM', 'Original_vs_Weighted']]
for _, row in biggest_weighted_changes.iterrows():
    print(f"  {row['CHM']}: {row['Original_vs_Weighted']:+.4f}")

# Biggest changes from original to collapsed
print("\nBiggest changes (Original → Synonym-Collapsed):")
biggest_collapsed_changes = comparison_df.nlargest(3, 'Original_vs_Collapsed')[['CHM', 'Original_vs_Collapsed']]
for _, row in biggest_collapsed_changes.iterrows():
    print(f"  {row['CHM']}: {row['Original_vs_Collapsed']:+.4f}")

# Biggest changes from original to weighted+collapsed
print("\nBiggest changes (Original → Weighted+Collapsed):")
biggest_weighted_collapsed_changes = comparison_df.nlargest(3, 'Original_vs_WeightedCollapsed')[['CHM', 'Original_vs_WeightedCollapsed']]
for _, row in biggest_weighted_collapsed_changes.iterrows():
    print(f"  {row['CHM']}: {row['Original_vs_WeightedCollapsed']:+.4f}")

# Show top performers in the final approach
print(f"\nTop 5 CHMs by Weighted+Collapsed Jaccard similarity:")
top_weighted_collapsed = comparison_df.nlargest(5, 'Weighted_Collapsed_Jaccard')[['CHM', 'Weighted_Collapsed_Jaccard']]
for i, (_, row) in enumerate(top_weighted_collapsed.iterrows(), 1):
    print(f"  {i}. {row['CHM']}: {row['Weighted_Collapsed_Jaccard']:.4f}")

print(f"\nAnalysis complete! All results saved to output/ directory.")
print(f"Key files:")
print(f"  - output/top25_synonym_analysis.csv (detailed synonym breakdown)")
print(f"  - output/jaccard_comparison_results.csv (original vs weighted)")
print(f"  - output/comprehensive_jaccard_comparison.csv (original 3 approaches)")
print(f"  - output/comprehensive_jaccard_comparison_all4.csv (all 4 approaches including weighted+collapsed)")




