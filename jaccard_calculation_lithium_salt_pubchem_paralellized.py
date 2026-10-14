import pandas as pd
import random
import re
from collections import defaultdict
from salt_synonyms_loader_for_pubchem import get_all_pubchem_synonyms
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 60)
print("JACCARD SIMILARITY ANALYSIS FOR PUBCHEM LITHIUM SALTS vs TFSI")
print("=" * 60)

# Load PubChem synonyms (mf -> synonyms)
print("Loading PubChem salt synonyms...")
all_pubchem_synonyms = get_all_pubchem_synonyms()
all_mfs = list(all_pubchem_synonyms.keys())
print(f"Loaded synonyms for {len(all_mfs)} salts")

# --- Batch processing setup ---
import os
BATCH_SIZE = 500
RESULTS_CSV = 'output/pubchem_lithium_salt_jaccard_vs_tfsi_results_parallel.csv'
TOP100_CSV = 'output/pubchem_lithium_salt_jaccard_vs_tfsi_top100_parallel.csv'

# Check for already processed mfs (checkpointing)
if os.path.exists(RESULTS_CSV):
    processed_df = pd.read_csv(RESULTS_CSV)
    processed_mfs = set(processed_df['mf'])
    print(f"Resuming: {len(processed_mfs)} formulas already processed.")
else:
    processed_mfs = set()
    print("No previous results found. Starting fresh.")

# Prepare batches
remaining_mfs = [mf for mf in all_mfs if mf not in processed_mfs]
print(f"{len(remaining_mfs)} formulas to process in batches of {BATCH_SIZE}.")
batches = [remaining_mfs[i:i+BATCH_SIZE] for i in range(0, len(remaining_mfs), BATCH_SIZE)]

# Load the knowledge graph
df = pd.read_csv('SUBRELOBJ.csv')
print(f"Loaded knowledge graph with {len(df)} triples")

# Remove unwanted relation types
rels_to_remove = ['CHM-CMT','CHM-SMT','CHM-SPL','CMT-CHM','SMT-CHM','SPL-CHM']
df = df[~df['Rel'].isin(rels_to_remove)].reset_index(drop=True)
print(f"After filtering relations: {len(df)} triples remain")

# Define allowed relations
allowed = {
    'CHM-CHM',
    'PRO-CHM','CHM-PRO',
    'APL-CHM','CHM-APL',
    'DSC-CHM','CHM-DSC',
}
df = df[df['Rel'].isin(allowed)].reset_index(drop=True)
print(f"After filtering to allowed relations: {len(df)} triples remain")

# --- Use TFSI synonyms from v2 script ---
tfsi_synonyms = [
    'TFSI', 'bis(trifluoromethanesulfonyl)imide', 'bis(trifluoromethane)sulfonimide',
    'bis(trifluoromethylsulfonyl)imide', 'bis(trifluoromethanesulfonyl)amide',
    'bis(trifluoromethanesulfonyl)azanide', 'bis(trifluoromethanesulfonyl)imido',
    'bis(trifluoromethanesulphonyl)imide', 'lithium bis(trifluoromethanesulfonyl)imide',
    'lithium bis(trifluoromethane)sulfonimide', 'lithium bis(trifluoromethylsulfonyl)imide',
    'lithium bis(trifluoromethanesulfonyl)amide', 'lithium bis(trifluoromethanesulfonyl)azanide',
    'lithium bis(trifluoromethanesulfonyl)imido', 'lithium bis(trifluoromethanesulphonyl)imide',
    'LiTFSI', 'LiN(SO2CF3)2', 'LiN(SO2Me)2', 'LiN(SO2CF3)2', 'LiTFS', 'LiTFMSI',
    'Lithium TFSI', 'Lithium bis(trifluoromethanesulfonyl)imide',
    'Lithium bis(trifluoromethane)sulfonimide', 'Lithium bis(trifluoromethylsulfonyl)imide',
    'Lithium bis(trifluoromethanesulfonyl)amide', 'Lithium bis(trifluoromethanesulfonyl)azanide',
    'Lithium bis(trifluoromethanesulfonyl)imido', 'Lithium bis(trifluoromethanesulphonyl)imide'
]
print(f"Using hardcoded TFSI synonyms from v2 script: {tfsi_synonyms[:5]} ... (total {len(tfsi_synonyms)})")

# --- Helper functions ---
def get_neighbors_for_salt(synonyms, df, max_synonyms=10):
    cleaned_synonyms = []
    for syn in synonyms:
        if ';' in syn:
            parts = [s.strip() for s in syn.split(';') if s.strip()]
            cleaned_synonyms.extend(parts)
        else:
            cleaned_synonyms.append(syn.strip())
    synonyms = cleaned_synonyms
    if len(synonyms) > max_synonyms:
        synonyms = synonyms[:max_synonyms]
    escaped_synonyms = [re.escape(syn) for syn in synonyms]
    pattern_str = '|'.join(escaped_synonyms)
    pattern = re.compile(f'(?i){pattern_str}')
    subject_mask = df['Subject'].str.contains(pattern, na=False)
    neighbors = set()
    for _, row in df[subject_mask].iterrows():
        target = row['Object']
        if not pattern.search(str(target)):
            neighbors.add(target)
    object_mask = df['Object'].str.contains(pattern, na=False)
    for _, row in df[object_mask].iterrows():
        target = row['Subject']
        if not pattern.search(str(target)):
            neighbors.add(target)
    return neighbors

def get_weighted_neighbors_for_salt(synonyms, df, freq_col='frequency', max_synonyms=10):
    cleaned_synonyms = []
    for syn in synonyms:
        if ';' in syn:
            parts = [s.strip() for s in syn.split(';') if s.strip()]
            cleaned_synonyms.extend(parts)
        else:
            cleaned_synonyms.append(syn.strip())
    synonyms = cleaned_synonyms
    if len(synonyms) > max_synonyms:
        synonyms = synonyms[:max_synonyms]
    escaped_synonyms = [re.escape(syn) for syn in synonyms]
    pattern_str = '|'.join(escaped_synonyms)
    pattern = re.compile(f'(?i){pattern_str}')
    if freq_col not in df.columns:
        df[freq_col] = 1
    neighbors = defaultdict(int)
    subject_mask = df['Subject'].str.contains(pattern, na=False)
    for _, row in df[subject_mask].iterrows():
        target = row['Object']
        weight = row[freq_col] if freq_col in df.columns else 1
        if not pattern.search(str(target)):
            neighbors[target] += weight
    object_mask = df['Object'].str.contains(pattern, na=False)
    for _, row in df[object_mask].iterrows():
        target = row['Subject']
        weight = row[freq_col] if freq_col in df.columns else 1
        if not pattern.search(str(target)):
            neighbors[target] += weight
    return dict(neighbors)

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def calculate_weighted_jaccard(dict1, dict2):
    all_neighbors = set(dict1.keys()) | set(dict2.keys())
    if not all_neighbors:
        return 0.0
    intersection_weight = 0.0
    union_weight = 0.0
    for neighbor in all_neighbors:
        weight1 = dict1.get(neighbor, 0)
        weight2 = dict2.get(neighbor, 0)
        intersection_weight += min(weight1, weight2)
        union_weight += max(weight1, weight2)
    return intersection_weight / union_weight if union_weight > 0 else 0.0

# ===================== MAIN BATCHED EXECUTION =====================

batch_size = 500
output_file = 'output/pubchem_lithium_salt_jaccard_vs_tfsi_results_parallel.csv'
already_done = set()
if os.path.exists(output_file):
    try:
        prev_df = pd.read_csv(output_file)
        already_done = set(prev_df['mf'].astype(str))
        print(f"Resuming: {len(already_done)} salts already processed.")
    except Exception as e:
        print(f"Could not read previous results: {e}")

print(f"\nFinding neighbors for TFSI...")
tfsi_neighbors = get_neighbors_for_salt(tfsi_synonyms, df)
print(f"TFSI has {len(tfsi_neighbors)} unique neighbors")
freq_col = None
possible_freq_cols = ['Count', 'Frequency', '# of relations', 'count', 'frequency', 'weight']
for col in df.columns:
    if col in possible_freq_cols:
        freq_col = col
        break
if freq_col:
    print(f"Found frequency column: '{freq_col}' - will calculate weighted Jaccard")
    tfsi_weighted_neighbors = get_weighted_neighbors_for_salt(tfsi_synonyms, df, freq_col)
else:
    print("No frequency column found - will use frequency = 1 for weighted calculations")
    tfsi_weighted_neighbors = get_weighted_neighbors_for_salt(tfsi_synonyms, df, 'frequency')

# --- Batched Jaccard calculation ---
from multiprocessing import Pool, cpu_count

# Use all available CPUs, or set a specific number if desired
num_workers = cpu_count()
print(f"Using {num_workers} parallel workers for batch processing.")

def process_salt(mf):
    synonyms = all_pubchem_synonyms[mf]
    salt_neighbors = get_neighbors_for_salt(synonyms, df)
    salt_weighted_neighbors = get_weighted_neighbors_for_salt(synonyms, df, freq_col or 'frequency')
    jaccard_unweighted = calculate_jaccard_similarity(tfsi_neighbors, salt_neighbors)
    jaccard_weighted = calculate_weighted_jaccard(tfsi_weighted_neighbors, salt_weighted_neighbors)
    intersection = tfsi_neighbors & salt_neighbors
    union = tfsi_neighbors | salt_neighbors
    return {
        'mf': mf,
        'synonyms_count': len(synonyms),
        'neighbors_count': len(salt_neighbors),
        'intersection_count': len(intersection),
        'union_count': len(union),
        'jaccard_unweighted': jaccard_unweighted,
        'jaccard_weighted': jaccard_weighted,
        'intersection_examples': ';'.join(map(str, list(intersection)[:10]))
    }

all_results = []
for batch_idx, batch_mfs in enumerate(batches):
    print(f"\nProcessing batch {batch_idx+1}/{len(batches)}: {len(batch_mfs)} formulas...")
    with Pool(processes=num_workers) as pool:
        batch_results = pool.map(process_salt, batch_mfs)
    batch_df = pd.DataFrame(batch_results)
    if os.path.exists(RESULTS_CSV):
        batch_df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        batch_df.to_csv(RESULTS_CSV, mode='w', header=True, index=False)
    print(f"Batch {batch_idx+1} results appended to {RESULTS_CSV}")
    all_results.extend(batch_results)

# --- After all batches: print and save top 100 ---
print("\nAll batches complete. Loading all results for final analysis...")
final_df = pd.read_csv(RESULTS_CSV)
final_df_sorted = final_df.sort_values('jaccard_unweighted', ascending=False).reset_index(drop=True)
print("\nTop 100 Most Similar Salts to TFSI:")
print(f"{'Rank':<4} {'MF':<15} {'Synonyms':<9} {'Neighbors':<9} {'Intersection':<12} {'Jaccard (Unweighted)':<20} {'Jaccard (Weighted)':<18}")
print("-" * 110)
for i, row in final_df_sorted.head(100).iterrows():
    print(f"{i+1:<4} {row['mf']:<15} {row['synonyms_count']:<9} {row['neighbors_count']:<9} "
          f"{row['intersection_count']:<12} {row['jaccard_unweighted']:<20.4f} {row['jaccard_weighted']:<18.4f}")

final_df_sorted.head(100).to_csv(TOP100_CSV, index=False)
print(f"\nTop 100 results saved to {TOP100_CSV}")
