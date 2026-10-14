import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from salt_synonyms_loader import get_all_salt_synonyms, load_salt_synonyms
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("JACCARD SIMILARITY ANALYSIS FOR LITHIUM SALTS")
print("=" * 60)

# Load salt synonyms
print("Loading salt synonyms...")
salt_synonyms_data = load_salt_synonyms()
salt_synonyms = get_all_salt_synonyms()
if not salt_synonyms:
    print("No salt synonyms found. Please run explore_lithium_salts.py first.")
    exit(1)
print(f"Loaded synonyms for {len(salt_synonyms)} salts")

# Create mapping from salt_id to formula for better display names
salt_id_to_formula = {}
for salt_id, data in salt_synonyms_data.items():
    salt_id_to_formula[salt_id] = data.get('formula', salt_id)

# Load the knowledge graph
print("Loading knowledge graph...")
df = pd.read_csv('SUBRELOBJ.csv')
print(f"Loaded knowledge graph with {len(df)} triples")

# Remove unwanted relation types (same as original jaccard_calculation.py)
rels_to_remove = ['CHM-CMT','CHM-SMT','CHM-SPL','CMT-CHM','SMT-CHM','SPL-CHM']
df = df[~df['Rel'].isin(rels_to_remove)].reset_index(drop=True)
print(f"After filtering relations: {len(df)} triples remain")

# Define allowed relations (same as original)
allowed = {
    'CHM-CHM',
    'PRO-CHM','CHM-PRO',
    'APL-CHM','CHM-APL',
    'DSC-CHM','CHM-DSC',
}

# Filter to only allowed relations
df = df[df['Rel'].isin(allowed)].reset_index(drop=True)
print(f"After filtering to allowed relations: {len(df)} triples remain")

# Find TFSI salt
tfsi_salt_id = None
tfsi_synonyms = []

# Load the full data to get additional info if needed
salt_data = load_salt_synonyms()
for salt_id, synonyms in salt_synonyms.items():
    if any('tfsi' in syn.lower() for syn in synonyms):
        tfsi_salt_id = salt_id
        tfsi_synonyms = synonyms
        break

if not tfsi_salt_id:
    print("TFSI salt not found in synonyms!")
    exit(1)

print(f"Found TFSI salt: {tfsi_salt_id}")
print(f"TFSI has {len(tfsi_synonyms)} synonyms")
if salt_data and tfsi_salt_id in salt_data:
    print(f"TFSI formula: {salt_data[tfsi_salt_id].get('formula', 'N/A')}")
print(f"Sample TFSI synonyms: {tfsi_synonyms[:5]}")  # Show first 5

def get_neighbors_for_salt(synonyms, df, max_synonyms=10):
    """
    Get all neighbors for a salt using its synonyms list
    Optimized approach similar to the original jaccard_calculation.py
    """
    neighbors = set()
    
    # Limit synonyms to avoid performance issues
    if len(synonyms) > max_synonyms:
        print(f"  Limiting to first {max_synonyms} synonyms (out of {len(synonyms)})")
        synonyms = synonyms[:max_synonyms]
    
    # Create regex pattern for all synonyms
    escaped_synonyms = [re.escape(syn) for syn in synonyms]
    pattern_str = '|'.join(escaped_synonyms)
    pattern = re.compile(f'(?i){pattern_str}')
    
    # Find matches where salt appears as Subject
    subject_mask = df['Subject'].str.contains(pattern, na=False)
    for _, row in df[subject_mask].iterrows():
        target = row['Object']
        # Don't add self-references
        if not pattern.search(str(target)):
            neighbors.add(target)
    
    # Find matches where salt appears as Object  
    object_mask = df['Object'].str.contains(pattern, na=False)
    for _, row in df[object_mask].iterrows():
        target = row['Subject']
        # Don't add self-references
        if not pattern.search(str(target)):
            neighbors.add(target)
    
    return neighbors

def get_weighted_neighbors_for_salt(synonyms, df, freq_col='frequency', max_synonyms=10):
    """
    Get weighted neighbors for a salt using its synonyms list
    """
    neighbors = defaultdict(int)
    
    # Limit synonyms to avoid performance issues
    if len(synonyms) > max_synonyms:
        synonyms = synonyms[:max_synonyms]
    
    # Create regex pattern for all synonyms
    escaped_synonyms = [re.escape(syn) for syn in synonyms]
    pattern_str = '|'.join(escaped_synonyms)
    pattern = re.compile(f'(?i){pattern_str}')
    
    # Check if frequency column exists
    if freq_col not in df.columns:
        print(f"Warning: Frequency column '{freq_col}' not found. Using frequency = 1 for all relations")
        df[freq_col] = 1
    
    # Find matches where salt appears as Subject
    subject_mask = df['Subject'].str.contains(pattern, na=False)
    for _, row in df[subject_mask].iterrows():
        target = row['Object']
        weight = row[freq_col] if freq_col in df.columns else 1
        # Don't add self-references
        if not pattern.search(str(target)):
            neighbors[target] += weight
    
    # Find matches where salt appears as Object  
    object_mask = df['Object'].str.contains(pattern, na=False)
    for _, row in df[object_mask].iterrows():
        target = row['Subject']
        weight = row[freq_col] if freq_col in df.columns else 1
        # Don't add self-references
        if not pattern.search(str(target)):
            neighbors[target] += weight
    
    return dict(neighbors)

def calculate_jaccard_similarity(set1, set2):
    """Calculate unweighted Jaccard similarity"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_weighted_jaccard(dict1, dict2):
    """Calculate weighted Jaccard similarity"""
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

# Get TFSI neighbors
print(f"\nFinding neighbors for TFSI...")
tfsi_neighbors = get_neighbors_for_salt(tfsi_synonyms, df)
print(f"TFSI has {len(tfsi_neighbors)} unique neighbors")

# Check if frequency column exists for weighted calculations
freq_col = None
possible_freq_cols = ['Count', 'Frequency', '# of relations', 'count', 'frequency', 'weight']
for col in possible_freq_cols:
    if col in df.columns:
        freq_col = col
        break

if freq_col:
    print(f"Found frequency column: '{freq_col}' - will calculate weighted Jaccard")
    tfsi_weighted_neighbors = get_weighted_neighbors_for_salt(tfsi_synonyms, df, freq_col)
else:
    print("No frequency column found - will use frequency = 1 for weighted calculations")
    tfsi_weighted_neighbors = get_weighted_neighbors_for_salt(tfsi_synonyms, df, 'frequency')

# Calculate Jaccard similarities for all other salts
print(f"\nCalculating Jaccard similarities for {len(salt_synonyms)-1} salts...")

results = []
for salt_id, synonyms in salt_synonyms.items():
    if salt_id == tfsi_salt_id:
        continue  # Skip TFSI itself
    
    print(f"Processing {salt_id}...")
    
    # Get neighbors for this salt
    salt_neighbors = get_neighbors_for_salt(synonyms, df)
    salt_weighted_neighbors = get_weighted_neighbors_for_salt(synonyms, df, freq_col or 'frequency')
    
    # Calculate unweighted Jaccard
    jaccard_unweighted = calculate_jaccard_similarity(tfsi_neighbors, salt_neighbors)
    
    # Calculate weighted Jaccard
    jaccard_weighted = calculate_weighted_jaccard(tfsi_weighted_neighbors, salt_weighted_neighbors)
    
    # Calculate intersection details
    intersection = tfsi_neighbors.intersection(salt_neighbors)
    union = tfsi_neighbors.union(salt_neighbors)
    
    results.append({
        'salt_id': salt_id,
        'formula': salt_data.get(salt_id, {}).get('formula', 'N/A') if salt_data else 'N/A',
        'synonyms_count': len(synonyms),
        'neighbors_count': len(salt_neighbors),
        'intersection_count': len(intersection),
        'union_count': len(union),
        'jaccard_unweighted': jaccard_unweighted,
        'jaccard_weighted': jaccard_weighted,
        'intersection_examples': list(intersection)[:10]  # First 10 examples
    })

# Sort by unweighted Jaccard similarity
results.sort(key=lambda x: x['jaccard_unweighted'], reverse=True)

# Display results
print("\n" + "=" * 80)
print("JACCARD SIMILARITY RESULTS")
print("=" * 80)

print(f"\nTFSI ({salt_id_to_formula.get(tfsi_salt_id, tfsi_salt_id)}) Analysis:")
print(f"  Synonyms: {len(tfsi_synonyms)}")
print(f"  Neighbors: {len(tfsi_neighbors)}")

print(f"\nTop 15 Most Similar Salts to TFSI:")
print(f"{'Rank':<4} {'Formula':<15} {'Synonyms':<9} {'Neighbors':<9} {'Intersection':<12} {'Jaccard (Unweighted)':<20} {'Jaccard (Weighted)':<18}")
print("-" * 110)

for i, result in enumerate(results[:15]):
    print(f"{i+1:<4} {result['formula']:<15} {result['synonyms_count']:<9} {result['neighbors_count']:<9} "
          f"{result['intersection_count']:<12} {result['jaccard_unweighted']:<20.4f} {result['jaccard_weighted']:<18.4f}")

# Show detailed intersection for top 5
print(f"\nDetailed intersection examples for top 5 salts:")
for i, result in enumerate(results[:5]):
    print(f"\n{i+1}. {result['formula']} (Jaccard: {result['jaccard_unweighted']:.4f})")
    print(f"   Shared neighbors ({result['intersection_count']} total): {result['intersection_examples']}")

# Save results
results_df = pd.DataFrame([{
    'salt_id': r['salt_id'],
    'formula': r['formula'],
    'synonyms_count': r['synonyms_count'], 
    'neighbors_count': r['neighbors_count'],
    'intersection_count': r['intersection_count'],
    'union_count': r['union_count'],
    'jaccard_unweighted': r['jaccard_unweighted'],
    'jaccard_weighted': r['jaccard_weighted']
} for r in results])

results_df.to_csv('output/lithium_salt_jaccard_results.csv', index=False)
print(f"\nDetailed results saved to output/lithium_salt_jaccard_results.csv")

# Create summary plots
print(f"\nCreating summary visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Top 10 Jaccard scores comparison
top_10_results = results[:10]
formulas = [r['formula'] for r in top_10_results]
jaccard_unweighted = [r['jaccard_unweighted'] for r in top_10_results]
jaccard_weighted = [r['jaccard_weighted'] for r in top_10_results]

x = range(len(formulas))
width = 0.35

bars1 = ax1.bar([i - width/2 for i in x], jaccard_unweighted, width, label='Unweighted', alpha=0.8)
bars2 = ax1.bar([i + width/2 for i in x], jaccard_weighted, width, label='Weighted', alpha=0.8)

ax1.set_xlabel('Salt Formula')
ax1.set_ylabel('Jaccard Similarity')
ax1.set_title('Top 10 Jaccard Similarities: TFSI vs Other Salts')
ax1.set_xticks(x)
ax1.set_xticklabels(formulas, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Intersection sizes
intersection_counts = [r['intersection_count'] for r in top_10_results]
bars3 = ax2.bar(formulas, intersection_counts, alpha=0.8, color='coral')

ax2.set_xlabel('Salt Formula')
ax2.set_ylabel('Number of Shared Neighbors')
ax2.set_title('Shared Knowledge Graph Neighbors with TFSI')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('output/lithium_salt_jaccard_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved summary plot: output/lithium_salt_jaccard_summary.png")

# Summary statistics
print(f"\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

avg_unweighted = sum(r['jaccard_unweighted'] for r in results) / len(results)
avg_weighted = sum(r['jaccard_weighted'] for r in results) / len(results)
max_unweighted = max(r['jaccard_unweighted'] for r in results)
max_weighted = max(r['jaccard_weighted'] for r in results)

print(f"Total salts analyzed: {len(results)}")
print(f"Average unweighted Jaccard: {avg_unweighted:.4f}")
print(f"Average weighted Jaccard: {avg_weighted:.4f}")
print(f"Max unweighted Jaccard: {max_unweighted:.4f}")
print(f"Max weighted Jaccard: {max_weighted:.4f}")

# Top performers
print(f"\nTop 3 most similar salts to TFSI:")
for i, result in enumerate(results[:3]):
    print(f"{i+1}. {result['formula']}: {result['jaccard_unweighted']:.4f} (unweighted), {result['jaccard_weighted']:.4f} (weighted)")

print(f"\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFiles created:")
print(f"- output/lithium_salt_jaccard_results.csv (detailed results)")
print(f"- output/lithium_salt_jaccard_summary.png (summary plots)")
