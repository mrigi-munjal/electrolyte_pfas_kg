#!/usr/bin/env python3
"""
Jaccard Similarity Calculation for Salt Knowledge Graph

This script calculates Jaccard similarity between LiTFSI and all other salts
using the consolidated salt knowledge graph triples.
"""

import json
import pandas as pd
import re
from collections import defaultdict, Counter
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_knowledge_graph_triples(file_path='salt_analysis_output/salt_knowledge_graph_triples.json'):
    """Load the knowledge graph triples from JSON file"""
    try:
        print(f"Loading knowledge graph triples from: {file_path}")
        with open(file_path, 'r') as f:
            triples = json.load(f)
        print(f"Successfully loaded {len(triples):,} triples")
        return triples
    except Exception as e:
        print(f"Error loading triples: {e}")
        return None

def convert_triples_to_dataframe(triples):
    """Convert triples list to DataFrame for easier processing"""
    df_data = []
    for triple in triples:
        df_data.append({
            'Subject': triple['subject'],
            'Predicate': triple['predicate'], 
            'Object': triple['object'],
            'SubjectType': triple['subject_type'],
            'ObjectType': triple['object_type']
        })
    
    df = pd.DataFrame(df_data)
    print(f"Created DataFrame with {len(df)} rows")
    print(f"Unique subjects (salts): {df['Subject'].nunique()}")
    print(f"Unique predicates: {df['Predicate'].nunique()}")
    print(f"Unique objects: {df['Object'].nunique()}")
    
    return df

def get_salt_neighbors(salt_name, df, relationship_types=None):
    """
    Get all neighbors (objects) connected to a salt through relationships
    
    Args:
        salt_name: Name of the salt to find neighbors for
        df: DataFrame with triples (Subject, Predicate, Object)
        relationship_types: List of relationship types to include (None = all)
    
    Returns:
        set: Set of neighbor objects
    """
    neighbors = set()
    
    # Filter by relationship types if specified
    if relationship_types:
        filtered_df = df[df['Predicate'].isin(relationship_types)]
    else:
        filtered_df = df
    
    # Find all objects where salt is the subject
    subject_mask = filtered_df['Subject'] == salt_name
    subject_neighbors = set(filtered_df[subject_mask]['Object'].values)
    neighbors.update(subject_neighbors)
    
    # Find all subjects where salt is the object (reverse relationships)
    object_mask = filtered_df['Object'] == salt_name
    object_neighbors = set(filtered_df[object_mask]['Subject'].values)
    neighbors.update(object_neighbors)
    
    # Remove self-references
    neighbors.discard(salt_name)
    
    return neighbors

def get_weighted_salt_neighbors(salt_name, df, relationship_types=None):
    """
    Get weighted neighbors based on relationship frequency
    
    Returns:
        dict: {neighbor: weight} where weight is the number of connections
    """
    neighbors = defaultdict(int)
    
    # Filter by relationship types if specified
    if relationship_types:
        filtered_df = df[df['Predicate'].isin(relationship_types)]
    else:
        filtered_df = df
    
    # Count connections where salt is the subject
    subject_mask = filtered_df['Subject'] == salt_name
    for _, row in filtered_df[subject_mask].iterrows():
        obj = row['Object']
        if obj != salt_name:  # Avoid self-references
            neighbors[obj] += 1
    
    # Count connections where salt is the object
    object_mask = filtered_df['Object'] == salt_name
    for _, row in filtered_df[object_mask].iterrows():
        subj = row['Subject']
        if subj != salt_name:  # Avoid self-references
            neighbors[subj] += 1
    
    return dict(neighbors)

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def calculate_weighted_jaccard(dict1, dict2):
    """Calculate weighted Jaccard similarity between two weighted neighbor dictionaries"""
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

def get_relationship_context(salt1, salt2, df, shared_neighbors):
    """Get the relationship context between two salts through shared neighbors"""
    context = []
    
    for neighbor in list(shared_neighbors)[:5]:  # Limit to first 5 for readability
        # Find relationships from salt1 to neighbor
        salt1_rels = df[(df['Subject'] == salt1) & (df['Object'] == neighbor)]['Predicate'].tolist()
        salt1_rels.extend(df[(df['Object'] == salt1) & (df['Subject'] == neighbor)]['Predicate'].tolist())
        
        # Find relationships from salt2 to neighbor
        salt2_rels = df[(df['Subject'] == salt2) & (df['Object'] == neighbor)]['Predicate'].tolist()
        salt2_rels.extend(df[(df['Object'] == salt2) & (df['Subject'] == neighbor)]['Predicate'].tolist())
        
        if salt1_rels and salt2_rels:
            common_rels = set(salt1_rels) & set(salt2_rels)
            if common_rels:
                context.append(f"{neighbor} ({', '.join(common_rels)})")
    
    return context

def calculate_salt_similarities(reference_salt, df, output_dir='salt_analysis_output'):
    """Calculate Jaccard similarities between reference salt and all other salts"""
    print(f"\n{'='*60}")
    print(f"CALCULATING JACCARD SIMILARITIES FOR {reference_salt}")
    print(f"{'='*60}")
    
    # Get all unique salts
    all_salts = set(df['Subject'].unique())
    all_salts.update(df['Object'].unique())
    
    # Filter to only keep actual salt names (subjects in our case)
    salt_subjects = set(df['Subject'].unique())
    all_salts = salt_subjects
    
    print(f"Found {len(all_salts)} unique salts in the knowledge graph")
    
    if reference_salt not in all_salts:
        print(f"ERROR: Reference salt '{reference_salt}' not found in knowledge graph!")
        print(f"Available salts: {sorted(list(all_salts))[:10]}...")
        return None
    
    # Get neighbors for reference salt
    print(f"Finding neighbors for {reference_salt}...")
    ref_neighbors = get_salt_neighbors(reference_salt, df)
    ref_weighted_neighbors = get_weighted_salt_neighbors(reference_salt, df)
    
    print(f"{reference_salt} has {len(ref_neighbors)} unique neighbors")
    print(f"Total weighted connections: {sum(ref_weighted_neighbors.values())}")
    
    # Calculate similarities with all other salts
    results = []
    other_salts = [salt for salt in all_salts if salt != reference_salt]
    
    print(f"Calculating similarities with {len(other_salts)} other salts...")
    
    for i, salt in enumerate(other_salts):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(other_salts)} salts...")
        
        # Get neighbors for current salt
        salt_neighbors = get_salt_neighbors(salt, df)
        salt_weighted_neighbors = get_weighted_salt_neighbors(salt, df)
        
        # Calculate similarities
        jaccard_unweighted = calculate_jaccard_similarity(ref_neighbors, salt_neighbors)
        jaccard_weighted = calculate_weighted_jaccard(ref_weighted_neighbors, salt_weighted_neighbors)
        
        # Find shared neighbors
        intersection = ref_neighbors & salt_neighbors
        union = ref_neighbors | salt_neighbors
        
        # Get relationship context
        context = get_relationship_context(reference_salt, salt, df, intersection)
        
        results.append({
            'salt': salt,
            'neighbors_count': len(salt_neighbors),
            'intersection_count': len(intersection),
            'union_count': len(union),
            'jaccard_unweighted': jaccard_unweighted,
            'jaccard_weighted': jaccard_weighted,
            'shared_neighbors_examples': list(intersection)[:10],
            'relationship_context': context[:3]  # Top 3 contexts
        })
    
    # Sort by unweighted Jaccard similarity
    results.sort(key=lambda x: x['jaccard_unweighted'], reverse=True)
    
    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame([{
        'salt': r['salt'],
        'neighbors_count': r['neighbors_count'],
        'intersection_count': r['intersection_count'],
        'union_count': r['union_count'],
        'jaccard_unweighted': r['jaccard_unweighted'],
        'jaccard_weighted': r['jaccard_weighted'],
        'shared_neighbors_examples': '; '.join(map(str, r['shared_neighbors_examples'])),
        'relationship_context': '; '.join(r['relationship_context'])
    } for r in results])
    
    # Save full results
    full_results_path = os.path.join(output_dir, f'jaccard_similarities_{reference_salt}_vs_all_salts.csv')
    results_df.to_csv(full_results_path, index=False)
    
    # Save top 50 results
    top_results_path = os.path.join(output_dir, f'jaccard_similarities_{reference_salt}_top50.csv')
    results_df.head(50).to_csv(top_results_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Full results: {full_results_path}")
    print(f"  Top 50: {top_results_path}")
    
    return results, results_df

def print_top_similarities(results, reference_salt, top_n=20):
    """Print top N most similar salts"""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} SALTS MOST SIMILAR TO {reference_salt}")
    print(f"{'='*80}")
    
    print(f"{'Rank':<4} {'Salt':<20} {'Neighbors':<9} {'Shared':<7} {'Jaccard (Unweighted)':<20} {'Jaccard (Weighted)':<18}")
    print("-" * 80)
    
    for i, result in enumerate(results[:top_n]):
        rank = i + 1
        salt = result['salt']
        neighbors = result['neighbors_count']
        shared = result['intersection_count']
        jaccard_unw = result['jaccard_unweighted']
        jaccard_w = result['jaccard_weighted']
        
        print(f"{rank:<4} {salt:<20} {neighbors:<9} {shared:<7} {jaccard_unw:<20.4f} {jaccard_w:<18.4f}")
    
    # Show some example shared relationships
    print(f"\nSample shared relationships for top salts:")
    for i, result in enumerate(results[:5]):
        if result['shared_neighbors_examples']:
            print(f"{i+1}. {result['salt']}: {', '.join(result['shared_neighbors_examples'][:5])}")

def analyze_relationship_types(df):
    """Analyze the distribution of relationship types in the knowledge graph"""
    print(f"\n{'='*60}")
    print("RELATIONSHIP TYPE ANALYSIS")
    print(f"{'='*60}")
    
    predicate_counts = Counter(df['Predicate'])
    
    print("Relationship types and frequencies:")
    for predicate, count in predicate_counts.most_common():
        print(f"  {predicate}: {count:,} instances")
    
    return predicate_counts

def main():
    """Main function to run the Jaccard similarity analysis"""
    print("Salt Knowledge Graph Jaccard Similarity Analysis")
    print("="*60)
    
    # Load knowledge graph triples
    triples = load_knowledge_graph_triples()
    if triples is None:
        return
    
    # Convert to DataFrame
    df = convert_triples_to_dataframe(triples)
    
    # Analyze relationship types
    predicate_counts = analyze_relationship_types(df)
    
    # Define reference salt
    reference_salt = "LiTFSI"
    
    # Calculate similarities
    results, results_df = calculate_salt_similarities(reference_salt, df)
    
    if results:
        # Print top results
        print_top_similarities(results, reference_salt, top_n=20)
        
        # Additional analysis
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Reference salt: {reference_salt}")
        print(f"Total salts compared: {len(results)}")
        print(f"Average Jaccard similarity: {results_df['jaccard_unweighted'].mean():.4f}")
        print(f"Median Jaccard similarity: {results_df['jaccard_unweighted'].median():.4f}")
        print(f"Max Jaccard similarity: {results_df['jaccard_unweighted'].max():.4f}")
        print(f"Salts with similarity > 0.1: {(results_df['jaccard_unweighted'] > 0.1).sum()}")
        print(f"Salts with similarity > 0.2: {(results_df['jaccard_unweighted'] > 0.2).sum()}")
        
        # Show relationship contexts for top salts
        print(f"\n{'='*60}")
        print("RELATIONSHIP CONTEXTS FOR TOP 5 SIMILAR SALTS")
        print(f"{'='*60}")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. {result['salt']} (Jaccard: {result['jaccard_unweighted']:.4f})")
            if result['relationship_context']:
                print(f"   Shared relationship contexts: {'; '.join(result['relationship_context'])}")
            else:
                print("   No specific relationship contexts found")

if __name__ == "__main__":
    main()
