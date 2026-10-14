#!/usr/bin/env python3
"""
ROUGE and Fuzzy Similarity Calculation for Salt Knowledge Graph

This script calculates ROUGE and other fuzzy similarity metrics between LiTFSI and all other salts
using the consolidated salt knowledge graph triples. Uses looser matching compared to exact Jaccard.
"""

import json
import pandas as pd
import re
from collections import defaultdict, Counter
import os
from datetime import datetime
import warnings
from difflib import SequenceMatcher
import math

warnings.filterwarnings('ignore')

# Try to import ROUGE and other similarity libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

class FuzzySimilarityCalculator:
    """Calculate various fuzzy similarity metrics for chemical entity matching"""
    
    def __init__(self):
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Chemical synonyms for preprocessing
        self.chemical_synonyms = {
            'li': 'lithium',
            'po4': 'phosphate',
            'co3': 'carbonate',
            'f': 'fluorine',
            'tfsi': 'trifluoromethanesulfonylimide',
            'fsi': 'fluorosulfonylimide',
            'bf4': 'tetrafluoroborate',
            'pf6': 'hexafluorophosphate',
            'clo4': 'perchlorate',
            'ec': 'ethylene carbonate',
            'dmc': 'dimethyl carbonate',
            'emc': 'ethyl methyl carbonate'
        }
    
    def preprocess_entity_name(self, text):
        """Preprocess entity names for better fuzzy matching"""
        if not isinstance(text, str):
            return str(text).lower()
        
        text = text.lower().strip()
        
        # Expand chemical abbreviations
        for abbrev, full_form in self.chemical_synonyms.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full_form, text)
        
        # Clean up special characters but preserve chemical notation
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def rouge_similarity(self, text1, text2):
        """Calculate ROUGE similarity between two texts"""
        if not ROUGE_AVAILABLE or not self.rouge_scorer:
            # Fallback to token overlap
            return self.token_overlap_similarity(text1, text2)
        
        text1_proc = self.preprocess_entity_name(text1)
        text2_proc = self.preprocess_entity_name(text2)
        
        if not text1_proc.strip() or not text2_proc.strip():
            return 0.0
        
        try:
            scores = self.rouge_scorer.score(text1_proc, text2_proc)
            # Use ROUGE-L F1 score as primary metric
            return scores['rougeL'].fmeasure
        except Exception as e:
            print(f"ROUGE calculation error for '{text1}' vs '{text2}': {e}")
            return self.token_overlap_similarity(text1, text2)
    
    def levenshtein_similarity(self, text1, text2):
        """Calculate normalized Levenshtein similarity"""
        text1_proc = self.preprocess_entity_name(text1)
        text2_proc = self.preprocess_entity_name(text2)
        
        if not text1_proc and not text2_proc:
            return 1.0
        if not text1_proc or not text2_proc:
            return 0.0
        
        return SequenceMatcher(None, text1_proc, text2_proc).ratio()
    
    def token_overlap_similarity(self, text1, text2):
        """Calculate token overlap similarity (Dice coefficient)"""
        text1_proc = self.preprocess_entity_name(text1)
        text2_proc = self.preprocess_entity_name(text2)
        
        tokens1 = set(text1_proc.split())
        tokens2 = set(text2_proc.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        return (2 * intersection) / (len(tokens1) + len(tokens2))
    
    def fuzzy_set_similarity(self, set1, set2, similarity_threshold=0.5):
        """Calculate fuzzy set similarity using entity name matching"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        list1 = list(set1)
        list2 = list(set2)
        
        # Create similarity matrix
        match_scores = []
        matched_indices_2 = set()
        
        for item1 in list1:
            best_match_score = 0.0
            best_match_idx = -1
            
            for idx2, item2 in enumerate(list2):
                if idx2 in matched_indices_2:
                    continue
                
                # Calculate combined similarity
                rouge_sim = self.rouge_similarity(item1, item2)
                token_sim = self.token_overlap_similarity(item1, item2)
                leven_sim = self.levenshtein_similarity(item1, item2)
                
                # Weighted combination
                combined_sim = 0.4 * rouge_sim + 0.3 * token_sim + 0.3 * leven_sim
                
                if combined_sim > best_match_score and combined_sim >= similarity_threshold:
                    best_match_score = combined_sim
                    best_match_idx = idx2
            
            if best_match_idx != -1:
                match_scores.append(best_match_score)
                matched_indices_2.add(best_match_idx)
        
        if not match_scores:
            return 0.0
        
        # Calculate fuzzy Jaccard: sum of match scores / union size
        # This gives partial credit for fuzzy matches
        intersection_weight = sum(match_scores)
        union_size = len(set1) + len(set2) - len(match_scores)
        
        return intersection_weight / union_size if union_size > 0 else 0.0

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

def calculate_fuzzy_similarity(set1, set2, similarity_calc):
    """Calculate fuzzy similarity between two sets using multiple metrics"""
    if not set1 and not set2:
        return {'fuzzy_jaccard': 1.0, 'rouge_avg': 1.0, 'token_overlap': 1.0, 'levenshtein_avg': 1.0}
    
    if not set1 or not set2:
        return {'fuzzy_jaccard': 0.0, 'rouge_avg': 0.0, 'token_overlap': 0.0, 'levenshtein_avg': 0.0}
    
    # Fuzzy Jaccard using threshold-based matching
    fuzzy_jaccard = similarity_calc.fuzzy_set_similarity(set1, set2, similarity_threshold=0.3)
    
    # Average pairwise similarities
    rouge_scores = []
    token_scores = []
    levenshtein_scores = []
    
    # Sample-based calculation for efficiency (if sets are large)
    list1 = list(set1)
    list2 = list(set2)
    
    # If sets are large, sample for efficiency
    if len(list1) > 20:
        list1 = list1[:20]
    if len(list2) > 20:
        list2 = list2[:20]
    
    for item1 in list1:
        for item2 in list2:
            rouge_scores.append(similarity_calc.rouge_similarity(item1, item2))
            token_scores.append(similarity_calc.token_overlap_similarity(item1, item2))
            levenshtein_scores.append(similarity_calc.levenshtein_similarity(item1, item2))
    
    rouge_avg = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    token_avg = sum(token_scores) / len(token_scores) if token_scores else 0.0
    levenshtein_avg = sum(levenshtein_scores) / len(levenshtein_scores) if levenshtein_scores else 0.0
    
    return {
        'fuzzy_jaccard': fuzzy_jaccard,
        'rouge_avg': rouge_avg,
        'token_overlap': token_avg,
        'levenshtein_avg': levenshtein_avg
    }

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets (kept for comparison)"""
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
    """Calculate ROUGE and fuzzy similarities between reference salt and all other salts"""
    print(f"\n{'='*60}")
    print(f"CALCULATING ROUGE & FUZZY SIMILARITIES FOR {reference_salt}")
    print(f"{'='*60}")
    
    # Initialize fuzzy similarity calculator
    similarity_calc = FuzzySimilarityCalculator()
    
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
    
    print(f"Calculating fuzzy similarities with {len(other_salts)} other salts...")
    print("Using ROUGE, token overlap, Levenshtein, and fuzzy Jaccard metrics...")
    
    for i, salt in enumerate(other_salts):
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{len(other_salts)} salts...")
        
        # Get neighbors for current salt
        salt_neighbors = get_salt_neighbors(salt, df)
        salt_weighted_neighbors = get_weighted_salt_neighbors(salt, df)
        
        # Calculate traditional similarities (for comparison)
        jaccard_unweighted = calculate_jaccard_similarity(ref_neighbors, salt_neighbors)
        jaccard_weighted = calculate_weighted_jaccard(ref_weighted_neighbors, salt_weighted_neighbors)
        
        # Calculate fuzzy similarities
        fuzzy_metrics = calculate_fuzzy_similarity(ref_neighbors, salt_neighbors, similarity_calc)
        
        # Calculate direct name similarity between salts
        direct_name_similarity = {
            'rouge': similarity_calc.rouge_similarity(reference_salt, salt),
            'token_overlap': similarity_calc.token_overlap_similarity(reference_salt, salt),
            'levenshtein': similarity_calc.levenshtein_similarity(reference_salt, salt)
        }
        
        # Find shared neighbors (exact and fuzzy)
        exact_intersection = ref_neighbors & salt_neighbors
        union = ref_neighbors | salt_neighbors
        
        # Get relationship context
        context = get_relationship_context(reference_salt, salt, df, exact_intersection)
        
        # Calculate combined similarity score
        # Weighted combination: 40% neighbor fuzzy similarity, 30% direct name similarity, 30% traditional Jaccard
        combined_score = (
            0.4 * fuzzy_metrics['fuzzy_jaccard'] +
            0.3 * max(direct_name_similarity.values()) +
            0.3 * jaccard_unweighted
        )
        
        results.append({
            'salt': salt,
            'neighbors_count': len(salt_neighbors),
            'exact_intersection_count': len(exact_intersection),
            'union_count': len(union),
            
            # Traditional metrics (for comparison)
            'jaccard_unweighted': jaccard_unweighted,
            'jaccard_weighted': jaccard_weighted,
            
            # Fuzzy neighbor-based metrics
            'fuzzy_jaccard': fuzzy_metrics['fuzzy_jaccard'],
            'neighbor_rouge_avg': fuzzy_metrics['rouge_avg'],
            'neighbor_token_overlap': fuzzy_metrics['token_overlap'],
            'neighbor_levenshtein_avg': fuzzy_metrics['levenshtein_avg'],
            
            # Direct name similarity
            'name_rouge': direct_name_similarity['rouge'],
            'name_token_overlap': direct_name_similarity['token_overlap'],
            'name_levenshtein': direct_name_similarity['levenshtein'],
            
            # Combined score
            'combined_similarity': combined_score,
            
            'shared_neighbors_examples': list(exact_intersection)[:10],
            'relationship_context': context[:3]  # Top 3 contexts
        })
    
    # Sort by combined similarity score
    results.sort(key=lambda x: x['combined_similarity'], reverse=True)
    
    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame([{
        'salt': r['salt'],
        'neighbors_count': r['neighbors_count'],
        'exact_intersection_count': r['exact_intersection_count'],
        'union_count': r['union_count'],
        
        # Traditional
        'jaccard_unweighted': r['jaccard_unweighted'],
        'jaccard_weighted': r['jaccard_weighted'],
        
        # Fuzzy neighbor-based
        'fuzzy_jaccard': r['fuzzy_jaccard'],
        'neighbor_rouge_avg': r['neighbor_rouge_avg'],
        'neighbor_token_overlap': r['neighbor_token_overlap'],
        'neighbor_levenshtein_avg': r['neighbor_levenshtein_avg'],
        
        # Direct name similarity
        'name_rouge': r['name_rouge'],
        'name_token_overlap': r['name_token_overlap'],
        'name_levenshtein': r['name_levenshtein'],
        
        # Combined
        'combined_similarity': r['combined_similarity'],
        
        'shared_neighbors_examples': '; '.join(map(str, r['shared_neighbors_examples'])),
        'relationship_context': '; '.join(r['relationship_context'])
    } for r in results])
    
    # Save full results
    full_results_path = os.path.join(output_dir, f'rouge_fuzzy_similarities_{reference_salt}_vs_all_salts.csv')
    results_df.to_csv(full_results_path, index=False)
    
    # Save top 50 results
    top_results_path = os.path.join(output_dir, f'rouge_fuzzy_similarities_{reference_salt}_top50.csv')
    results_df.head(50).to_csv(top_results_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Full results: {full_results_path}")
    print(f"  Top 50: {top_results_path}")
    
    return results, results_df

def print_top_similarities(results, reference_salt, top_n=20):
    """Print top N most similar salts using fuzzy metrics"""
    print(f"\n{'='*100}")
    print(f"TOP {top_n} SALTS MOST SIMILAR TO {reference_salt} (ROUGE & FUZZY METRICS)")
    print(f"{'='*100}")
    
    print(f"{'Rank':<4} {'Salt':<20} {'Combined':<9} {'Fuzzy Jaccard':<13} {'Name ROUGE':<11} {'Exact Jaccard':<13} {'Shared':<7}")
    print("-" * 100)
    
    for i, result in enumerate(results[:top_n]):
        rank = i + 1
        salt = result['salt']
        combined = result['combined_similarity']
        fuzzy_jaccard = result['fuzzy_jaccard']
        name_rouge = result['name_rouge']
        exact_jaccard = result['jaccard_unweighted']
        shared = result['exact_intersection_count']
        
        print(f"{rank:<4} {salt:<20} {combined:<9.4f} {fuzzy_jaccard:<13.4f} {name_rouge:<11.4f} {exact_jaccard:<13.4f} {shared:<7}")
    
    # Show detailed similarity breakdown for top 5
    print(f"\nDetailed similarity breakdown for top 5 salts:")
    print("-" * 100)
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. {result['salt']} (Combined Score: {result['combined_similarity']:.4f})")
        print(f"   • Fuzzy Neighbor Jaccard: {result['fuzzy_jaccard']:.4f}")
        print(f"   • Neighbor ROUGE Avg: {result['neighbor_rouge_avg']:.4f}")
        print(f"   • Neighbor Token Overlap: {result['neighbor_token_overlap']:.4f}")
        print(f"   • Name ROUGE: {result['name_rouge']:.4f}")
        print(f"   • Name Levenshtein: {result['name_levenshtein']:.4f}")
        print(f"   • Traditional Jaccard: {result['jaccard_unweighted']:.4f}")
        
        if result['shared_neighbors_examples']:
            print(f"   • Shared neighbors: {', '.join(result['shared_neighbors_examples'][:5])}")
        if result['relationship_context']:
            print(f"   • Relationship context: {'; '.join(result['relationship_context'])}")

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
    """Main function to run the ROUGE and fuzzy similarity analysis"""
    print("Salt Knowledge Graph ROUGE & Fuzzy Similarity Analysis")
    print("="*70)
    print("Using ROUGE, Levenshtein, Token Overlap, and Fuzzy Jaccard metrics")
    print("="*70)
    
    # Check library availability
    print("📋 Checking available libraries:")
    print(f"   - ROUGE: {'✅ Available' if ROUGE_AVAILABLE else '❌ Not available (using fallback)'}")
    print(f"   - scikit-learn: {'✅ Available' if SKLEARN_AVAILABLE else '❌ Not available'}")
    print()
    
    # Load knowledge graph triples
    triples = load_knowledge_graph_triples()
    if triples is None:
        return
    
    # Convert to DataFrame
    df = convert_triples_to_dataframe(triples)
    
    # Analyze relationship types
    predicate_counts = analyze_relationship_types(df)
    
    # Define reference salt (still focusing on LiTFSI/TFSI as requested)
    reference_salt = "LiTFSI"
    
    # Calculate similarities
    results, results_df = calculate_salt_similarities(reference_salt, df)
    
    if results:
        # Print top results
        print_top_similarities(results, reference_salt, top_n=20)
        
        # Additional analysis
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS - FUZZY SIMILARITY METRICS")
        print(f"{'='*80}")
        print(f"Reference salt: {reference_salt}")
        print(f"Total salts compared: {len(results)}")
        print()
        print("Combined Similarity Score:")
        print(f"  Average: {results_df['combined_similarity'].mean():.4f}")
        print(f"  Median:  {results_df['combined_similarity'].median():.4f}")
        print(f"  Max:     {results_df['combined_similarity'].max():.4f}")
        print(f"  Salts with score > 0.2: {(results_df['combined_similarity'] > 0.2).sum()}")
        print(f"  Salts with score > 0.1: {(results_df['combined_similarity'] > 0.1).sum()}")
        print()
        print("Fuzzy Jaccard (Neighbor-based):")
        print(f"  Average: {results_df['fuzzy_jaccard'].mean():.4f}")
        print(f"  Median:  {results_df['fuzzy_jaccard'].median():.4f}")
        print(f"  Max:     {results_df['fuzzy_jaccard'].max():.4f}")
        print()
        print("Name-based ROUGE Similarity:")
        print(f"  Average: {results_df['name_rouge'].mean():.4f}")
        print(f"  Median:  {results_df['name_rouge'].median():.4f}")
        print(f"  Max:     {results_df['name_rouge'].max():.4f}")
        print()
        print("Traditional Jaccard (for comparison):")
        print(f"  Average: {results_df['jaccard_unweighted'].mean():.4f}")
        print(f"  Median:  {results_df['jaccard_unweighted'].median():.4f}")
        print(f"  Max:     {results_df['jaccard_unweighted'].max():.4f}")
        
        # Improvement analysis
        print(f"\n{'='*80}")
        print("IMPROVEMENT ANALYSIS: Fuzzy vs Traditional Jaccard")
        print(f"{'='*80}")
        
        improved_count = (results_df['combined_similarity'] > results_df['jaccard_unweighted']).sum()
        print(f"Salts with improved similarity scores: {improved_count}/{len(results)} ({100*improved_count/len(results):.1f}%)")
        
        # Show biggest improvements
        results_df['improvement'] = results_df['combined_similarity'] - results_df['jaccard_unweighted']
        top_improvements = results_df.nlargest(5, 'improvement')
        
        print("\nTop 5 biggest improvements (Combined - Traditional Jaccard):")
        for idx, row in top_improvements.iterrows():
            print(f"  {row['salt']}: +{row['improvement']:.4f} (Combined: {row['combined_similarity']:.4f}, Jaccard: {row['jaccard_unweighted']:.4f})")
        
        # Show relationship contexts for top salts
        print(f"\n{'='*80}")
        print("RELATIONSHIP CONTEXTS FOR TOP 5 SIMILAR SALTS")
        print(f"{'='*80}")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. {result['salt']} (Combined Score: {result['combined_similarity']:.4f})")
            if result['relationship_context']:
                print(f"   Shared relationship contexts: {'; '.join(result['relationship_context'])}")
            else:
                print("   No specific relationship contexts found")
            
            # Show fuzzy vs exact neighbor matching
            exact_shared = result['exact_intersection_count']
            neighbor_fuzzy = result['fuzzy_jaccard']
            print(f"   Exact shared neighbors: {exact_shared}")
            print(f"   Fuzzy neighbor similarity: {neighbor_fuzzy:.4f}")

if __name__ == "__main__":
    main()
