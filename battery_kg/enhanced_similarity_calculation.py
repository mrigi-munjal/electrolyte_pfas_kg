#!/usr/bin/env python3
"""
Enhanced Similarity Calculation for Battery Ontology

This script implements multiple similarity measures beyond Jaccard for matching
chemical entities and battery components, including ROUGE, semantic similarity,
and domain-specific matching strategies.
"""

import json
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import os
from datetime import datetime
import warnings
from difflib import SequenceMatcher
from typing import Set, Dict, List, Tuple, Any
import math

warnings.filterwarnings('ignore')

# Try to import additional libraries for advanced similarity
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

class EnhancedSimilarityCalculator:
    """Enhanced similarity calculator with multiple algorithms"""
    
    def __init__(self):
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        
        # Chemical/battery domain-specific patterns
        self.chemical_patterns = {
            'lithium_compounds': re.compile(r'\b(li|lithium)[a-z]*\b', re.IGNORECASE),
            'phosphate_compounds': re.compile(r'\b(po4|phosphate|phos)[a-z]*\b', re.IGNORECASE),
            'carbonate_compounds': re.compile(r'\b(co3|carbonate)[a-z]*\b', re.IGNORECASE),
            'fluorine_compounds': re.compile(r'\b(f|fluorine|fluor)[a-z]*\b', re.IGNORECASE),
            'battery_terms': re.compile(r'\b(battery|electrolyte|cathode|anode|cell)[a-z]*\b', re.IGNORECASE)
        }
        
        # Common chemical abbreviations and synonyms
        self.chemical_synonyms = {
            'li': 'lithium',
            'po4': 'phosphate',
            'co3': 'carbonate',
            'f': 'fluorine',
            'lifepo4': 'lithium iron phosphate',
            'licoo2': 'lithium cobalt oxide',
            'limn2o4': 'lithium manganese oxide',
            'litfsi': 'lithium bis(trifluoromethanesulfonyl)imide',
            'lipf6': 'lithium hexafluorophosphate',
            'ec': 'ethylene carbonate',
            'dmc': 'dimethyl carbonate',
            'emc': 'ethyl methyl carbonate'
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        if not isinstance(text, str):
            return str(text).lower()
        
        text = text.lower().strip()
        
        # Expand common abbreviations
        for abbrev, full_form in self.chemical_synonyms.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full_form, text)
        
        # Remove special characters but keep chemical notation
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Standard Jaccard similarity"""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def dice_similarity(self, set1: Set, set2: Set) -> float:
        """Dice similarity coefficient (often better for text)"""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0

    def overlap_coefficient(self, set1: Set, set2: Set) -> float:
        """Overlap coefficient (good for subset relationships)"""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        min_size = min(len(set1), len(set2))
        return intersection / min_size if min_size > 0 else 0.0

    def rouge_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate ROUGE similarity scores"""
        if not ROUGE_AVAILABLE or not self.rouge_scorer:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        try:
            scores = self.rouge_scorer.score(text1, text2)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Normalized Levenshtein similarity"""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for ratio
        return SequenceMatcher(None, text1, text2).ratio()

    def token_overlap_similarity(self, text1: str, text2: str) -> float:
        """Token-based overlap with preprocessing"""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        return self.dice_similarity(tokens1, tokens2)

    def ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """N-gram based similarity"""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        def get_ngrams(text: str, n: int) -> Set[str]:
            if len(text) < n:
                return {text}
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        return self.dice_similarity(ngrams1, ngrams2)

    def chemical_domain_similarity(self, text1: str, text2: str) -> float:
        """Domain-specific similarity for chemical compounds"""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        similarity_score = 0.0
        pattern_matches = 0
        
        # Check for matching chemical patterns
        for pattern_name, pattern in self.chemical_patterns.items():
            matches1 = bool(pattern.search(text1))
            matches2 = bool(pattern.search(text2))
            
            if matches1 and matches2:
                similarity_score += 1.0
            elif matches1 or matches2:
                similarity_score += 0.3  # Partial credit
            
            pattern_matches += 1
        
        # Normalize by number of patterns
        pattern_similarity = similarity_score / pattern_matches if pattern_matches > 0 else 0.0
        
        # Combine with token overlap
        token_similarity = self.token_overlap_similarity(text1, text2)
        
        # Weighted combination
        return 0.6 * token_similarity + 0.4 * pattern_similarity

    def tfidf_cosine_similarity(self, texts: List[str]) -> np.ndarray:
        """Calculate TF-IDF cosine similarity matrix"""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            # Fallback to pairwise token similarity
            n = len(texts)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        similarity_matrix[i, j] = self.token_overlap_similarity(texts[i], texts[j])
            return similarity_matrix
        
        try:
            processed_texts = [self.preprocess_text(text) for text in texts]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
            return cosine_similarity(tfidf_matrix)
        except Exception as e:
            print(f"TF-IDF calculation error: {e}")
            # Fallback
            n = len(texts)
            return np.eye(n)

    def comprehensive_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate multiple similarity measures (optimized)"""
        results = {}
        
        # Early exit for identical strings
        if text1 == text2:
            return {key: 1.0 for key in ['jaccard', 'dice', 'overlap', 'levenshtein', 
                                        'token_overlap', 'bigram', 'trigram', 
                                        'chemical_domain', 'rouge1', 'rouge2', 'rougeL', 'combined']}
        
        # Preprocess once
        proc_text1 = self.preprocess_text(text1)
        proc_text2 = self.preprocess_text(text2)
        
        # Early exit for empty strings
        if not proc_text1.strip() or not proc_text2.strip():
            return {key: 0.0 for key in ['jaccard', 'dice', 'overlap', 'levenshtein', 
                                        'token_overlap', 'bigram', 'trigram', 
                                        'chemical_domain', 'rouge1', 'rouge2', 'rougeL', 'combined']}
        
        # Token-based similarities (computed once)
        tokens1 = set(proc_text1.split())
        tokens2 = set(proc_text2.split())
        
        results['jaccard'] = self.jaccard_similarity(tokens1, tokens2)
        results['dice'] = self.dice_similarity(tokens1, tokens2)
        results['overlap'] = self.overlap_coefficient(tokens1, tokens2)
        
        # Only compute expensive similarities if basic similarity is promising
        basic_similarity = (results['jaccard'] + results['dice'] + results['overlap']) / 3
        
        if basic_similarity > 0.01:  # Only compute expensive metrics if there's some basic similarity
            # String-based similarities
            results['levenshtein'] = self.levenshtein_similarity(text1, text2)
            results['token_overlap'] = self.token_overlap_similarity(text1, text2)
            results['bigram'] = self.ngram_similarity(text1, text2, n=2)
            results['trigram'] = self.ngram_similarity(text1, text2, n=3)
            
            # Domain-specific similarity
            results['chemical_domain'] = self.chemical_domain_similarity(text1, text2)
            
            # ROUGE scores (only if basic similarity is reasonable)
            if basic_similarity > 0.05:
                rouge_scores = self.rouge_similarity(text1, text2)
                results.update(rouge_scores)
            else:
                # Use fast approximations
                results['rouge1'] = results['token_overlap']
                results['rouge2'] = results['bigram']
                results['rougeL'] = results['levenshtein']
        else:
            # Set remaining metrics to low values to save computation
            results['levenshtein'] = basic_similarity
            results['token_overlap'] = basic_similarity
            results['bigram'] = basic_similarity
            results['trigram'] = basic_similarity
            results['chemical_domain'] = 0.0
            results['rouge1'] = basic_similarity
            results['rouge2'] = basic_similarity
            results['rougeL'] = basic_similarity
        
        # Combined score (weighted average of best performing measures)
        results['combined'] = (
            0.25 * results['dice'] +
            0.20 * results['chemical_domain'] +
            0.15 * results['rougeL'] +
            0.15 * results['overlap'] +
            0.15 * results['levenshtein'] +
            0.10 * results['trigram']
        )
        
        return results

def load_knowledge_graph_triples(file_path='salt_analysis_output/salt_knowledge_graph_triples.json'):
    """Load the knowledge graph triples from JSON file"""
    try:
        print(f"📥 Loading knowledge graph triples from: {file_path}")
        with open(file_path, 'r') as f:
            triples = json.load(f)
        print(f"✅ Successfully loaded {len(triples):,} triples")
        return triples
    except Exception as e:
        print(f"❌ Error loading triples: {e}")
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
    print(f"📊 Created DataFrame with {len(df)} rows")
    print(f"   - Unique subjects (salts): {df['Subject'].nunique()}")
    print(f"   - Unique predicates: {df['Predicate'].nunique()}")
    print(f"   - Unique objects: {df['Object'].nunique()}")
    
    return df

def get_entity_neighbors(entity_name, df, relationship_types=None):
    """Get all neighbors connected to an entity through relationships"""
    neighbors = set()
    
    # Filter by relationship types if specified
    if relationship_types:
        filtered_df = df[df['Predicate'].isin(relationship_types)]
    else:
        filtered_df = df
    
    # Find all objects where entity is the subject
    subject_mask = filtered_df['Subject'] == entity_name
    subject_neighbors = set(filtered_df[subject_mask]['Object'].values)
    neighbors.update(subject_neighbors)
    
    # Find all subjects where entity is the object (reverse relationships)
    object_mask = filtered_df['Object'] == entity_name
    object_neighbors = set(filtered_df[object_mask]['Subject'].values)
    neighbors.update(object_neighbors)
    
    # Remove self-references
    neighbors.discard(entity_name)
    
    return neighbors

def calculate_enhanced_entity_similarity(entity1, entity2, df, similarity_calc, use_neighbors=True):
    """Calculate enhanced similarity between two entities"""
    results = {}
    
    # Direct name similarity
    name_similarity = similarity_calc.comprehensive_similarity(entity1, entity2)
    results['name_similarity'] = name_similarity
    
    if use_neighbors:
        # Neighbor-based similarity
        neighbors1 = get_entity_neighbors(entity1, df)
        neighbors2 = get_entity_neighbors(entity2, df)
        
        # Convert neighbor sets to text for similarity calculation
        neighbors_text1 = ' '.join(sorted(neighbors1)) if neighbors1 else ''
        neighbors_text2 = ' '.join(sorted(neighbors2)) if neighbors2 else ''
        
        neighbor_similarity = similarity_calc.comprehensive_similarity(neighbors_text1, neighbors_text2)
        results['neighbor_similarity'] = neighbor_similarity
        
        # Combined similarity (name + neighbors)
        results['combined_similarity'] = {}
        for metric in name_similarity:
            name_score = name_similarity[metric]
            neighbor_score = neighbor_similarity.get(metric, 0.0)
            # Weight: 70% name similarity, 30% neighbor similarity
            results['combined_similarity'][metric] = 0.7 * name_score + 0.3 * neighbor_score
    
    return results

def find_similar_entities(target_entity, df, similarity_calc, top_k=10, min_similarity=0.1):
    """Find entities most similar to the target entity (optimized)"""
    print(f"🔍 Finding entities similar to: {target_entity}")
    
    # Get all unique entities
    all_subjects = set(df['Subject'].unique())
    all_objects = set(df['Object'].unique())
    all_entities = all_subjects | all_objects
    all_entities.discard(target_entity)  # Remove target itself
    
    print(f"   📊 Comparing against {len(all_entities):,} entities")
    
    # Precompute target entity neighbors once
    target_neighbors = get_entity_neighbors(target_entity, df)
    target_neighbors_text = ' '.join(sorted(target_neighbors)) if target_neighbors else ''
    
    similarities = []
    processed_count = 0
    start_time = datetime.now()
    
    for i, entity in enumerate(all_entities):
        if (i + 1) % 1000 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(all_entities) - i - 1) / rate if rate > 0 else 0
            print(f"   Processing entity {i+1:,}/{len(all_entities):,} (Rate: {rate:.1f}/sec, ETA: {remaining/60:.1f} min)")
        
        try:
            # Quick name similarity check first
            name_similarity = similarity_calc.comprehensive_similarity(target_entity, entity)
            combined_name_score = name_similarity.get('combined', 0.0)
            
            # Skip entities with very low name similarity to save time
            if combined_name_score < 0.02:  # Early exit for very dissimilar names
                continue
            
            # Calculate neighbor similarity only if name similarity is promising
            entity_neighbors = get_entity_neighbors(entity, df)
            entity_neighbors_text = ' '.join(sorted(entity_neighbors)) if entity_neighbors else ''
            
            neighbor_similarity = similarity_calc.comprehensive_similarity(target_neighbors_text, entity_neighbors_text)
            
            # Combined similarity (name + neighbors)
            combined_similarity = {}
            for metric in name_similarity:
                name_score = name_similarity[metric]
                neighbor_score = neighbor_similarity.get(metric, 0.0)
                # Weight: 70% name similarity, 30% neighbor similarity
                combined_similarity[metric] = 0.7 * name_score + 0.3 * neighbor_score
            
            combined_score = combined_similarity.get('combined', 0.0)
            
            if combined_score >= min_similarity:
                similarities.append({
                    'entity': entity,
                    'combined_score': combined_score,
                    'name_similarity': name_similarity,
                    'neighbor_similarity': neighbor_similarity,
                    'combined_similarity': combined_similarity
                })
                processed_count += 1
        
        except Exception as e:
            print(f"   ⚠️ Error processing {entity}: {e}")
            continue
    
    # Sort by combined score
    similarities.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print(f"✅ Found {len(similarities)} entities with similarity >= {min_similarity}")
    print(f"   📈 Processed {processed_count} entities that passed initial filtering")
    
    return similarities

def main():
    """Main execution function"""
    print("🚀 ENHANCED SIMILARITY CALCULATION FOR BATTERY ONTOLOGY")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check library availability
    print("📋 Checking available libraries:")
    print(f"   - ROUGE: {'✅ Available' if ROUGE_AVAILABLE else '❌ Not available'}")
    print(f"   - scikit-learn: {'✅ Available' if SKLEARN_AVAILABLE else '❌ Not available'}")
    print()
    
    # Initialize similarity calculator
    similarity_calc = EnhancedSimilarityCalculator()
    
    # Load data
    triples = load_knowledge_graph_triples()
    if not triples:
        print("❌ Failed to load knowledge graph triples")
        return
    
    df = convert_triples_to_dataframe(triples)
    print()
    
    # Example: Find entities similar to LiTFSI using enhanced similarity
    target_entities = ['LiTFSI', 'LiFePO4', 'LiPF6']
    
    print(f"🎯 Target entities to analyze: {target_entities}")
    print(f"⏱️  Each entity will be compared against ~{len(set(df['Subject'].unique()) | set(df['Object'].unique())):,} entities")
    print(f"🚀 Performance optimizations enabled: early filtering, progress tracking, checkpointing")
    print()
    
    # Create output directory
    output_dir = 'enhanced_similarity_output'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Check for existing results to resume from
    for target in target_entities:
        checkpoint_file = os.path.join(output_dir, f'{target}_enhanced_similarities.json')
        if os.path.exists(checkpoint_file):
            print(f"📂 Found existing results for {target}, skipping...")
            try:
                with open(checkpoint_file, 'r') as f:
                    all_results[target] = json.load(f)
                continue
            except:
                print(f"⚠️ Could not load {checkpoint_file}, will recalculate...")
    
    # Process remaining targets
    for target in target_entities:
        if target in all_results:
            continue  # Skip already processed
            
        print(f"\n🎯 Analyzing similarities for: {target}")
        print("-" * 40)
        
        similar_entities = find_similar_entities(
            target, df, similarity_calc, 
            top_k=50, min_similarity=0.05  # Increased top_k and lowered min_similarity
        )
        
        all_results[target] = similar_entities
        
        # Save results for this target immediately (checkpoint)
        output_file = os.path.join(output_dir, f'{target}_enhanced_similarities.json')
        with open(output_file, 'w') as f:
            json.dump(similar_entities, f, indent=2, default=str)
        print(f"💾 Saved results to: {output_file}")
        
        # Print top 5 results
        print(f"\n📊 Top 5 most similar entities to {target}:")
        for i, result in enumerate(similar_entities[:5], 1):
            print(f"   {i}. {result['entity']}")
            print(f"      Combined Score: {result['combined_score']:.4f}")
            print(f"      Name Similarity (combined): {result['name_similarity']['combined']:.4f}")
            print(f"      Neighbor Similarity (combined): {result['neighbor_similarity']['combined']:.4f}")
            print()
    
    # Create summary comparison
    print("\n📈 Creating enhanced similarity comparison...")
    
    # Comparison across different similarity metrics
    comparison_data = []
    
    for target, results in all_results.items():
        for result in results[:10]:  # Top 10 for each target
            row = {
                'target_entity': target,
                'similar_entity': result['entity'],
                'combined_score': result['combined_score']
            }
            
            # Add all similarity metrics
            for metric, score in result['combined_similarity'].items():
                row[f'combined_{metric}'] = score
            
            for metric, score in result['name_similarity'].items():
                row[f'name_{metric}'] = score
            
            comparison_data.append(row)
    
    # Save comprehensive comparison
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = os.path.join(output_dir, 'enhanced_similarity_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"💾 Saved comprehensive comparison to: {comparison_file}")
    
    # Summary statistics
    print(f"\n📊 ANALYSIS SUMMARY")
    print("=" * 40)
    for target, results in all_results.items():
        print(f"🎯 {target}:")
        if results:
            scores = [r['combined_score'] for r in results]
            print(f"   - Found {len(results)} similar entities")
            print(f"   - Max similarity: {max(scores):.4f}")
            print(f"   - Avg similarity: {np.mean(scores):.4f}")
            print(f"   - Min similarity: {min(scores):.4f}")
        else:
            print(f"   - No similar entities found")
        print()
    
    print(f"✅ Enhanced similarity analysis completed!")
    print(f"📁 All results saved in: {output_dir}/")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
