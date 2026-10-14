#!/usr/bin/env python3
"""
Optimized Battery Ontology Link Plausibility Analysis

This script performs link plausibility scoring and relation prediction tasks on the
canonical battery ontology relationships dataset. It includes:

1. Link plausibility scoring using simple baselines:
   - PMI Baseline: Measures statistical association between entities via relationship co-occurrence patterns
   - TF-IDF + k-NN: Creates relationship fingerprints and finds entities with similar patterns
   - Graph Neighborhood: Predicts links using structural similarity and shared graph neighbors
   
2. Evaluation metrics for link prediction tasks

"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import math
import random
import warnings
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
from itertools import combinations
import scipy.sparse as sp

warnings.filterwarnings('ignore')

class OptimizedBatteryOntologyLinkAnalyzer:
    """Optimized class for memory-efficient link plausibility analysis"""
    
    def __init__(self, data_path='battery_ontology_analysis/relationships_data_canonical.csv'):
        self.data_path = data_path
        self.df = None
        self.graph = None
        self.entity_counts = {}
        self.relation_counts = {}
        self.cooccurrence_matrix = {}
        self.pmi_scores = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.entity_to_idx = {}
        
        # Optimization parameters
        self.max_entities_for_pmi = 10000  # Limit PMI calculation to top entities
        self.chunk_size = 1000  # Process in chunks to save memory
        self.min_entity_frequency = 5  # Filter out rare entities
        
    def print_memory_usage(self, stage_name):
        """Print current memory usage"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024
        print(f"💾 Memory usage at {stage_name}: {memory_gb:.2f} GB")
        
    def save_progress_checkpoint(self, stage_name, data_dict):
        """Save intermediate results to avoid losing progress"""
        checkpoint_file = f"battery_ontology_analysis/checkpoint_{stage_name}.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(data_dict, f, indent=2, default=str)
            print(f"✅ Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
        
    def load_and_filter_data(self):
        """Load data and filter for efficiency"""
        print("📊 Loading canonical relationships data...")
        self.df = pd.read_csv(self.data_path)
        
        initial_size = len(self.df)
        print(f"✅ Loaded {initial_size:,} canonical relationships")
        
        # Filter out rows with missing values
        self.df = self.df.dropna(subset=['subject', 'object', 'relation'])
        
        # Get entity frequency statistics
        all_entities = list(self.df['subject']) + list(self.df['object'])
        entity_counts = Counter(all_entities)
        
        # Filter entities by minimum frequency
        frequent_entities = {entity for entity, count in entity_counts.items() 
                           if count >= self.min_entity_frequency}
        
        # Filter dataframe to only include frequent entities
        mask = (self.df['subject'].isin(frequent_entities)) & (self.df['object'].isin(frequent_entities))
        self.df = self.df[mask].copy()
        
        print(f"   - After filtering: {len(self.df):,} relationships")
        print(f"   - Unique subjects: {self.df['subject'].nunique():,}")
        print(f"   - Unique objects: {self.df['object'].nunique():,}")
        print(f"   - Unique relations: {self.df['relation'].nunique():,}")
        
        # Force garbage collection
        gc.collect()
        
    def create_knowledge_graph(self):
        """Create knowledge graph efficiently"""
        print("🕸️  Creating knowledge graph...")
        
        self.graph = nx.MultiDiGraph()
        
        # Add edges in chunks to manage memory
        chunk_size = 10000
        for i in range(0, len(self.df), chunk_size):
            chunk = self.df.iloc[i:i+chunk_size]
            for _, row in chunk.iterrows():
                self.graph.add_edge(row['subject'], row['object'], relation=row['relation'])
            
            if i % 50000 == 0:
                print(f"   Added {min(i+chunk_size, len(self.df)):,}/{len(self.df):,} edges")
                
        print(f"✅ Created graph with {self.graph.number_of_nodes():,} nodes and {self.graph.number_of_edges():,} edges")
        print(f"   - Average degree: {(2 * self.graph.number_of_edges()) / self.graph.number_of_nodes():.2f}")
        print(f"   - Number of connected components: {nx.number_weakly_connected_components(self.graph):,}")
        
    def calculate_entity_statistics(self):
        """Calculate entity and relation statistics efficiently"""
        print("📈 Calculating entity and relation statistics...")
        
        # Entity counts
        all_entities = list(self.df['subject']) + list(self.df['object'])
        self.entity_counts = Counter(all_entities)
        
        # Relation counts
        self.relation_counts = Counter(self.df['relation'])
        
        print(f"✅ Statistics calculated:")
        print(f"   - Total unique entities: {len(self.entity_counts):,}")
        print(f"   - Most frequent entity: {self.entity_counts.most_common(1)[0]}")
        print(f"   - Most frequent relation: {self.relation_counts.most_common(1)[0]}")
        
    def calculate_optimized_pmi(self):
        """Calculate PMI scores efficiently for top entities only"""
        print("🔢 Calculating co-occurrence and PMI scores (optimized)...")
        
        # Get top entities for PMI calculation
        top_entities = [entity for entity, count in self.entity_counts.most_common(self.max_entities_for_pmi)]
        print(f"   - Focusing on top {len(top_entities):,} entities for PMI calculation")
        
        # Build relation contexts for top entities only
        entity_relation_contexts = defaultdict(set)
        
        # Process in chunks to manage memory
        chunk_size = 10000
        for i in range(0, len(self.df), chunk_size):
            chunk = self.df.iloc[i:i+chunk_size]
            for _, row in chunk.iterrows():
                subject, relation, obj = row['subject'], row['relation'], row['object']
                if subject in top_entities:
                    entity_relation_contexts[subject].add(relation)
                if obj in top_entities:
                    entity_relation_contexts[obj].add(relation)
                    
        # Calculate co-occurrence efficiently using combinations
        print("   - Calculating co-occurrence matrix...")
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Process entities in chunks
        entity_chunks = [top_entities[i:i+self.chunk_size] for i in range(0, len(top_entities), self.chunk_size)]
        
        total_pairs = 0
        for chunk_idx, entity_chunk in enumerate(entity_chunks):
            print(f"   - Processing chunk {chunk_idx+1:,}/{len(entity_chunks):,}")
            
            for i, entity1 in enumerate(entity_chunk):
                # Only compare within chunk and with remaining entities to avoid duplicates
                for entity2 in entity_chunk[i+1:]:
                    shared_relations = entity_relation_contexts[entity1] & entity_relation_contexts[entity2]
                    if shared_relations:
                        count = len(shared_relations)
                        cooccurrence[entity1][entity2] = count
                        cooccurrence[entity2][entity1] = count
                        total_pairs += 1
                        
                # Compare with entities in later chunks
                for other_chunk in entity_chunks[chunk_idx+1:]:
                    for entity2 in other_chunk:
                        shared_relations = entity_relation_contexts[entity1] & entity_relation_contexts[entity2]
                        if shared_relations:
                            count = len(shared_relations)
                            cooccurrence[entity1][entity2] = count
                            cooccurrence[entity2][entity1] = count
                            total_pairs += 1
                            
            # Force garbage collection after each chunk
            gc.collect()
        
        self.cooccurrence_matrix = dict(cooccurrence)
        
        # Calculate PMI scores efficiently
        print("   - Calculating PMI scores...")
        self.pmi_scores = {}
        
        total_contexts = sum(len(contexts) for contexts in entity_relation_contexts.values())
        
        for entity1 in self.cooccurrence_matrix:
            self.pmi_scores[entity1] = {}
            p_x = len(entity_relation_contexts[entity1]) / total_contexts if total_contexts > 0 else 0
            
            for entity2, coocurr_count in self.cooccurrence_matrix[entity1].items():
                p_y = len(entity_relation_contexts[entity2]) / total_contexts if total_contexts > 0 else 0
                p_xy = coocurr_count / total_pairs if total_pairs > 0 else 0
                # co-occurrence example:
                # entity_relation_contexts["lithium_ion_battery"] = {"contains", "has_component"}
                # entity_relation_contexts["electrolyte"] = {"contains", "enables"}
                # coocurr_count = 1 (shared "contains")
                
                if p_x > 0 and p_y > 0 and p_xy > 0:
                    pmi = math.log(p_xy / (p_x * p_y))
                    self.pmi_scores[entity1][entity2] = pmi
                else:
                    self.pmi_scores[entity1][entity2] = 0
        
        print(f"✅ Co-occurrence and PMI calculation complete")
        print(f"   - Entity pairs with co-occurrence: {total_pairs:,}")
        print(f"   - Entities with PMI scores: {len(self.pmi_scores):,}")
        
        # Force garbage collection
        gc.collect()
        
    def calculate_tfidf_features(self):
        """Calculate TF-IDF features over entity neighborhoods (sequential for reliability)"""
        print("📝 Calculating TF-IDF features over neighborhoods...")
        # for this TFIDF, let's say document is LiFePo4, and document has content like: "HAS_PROPERTY_low_specific_capacity IMPROVES_energy_density inv_CONTAINS_lithium_ion_battery inv_HAS_COMPONENT_cathode"

        # Get entities with sufficient frequency
        frequent_entities = [entity for entity, count in self.entity_counts.items() 
                           if count >= self.min_entity_frequency]
        
        print(f"   - Processing {len(frequent_entities):,} frequent entities for TF-IDF")
        
        # Create neighborhood documents sequentially (more reliable)
        entity_documents = {}
        
        for i, entity in enumerate(frequent_entities):
            if i % 1000 == 0:
                print(f"   - Processing entity {i+1:,}/{len(frequent_entities):,}")
                
            neighbors = []
            
            # Get neighbors from graph (more efficient than DataFrame filtering)
            if entity in self.graph:
                # Outgoing edges
                for neighbor in self.graph.successors(entity):
                    edge_data = self.graph.get_edge_data(entity, neighbor)
                    for key, attrs in edge_data.items():
                        neighbors.append(f"{attrs['relation']}_{neighbor}")
                
                # Incoming edges
                for neighbor in self.graph.predecessors(entity):
                    edge_data = self.graph.get_edge_data(neighbor, entity)
                    for key, attrs in edge_data.items():
                        neighbors.append(f"inv_{attrs['relation']}_{neighbor}")
            
            document = ' '.join(neighbors)
            if document.strip():  # Only keep non-empty documents
                entity_documents[entity] = document
                
            # Periodic garbage collection
            if i % 5000 == 0:
                gc.collect()

        # Calculate TF-IDF; for this part we DO need the graph representation unlike the PMI part
        if entity_documents:
            entities = list(entity_documents.keys())
            documents = [entity_documents[entity] for entity in entities]
            
            print(f"   - Computing TF-IDF for {len(documents):,} entity documents")
            
            # Use memory-efficient TF-IDF parameters
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=20000,  # Reduced from 50000 to save memory
                min_df=3,  # Increased from 2 to ignore very rare terms; Minimum Document Frequency
                # Ignores terms that appear in fewer than 3 entity documents
                max_df=0.9,  # Reduced from 0.95 to ignore common terms; Maximum Document Frequency
                # Ignores terms that appear in more than 90% of entity documents
                # Purpose: Filters out very common patterns that don't help distinguish between entities
                ngram_range=(1, 1),  # Only unigrams to save memory
                token_pattern=r'\S+'  # Match any non-whitespace sequence
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            self.entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
            
            print(f"✅ TF-IDF calculation complete")
            print(f"   - Entities with TF-IDF features: {len(entities):,}")
            print(f"   - Feature dimensions: {self.tfidf_matrix.shape[1]:,}")
            print(f"   - Matrix sparsity: {(1 - self.tfidf_matrix.nnz / np.prod(self.tfidf_matrix.shape)):.3f}")
        else:
            print("❌ No valid entity documents found for TF-IDF")
            
        # Force garbage collection
        gc.collect()
        
    def create_test_set(self, test_ratio=0.1, random_seed=42):
        """Create train/test split for link prediction evaluation"""
        print(f"🔄 Creating train/test split (test ratio: {test_ratio})...")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create positive examples (existing relationships)
        positive_examples = []
        for _, row in self.df.iterrows():
            positive_examples.append((row['subject'], row['relation'], row['object'], 1))
        
        # Create negative examples (non-existing relationships)
        print("🔄 Generating negative examples...")
        negative_examples = []
        entities = list(self.entity_counts.keys())
        relations = list(self.relation_counts.keys())
        
        # Generate negative examples by random sampling
        existing_triples = set((row['subject'], row['relation'], row['object']) for _, row in self.df.iterrows())
        
        target_negative_count = len(positive_examples)
        attempts = 0
        max_attempts = target_negative_count * 10
        
        while len(negative_examples) < target_negative_count and attempts < max_attempts:
            subj = random.choice(entities)
            rel = random.choice(relations)
            obj = random.choice(entities)
            
            if (subj, rel, obj) not in existing_triples and subj != obj:
                negative_examples.append((subj, rel, obj, 0))
            
            attempts += 1
        
        # Combine positive and negative examples
        all_examples = positive_examples + negative_examples
        random.shuffle(all_examples)
        # Both train and test have equal positive/negative ratios
        
        # Split into train/test
        train_examples, test_examples = train_test_split(
            all_examples, test_size=test_ratio, random_state=random_seed, stratify=[ex[3] for ex in all_examples]
            # What it does:

            # ex[3] = the label (1 for positive, 0 for negative)
            # Extracts all labels: [1, 0, 1, 0, 1, 0, ...]
            # Forces train and test to have same positive/negative ratio
        )
        
        print(f"✅ Test set created:")
        print(f"   - Training examples: {len(train_examples):,} ({len([ex for ex in train_examples if ex[3]==1]):,} positive)")
        print(f"   - Test examples: {len(test_examples):,} ({len([ex for ex in test_examples if ex[3]==1]):,} positive)")
        
        return train_examples, test_examples
    
    def evaluate_pmi_baseline(self, test_examples):
        """Evaluate PMI-based link prediction"""
        print("🔍 Evaluating PMI baseline...")
        
        predictions = []
        true_labels = []
        
        for subj, rel, obj, label in test_examples:
            # PMI-based score
            if subj in self.pmi_scores and obj in self.pmi_scores[subj]:
                score = self.pmi_scores[subj][obj]
            elif obj in self.pmi_scores and subj in self.pmi_scores[obj]:
                score = self.pmi_scores[obj][subj]
            else:
                score = 0
            
            predictions.append(score)
            true_labels.append(label)
        
        # Convert scores to binary predictions using median threshold
        threshold = np.median([p for p in predictions if p > 0]) if any(p > 0 for p in predictions) else 0
        # Recommendation: Use optimal F1 threshold for best research results, but also report median for comparison. So for that I will need a validation set
        # For now, using median threshold for simplicity
        binary_predictions = [1 if p > threshold else 0 for p in predictions]
        
        # Calculate metrics
        precision = precision_score(true_labels, binary_predictions, zero_division=0)
        recall = recall_score(true_labels, binary_predictions, zero_division=0)
        f1 = f1_score(true_labels, binary_predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(true_labels, predictions)
        except:
            auc = 0.5
        
        results = {
            'method': 'PMI Baseline',
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': threshold
        }
        
        print(f"✅ PMI Baseline Results:")
        print(f"   - Precision: {precision:.3f}")
        print(f"   - Recall: {recall:.3f}")
        print(f"   - F1-Score: {f1:.3f}")
        print(f"   - AUC: {auc:.3f}")
        
        return results
    
    def evaluate_tfidf_knn_baseline(self, test_examples, k=5):
        """Evaluate TF-IDF + k-NN baseline"""
        # If "low specific capacity" is in top-k, score=1.0; else score=cosine_similarity
        print(f"🔍 Evaluating TF-IDF + {k}-NN baseline...")
        
        # Check if TF-IDF data is available
        if self.tfidf_matrix is None or self.entity_to_idx is None:
            print(f"❌ TF-IDF features not available")
            return None
        
        # Prepare data for k-NN
        entities_with_features = list(self.entity_to_idx.keys())
        if len(entities_with_features) < k:
            print(f"❌ Not enough entities with TF-IDF features for k={k}")
            return None
        
        # Train k-NN model
        knn_model = NearestNeighbors(n_neighbors=min(k, len(entities_with_features)), metric='cosine')
        knn_model.fit(self.tfidf_matrix)
        
        predictions = []
        true_labels = []
        
        for subj, rel, obj, label in test_examples:
            score = 0
            
            # Check if both entities have TF-IDF features
            if subj in self.entity_to_idx and obj in self.entity_to_idx:
                subj_idx = self.entity_to_idx[subj]
                obj_idx = self.entity_to_idx[obj]
                
                subj_features = self.tfidf_matrix[subj_idx:subj_idx+1]
                obj_features = self.tfidf_matrix[obj_idx:obj_idx+1]
                
                # Find k-nearest neighbors for subject
                distances, indices = knn_model.kneighbors(subj_features)
                
                # Check if object is among the neighbors
                if obj_idx in indices[0]:
                    score = 1.0
                else:
                    # Calculate cosine similarity using sklearn's implementation
                    # This handles sparse matrices correctly
                    cosine_sim = cosine_similarity(subj_features, obj_features)[0, 0]
                    score = max(0, cosine_sim)
            
            predictions.append(score)
            true_labels.append(label)
        
        # Convert scores to binary predictions
        threshold = np.median([p for p in predictions if p > 0]) if any(p > 0 for p in predictions) else 0.5
        binary_predictions = [1 if p > threshold else 0 for p in predictions]
        
        # Calculate metrics
        precision = precision_score(true_labels, binary_predictions, zero_division=0)
        recall = recall_score(true_labels, binary_predictions, zero_division=0)
        f1 = f1_score(true_labels, binary_predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(true_labels, predictions)
        except:
            auc = 0.5
        
        results = {
            'method': f'TF-IDF + {k}-NN',
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': threshold,
            'k': k
        }
        
        print(f"✅ TF-IDF + {k}-NN Results:")
        print(f"   - Precision: {precision:.3f}")
        print(f"   - Recall: {recall:.3f}")
        print(f"   - F1-Score: {f1:.3f}")
        print(f"   - AUC: {auc:.3f}")
        
        return results
    
    def evaluate_graph_neighborhood_baseline(self, test_examples):
        """Evaluate graph neighborhood-based prediction"""
        print("🔍 Evaluating Graph Neighborhood baseline...")
        
        predictions = []
        true_labels = []
        
        for subj, rel, obj, label in test_examples:
            score = 0
            
            if self.graph.has_node(subj) and self.graph.has_node(obj):
                # Calculate neighborhood overlap
                subj_neighbors = set(self.graph.neighbors(subj))
                obj_neighbors = set(self.graph.predecessors(obj))  # Entities that point to obj
                
                # Jaccard similarity of neighborhoods
                intersection = len(subj_neighbors & obj_neighbors)
                union = len(subj_neighbors | obj_neighbors)
                
                if union > 0:
                    score = intersection / union
                
                # Bonus for direct connection
                if self.graph.has_edge(subj, obj):
                    score += 0.5
            
            predictions.append(score)
            true_labels.append(label)
        
        # Convert scores to binary predictions
        threshold = np.median([p for p in predictions if p > 0]) if any(p > 0 for p in predictions) else 0.1
        binary_predictions = [1 if p > threshold else 0 for p in predictions]
        
        # Calculate metrics
        precision = precision_score(true_labels, binary_predictions, zero_division=0)
        recall = recall_score(true_labels, binary_predictions, zero_division=0)
        f1 = f1_score(true_labels, binary_predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(true_labels, predictions)
        except:
            auc = 0.5
        
        results = {
            'method': 'Graph Neighborhood',
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': threshold
        }
        
        print(f"✅ Graph Neighborhood Results:")
        print(f"   - Precision: {precision:.3f}")
        print(f"   - Recall: {recall:.3f}")
        print(f"   - F1-Score: {f1:.3f}")
        print(f"   - AUC: {auc:.3f}")
        
        return results
    
    def analyze_relation_patterns(self):
        """Analyze patterns in relations for different entity types"""
        print("🔍 Analyzing relation patterns...")
        
        # Group relations by frequency
        relation_stats = {}
        for relation, count in self.relation_counts.most_common(20):
            # Find common subject and object types for this relation
            rel_data = self.df[self.df['relation'] == relation]
            
            top_subjects = rel_data['subject'].value_counts().head(5)
            top_objects = rel_data['object'].value_counts().head(5)
            
            relation_stats[relation] = {
                'count': count,
                'top_subjects': top_subjects.to_dict(),
                'top_objects': top_objects.to_dict(),
                'unique_subjects': rel_data['subject'].nunique(),
                'unique_objects': rel_data['object'].nunique()
            }
        
        print(f"✅ Relation pattern analysis complete for top 20 relations")
        return relation_stats
    
    def create_visualizations(self, all_results):
        """Create visualizations of the results"""
        print("📊 Creating result visualizations...")
        
        # Performance comparison plot
        methods = [result['method'] for result in all_results if result is not None]
        f1_scores = [result['f1'] for result in all_results if result is not None]
        precision_scores = [result['precision'] for result in all_results if result is not None]
        recall_scores = [result['recall'] for result in all_results if result is not None]
        auc_scores = [result['auc'] for result in all_results if result is not None]
        
        if methods:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Battery Ontology Link Prediction Baseline Results', fontsize=16, fontweight='bold')
            
            # F1 Score
            axes[0, 0].bar(methods, f1_scores, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('F1 Score')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Precision
            axes[0, 1].bar(methods, precision_scores, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Precision')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Recall
            axes[1, 0].bar(methods, recall_scores, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Recall')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # AUC
            axes[1, 1].bar(methods, auc_scores, color='gold', alpha=0.7)
            axes[1, 1].set_title('AUC')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('battery_ontology_analysis/link_analysis/baseline_comparison.png', 
                       dpi=300, bbox_inches='tight')
            print("✅ Saved baseline comparison plot")
        
        # Entity frequency distribution
        plt.figure(figsize=(12, 8))
        top_entities = Counter(self.entity_counts).most_common(20)
        entities, counts = zip(*top_entities)
        
        plt.barh(range(len(entities)), counts)
        plt.yticks(range(len(entities)), entities)
        plt.xlabel('Frequency')
        plt.title('Top 20 Most Frequent Entities in Battery Ontology')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('battery_ontology_analysis/link_analysis/entity_frequency.png', 
                   dpi=300, bbox_inches='tight')
        print("✅ Saved entity frequency plot")
    
    def save_results(self, all_results, relation_stats):
        """Save all results to files"""
        print("💾 Saving results...")
        
        # Save baseline results
        results_df = pd.DataFrame([r for r in all_results if r is not None])
        results_df.to_csv('battery_ontology_analysis/link_analysis/baseline_results.csv', index=False)
        
        # Save relation patterns
        with open('battery_ontology_analysis/link_analysis/relation_patterns.json', 'w') as f:
            json.dump(relation_stats, f, indent=2)
        
        # Save detailed statistics
        stats = {
            'dataset_stats': {
                'total_relationships': len(self.df),
                'unique_entities': len(self.entity_counts),
                'unique_relations': len(self.relation_counts),
                'graph_nodes': self.graph.number_of_nodes(),
                'graph_edges': self.graph.number_of_edges()
            },
            'top_entities': dict(Counter(self.entity_counts).most_common(20)),
            'top_relations': dict(self.relation_counts.most_common(20))
        }
        
        with open('battery_ontology_analysis/link_analysis/dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate report
        report_path = 'battery_ontology_analysis/link_analysis/link_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("BATTERY ONTOLOGY LINK PLAUSIBILITY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total relationships: {len(self.df):,}\n")
            f.write(f"Unique entities: {len(self.entity_counts):,}\n")
            f.write(f"Unique relations: {len(self.relation_counts):,}\n")
            f.write(f"Graph nodes: {self.graph.number_of_nodes():,}\n")
            f.write(f"Graph edges: {self.graph.number_of_edges():,}\n\n")
            
            f.write("BASELINE PERFORMANCE RESULTS\n")
            f.write("-" * 30 + "\n")
            for result in all_results:
                if result is not None:
                    f.write(f"{result['method']}:\n")
                    f.write(f"  Precision: {result['precision']:.3f}\n")
                    f.write(f"  Recall: {result['recall']:.3f}\n")
                    f.write(f"  F1-Score: {result['f1']:.3f}\n")
                    f.write(f"  AUC: {result['auc']:.3f}\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 15 + "\n")
            f.write("- baseline_results.csv\n")
            f.write("- relation_patterns.json\n")
            f.write("- dataset_statistics.json\n")
            f.write("- baseline_comparison.png\n")
            f.write("- entity_frequency.png\n")
            f.write("- link_analysis_report.txt (this file)\n")
        
        print(f"✅ Results saved to battery_ontology_analysis/link_analysis/")

def main():
    """Main function to run the link plausibility analysis"""
    print("🔗 BATTERY ONTOLOGY LINK PLAUSIBILITY ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize optimized analyzer
    analyzer = OptimizedBatteryOntologyLinkAnalyzer()
    analyzer.print_memory_usage("initialization")
    
    # Create output directory early
    os.makedirs('battery_ontology_analysis/link_analysis', exist_ok=True)
    print(f"📁 Created output directory: battery_ontology_analysis/link_analysis")
    
    # Load and filter data
    analyzer.load_and_filter_data()
    analyzer.print_memory_usage("data loading")
    
    # Create knowledge graph
    analyzer.create_knowledge_graph()
    analyzer.print_memory_usage("graph creation")
    
    # Calculate statistics
    analyzer.calculate_entity_statistics()
    analyzer.print_memory_usage("statistics calculation")
    
    # Save checkpoint
    analyzer.save_progress_checkpoint("after_stats", {
        "num_entities": len(analyzer.entity_counts),
        "num_relations": len(analyzer.relation_counts),
        "graph_nodes": analyzer.graph.number_of_nodes(),
        "graph_edges": analyzer.graph.number_of_edges()
    })
    
    # Calculate optimized PMI
    analyzer.calculate_optimized_pmi()
    analyzer.print_memory_usage("PMI calculation")
    
    # Calculate TF-IDF features
    analyzer.calculate_tfidf_features()
    analyzer.print_memory_usage("TF-IDF calculation")
    
    # Create test set
    train_examples, test_examples = analyzer.create_test_set()
    
    # Evaluate baselines
    print("\n🔬 EVALUATING BASELINE METHODS")
    print("=" * 40)
    
    results = []
    
    # PMI baseline
    pmi_results = analyzer.evaluate_pmi_baseline(test_examples)
    results.append(pmi_results)
    
    # TF-IDF + k-NN baseline
    for k in [3, 5, 10]:
        tfidf_results = analyzer.evaluate_tfidf_knn_baseline(test_examples, k=k)
        if tfidf_results:
            results.append(tfidf_results)
    
    # Graph neighborhood baseline
    graph_results = analyzer.evaluate_graph_neighborhood_baseline(test_examples)
    results.append(graph_results)
    
    # Analyze relation patterns
    relation_stats = analyzer.analyze_relation_patterns()
    
    # Create visualizations
    analyzer.create_visualizations(results)
    
    # Save results
    analyzer.save_results(results, relation_stats)
    
    print("\n" + "=" * 60)
    print("✅ LINK PLAUSIBILITY ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files in 'battery_ontology_analysis/link_analysis/':")
    print("- baseline_results.csv")
    print("- relation_patterns.json") 
    print("- dataset_statistics.json")
    print("- baseline_comparison.png")
    print("- entity_frequency.png")
    print("- link_analysis_report.txt")
    
    # Print summary of best performing method
    if results:
        best_f1 = max(results, key=lambda x: x['f1'] if x else 0)
        print(f"\n🏆 Best performing method: {best_f1['method']} (F1: {best_f1['f1']:.3f})")

if __name__ == "__main__":
    main()

