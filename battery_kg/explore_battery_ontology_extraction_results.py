#!/usr/bin/env python3
"""
Explore Battery Ontology Extraction Results

This script analyzes the battery_ontology_extraction_results_v2.json file to:
1. Understand the data structure and quality
2. Generate comprehensive statistics
3. Visualize knowledge graphs for significant entries
4. Analyze trends in relationships and entities
5. Create network visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
import warnings
import os
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('battery_ontology_analysis', exist_ok=True)

def load_data(file_path='battery_ontology_extraction_results_v2.json'):
    """Load and parse the JSON data"""
    print("🔄 Loading battery ontology extraction results...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded {len(data):,} entries")
        return data
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_basic_stats(data):
    """Analyze basic statistics about the dataset"""
    print("\n" + "="*60)
    print("📊 BASIC STATISTICS")
    print("="*60)
    
    total_entries = len(data)
    print(f"Total entries: {total_entries:,}")
    
    # Check for parsed analysis
    parsed_count = sum(1 for entry in data if entry.get('parsed_analysis'))
    print(f"Entries with parsed analysis: {parsed_count:,} ({parsed_count/total_entries*100:.1f}%)")
    
    # Check processing times
    processing_times = [entry.get('processing_time_seconds', 0) for entry in data if entry.get('processing_time_seconds')]
    if processing_times:
        print(f"Average processing time: {np.mean(processing_times):.2f} seconds")
        print(f"Total processing time: {sum(processing_times)/3600:.2f} hours")
    
    # Check batch distribution
    batch_ids = [entry.get('batch_id', 0) for entry in data]
    unique_batches = len(set(batch_ids))
    print(f"Number of batches: {unique_batches}")
    
    # Abstract lengths
    abstract_lengths = [len(entry.get('abstract', '')) for entry in data]
    print(f"Average abstract length: {np.mean(abstract_lengths):.0f} characters")
    print(f"Abstract length range: {min(abstract_lengths)} - {max(abstract_lengths)} characters")
    
    return {
        'total_entries': total_entries,
        'parsed_count': parsed_count,
        'processing_times': processing_times,
        'batch_count': unique_batches,
        'abstract_lengths': abstract_lengths
    }

def extract_relationships(data):
    """Extract all relationships from the parsed data"""
    print("\n🔗 Extracting relationships...")
    
    all_relationships = []
    entries_with_relationships = 0
    
    for entry in data:
        if entry.get('parsed_analysis') and entry['parsed_analysis'].get('relationships'):
            relationships = entry['parsed_analysis']['relationships']
            entries_with_relationships += 1
            
            for rel in relationships:
                if isinstance(rel, dict) and all(key in rel for key in ['subject', 'relation', 'object']):
                    all_relationships.append({
                        'subject': rel['subject'],
                        'relation': rel['relation'],
                        'object': rel['object'],
                        'entry_index': entry.get('index', 0),
                        'batch_id': entry.get('batch_id', 0)
                    })
    
    print(f"✅ Extracted {len(all_relationships):,} relationships from {entries_with_relationships:,} entries")
    return all_relationships

def analyze_relationships(relationships):
    """Analyze the extracted relationships"""
    print("\n" + "="*60)
    print("🔗 RELATIONSHIP ANALYSIS")
    print("="*60)
    
    if not relationships:
        print("❌ No relationships found to analyze")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(relationships)
    
    # Basic stats
    print(f"Total relationships: {len(df):,}")
    print(f"Unique subjects: {df['subject'].nunique():,}")
    print(f"Unique relations: {df['relation'].nunique():,}")
    print(f"Unique objects: {df['object'].nunique():,}")
    
    # Top relations
    print("\n🔝 Top 10 Relation Types:")
    top_relations = df['relation'].value_counts().head(10)
    for relation, count in top_relations.items():
        print(f"  {relation}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Top subjects
    print("\n🔝 Top 10 Subjects:")
    top_subjects = df['subject'].value_counts().head(10)
    for subject, count in top_subjects.items():
        print(f"  {subject}: {count:,} relationships")
    
    # Top objects
    print("\n🔝 Top 10 Objects:")
    top_objects = df['object'].value_counts().head(10)
    for obj, count in top_objects.items():
        print(f"  {obj}: {count:,} relationships")
    
    return df

def create_relationship_visualizations(df):
    """Create visualizations for relationship data"""
    print("\n📈 Creating relationship visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Battery Ontology Relationship Analysis', fontsize=16, fontweight='bold')
    
    # Top relations
    top_relations = df['relation'].value_counts().head(15)
    axes[0, 0].barh(range(len(top_relations)), top_relations.values)
    axes[0, 0].set_yticks(range(len(top_relations)))
    axes[0, 0].set_yticklabels(top_relations.index, fontsize=8)
    axes[0, 0].set_xlabel('Count')
    axes[0, 0].set_title('Top 15 Relation Types')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top subjects
    top_subjects = df['subject'].value_counts().head(15)
    axes[0, 1].barh(range(len(top_subjects)), top_subjects.values)
    axes[0, 1].set_yticks(range(len(top_subjects)))
    axes[0, 1].set_yticklabels(top_subjects.index, fontsize=8)
    axes[0, 1].set_xlabel('Count')
    axes[0, 1].set_title('Top 15 Subjects')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relationship count per entry distribution
    entry_rel_counts = df.groupby('entry_index').size()
    axes[1, 0].hist(entry_rel_counts, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Relationships per Entry')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Relationships per Entry')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Batch distribution
    batch_counts = df['batch_id'].value_counts().sort_index()
    axes[1, 1].bar(batch_counts.index, batch_counts.values)
    axes[1, 1].set_xlabel('Batch ID')
    axes[1, 1].set_ylabel('Number of Relationships')
    axes[1, 1].set_title('Relationships by Batch')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('battery_ontology_analysis/relationship_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved relationship analysis plot")
    
    return fig

def create_knowledge_graph_sample(relationships, top_n=20):
    """Create a knowledge graph visualization for top entities"""
    print(f"\n🕸️  Creating knowledge graph for top {top_n} entities...")
    
    # Get top subjects by relationship count
    df = pd.DataFrame(relationships)
    top_subjects = df['subject'].value_counts().head(top_n).index.tolist()
    
    # Filter relationships for top subjects
    filtered_rels = df[df['subject'].isin(top_subjects)]
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add edges (subject -> object with relation as edge attribute)
    for _, row in filtered_rels.iterrows():
        G.add_edge(row['subject'], row['object'], 
                  relation=row['relation'], 
                  weight=1)
    
    # Calculate layout
    print("🔄 Calculating graph layout...")
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Create visualization
        plt.figure(figsize=(20, 16))
        
        # Draw nodes
        node_sizes = [G.degree(node) * 20 + 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', 
                              arrows=True, arrowsize=10, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f'Knowledge Graph - Top {top_n} Entities by Relationship Count', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('battery_ontology_analysis/knowledge_graph_sample.png', 
                   dpi=300, bbox_inches='tight')
        print("✅ Saved knowledge graph visualization")
        
        # Print graph statistics
        print(f"Graph nodes: {G.number_of_nodes()}")
        print(f"Graph edges: {G.number_of_edges()}")
        
        # Most connected nodes
        degree_dict = dict(G.degree())
        top_connected = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n🔗 Most Connected Entities:")
        for entity, degree in top_connected:
            print(f"  {entity}: {degree} connections")
    
    return G

def analyze_semantic_patterns(relationships):
    """Analyze semantic patterns in the relationships"""
    print("\n" + "="*60)
    print("🧠 SEMANTIC PATTERN ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(relationships)
    
    # Analyze relation patterns
    relation_patterns = defaultdict(list)
    
    for _, row in df.iterrows():
        relation = row['relation']
        subject = row['subject']
        obj = row['object']
        
        relation_patterns[relation].append((subject, obj))
    
    print("\n🔍 Relation Pattern Analysis:")
    for relation, pairs in list(relation_patterns.items())[:10]:
        unique_subjects = set(pair[0] for pair in pairs)
        unique_objects = set(pair[1] for pair in pairs)
        print(f"\n{relation}:")
        print(f"  Used {len(pairs)} times")
        print(f"  {len(unique_subjects)} unique subjects, {len(unique_objects)} unique objects")
        
        # Most common subject-object pairs for this relation
        pair_counts = Counter(pairs)
        top_pairs = pair_counts.most_common(3)
        if top_pairs:
            print("  Top pairs:")
            for (subj, obj), count in top_pairs:
                print(f"    {subj} → {obj} ({count} times)")

def create_entity_network(relationships, min_connections=5):
    """Create a network of entities based on co-occurrence"""
    print(f"\n🌐 Creating entity co-occurrence network (min {min_connections} connections)...")
    
    df = pd.DataFrame(relationships)
    
    # Count entity co-occurrences within entries
    entity_cooccurrence = defaultdict(int)
    
    # Group by entry_index to find entities that appear together
    for entry_idx, group in df.groupby('entry_index'):
        entities = set()
        entities.update(group['subject'].tolist())
        entities.update(group['object'].tolist())
        
        # Create pairs of co-occurring entities
        entities = list(entities)
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                pair = tuple(sorted([entities[i], entities[j]]))
                entity_cooccurrence[pair] += 1
    
    # Filter by minimum connections
    filtered_cooccurrence = {pair: count for pair, count in entity_cooccurrence.items() 
                           if count >= min_connections}
    
    print(f"Found {len(filtered_cooccurrence)} entity pairs with {min_connections}+ co-occurrences")
    
    # Create network
    G = nx.Graph()
    for (entity1, entity2), weight in filtered_cooccurrence.items():
        G.add_edge(entity1, entity2, weight=weight)
    
    if G.number_of_nodes() > 0:
        # Create visualization
        plt.figure(figsize=(16, 12))
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node sizes based on degree
        node_sizes = [G.degree(node, weight='weight') * 5 + 50 for node in G.nodes()]
        
        # Edge widths based on weight
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightcoral', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f'Entity Co-occurrence Network (≥{min_connections} co-occurrences)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('battery_ontology_analysis/entity_cooccurrence_network.png', 
                   dpi=300, bbox_inches='tight')
        print("✅ Saved entity co-occurrence network")
        
        # Network statistics
        print(f"Network nodes: {G.number_of_nodes()}")
        print(f"Network edges: {G.number_of_edges()}")
        
        if G.number_of_nodes() > 0:
            # Most central entities
            centrality = nx.degree_centrality(G)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\n🎯 Most Central Entities:")
            for entity, centrality_score in top_central:
                print(f"  {entity}: {centrality_score:.3f}")
    
    return G

def generate_summary_report(data, relationships_df, stats):
    """Generate a comprehensive summary report"""
    print("\n📋 Generating summary report...")
    
    report_path = 'battery_ontology_analysis/summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("BATTERY ONTOLOGY EXTRACTION RESULTS - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total entries: {stats['total_entries']:,}\n")
        f.write(f"Successfully parsed: {stats['parsed_count']:,} ({stats['parsed_count']/stats['total_entries']*100:.1f}%)\n")
        f.write(f"Total processing batches: {stats['batch_count']}\n")
        f.write(f"Average abstract length: {np.mean(stats['abstract_lengths']):.0f} characters\n")
        
        if stats['processing_times']:
            f.write(f"Total processing time: {sum(stats['processing_times'])/3600:.2f} hours\n")
            f.write(f"Average time per entry: {np.mean(stats['processing_times']):.2f} seconds\n")
        
        # Relationship statistics
        if relationships_df is not None and len(relationships_df) > 0:
            f.write("\nRELATIONSHIP STATISTICS\n")
            f.write("-" * 23 + "\n")
            f.write(f"Total relationships extracted: {len(relationships_df):,}\n")
            f.write(f"Unique subjects: {relationships_df['subject'].nunique():,}\n")
            f.write(f"Unique relations: {relationships_df['relation'].nunique():,}\n")
            f.write(f"Unique objects: {relationships_df['object'].nunique():,}\n")
            f.write(f"Average relationships per entry: {len(relationships_df)/stats['parsed_count']:.1f}\n")
            
            # Top entities and relations
            f.write("\nTOP 10 RELATION TYPES\n")
            f.write("-" * 21 + "\n")
            top_relations = relationships_df['relation'].value_counts().head(10)
            for i, (relation, count) in enumerate(top_relations.items(), 1):
                f.write(f"{i:2d}. {relation}: {count:,} ({count/len(relationships_df)*100:.1f}%)\n")
            
            f.write("\nTOP 10 SUBJECTS\n")
            f.write("-" * 16 + "\n")
            top_subjects = relationships_df['subject'].value_counts().head(10)
            for i, (subject, count) in enumerate(top_subjects.items(), 1):
                f.write(f"{i:2d}. {subject}: {count:,} relationships\n")
            
            f.write("\nTOP 10 OBJECTS\n")
            f.write("-" * 15 + "\n")
            top_objects = relationships_df['object'].value_counts().head(10)
            for i, (obj, count) in enumerate(top_objects.items(), 1):
                f.write(f"{i:2d}. {obj}: {count:,} relationships\n")
        
        # Files generated
        f.write("\nGENERATED FILES\n")
        f.write("-" * 15 + "\n")
        f.write("- summary_report.txt (this file)\n")
        f.write("- relationship_analysis.png\n")
        f.write("- knowledge_graph_sample.png\n")
        f.write("- entity_cooccurrence_network.png\n")
        f.write("- relationships_data.csv\n")
    
    print(f"✅ Summary report saved to: {report_path}")

def main():
    """Main function to run the complete analysis"""
    print("🔬 BATTERY ONTOLOGY EXTRACTION RESULTS EXPLORER")
    print("=" * 60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_data()
    if not data:
        return
    
    # Basic statistics
    stats = analyze_basic_stats(data)
    
    # Extract relationships
    relationships = extract_relationships(data)
    
    if not relationships:
        print("❌ No relationships found in the data!")
        return
    
    # Analyze relationships
    relationships_df = analyze_relationships(relationships)
    
    # Save relationships to CSV for further analysis
    if relationships_df is not None:
        csv_path = 'battery_ontology_analysis/relationships_data.csv'
        relationships_df.to_csv(csv_path, index=False)
        print(f"✅ Relationships data saved to: {csv_path}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Relationship analysis plots
    create_relationship_visualizations(relationships_df)
    
    # Knowledge graph sample
    knowledge_graph = create_knowledge_graph_sample(relationships, top_n=25)
    
    # Entity co-occurrence network
    cooccurrence_network = create_entity_network(relationships, min_connections=3)
    
    # Semantic pattern analysis
    analyze_semantic_patterns(relationships)
    
    # Generate summary report
    generate_summary_report(data, relationships_df, stats)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Analysis finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files in 'battery_ontology_analysis/' directory:")
    print("- summary_report.txt")
    print("- relationship_analysis.png")
    print("- knowledge_graph_sample.png") 
    print("- entity_cooccurrence_network.png")
    print("- relationships_data.csv")
    print("\n🎉 Check the 'battery_ontology_analysis' directory for all outputs!")

if __name__ == "__main__":
    main()
