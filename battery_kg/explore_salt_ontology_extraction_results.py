#!/usr/bin/env python3
"""
Salt Ontology Extraction Results Explorer

This script analyzes and visualizes the salt ontology extraction results,
including knowledge graph visualization and statistical analysis.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
import numpy as np
from datetime import datetime
import re
from difflib import SequenceMatcher
import warnings

# Suppress NetworkX backend warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(file_path):
    """Load the salt ontology extraction results from JSON file"""
    try:
        print(f"Loading results from: {file_path}")
        with open(file_path, 'r') as f:
            results = json.load(f)
        print(f"Successfully loaded {len(results)} entries")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def analyze_basic_stats(results):
    """Analyze basic statistics of the extraction results"""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    total_entries = len(results)
    successful_parses = sum(1 for r in results if r.get('parsed_analysis') is not None)
    failed_parses = total_entries - successful_parses
    
    print(f"Total entries: {total_entries:,}")
    print(f"Successfully parsed: {successful_parses:,} ({successful_parses/total_entries*100:.1f}%)")
    print(f"Failed to parse: {failed_parses:,} ({failed_parses/total_entries*100:.1f}%)")
    
    # Processing time analysis
    processing_times = [r.get('processing_time_seconds', 0) for r in results if r.get('processing_time_seconds')]
    if processing_times:
        print(f"Average processing time: {np.mean(processing_times):.2f} seconds")
        print(f"Total processing time: {sum(processing_times)/3600:.2f} hours")
    
    return {
        'total_entries': total_entries,
        'successful_parses': successful_parses,
        'failed_parses': failed_parses,
        'success_rate': successful_parses/total_entries*100
    }

def extract_salt_data(results):
    """Extract all salt information from the results"""
    print("\n" + "="*60)
    print("EXTRACTING SALT DATA")
    print("="*60)
    
    all_salts = []
    salt_counter = Counter()
    relationship_counter = Counter()
    salt_relationships = defaultdict(list)
    
    for entry in results:
        parsed = entry.get('parsed_analysis')
        if parsed and parsed.get('salts'):
            for salt_entry in parsed['salts']:
                salt_name = salt_entry.get('salt', 'Unknown')
                relationships = salt_entry.get('relationships', [])
                
                # Count salt occurrences
                salt_counter[salt_name] += 1
                
                # Store salt with its relationships
                salt_data = {
                    'salt': salt_name,
                    'relationships': relationships,
                    'num_relationships': len(relationships),
                    'entry_index': entry.get('index', 0),
                    'abstract': entry.get('abstract', '')[:200] + '...'  # First 200 chars
                }
                all_salts.append(salt_data)
                
                # Count relationship types
                for rel in relationships:
                    rel_type = rel.get('relation', 'Unknown')
                    relationship_counter[rel_type] += 1
                    salt_relationships[salt_name].append(rel)
    
    print(f"Total salt entries found: {len(all_salts):,}")
    print(f"Unique salts identified: {len(salt_counter):,}")
    print(f"Total relationships: {sum(relationship_counter.values()):,}")
    
    return all_salts, salt_counter, relationship_counter, salt_relationships

def visualize_salt_statistics(salt_counter, relationship_counter, output_dir):
    """Create visualizations for salt statistics"""
    print("\n" + "="*60)
    print("CREATING STATISTICAL VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Top 20 most frequent salts
    plt.figure(figsize=(12, 8))
    top_salts = salt_counter.most_common(20)
    salts, counts = zip(*top_salts)
    
    plt.barh(range(len(salts)), counts)
    plt.yticks(range(len(salts)), salts)
    plt.xlabel('Number of Occurrences')
    plt.title('Top 20 Most Frequently Mentioned Salts')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_20_salts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Relationship types distribution
    plt.figure(figsize=(12, 8))
    rel_types, rel_counts = zip(*relationship_counter.most_common())
    
    plt.barh(range(len(rel_types)), rel_counts)
    plt.yticks(range(len(rel_types)), rel_types)
    plt.xlabel('Number of Relationships')
    plt.title('Distribution of Relationship Types')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/relationship_types.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Salt occurrence distribution
    plt.figure(figsize=(10, 6))
    occurrence_counts = list(salt_counter.values())
    plt.hist(occurrence_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Occurrences per Salt')
    plt.ylabel('Number of Salts')
    plt.title('Distribution of Salt Mention Frequencies')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/salt_frequency_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical visualizations saved to: {output_dir}/")

def create_knowledge_graph(salt_name, relationships, output_path):
    """Create a knowledge graph for a specific salt"""
    G = nx.DiGraph()
    
    # Add the central salt node
    G.add_node(salt_name, node_type='salt', color='red', size=1000)
    
    # Add relationship nodes and edges
    for rel in relationships:
        relation = rel.get('relation', 'Unknown')
        obj = rel.get('object', 'Unknown')
        
        # Add object node
        G.add_node(obj, node_type='object', color='lightblue', size=500)
        
        # Add edge with relationship as label
        G.add_edge(salt_name, obj, label=relation)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Position nodes using layout (fallback for missing scipy)
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except ImportError:
        # Fallback to simple circular layout if scipy is not available
        print(f"Warning: scipy not available, using circular layout for {salt_name}")
        pos = nx.circular_layout(G)
    
    # Draw nodes
    salt_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'salt']
    object_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'object']
    
    nx.draw_networkx_nodes(G, pos, nodelist=salt_nodes, node_color='red', 
                          node_size=1000, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=object_nodes, node_color='lightblue', 
                          node_size=500, alpha=0.6)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Draw edge labels (relationships)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, alpha=0.7)
    
    plt.title(f"Knowledge Graph for Salt: {salt_name}\n({len(relationships)} relationships)", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return G

def visualize_complex_knowledge_graphs(salt_relationships, output_dir, min_edges=4):
    """Create knowledge graphs for salts with many relationships"""
    print(f"\n" + "="*60)
    print(f"CREATING KNOWLEDGE GRAPHS (min {min_edges} edges)")
    print("="*60)
    
    kg_dir = os.path.join(output_dir, "knowledge_graphs")
    os.makedirs(kg_dir, exist_ok=True)
    
    # Find salts with at least min_edges relationships
    complex_salts = {salt: rels for salt, rels in salt_relationships.items() 
                    if len(rels) >= min_edges}
    
    print(f"Found {len(complex_salts)} salts with {min_edges}+ relationships")
    
    # Sort by number of relationships (descending)
    sorted_salts = sorted(complex_salts.items(), key=lambda x: len(x[1]), reverse=True)
    
    graphs_created = 0
    for salt, relationships in sorted_salts[:20]:  # Limit to top 20
        try:
            output_path = os.path.join(kg_dir, f"{salt.replace('/', '_').replace(' ', '_')}_graph.png")
            graph = create_knowledge_graph(salt, relationships, output_path)
            graphs_created += 1
            print(f"Created graph for {salt}: {len(relationships)} relationships")
        except Exception as e:
            print(f"Error creating graph for {salt}: {e}")
    
    print(f"\nCreated {graphs_created} knowledge graphs in: {kg_dir}/")
    return sorted_salts

def create_summary_report(basic_stats, salt_counter, relationship_counter, 
                         complex_salts, output_dir):
    """Create a comprehensive summary report"""
    print("\n" + "="*60)
    print("CREATING SUMMARY REPORT")
    print("="*60)
    
    report_path = os.path.join(output_dir, "salt_ontology_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("SALT ONTOLOGY EXTRACTION ANALYSIS REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BASIC STATISTICS\n")
        f.write("-"*20 + "\n")
        f.write(f"Total entries: {basic_stats['total_entries']:,}\n")
        f.write(f"Successfully parsed: {basic_stats['successful_parses']:,}\n")
        f.write(f"Success rate: {basic_stats['success_rate']:.1f}%\n\n")
        
        f.write("SALT ANALYSIS\n")
        f.write("-"*20 + "\n")
        f.write(f"Unique salts identified: {len(salt_counter):,}\n")
        f.write(f"Total salt mentions: {sum(salt_counter.values()):,}\n")
        f.write(f"Average mentions per salt: {sum(salt_counter.values())/len(salt_counter):.1f}\n\n")
        
        f.write("TOP 10 MOST FREQUENT SALTS\n")
        f.write("-"*30 + "\n")
        for i, (salt, count) in enumerate(salt_counter.most_common(10), 1):
            f.write(f"{i:2d}. {salt}: {count:,} mentions\n")
        f.write("\n")
        
        f.write("RELATIONSHIP ANALYSIS\n")
        f.write("-"*20 + "\n")
        f.write(f"Total relationships: {sum(relationship_counter.values()):,}\n")
        f.write(f"Unique relationship types: {len(relationship_counter)}\n\n")
        
        f.write("RELATIONSHIP TYPES (by frequency)\n")
        f.write("-"*35 + "\n")
        for i, (rel_type, count) in enumerate(relationship_counter.most_common(), 1):
            f.write(f"{i:2d}. {rel_type}: {count:,} instances\n")
        f.write("\n")
        
        f.write("COMPLEX SALTS (4+ relationships)\n")
        f.write("-"*35 + "\n")
        for i, (salt, relationships) in enumerate(complex_salts[:10], 1):
            f.write(f"{i:2d}. {salt}: {len(relationships)} relationships\n")
    
    print(f"Summary report saved to: {report_path}")

# Add synonym consolidation functionality
def load_pubchem_synonyms(file_path='../PubChem_filtered_compounds_mw_250_or_below.csv'):
    """Load PubChem synonyms from the filtered CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"PubChem synonyms file not found: {file_path}")
        return pd.DataFrame()

def create_salt_synonym_mapping():
    """Create a mapping of salt synonyms using PubChem data and manual rules"""
    print("\n" + "="*60)
    print("CREATING SALT SYNONYM MAPPING")
    print("="*60)
    
    # Load PubChem data
    pubchem_df = load_pubchem_synonyms()
    
    # Create synonym mapping
    synonym_map = {}
    canonical_names = {}
    
    # Process PubChem data
    if not pubchem_df.empty:
        for idx, row in pubchem_df.iterrows():
            if pd.notna(row['cmpdsynonym']):
                synonyms = [s.strip() for s in str(row['cmpdsynonym']).split('|') if s.strip()]
                if synonyms:
                    # Use the shortest reasonable name as canonical
                    canonical = min(synonyms, key=lambda x: (len(x), x))
                    canonical_names[row['mf']] = canonical
                    
                    for syn in synonyms:
                        synonym_map[syn.lower()] = canonical
            
            # Also map the molecular formula
            if pd.notna(row['mf']) and pd.notna(row['cmpdname']):
                mf = str(row['mf'])
                name = str(row['cmpdname']).split('/')[0]  # Take first name part
                synonym_map[mf.lower()] = name
                synonym_map[name.lower()] = name
    
    # Add manual lithium salt mappings (common variations)
    manual_mappings = {
        # LiPF6 variations
        'lipf6': 'LiPF6',
        'lithium hexafluorophosphate': 'LiPF6',
        'lithium (hexafluoro) phosphate': 'LiPF6',
        
        # LiTFSI variations (only true synonyms)
        'litfsi': 'LiTFSI',
        'lithium bis(trifluoromethanesulfonyl)imide': 'LiTFSI',
        'lithium bis(trifluoromethylsulfonyl)imide': 'LiTFSI',
        'lin(so2cf3)2': 'LiTFSI',
        'lin(cf3so2)2': 'LiTFSI',
        'litfsa': 'LiTFSI',
        'li[ntf2]': 'LiTFSI',
        
        # LiFTFSI (distinct salt - do not merge with LiTFSI)
        'liftfsi': 'LiFTFSI',
        
        # LiTNFSI (distinct salt - do not merge with LiTFSI)
        'litnfsi': 'LiTNFSI',
        
        # LiATFSI (distinct salt - do not merge with LiTFSI)
        'liatfsi': 'LiATFSI',
        
        # LiSTFSI (distinct salt - do not merge with LiTFSI)
        'listfsi': 'LiSTFSI',
        
        # LiSFSI (distinct salt - do not merge with LiFSI)
        'lisfsi': 'LiSFSI',
        
        # LiBF4 variations
        'libf4': 'LiBF4',
        'lithium tetrafluoroborate': 'LiBF4',
        
        # LiClO4 variations
        'liclo4': 'LiClO4',
        'lithium perchlorate': 'LiClO4',
        
        # LiFSI variations
        'lifsi': 'LiFSI',
        'lithium bis(fluorosulfonyl)imide': 'LiFSI',
        'lin(so2f)2': 'LiFSI',
        
        # LiAsF6 variations
        'liasf6': 'LiAsF6',
        'lithium hexafluoroarsenate': 'LiAsF6',
        
        # LiBOB variations
        'libob': 'LiBOB',
        'lithium bis(oxalato)borate': 'LiBOB',
        'lithium [bis (oxalate) borate]': 'LiBOB',
        
        # LiOTF variations
        'liotf': 'LiOTF',
        'licf3so3': 'LiOTF',
        'lithium triflate': 'LiOTF',
        'lithium trifluoromethanesulfonate': 'LiOTF',
        
        # LiDFOB variations (same compound with different names)
        'lidfob': 'LiDFOB',
        'liodib': 'LiDFOB',
        'lifob': 'LiDFOB',
        'lithium difluorooxalate borate': 'LiDFOB',
        
        # Generic terms
        'li salt': 'Lithium salt',
        'li salts': 'Lithium salt',
        'lithium salts': 'Lithium salt',
        'li-salt': 'Lithium salt',
        'li-salts': 'Lithium salt',
        'lithium salt': 'Lithium salt',
        'electrolyte salt': 'Electrolyte salt',
        'supporting lithium salt': 'Lithium salt',
    }
    
    # Add manual mappings
    synonym_map.update(manual_mappings)
    
    print(f"Created synonym mapping with {len(synonym_map)} entries")
    return synonym_map

def normalize_salt_name(salt_name, synonym_map):
    """Normalize a salt name using the synonym mapping"""
    if not salt_name or salt_name.lower() == 'unknown':
        return salt_name
    
    # Direct lookup
    normalized = synonym_map.get(salt_name.lower())
    if normalized:
        return normalized
    
    # Try removing common prefixes/suffixes and extra spaces
    cleaned = re.sub(r'\s+', ' ', salt_name.strip())
    normalized = synonym_map.get(cleaned.lower())
    if normalized:
        return normalized
    
    # Try fuzzy matching for close variants (more conservative threshold)
    for synonym, canonical in synonym_map.items():
        if len(synonym) > 5 and len(salt_name) > 5:  # Only for longer names
            similarity = SequenceMatcher(None, synonym.lower(), salt_name.lower()).ratio()
            if similarity > 0.95:  # Increased threshold to 95% for more precision
                return canonical
    
    # Return original if no match found
    return salt_name

def consolidate_salt_data_with_synonyms(all_salts, salt_counter, relationship_counter, salt_relationships):
    """Consolidate salt data by merging synonyms"""
    print("\n" + "="*60)
    print("CONSOLIDATING SALTS USING SYNONYMS")
    print("="*60)
    
    # Create synonym mapping
    synonym_map = create_salt_synonym_mapping()
    
    # Create consolidated data structures
    consolidated_salt_counter = Counter()
    consolidated_relationship_counter = Counter(relationship_counter)  # Keep relationship counts the same
    consolidated_salt_relationships = defaultdict(list)
    consolidated_all_salts = []
    
    # Track the consolidation
    consolidation_map = {}
    
    for salt_data in all_salts:
        original_salt = salt_data['salt']
        normalized_salt = normalize_salt_name(original_salt, synonym_map)
        
        # Track consolidation
        if original_salt != normalized_salt:
            if normalized_salt not in consolidation_map:
                consolidation_map[normalized_salt] = []
            consolidation_map[normalized_salt].append(original_salt)
        
        # Update consolidated data
        consolidated_salt_counter[normalized_salt] += 1
        
        # Update relationships
        consolidated_salt_relationships[normalized_salt].extend(salt_data['relationships'])
        
        # Update salt data
        new_salt_data = salt_data.copy()
        new_salt_data['salt'] = normalized_salt
        new_salt_data['original_salt'] = original_salt
        consolidated_all_salts.append(new_salt_data)
    
    # Print consolidation summary
    print(f"Original unique salts: {len(salt_counter)}")
    print(f"Consolidated unique salts: {len(consolidated_salt_counter)}")
    print(f"Reduction: {len(salt_counter) - len(consolidated_salt_counter)} salts merged")
    
    # Show some consolidation examples
    print(f"\nConsolidation examples:")
    consolidation_examples = 0
    for canonical, variants in consolidation_map.items():
        if len(variants) > 1 and consolidation_examples < 10:
            print(f"  '{canonical}' ← {variants[:5]}")
            consolidation_examples += 1
    
    return (consolidated_all_salts, consolidated_salt_counter, 
            consolidated_relationship_counter, consolidated_salt_relationships, 
            consolidation_map)

def create_consolidation_report(consolidation_map, output_dir):
    """Create a detailed report of salt consolidations"""
    report_path = os.path.join(output_dir, "salt_consolidation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("SALT CONSOLIDATION REPORT\n")
        f.write("="*40 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total canonical salts with variants: {len(consolidation_map)}\n")
        f.write(f"Total original variants consolidated: {sum(len(variants) for variants in consolidation_map.values())}\n\n")
        
        f.write("CONSOLIDATION DETAILS\n")
        f.write("-"*30 + "\n")
        
        # Sort by number of variants
        sorted_consolidations = sorted(consolidation_map.items(), 
                                     key=lambda x: len(x[1]), reverse=True)
        
        for canonical, variants in sorted_consolidations:
            f.write(f"\nCanonical: {canonical}\n")
            f.write(f"Variants ({len(variants)}):\n")
            for variant in sorted(variants):
                f.write(f"  - {variant}\n")
    
    print(f"Consolidation report saved to: {report_path}")

def export_subject_predicate_object_json(consolidated_salt_relationships, output_dir):
    """Export consolidated salt relationships as subject-predicate-object JSON triples"""
    print("\n" + "="*60)
    print("EXPORTING SUBJECT-PREDICATE-OBJECT TRIPLES")
    print("="*60)
    
    triples = []
    triple_id = 1
    
    for salt_name, relationships in consolidated_salt_relationships.items():
        for relationship in relationships:
            triple = {
                "id": triple_id,
                "subject": salt_name,
                "predicate": relationship.get('relation', 'Unknown'),
                "object": relationship.get('object', 'Unknown'),
                "subject_type": "SALT",
                "object_type": "ENTITY"  # Could be refined based on predicate type
            }
            triples.append(triple)
            triple_id += 1
    
    # Export to JSON
    export_path = os.path.join(output_dir, "salt_knowledge_graph_triples.json")
    with open(export_path, 'w') as f:
        json.dump(triples, f, indent=2)
    
    # Create summary statistics
    predicate_counts = Counter([t['predicate'] for t in triples])
    subject_counts = Counter([t['subject'] for t in triples])
    
    # Export summary
    summary_path = os.path.join(output_dir, "triples_summary.json")
    summary = {
        "total_triples": len(triples),
        "unique_subjects": len(subject_counts),
        "unique_predicates": len(predicate_counts),
        "predicate_distribution": dict(predicate_counts),
        "top_10_subjects_by_relationships": dict(subject_counts.most_common(10)),
        "export_timestamp": datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Exported {len(triples):,} knowledge graph triples to: {export_path}")
    print(f"Summary statistics saved to: {summary_path}")
    print(f"Unique subjects (salts): {len(subject_counts)}")
    print(f"Unique predicates (relationships): {len(predicate_counts)}")
    
    # Print top predicates
    print("\nTop 10 relationship types:")
    for predicate, count in predicate_counts.most_common(10):
        print(f"  {predicate}: {count:,} triples")
    
    return triples, summary

def main():
    """Main function to run the complete analysis"""
    print("Salt Ontology Extraction Results Explorer")
    print("="*60)
    
    # Configuration
    results_files = [
        "salt_ontology_extraction_results_v2.json",
        "battery_salt_extraction_results.json",
        "../salt_ontology_extraction_results_v2.json",
        "../battery_salt_extraction_results.json"
    ]
    
    output_dir = "salt_analysis_output"
    
    # Load results
    results = None
    for file_path in results_files:
        if os.path.exists(file_path):
            results = load_results(file_path)
            break
    
    if results is None:
        print("ERROR: Could not find any results file!")
        print("Looked for files:")
        for file_path in results_files:
            print(f"  - {file_path}")
        return
    
    # Run analysis
    basic_stats = analyze_basic_stats(results)
    all_salts, salt_counter, relationship_counter, salt_relationships = extract_salt_data(results)
    
    # Consolidate synonyms
    (consolidated_all_salts, consolidated_salt_counter, 
     consolidated_relationship_counter, consolidated_salt_relationships, 
     consolidation_map) = consolidate_salt_data_with_synonyms(all_salts, salt_counter, 
                                                              relationship_counter, salt_relationships)
    
    # Create visualizations with both original and consolidated data
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS (ORIGINAL DATA)")
    print("="*60)
    visualize_salt_statistics(salt_counter, relationship_counter, output_dir)
    complex_salts = visualize_complex_knowledge_graphs(salt_relationships, output_dir, min_edges=4)
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS (CONSOLIDATED DATA)")
    print("="*60)
    consolidated_output_dir = os.path.join(output_dir, "consolidated")
    os.makedirs(consolidated_output_dir, exist_ok=True)
    visualize_salt_statistics(consolidated_salt_counter, consolidated_relationship_counter, consolidated_output_dir)
    consolidated_complex_salts = visualize_complex_knowledge_graphs(consolidated_salt_relationships, consolidated_output_dir, min_edges=4)
    
    # Create reports
    create_summary_report(basic_stats, salt_counter, relationship_counter, 
                         complex_salts, output_dir)
    create_summary_report(basic_stats, consolidated_salt_counter, consolidated_relationship_counter, 
                         consolidated_complex_salts, consolidated_output_dir)
    create_consolidation_report(consolidation_map, output_dir)
    
    # Export subject-predicate-object triples
    triples, triples_summary = export_subject_predicate_object_json(consolidated_salt_relationships, output_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"Original unique salts found: {len(salt_counter):,}")
    print(f"Consolidated unique salts: {len(consolidated_salt_counter):,}")
    print(f"Salts merged through consolidation: {len(salt_counter) - len(consolidated_salt_counter)}")
    print(f"Original salts with 4+ relationships: {len([s for s, r in salt_relationships.items() if len(r) >= 4])}")
    print(f"Consolidated salts with 4+ relationships: {len([s for s, r in consolidated_salt_relationships.items() if len(r) >= 4])}")
    print(f"Knowledge graphs created: {min(20, len([s for s, r in salt_relationships.items() if len(r) >= 4]))}")
    print(f"Knowledge graph triples exported: {len(triples):,}")
    print("")
    print("Key output files:")
    print(f"  - Original data plots: {output_dir}/*.png")
    print(f"  - Consolidated data plots: {output_dir}/consolidated/*.png")
    print(f"  - Knowledge graphs: {output_dir}/knowledge_graphs/*.png")
    print(f"  - Consolidated knowledge graphs: {output_dir}/consolidated/knowledge_graphs/*.png")
    print(f"  - Summary reports: {output_dir}/salt_ontology_analysis_report.txt")
    print(f"  - Consolidation report: {output_dir}/salt_consolidation_report.txt")
    print(f"  - Knowledge graph triples: {output_dir}/salt_knowledge_graph_triples.json")
    print(f"  - Triples summary: {output_dir}/triples_summary.json")

if __name__ == "__main__":
    main()
