#!/usr/bin/env python3
"""
Salt Ontology Cleaning and Canonicalization Script WITH SOURCE TEXT

This script cleans and canonicalizes salt names and chemical entities from the extracted
battery/salt ontology data using both PubChem API lookups and manual mapping rules.

KEY DIFFERENCE FROM ORIGINAL: This version preserves the COMPLETE source paragraph
for each triple, enabling downstream evaluation tasks.

Input: enhanced_salt_extraction_raw JSON file (contains paragraphs + nested triples)
Output: CSV with canonicalized triples AND full source paragraphs

Author: AI Assistant
Date: 2025-08-14, Updated: 2026-04-05
"""

import pandas as pd
import numpy as np
import json
import re
import requests
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import logging
from urllib.parse import quote
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SaltOntologyCleanerWithSource:
    """Main class for cleaning and canonicalizing salt ontology data with source text preservation."""
    
    def __init__(self, input_file: str, output_dir: str = "salt_analysis_output/canonical"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.pubchem_cache = {}
        self.manual_mappings = {}
        self.entity_stats = {}
        self.entity_mappings = {}
        self.load_manual_mappings()
        
    def load_manual_mappings(self):
        """Load manual entity mappings and synonyms."""
        logger.info("🔧 Loading manual entity mappings...")
        
        # Common chemical name mappings
        self.manual_mappings = {
            # Lithium salts
            "LiTFSI": "Lithium bis(trifluoromethanesulfonyl)imide",
            "Li TFSI": "Lithium bis(trifluoromethanesulfonyl)imide",
            "LiN(SO2CF3)2": "Lithium bis(trifluoromethanesulfonyl)imide",
            "LiPF6": "Lithium hexafluorophosphate",
            "Li PF6": "Lithium hexafluorophosphate",
            "LiBF4": "Lithium tetrafluoroborate",
            "Li BF4": "Lithium tetrafluoroborate",
            "LiClO4": "Lithium perchlorate",
            "Li ClO4": "Lithium perchlorate",
            "LiCF3SO3": "Lithium trifluoromethanesulfonate",
            "LiOTf": "Lithium trifluoromethanesulfonate",
            "Li OTf": "Lithium trifluoromethanesulfonate",
            "LiFSI": "Lithium bis(fluorosulfonyl)imide",
            "Li FSI": "Lithium bis(fluorosulfonyl)imide",
            "LiN(SO2F)2": "Lithium bis(fluorosulfonyl)imide",
            "LiDFOB": "Lithium difluoro(oxalato)borate",
            "Li DFOB": "Lithium difluoro(oxalato)borate",
            "LiBOB": "Lithium bis(oxalato)borate",
            "Li BOB": "Lithium bis(oxalato)borate",
            
            # Sodium salts
            "NaTFSI": "Sodium bis(trifluoromethanesulfonyl)imide",
            "Na TFSI": "Sodium bis(trifluoromethanesulfonyl)imide",
            "NaPF6": "Sodium hexafluorophosphate",
            "Na PF6": "Sodium hexafluorophosphate",
            "NaClO4": "Sodium perchlorate",
            "Na ClO4": "Sodium perchlorate",
            "NaFSI": "Sodium bis(fluorosulfonyl)imide",
            "Na FSI": "Sodium bis(fluorosulfonyl)imide",
            
            # Battery cathode materials
            "LiFePO4": "Lithium iron phosphate",
            "LiCoO2": "Lithium cobalt oxide",
            "LiMn2O4": "Lithium manganese oxide",
            "LiNiO2": "Lithium nickel oxide",
            "NCM": "Nickel cobalt manganese oxide",
            "NMC": "Nickel cobalt manganese oxide",
            "NCA": "Nickel cobalt aluminum oxide",
            "LTO": "Lithium titanate",
            "Li4Ti5O12": "Lithium titanate",
            
            # Electrolyte solvents
            "EC": "Ethylene carbonate",
            "PC": "Propylene carbonate",
            "DMC": "Dimethyl carbonate",
            "DEC": "Diethyl carbonate",
            "EMC": "Ethyl methyl carbonate",
            "DME": "1,2-Dimethoxyethane",
            "DOL": "1,3-Dioxolane",
            "THF": "Tetrahydrofuran",
            "ACN": "Acetonitrile",
            "AN": "Acetonitrile",
            "DMSO": "Dimethyl sulfoxide",
            "NMP": "N-Methyl-2-pyrrolidone",
            
            # Common chemical terms
            "conc.": "concentration",
            "temp.": "temperature",
            "mol.": "molecular",
            "wt.": "weight",
            "vol.": "volume",
            "aq.": "aqueous",
            "org.": "organic",
            "sol.": "solution",
            "soln.": "solution",
            "electrolyte": "electrolyte",
            "electrode": "electrode",
            "cathode": "cathode",
            "anode": "anode",
            "separator": "separator",
            "binder": "binder",
            "additive": "additive",
            "conductivity": "ionic conductivity",
            "capacity": "specific capacity",
            "voltage": "voltage",
            "potential": "electrode potential"
        }
        
        logger.info(f"✅ Loaded {len(self.manual_mappings)} manual mappings")
        
    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistent processing."""
        if pd.isna(name) or name == "":
            return ""
            
        # Convert to string and strip whitespace
        name = str(name).strip()
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Remove common prefixes/suffixes that don't add semantic value
        name = re.sub(r'^(the\s+|a\s+|an\s+)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+(material|compound|salt|electrolyte|solution)$', '', name, flags=re.IGNORECASE)
        
        return name.strip()
        
    def query_pubchem_by_name(self, name: str) -> Optional[Dict]:
        """Query PubChem for compound information by name."""
        if name in self.pubchem_cache:
            return self.pubchem_cache[name]
            
        try:
            # Encode the name for URL
            encoded_name = quote(name)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/JSON"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'PC_Compounds' in data and len(data['PC_Compounds']) > 0:
                    compound = data['PC_Compounds'][0]
                    
                    # Extract relevant information
                    result = {
                        'cid': compound.get('id', {}).get('id', {}).get('cid'),
                        'molecular_formula': None,
                        'molecular_weight': None,
                        'iupac_name': None,
                        'synonyms': []
                    }
                    
                    # Get properties
                    if 'props' in compound:
                        for prop in compound['props']:
                            if prop.get('urn', {}).get('label') == 'Molecular Formula':
                                result['molecular_formula'] = prop.get('value', {}).get('sval')
                            elif prop.get('urn', {}).get('label') == 'Molecular Weight':
                                result['molecular_weight'] = prop.get('value', {}).get('fval')
                            elif prop.get('urn', {}).get('label') == 'IUPAC Name':
                                result['iupac_name'] = prop.get('value', {}).get('sval')
                    
                    self.pubchem_cache[name] = result
                    time.sleep(0.2)  # Rate limiting
                    return result
                    
        except Exception as e:
            logger.warning(f"PubChem query failed for '{name}': {e}")
            
        self.pubchem_cache[name] = None
        return None
        
    def get_canonical_name(self, entity: str) -> str:
        """Get canonical name for an entity using manual mappings and PubChem."""
        if pd.isna(entity) or entity == "":
            return ""
            
        # Normalize the input
        normalized = self.normalize_entity_name(entity)
        if not normalized:
            return ""
            
        # Check manual mappings first
        if normalized in self.manual_mappings:
            return self.manual_mappings[normalized]
            
        # Check case-insensitive manual mappings
        normalized_lower = normalized.lower()
        for key, value in self.manual_mappings.items():
            if key.lower() == normalized_lower:
                return value
                
        # Try PubChem lookup for chemical-looking names
        if self.looks_like_chemical(normalized):
            pubchem_result = self.query_pubchem_by_name(normalized)
            if pubchem_result and pubchem_result.get('iupac_name'):
                canonical = pubchem_result['iupac_name']
                # Cache this mapping
                self.manual_mappings[normalized] = canonical
                return canonical
                
        # Return normalized version if no mapping found
        return normalized
        
    def looks_like_chemical(self, name: str) -> bool:
        """Determine if a name looks like a chemical compound."""
        chemical_indicators = [
            r'Li[A-Z]',  # Lithium compounds
            r'Na[A-Z]',  # Sodium compounds
            r'[A-Z][a-z]*\d+',  # Chemical formulas with numbers
            r'[A-Z]{2,}$',  # Abbreviations like TFSI, EC, etc.
            r'(oxide|carbonate|phosphate|sulfate|fluoride|chloride|bromide|iodide)',
            r'(acid|amine|ether|ester|alcohol)',
            r'(bis|tri|tetra|penta|hexa|poly)',
            r'\([IVX]+\)',  # Roman numerals in parentheses
        ]
        
        for pattern in chemical_indicators:
            if re.search(pattern, name, re.IGNORECASE):
                return True
                
        return False

    def load_and_flatten_json(self) -> pd.DataFrame:
        """
        Load the raw extraction JSON and flatten it into a DataFrame.
        Each triple becomes a row with its source paragraph preserved.
        """
        logger.info(f"📂 Loading data from {self.input_file}...")
        
        with open(self.input_file, 'r') as f:
            raw_data = json.load(f)
        
        logger.info(f"✅ Loaded {len(raw_data)} paragraph records")
        
        # Flatten the nested structure
        flattened_rows = []
        triple_id = 0
        
        for record in raw_data:
            doc_id = record.get('doc_id', '')
            paragraph_index = record.get('paragraph_index', -1)
            paragraph = record.get('paragraph', '')  # FULL source text
            processed_at = record.get('processed_at', '')
            
            salts = record.get('salts', [])
            
            for salt_entry in salts:
                salt_name = salt_entry.get('salt', '')
                relationships = salt_entry.get('relationships', [])
                
                for rel in relationships:
                    relation = rel.get('relation', '')
                    obj = rel.get('object', '')
                    
                    flattened_rows.append({
                        'triple_id': triple_id,
                        'doc_id': doc_id,
                        'paragraph_index': paragraph_index,
                        'subject': salt_name,
                        'relation': relation,
                        'object': obj,
                        'source_text': paragraph,  # COMPLETE non-truncated paragraph
                        'processed_at': processed_at
                    })
                    triple_id += 1
        
        df = pd.DataFrame(flattened_rows)
        logger.info(f"✅ Flattened into {len(df)} triples with source text")
        
        return df
        
    def process_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and canonicalize all relationships in the dataframe."""
        logger.info("🔄 Processing and canonicalizing relationships...")
        
        # Create canonical columns
        df_canonical = df.copy()
        
        # Track entity mappings
        entity_mappings = {}
        
        # Process subjects
        logger.info("  📝 Canonicalizing subjects...")
        subjects_canonical = []
        for i, subject in enumerate(df['subject']):
            if i % 1000 == 0 and i > 0:
                logger.info(f"    Processing subject {i+1}/{len(df)}")
            canonical = self.get_canonical_name(subject)
            subjects_canonical.append(canonical)
            if subject != canonical:
                entity_mappings[subject] = canonical
                
        df_canonical['subject_canonical'] = subjects_canonical
        
        # Process objects
        logger.info("  📝 Canonicalizing objects...")
        objects_canonical = []
        for i, obj in enumerate(df['object']):
            if i % 1000 == 0 and i > 0:
                logger.info(f"    Processing object {i+1}/{len(df)}")
            canonical = self.get_canonical_name(obj)
            objects_canonical.append(canonical)
            if obj != canonical:
                entity_mappings[obj] = canonical
                
        df_canonical['object_canonical'] = objects_canonical
        
        # Process relations (normalize but don't necessarily canonicalize)
        logger.info("  📝 Normalizing relations...")
        relations_normalized = []
        for relation in df['relation']:
            normalized = self.normalize_entity_name(relation).upper()
            relations_normalized.append(normalized)
            
        df_canonical['relation_normalized'] = relations_normalized
        
        # Store entity mappings
        self.entity_mappings = entity_mappings
        
        logger.info(f"✅ Processed {len(df)} relationships")
        logger.info(f"   - Created {len(entity_mappings)} entity mappings")
        logger.info(f"   - Unique canonical subjects: {df_canonical['subject_canonical'].nunique()}")
        logger.info(f"   - Unique canonical objects: {df_canonical['object_canonical'].nunique()}")
        logger.info(f"   - Unique normalized relations: {df_canonical['relation_normalized'].nunique()}")
        
        return df_canonical
        
    def calculate_statistics(self, df_original: pd.DataFrame, df_canonical: pd.DataFrame):
        """Calculate statistics about the canonicalization process."""
        logger.info("📊 Calculating canonicalization statistics...")
        
        stats = {
            'original_data': {
                'total_relationships': len(df_original),
                'unique_subjects': df_original['subject'].nunique(),
                'unique_objects': df_original['object'].nunique(),
                'unique_relations': df_original['relation'].nunique(),
                'total_unique_entities': len(set(df_original['subject'].dropna()) | 
                                          set(df_original['object'].dropna())),
                'unique_paragraphs': df_original['source_text'].nunique()
            },
            'canonical_data': {
                'total_relationships': len(df_canonical),
                'unique_canonical_subjects': df_canonical['subject_canonical'].nunique(),
                'unique_canonical_objects': df_canonical['object_canonical'].nunique(),
                'unique_normalized_relations': df_canonical['relation_normalized'].nunique(),
                'total_unique_canonical_entities': len(set(df_canonical['subject_canonical'].dropna()) | 
                                                    set(df_canonical['object_canonical'].dropna()))
            },
            'entity_mappings': {
                'total_mappings': len(self.entity_mappings),
                'pubchem_lookups': len([k for k, v in self.pubchem_cache.items() if v is not None]),
                'manual_mappings_used': len([k for k in self.entity_mappings.keys() 
                                          if k in self.manual_mappings])
            },
            'compression_ratios': {}
        }
        
        # Calculate compression ratios safely
        if stats['original_data']['unique_subjects'] > 0:
            stats['compression_ratios']['subject_compression'] = (
                stats['original_data']['unique_subjects'] - 
                stats['canonical_data']['unique_canonical_subjects']
            ) / stats['original_data']['unique_subjects']
        else:
            stats['compression_ratios']['subject_compression'] = 0.0
            
        if stats['original_data']['unique_objects'] > 0:
            stats['compression_ratios']['object_compression'] = (
                stats['original_data']['unique_objects'] - 
                stats['canonical_data']['unique_canonical_objects']
            ) / stats['original_data']['unique_objects']
        else:
            stats['compression_ratios']['object_compression'] = 0.0
            
        if stats['original_data']['total_unique_entities'] > 0:
            stats['compression_ratios']['total_entity_compression'] = (
                stats['original_data']['total_unique_entities'] - 
                stats['canonical_data']['total_unique_canonical_entities']
            ) / stats['original_data']['total_unique_entities']
        else:
            stats['compression_ratios']['total_entity_compression'] = 0.0
        
        # Calculate most frequent entities and relations
        stats['top_entities'] = {
            'original_subjects': df_original['subject'].value_counts().head(10).to_dict(),
            'canonical_subjects': df_canonical['subject_canonical'].value_counts().head(10).to_dict(),
            'original_objects': df_original['object'].value_counts().head(10).to_dict(),
            'canonical_objects': df_canonical['object_canonical'].value_counts().head(10).to_dict(),
            'normalized_relations': df_canonical['relation_normalized'].value_counts().head(10).to_dict()
        }
        
        self.entity_stats = stats
        return stats
        
    def save_outputs(self, df_canonical: pd.DataFrame, stats: Dict):
        """Save all outputs to files."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"💾 Saving outputs to {self.output_dir}/...")
        
        # Save canonical relationships WITH SOURCE TEXT
        # This is the main output for evaluation
        canonical_file = f"{self.output_dir}/salt_relationships_canonical_with_source.csv"
        df_canonical.to_csv(canonical_file, index=False)
        logger.info(f"   ✅ Saved canonical relationships with source: {canonical_file}")
        
        # Also save a timestamped version
        canonical_file_ts = f"{self.output_dir}/salt_relationships_canonical_with_source_{timestamp}.csv"
        df_canonical.to_csv(canonical_file_ts, index=False)
        logger.info(f"   ✅ Saved timestamped version: {canonical_file_ts}")
        
        # Save entity mappings
        mappings_file = f"{self.output_dir}/salt_entity_canonicalization_mappings.json"
        with open(mappings_file, 'w') as f:
            json.dump(self.entity_mappings, f, indent=2)
        logger.info(f"   ✅ Saved entity mappings: {mappings_file}")
        
        # Save PubChem cache
        cache_file = f"{self.output_dir}/pubchem_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.pubchem_cache, f, indent=2)
        logger.info(f"   ✅ Saved PubChem cache: {cache_file}")
        
        # Save statistics
        stats_file = f"{self.output_dir}/canonicalization_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"   ✅ Saved statistics: {stats_file}")
        
        # Create human-readable report
        report_file = f"{self.output_dir}/salt_canonicalization_report.txt"
        with open(report_file, 'w') as f:
            f.write("🧪 SALT ONTOLOGY CANONICALIZATION REPORT (WITH SOURCE TEXT)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Input file: {self.input_file}\n\n")
            
            f.write("📊 DATASET STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Original relationships: {stats['original_data']['total_relationships']:,}\n")
            f.write(f"Original unique entities: {stats['original_data']['total_unique_entities']:,}\n")
            f.write(f"  - Subjects: {stats['original_data']['unique_subjects']:,}\n")
            f.write(f"  - Objects: {stats['original_data']['unique_objects']:,}\n")
            f.write(f"  - Relations: {stats['original_data']['unique_relations']:,}\n")
            f.write(f"  - Unique source paragraphs: {stats['original_data']['unique_paragraphs']:,}\n\n")
            
            f.write(f"Canonical unique entities: {stats['canonical_data']['total_unique_canonical_entities']:,}\n")
            f.write(f"  - Subjects: {stats['canonical_data']['unique_canonical_subjects']:,}\n")
            f.write(f"  - Objects: {stats['canonical_data']['unique_canonical_objects']:,}\n")
            f.write(f"  - Relations: {stats['canonical_data']['unique_normalized_relations']:,}\n\n")
            
            f.write("🔄 CANONICALIZATION RESULTS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total entity mappings created: {stats['entity_mappings']['total_mappings']:,}\n")
            f.write(f"PubChem successful lookups: {stats['entity_mappings']['pubchem_lookups']:,}\n")
            f.write(f"Manual mappings used: {stats['entity_mappings']['manual_mappings_used']:,}\n\n")
            
            f.write("📉 COMPRESSION RATIOS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Subject compression: {stats['compression_ratios']['subject_compression']:.2%}\n")
            f.write(f"Object compression: {stats['compression_ratios']['object_compression']:.2%}\n")
            f.write(f"Total entity compression: {stats['compression_ratios']['total_entity_compression']:.2%}\n\n")
            
            f.write("🔝 TOP CANONICAL ENTITIES\n")
            f.write("-" * 30 + "\n")
            f.write("Top canonical subjects:\n")
            for entity, count in list(stats['top_entities']['canonical_subjects'].items())[:5]:
                f.write(f"  - {entity}: {count:,}\n")
            f.write("\nTop canonical objects:\n")
            for entity, count in list(stats['top_entities']['canonical_objects'].items())[:5]:
                f.write(f"  - {entity}: {count:,}\n")
            f.write("\nTop normalized relations:\n")
            for relation, count in list(stats['top_entities']['normalized_relations'].items())[:5]:
                f.write(f"  - {relation}: {count:,}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("-" * 30 + "\n")
            f.write(f"Main output: {canonical_file}\n")
            f.write(f"  - Contains COMPLETE source paragraphs for each triple\n")
            f.write(f"  - Ready for LLM-as-Judge evaluation\n")
                
        logger.info(f"   ✅ Saved report: {report_file}")
        
    def run(self):
        """Main execution method."""
        logger.info("🧪 STARTING SALT ONTOLOGY CANONICALIZATION (WITH SOURCE TEXT)")
        logger.info("=" * 70)
        
        # Load and flatten JSON data
        df = self.load_and_flatten_json()
        
        # Process relationships
        df_canonical = self.process_relationships(df)
        
        # Calculate statistics
        stats = self.calculate_statistics(df, df_canonical)
        
        # Save outputs
        self.save_outputs(df_canonical, stats)
        
        logger.info("\n🎉 CANONICALIZATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Total triples: {stats['original_data']['total_relationships']:,}")
        logger.info(f"   - Original entities: {stats['original_data']['total_unique_entities']:,}")
        logger.info(f"   - Canonical entities: {stats['canonical_data']['total_unique_canonical_entities']:,}")
        logger.info(f"   - Entity compression: {stats['compression_ratios']['total_entity_compression']:.2%}")
        logger.info(f"   - Mappings created: {stats['entity_mappings']['total_mappings']:,}")
        logger.info(f"   - Source paragraphs preserved: {stats['original_data']['unique_paragraphs']:,}")
        logger.info(f"📁 Outputs saved to: {self.output_dir}/")
        
        return df_canonical


def main():
    """Main function."""
    # Configuration - using the raw extraction JSON that has paragraphs
    input_file = "salt_analysis_output/enhanced_salt_extraction_raw_20250815_225432.json"
    output_dir = "salt_analysis_output/canonical"
    
    # Create and run cleaner
    cleaner = SaltOntologyCleanerWithSource(input_file, output_dir)
    df_result = cleaner.run()
    
    # Print sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (first 3 rows)")
    print("=" * 70)
    print(df_result[['subject', 'subject_canonical', 'relation_normalized', 'object_canonical', 'source_text']].head(3).to_string())
    print("\nSource text column contains COMPLETE paragraphs (not truncated)")


if __name__ == "__main__":
    main()
