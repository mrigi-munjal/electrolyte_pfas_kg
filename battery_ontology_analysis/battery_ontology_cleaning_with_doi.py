#!/usr/bin/env python3
"""
Battery Ontology Cleaning and Canonicalization

This script cleans and canonicalizes the battery ontology relationships data by:
1. Using PubChem compound data to standardize chemical names
2. Applying manual synonym mappings for common battery terms
3. Standardizing chemical formulas and structures
4. Removing duplicates and inconsistencies
5. Generating a cleaned relationships dataset with canonical names
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class BatteryOntologyCanonicalizer:
    """Main class for canonicalizing battery ontology data"""
    
    def __init__(self):
        self.pubchem_synonyms = {}
        self.manual_synonyms = {}
        self.canonical_mappings = {}
        self.stats = {
            'original_subjects': 0,
            'original_objects': 0,
            'original_relationships': 0,
            'canonical_subjects': 0,
            'canonical_objects': 0,
            'canonical_relationships': 0,
            'duplicates_removed': 0
        }
    
    def load_pubchem_data(self, file_path='PubChem_filtered_compounds_mw_250_or_below.csv'):
        """Load PubChem compound data and create synonym mappings"""
        print("📊 Loading PubChem compound data...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df):,} PubChem compounds")
            
            # Create synonym mappings
            for _, row in df.iterrows():
                canonical_name = str(row['cmpdname']).strip()
                if pd.isna(canonical_name) or canonical_name in ['nan', '']:
                    continue
                
                # Add molecular formula as synonym
                mf = str(row['mf']).strip()
                if not pd.isna(mf) and mf not in ['nan', '']:
                    self.pubchem_synonyms[mf] = canonical_name
                
                # Process synonyms
                synonyms_str = str(row['cmpdsynonym'])
                if not pd.isna(synonyms_str) and synonyms_str not in ['nan', '']:
                    # Split by pipe and clean
                    synonyms = [s.strip() for s in synonyms_str.split('|') if s.strip()]
                    for synonym in synonyms:
                        # Skip CAS numbers, CHEBI IDs, etc.
                        if not re.match(r'^(\d{2,}-\d{2}-\d|CHEBI:|DTXSID|UNII-|CHEMBL|DB\d+|Q\d+)', synonym):
                            self.pubchem_synonyms[synonym] = canonical_name
                
                # Add the canonical name itself
                self.pubchem_synonyms[canonical_name] = canonical_name
            
            print(f"✅ Created {len(self.pubchem_synonyms):,} PubChem synonym mappings")
            
        except Exception as e:
            print(f"❌ Error loading PubChem data: {e}")
    
    def create_manual_synonym_mappings(self):
        """Create manual synonym mappings for common battery terms"""
        print("🔧 Creating manual synonym mappings...")
        
        # Lithium salts standardization
        lithium_salts = {
            # LiPF6 variants
            'LiPF6': 'LiPF6',
            'lithium hexafluorophosphate': 'LiPF6',
            'lithium phosphate hexafluoride': 'LiPF6',
            'LiPF₆': 'LiPF6',
            
            # LiTFSI variants
            'LiTFSI': 'LiTFSI',
            'lithium bis(trifluoromethanesulfonyl)imide': 'LiTFSI',
            'lithium triflimide': 'LiTFSI',
            'LiN(SO2CF3)2': 'LiTFSI',
            'LiNTf2': 'LiTFSI',
            'LiTf2N': 'LiTFSI',
            
            # LiFSI variants
            'LiFSI': 'LiFSI',
            'lithium bis(fluorosulfonyl)imide': 'LiFSI',
            'LiN(SO2F)2': 'LiFSI',
            'LiFTFSI': 'LiFSI',  # Common misnaming
            
            # LiBF4 variants
            'LiBF4': 'LiBF4',
            'lithium tetrafluoroborate': 'LiBF4',
            'LiBF₄': 'LiBF4',
            
            # LiClO4 variants
            'LiClO4': 'LiClO4',
            'lithium perchlorate': 'LiClO4',
            'LiClO₄': 'LiClO4',
            
            # LiAsF6 variants
            'LiAsF6': 'LiAsF6',
            'lithium hexafluoroarsenate': 'LiAsF6',
            'LiAsF₆': 'LiAsF6',
            
            # Other lithium salts
            'LiDFOB': 'LiDFOB',
            'LiODFB': 'LiDFOB',  # Same compound, different naming
            'LiFOB': 'LiFOB',
            'LiPO2F2': 'LiPO2F2',
            'LiTFPB': 'LiTFPB',
            'LiB(CN)4': 'LiB(CN)4',
        }
        
        # Cathode materials
        cathode_materials = {
            # LiFePO4 variants
            'LiFePO4': 'LiFePO4',
            'LFP': 'LiFePO4',
            'lithium iron phosphate': 'LiFePO4',
            'LiFePO₄': 'LiFePO4',
            'LiFePO4/C': 'LiFePO4/C',
            'LiFePO4-C': 'LiFePO4/C',
            'carbon-coated LiFePO4': 'LiFePO4/C',
            
            # LiCoO2 variants
            'LiCoO2': 'LiCoO2',
            'LCO': 'LiCoO2',
            'lithium cobalt oxide': 'LiCoO2',
            'LiCoO₂': 'LiCoO2',
            
            # LiMn2O4 variants
            'LiMn2O4': 'LiMn2O4',
            'LMO': 'LiMn2O4',
            'lithium manganese oxide': 'LiMn2O4',
            'LiMn₂O₄': 'LiMn2O4',
            
            # NCM variants
            'NCM': 'LiNi1/3Co1/3Mn1/3O2',
            'NMC': 'LiNi1/3Co1/3Mn1/3O2',
            'LiNiCoMnO2': 'LiNi1/3Co1/3Mn1/3O2',
            'NCM111': 'LiNi1/3Co1/3Mn1/3O2',
            'NCM811': 'LiNi0.8Co0.1Mn0.1O2',
            'NCM622': 'LiNi0.6Co0.2Mn0.2O2',
            'NCM523': 'LiNi0.5Co0.2Mn0.3O2',
            
            # NCA variants
            'NCA': 'LiNiCoAlO2',
            'LiNiCoAlO2': 'LiNiCoAlO2',
            'lithium nickel cobalt aluminum oxide': 'LiNiCoAlO2',
        }
        
        # Anode materials
        anode_materials = {
            'graphite': 'graphite',
            'natural graphite': 'graphite',
            'artificial graphite': 'graphite',
            'synthetic graphite': 'graphite',
            'Li': 'lithium metal',
            'lithium': 'lithium metal',
            'lithium metal': 'lithium metal',
            'Li metal': 'lithium metal',
            'silicon': 'silicon',
            'Si': 'silicon',
            'Li4Ti5O12': 'Li4Ti5O12',
            'LTO': 'Li4Ti5O12',
            'lithium titanate': 'Li4Ti5O12',
        }
        
        # Solvents
        solvents = {
            # Carbonate solvents
            'EC': 'ethylene carbonate',
            'ethylene carbonate': 'ethylene carbonate',
            'DMC': 'dimethyl carbonate',
            'dimethyl carbonate': 'dimethyl carbonate',
            'DEC': 'diethyl carbonate',
            'diethyl carbonate': 'diethyl carbonate',
            'EMC': 'ethyl methyl carbonate',
            'ethyl methyl carbonate': 'ethyl methyl carbonate',
            'PC': 'propylene carbonate',
            'propylene carbonate': 'propylene carbonate',
            'VC': 'vinylene carbonate',
            'vinylene carbonate': 'vinylene carbonate',
            'FEC': 'fluoroethylene carbonate',
            'fluoroethylene carbonate': 'fluoroethylene carbonate',
            
            # Other solvents
            'NMP': 'N-methyl-2-pyrrolidone',
            'N-methyl-2-pyrrolidone': 'N-methyl-2-pyrrolidone',
            'DMSO': 'dimethyl sulfoxide',
            'dimethyl sulfoxide': 'dimethyl sulfoxide',
            'DOL': '1,3-dioxolane',
            '1,3-dioxolane': '1,3-dioxolane',
            'DME': '1,2-dimethoxyethane',
            '1,2-dimethoxyethane': '1,2-dimethoxyethane',
            'THF': 'tetrahydrofuran',
            'tetrahydrofuran': 'tetrahydrofuran',
        }
        
        # General terms
        general_terms = {
            'electrolyte': 'electrolyte',
            'separator': 'separator',
            'carbon': 'carbon',
            'binder': 'binder',
            'PVDF': 'PVDF',
            'polyvinylidene fluoride': 'PVDF',
            'CMC': 'carboxymethyl cellulose',
            'carboxymethyl cellulose': 'carboxymethyl cellulose',
            'SBR': 'styrene-butadiene rubber',
            'styrene-butadiene rubber': 'styrene-butadiene rubber',
            'CNT': 'carbon nanotube',
            'carbon nanotube': 'carbon nanotube',
            'carbon nanotubes': 'carbon nanotube',
            'BATTERY_CELL': 'battery cell',
            'battery cell': 'battery cell',
            'lithium-ion battery': 'lithium-ion battery',
            'Li-ion battery': 'lithium-ion battery',
            'LIB': 'lithium-ion battery',
        }
        
        # Combine all mappings
        self.manual_synonyms.update(lithium_salts)
        self.manual_synonyms.update(cathode_materials)
        self.manual_synonyms.update(anode_materials)
        self.manual_synonyms.update(solvents)
        self.manual_synonyms.update(general_terms)
        
        print(f"✅ Created {len(self.manual_synonyms):,} manual synonym mappings")
    
    def canonicalize_name(self, name):
        """Canonicalize a single entity name using all available mappings"""
        if pd.isna(name) or str(name).strip() == '':
            return name
        
        name = str(name).strip()
        
        # First try manual synonyms (more specific)
        if name in self.manual_synonyms:
            return self.manual_synonyms[name]
        
        # Then try PubChem synonyms
        if name in self.pubchem_synonyms:
            return self.pubchem_synonyms[name]
        
        # Apply some basic cleaning rules
        cleaned_name = self.apply_cleaning_rules(name)
        
        # Check again after cleaning
        if cleaned_name in self.manual_synonyms:
            return self.manual_synonyms[cleaned_name]
        
        if cleaned_name in self.pubchem_synonyms:
            return self.pubchem_synonyms[cleaned_name]
        
        return cleaned_name
    
    def apply_cleaning_rules(self, name):
        """Apply basic cleaning rules to entity names"""
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Standardize chemical subscripts
        name = re.sub(r'([A-Za-z])([0-9]+)', r'\1\2', name)  # Remove spaces before numbers
        
        # Convert common Unicode subscripts to regular numbers
        subscript_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        for unicode_sub, regular in subscript_map.items():
            name = name.replace(unicode_sub, regular)
        
        # Standardize temperature ranges
        name = re.sub(r'(\d+)\s*[°º]\s*C', r'\1°C', name)
        name = re.sub(r'(\d+)\s*K', r'\1 K', name)
        
        # Standardize concentration units
        name = re.sub(r'(\d+(?:\.\d+)?)\s*M\b', r'\1 M', name)
        name = re.sub(r'(\d+(?:\.\d+)?)\s*wt%', r'\1 wt%', name)
        name = re.sub(r'(\d+(?:\.\d+)?)\s*mol%', r'\1 mol%', name)
        
        # Standardize voltage
        name = re.sub(r'(\d+(?:\.\d+)?)\s*V\b', r'\1 V', name)
        
        return name.strip()
    
    def load_relationships_data(self, file_path='battery_ontology_triples_with_doi_clean_20250819_214710.csv'):
        """Load the relationships data with DOI and paragraph information"""
        print("📊 Loading relationships data with DOI information...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df):,} relationships with DOI information")
            
            # Check for expected columns
            required_cols = ['subject', 'relation', 'object']
            doi_cols = ['doi', 'paragraph']
            
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                print(f"❌ Missing required columns: {missing_required}")
                return None
            
            missing_doi = [col for col in doi_cols if col not in df.columns]
            if missing_doi:
                print(f"⚠️  Missing DOI columns: {missing_doi} - will proceed without them")
            else:
                print(f"✅ Found DOI and paragraph columns")
                unique_dois = df['doi'].nunique()
                known_dois = len(df[df['doi'] != 'Unknown'])
                print(f"  - Unique DOIs: {unique_dois:,}")
                print(f"  - Triples with known DOI: {known_dois:,} ({known_dois/len(df)*100:.1f}%)")
            
            # Store original stats
            self.stats['original_relationships'] = len(df)
            self.stats['original_subjects'] = df['subject'].nunique()
            self.stats['original_objects'] = df['object'].nunique()
            
            print(f"Original stats:")
            print(f"  - Unique subjects: {self.stats['original_subjects']:,}")
            print(f"  - Unique objects: {self.stats['original_objects']:,}")
            print(f"  - Total relationships: {self.stats['original_relationships']:,}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading relationships data: {e}")
            return None
    
    def canonicalize_relationships(self, df):
        """Canonicalize all entity names in the relationships while preserving DOI and metadata"""
        print("🔄 Canonicalizing entity names...")
        
        # Create canonical mappings
        all_entities = set(df['subject'].unique()) | set(df['object'].unique())
        print(f"Processing {len(all_entities):,} unique entities...")
        
        canonical_map = {}
        for entity in all_entities:
            canonical_map[entity] = self.canonicalize_name(entity)
        
        # Apply canonicalization while preserving all original columns
        df_canonical = df.copy()
        df_canonical['subject'] = df_canonical['subject'].map(canonical_map)
        df_canonical['object'] = df_canonical['object'].map(canonical_map)
        
        # Store canonical mappings for reporting
        self.canonical_mappings = canonical_map
        
        print("✅ Entity canonicalization complete")
        print(f"✅ Preserved metadata columns: {[col for col in df.columns if col not in ['subject', 'relation', 'object']]}")
        return df_canonical
    
    def remove_duplicates_and_clean(self, df):
        """Remove duplicates and invalid relationships"""
        print("🧹 Removing duplicates and cleaning data...")
        
        initial_count = len(df)
        
        # Remove rows with NaN values
        df_clean = df.dropna(subset=['subject', 'relation', 'object'])
        
        # Remove relationships where subject equals object
        df_clean = df_clean[df_clean['subject'] != df_clean['object']]
        
        # Remove duplicate relationships (same subject-relation-object)
        df_clean = df_clean.drop_duplicates(subset=['subject', 'relation', 'object'])
        
        # Remove very short or uninformative names
        df_clean = df_clean[
            (df_clean['subject'].str.len() > 1) & 
            (df_clean['object'].str.len() > 1)
        ]
        
        final_count = len(df_clean)
        self.stats['duplicates_removed'] = initial_count - final_count
        
        print(f"✅ Removed {self.stats['duplicates_removed']:,} duplicates/invalid relationships")
        print(f"Final dataset: {final_count:,} relationships")
        
        return df_clean
    
    def analyze_canonicalization_results(self, df_original, df_canonical):
        """Analyze the results of canonicalization"""
        print("\n" + "="*60)
        print("📊 CANONICALIZATION RESULTS ANALYSIS")
        print("="*60)
        
        # Update final stats
        self.stats['canonical_relationships'] = len(df_canonical)
        self.stats['canonical_subjects'] = df_canonical['subject'].nunique()
        self.stats['canonical_objects'] = df_canonical['object'].nunique()
        
        # Print comparison
        print(f"BEFORE CANONICALIZATION:")
        print(f"  - Unique subjects: {self.stats['original_subjects']:,}")
        print(f"  - Unique objects: {self.stats['original_objects']:,}")
        print(f"  - Total relationships: {self.stats['original_relationships']:,}")
        
        print(f"\nAFTER CANONICALIZATION:")
        print(f"  - Unique subjects: {self.stats['canonical_subjects']:,}")
        print(f"  - Unique objects: {self.stats['canonical_objects']:,}")
        print(f"  - Total relationships: {self.stats['canonical_relationships']:,}")
        
        print(f"\nREDUCTION:")
        subj_reduction = self.stats['original_subjects'] - self.stats['canonical_subjects']
        obj_reduction = self.stats['original_objects'] - self.stats['canonical_objects']
        rel_reduction = self.stats['original_relationships'] - self.stats['canonical_relationships']
        
        print(f"  - Subjects reduced by: {subj_reduction:,} ({subj_reduction/self.stats['original_subjects']*100:.1f}%)")
        print(f"  - Objects reduced by: {obj_reduction:,} ({obj_reduction/self.stats['original_objects']*100:.1f}%)")
        print(f"  - Relationships reduced by: {rel_reduction:,} ({rel_reduction/self.stats['original_relationships']*100:.1f}%)")
        
        # Show most frequent canonicalizations
        print(f"\n🔄 TOP 10 ENTITY CANONICALIZATIONS:")
        canonicalization_counts = defaultdict(list)
        for original, canonical in self.canonical_mappings.items():
            if original != canonical:
                canonicalization_counts[canonical].append(original)
        
        # Sort by number of synonyms mapped to each canonical name
        top_canonicalizations = sorted(canonicalization_counts.items(), 
                                     key=lambda x: len(x[1]), reverse=True)[:10]
        
        for canonical, synonyms in top_canonicalizations:
            print(f"  {canonical}: {len(synonyms)} variants")
            print(f"    Examples: {', '.join(synonyms[:5])}")
            if len(synonyms) > 5:
                print(f"    ... and {len(synonyms)-5} more")
            print()
        
        # Top entities after canonicalization
        print(f"🔝 TOP 10 SUBJECTS AFTER CANONICALIZATION:")
        top_subjects = df_canonical['subject'].value_counts().head(10)
        for i, (subject, count) in enumerate(top_subjects.items(), 1):
            print(f"  {i:2d}. {subject}: {count:,} relationships")
        
        print(f"\n🔝 TOP 10 OBJECTS AFTER CANONICALIZATION:")
        top_objects = df_canonical['object'].value_counts().head(10)
        for i, (obj, count) in enumerate(top_objects.items(), 1):
            print(f"  {i:2d}. {obj}: {count:,} relationships")
    
    def save_results(self, df_canonical):
        """Save the canonicalized results with DOI information"""
        print("\n💾 Saving canonicalized results with DOI information...")
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save canonical relationships with DOI
        output_path = f'battery_ontology_analysis/relationships_data_canonical_with_doi_{timestamp}.csv'
        df_canonical.to_csv(output_path, index=False)
        print(f"✅ Saved canonical relationships with DOI to: {output_path}")
        
        # Save canonicalization mappings
        mappings_path = f'battery_ontology_analysis/entity_canonicalization_mappings_with_doi_{timestamp}.json'
        with open(mappings_path, 'w') as f:
            json.dump(self.canonical_mappings, f, indent=2)
        print(f"✅ Saved canonicalization mappings to: {mappings_path}")
        
        # Save statistics report
        report_path = f'battery_ontology_analysis/canonicalization_report_with_doi_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("BATTERY ONTOLOGY CANONICALIZATION REPORT (WITH DOI)\n")
            f.write("=" * 55 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INPUT FILE\n")
            f.write("-" * 15 + "\n")
            f.write("battery_ontology_triples_with_doi_clean_20250819_214710.csv\n")
            f.write("(Contains DOI and paragraph provenance information)\n\n")
            
            f.write("STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original unique subjects: {self.stats['original_subjects']:,}\n")
            f.write(f"Original unique objects: {self.stats['original_objects']:,}\n")
            f.write(f"Original total relationships: {self.stats['original_relationships']:,}\n\n")
            
            f.write(f"Canonical unique subjects: {self.stats['canonical_subjects']:,}\n")
            f.write(f"Canonical unique objects: {self.stats['canonical_objects']:,}\n")
            f.write(f"Canonical total relationships: {self.stats['canonical_relationships']:,}\n\n")
            
            subj_reduction = self.stats['original_subjects'] - self.stats['canonical_subjects']
            obj_reduction = self.stats['original_objects'] - self.stats['canonical_objects']
            rel_reduction = self.stats['original_relationships'] - self.stats['canonical_relationships']
            
            f.write(f"Subjects reduced by: {subj_reduction:,} ({subj_reduction/self.stats['original_subjects']*100:.1f}%)\n")
            f.write(f"Objects reduced by: {obj_reduction:,} ({obj_reduction/self.stats['original_objects']*100:.1f}%)\n")
            f.write(f"Relationships reduced by: {rel_reduction:,} ({rel_reduction/self.stats['original_relationships']*100:.1f}%)\n")
            f.write(f"Duplicates/invalid relationships removed: {self.stats['duplicates_removed']:,}\n\n")
            
            f.write("DOI INFORMATION PRESERVED\n")
            f.write("-" * 30 + "\n")
            if 'doi' in df_canonical.columns:
                unique_dois = df_canonical['doi'].nunique()
                known_dois = len(df_canonical[df_canonical['doi'] != 'Unknown'])
                f.write(f"Unique DOIs in final dataset: {unique_dois:,}\n")
                f.write(f"Triples with known DOI: {known_dois:,} ({known_dois/len(df_canonical)*100:.1f}%)\n")
                f.write(f"Triples with unknown DOI: {len(df_canonical) - known_dois:,}\n\n")
            
            f.write("METADATA COLUMNS PRESERVED\n")
            f.write("-" * 30 + "\n")
            metadata_cols = [col for col in df_canonical.columns if col not in ['subject', 'relation', 'object']]
            for col in metadata_cols:
                f.write(f"  - {col}\n")
            f.write("\n")
            
            f.write("CANONICALIZATION SOURCES\n")
            f.write("-" * 25 + "\n")
            f.write(f"PubChem synonym mappings: {len(self.pubchem_synonyms):,}\n")
            f.write(f"Manual synonym mappings: {len(self.manual_synonyms):,}\n")
            f.write(f"Total entity mappings created: {len(self.canonical_mappings):,}\n")
        
        print(f"✅ Saved canonicalization report to: {report_path}")
        
        return output_path, mappings_path, report_path

def main():
    """Main function to run the canonicalization process with DOI information"""
    print("🧹 BATTERY ONTOLOGY CANONICALIZATION WITH DOI INFORMATION")
    print("=" * 65)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📁 Input: battery_ontology_triples_with_doi_clean_20250819_214710.csv")
    print("🔗 Preserving DOI and paragraph provenance information")
    print()
    
    # Initialize canonicalizer
    canonicalizer = BatteryOntologyCanonicalizer()
    
    # Load PubChem data
    canonicalizer.load_pubchem_data()
    
    # Create manual synonym mappings
    canonicalizer.create_manual_synonym_mappings()
    
    # Load relationships data with DOI
    df_original = canonicalizer.load_relationships_data()
    if df_original is None:
        return
    
    # Canonicalize entity names
    df_canonical = canonicalizer.canonicalize_relationships(df_original)
    
    # Remove duplicates and clean
    df_canonical_clean = canonicalizer.remove_duplicates_and_clean(df_canonical)
    
    # Analyze results
    canonicalizer.analyze_canonicalization_results(df_original, df_canonical_clean)
    
    # Save results
    output_files = canonicalizer.save_results(df_canonical_clean)
    
    print("\n" + "=" * 65)
    print("✅ CANONICALIZATION WITH DOI INFORMATION COMPLETE!")
    print("=" * 65)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    for file_path in output_files:
        print(f"  - {file_path}")
    
    print(f"\n🎉 Canonicalization reduced entity count by:")
    subj_reduction = canonicalizer.stats['original_subjects'] - canonicalizer.stats['canonical_subjects']
    obj_reduction = canonicalizer.stats['original_objects'] - canonicalizer.stats['canonical_objects']
    print(f"  - Subjects: {subj_reduction:,} ({subj_reduction/canonicalizer.stats['original_subjects']*100:.1f}%)")
    print(f"  - Objects: {obj_reduction:,} ({obj_reduction/canonicalizer.stats['original_objects']*100:.1f}%)")
    
    # Report on DOI information preservation
    if 'doi' in df_canonical_clean.columns:
        unique_dois = df_canonical_clean['doi'].nunique()
        known_dois = len(df_canonical_clean[df_canonical_clean['doi'] != 'Unknown'])
        print(f"\n🔗 DOI Information Preserved:")
        print(f"  - Unique DOIs: {unique_dois:,}")
        print(f"  - Triples with known DOI: {known_dois:,} ({known_dois/len(df_canonical_clean)*100:.1f}%)")
        print(f"  - Full provenance chain maintained from original papers to canonicalized entities")

if __name__ == "__main__":
    main()
