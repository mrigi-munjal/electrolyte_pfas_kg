#!/usr/bin/env python3
"""
Enhanced Salt Ontology Extraction with DOI Integration

This script extracts subject-predicate-object triples from battery/salt knowledge graphs
with enhanced DOI tracking, provenance information, and improved entity resolution.

Author: AI Assistant
Date: 2025-08-14
"""

import pandas as pd
import numpy as np
import json
import re
import os
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import logging
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSaltOntologyExtractor:
    """Enhanced extractor for salt ontology with DOI integration."""
    
    def __init__(self, input_file: str, output_dir: str = "salt_analysis_output"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.salt_synonyms = {}
        self.doi_metadata = {}
        self.entity_resolution_cache = {}
        self.load_salt_synonyms()
        
    def load_salt_synonyms(self):
        """Load comprehensive salt synonyms and chemical name mappings."""
        logger.info("📚 Loading enhanced salt synonyms and chemical mappings...")
        
        # Expanded salt synonym dictionary with chemical formulas and common names
        self.salt_synonyms = {
            # Lithium salts - comprehensive mappings
            "LiTFSI": ["Lithium bis(trifluoromethanesulfonyl)imide", "Li-TFSI", "LiN(SO2CF3)2", 
                      "lithium bistriflimide", "LiTf2N", "Li TFSI"],
            "LiPF6": ["Lithium hexafluorophosphate", "Li-PF6", "LiPF₆", "Li PF6", 
                     "lithium hexafluorophosphate salt"],
            "LiBF4": ["Lithium tetrafluoroborate", "Li-BF4", "LiBF₄", "Li BF4", 
                     "lithium fluoroborate"],
            "LiClO4": ["Lithium perchlorate", "Li-ClO4", "LiClO₄", "Li ClO4", 
                      "lithium perchlorate salt"],
            "LiCF3SO3": ["Lithium trifluoromethanesulfonate", "LiOTf", "Li-OTf", "Li OTf", 
                        "lithium triflate", "LiTfO"],
            "LiFSI": ["Lithium bis(fluorosulfonyl)imide", "Li-FSI", "LiN(SO2F)2", 
                     "lithium bis(fluorosulfonyl)imide", "Li FSI"],
            "LiDFOB": ["Lithium difluoro(oxalato)borate", "Li-DFOB", "Li DFOB", 
                      "lithium difluorooxalatoborate"],
            "LiBOB": ["Lithium bis(oxalato)borate", "Li-BOB", "Li BOB", 
                     "lithium bisoxalatoborate"],
            "LiDFP": ["Lithium difluorophosphate", "Li-DFP", "LiPO2F2", "Li DFP"],
            "LiODFB": ["Lithium oxalyldifluoroborate", "Li-ODFB", "Li ODFB"],
            "LiTDI": ["Lithium 4,5-dicyano-2-(trifluoromethyl)imidazolide", "Li-TDI", "Li TDI"],
            "LiFTFSI": ["Lithium (fluorosulfonyl)(trifluoromethanesulfonyl)imide", "Li-FTFSI"],
            
            # Sodium salts
            "NaTFSI": ["Sodium bis(trifluoromethanesulfonyl)imide", "Na-TFSI", "NaN(SO2CF3)2", 
                      "sodium bistriflimide", "Na TFSI"],
            "NaPF6": ["Sodium hexafluorophosphate", "Na-PF6", "NaPF₆", "Na PF6"],
            "NaClO4": ["Sodium perchlorate", "Na-ClO4", "NaClO₄", "Na ClO4"],
            "NaFSI": ["Sodium bis(fluorosulfonyl)imide", "Na-FSI", "NaN(SO2F)2", "Na FSI"],
            "NaOTf": ["Sodium trifluoromethanesulfonate", "Na-OTf", "NaCF3SO3", "Na OTf"],
            "NaBF4": ["Sodium tetrafluoroborate", "Na-BF4", "NaBF₄", "Na BF4"],
            
            # Potassium salts
            "KTFSI": ["Potassium bis(trifluoromethanesulfonyl)imide", "K-TFSI", "KN(SO2CF3)2", "K TFSI"],
            "KPF6": ["Potassium hexafluorophosphate", "K-PF6", "KPF₆", "K PF6"],
            "KFSI": ["Potassium bis(fluorosulfonyl)imide", "K-FSI", "KN(SO2F)2", "K FSI"],
            
            # Magnesium salts
            "Mg(TFSI)2": ["Magnesium bis(trifluoromethanesulfonyl)imide", "MgTFSI2", "Mg-TFSI2", 
                         "Mg(N(SO2CF3)2)2"],
            "Mg(ClO4)2": ["Magnesium perchlorate", "MgClO42", "Mg-ClO42", "Mg(ClO₄)₂"],
            "Mg(BF4)2": ["Magnesium tetrafluoroborate", "MgBF42", "Mg-BF42", "Mg(BF₄)₂"],
            
            # Calcium salts
            "Ca(TFSI)2": ["Calcium bis(trifluoromethanesulfonyl)imide", "CaTFSI2", "Ca-TFSI2"],
            "Ca(ClO4)2": ["Calcium perchlorate", "CaClO42", "Ca-ClO42"],
            
            # Aluminum salts
            "Al(TFSI)3": ["Aluminum bis(trifluoromethanesulfonyl)imide", "AlTFSI3", "Al-TFSI3"],
            "AlCl3": ["Aluminum chloride", "Al-Cl3", "aluminum trichloride"],
            
            # Ionic liquid cations
            "EMI": ["1-Ethyl-3-methylimidazolium", "EMIM", "[EMI]", "1-ethyl-3-methylimidazolium"],
            "BMIM": ["1-Butyl-3-methylimidazolium", "BMI", "[BMIM]", "1-butyl-3-methylimidazolium"],
            "HMIM": ["1-Hexyl-3-methylimidazolium", "HMI", "[HMIM]", "1-hexyl-3-methylimidazolium"],
            "Pyr14": ["1-Butyl-1-methylpyrrolidinium", "BMP", "[Pyr14]", "N-butyl-N-methylpyrrolidinium"],
            "DEME": ["N,N-diethyl-N-methyl-N-(2-methoxyethyl)ammonium", "[DEME]"],
            
            # Battery cathode materials
            "LiFePO4": ["Lithium iron phosphate", "LFP", "olivine LiFePO4", "lithium iron phosphate cathode"],
            "LiCoO2": ["Lithium cobalt oxide", "LCO", "lithium cobalt dioxide", "LiCoO₂"],
            "LiMn2O4": ["Lithium manganese oxide", "LMO", "spinel LiMn2O4", "LiMn₂O₄"],
            "LiNiO2": ["Lithium nickel oxide", "LNO", "lithium nickel dioxide", "LiNiO₂"],
            "NCM": ["Nickel cobalt manganese oxide", "NCM cathode", "Li(NiCoMn)O2", "NMC"],
            "NMC": ["Nickel manganese cobalt oxide", "NMC cathode", "Li(NiMnCo)O2", "NCM"],
            "NCA": ["Nickel cobalt aluminum oxide", "NCA cathode", "Li(NiCoAl)O2"],
            "LiNi0.8Co0.1Mn0.1O2": ["NCM811", "NCM-811", "Li(Ni0.8Co0.1Mn0.1)O2"],
            "LiNi0.6Co0.2Mn0.2O2": ["NCM622", "NCM-622", "Li(Ni0.6Co0.2Mn0.2)O2"],
            "LiNi0.5Co0.3Mn0.2O2": ["NCM532", "NCM-532", "Li(Ni0.5Co0.3Mn0.2)O2"],
            
            # Anode materials
            "Li4Ti5O12": ["Lithium titanate", "LTO", "Li₄Ti₅O₁₂", "lithium titanium oxide"],
            "LiC6": ["Lithiated graphite", "Li-C6", "lithium intercalated graphite"],
            "Si": ["Silicon", "silicon anode", "Si anode"],
            "SiO": ["Silicon monoxide", "silicon oxide", "SiO anode"],
            "Sn": ["Tin", "tin anode", "Sn anode"],
            "Li": ["Lithium metal", "Li metal", "lithium metal anode"],
            
            # Electrolyte solvents
            "EC": ["Ethylene carbonate", "ethylene carbonate solvent"],
            "PC": ["Propylene carbonate", "propylene carbonate solvent"],
            "DMC": ["Dimethyl carbonate", "dimethyl carbonate solvent"],
            "DEC": ["Diethyl carbonate", "diethyl carbonate solvent"],
            "EMC": ["Ethyl methyl carbonate", "ethyl methyl carbonate solvent"],
            "DME": ["1,2-Dimethoxyethane", "dimethoxyethane", "glyme"],
            "DOL": ["1,3-Dioxolane", "dioxolane"],
            "THF": ["Tetrahydrofuran", "tetrahydrofuran solvent"],
            "DMSO": ["Dimethyl sulfoxide", "dimethyl sulfoxide solvent"],
            "ACN": ["Acetonitrile", "acetonitrile solvent", "AN"],
            "AN": ["Acetonitrile", "acetonitrile solvent", "ACN"],
            "GBL": ["γ-Butyrolactone", "gamma-butyrolactone", "butyrolactone"],
            "NMP": ["N-Methyl-2-pyrrolidone", "N-methylpyrrolidone"],
            "TEGDME": ["Tetraethylene glycol dimethyl ether", "tetraglyme"],
            "DIGLYME": ["Diethylene glycol dimethyl ether", "diglyme"],
            "TRIGLYME": ["Triethylene glycol dimethyl ether", "triglyme"],
            
            # Electrolyte additives
            "VC": ["Vinylene carbonate", "vinylene carbonate additive"],
            "FEC": ["Fluoroethylene carbonate", "fluoroethylene carbonate additive"],
            "VEC": ["Vinyl ethylene carbonate", "vinyl ethylene carbonate additive"],
            "ES": ["Ethylene sulfite", "ethylene sulfite additive"],
            "PS": ["Propane sultone", "1,3-propane sultone", "propane sultone additive"],
            "DTD": ["Ditert-butyl disulfide", "di-tert-butyl disulfide"],
            "LiNO3": ["Lithium nitrate", "Li-NO3", "LiNO₃", "lithium nitrate additive"],
            "LiI": ["Lithium iodide", "Li-I", "lithium iodide additive"],
            "LiBr": ["Lithium bromide", "Li-Br", "lithium bromide additive"],
            
            # Separator materials
            "PP": ["Polypropylene", "polypropylene separator"],
            "PE": ["Polyethylene", "polyethylene separator"],
            "PET": ["Polyethylene terephthalate", "PET separator"],
            "PVDF": ["Polyvinylidene fluoride", "PVDF separator", "polyvinylidene difluoride"],
            "Celgard": ["Celgard separator", "Celgard membrane"],
            
            # Binder materials
            "PVDF": ["Polyvinylidene fluoride", "PVDF binder", "polyvinylidene difluoride binder"],
            "CMC": ["Carboxymethyl cellulose", "CMC binder"],
            "SBR": ["Styrene-butadiene rubber", "SBR binder"],
            "PAA": ["Polyacrylic acid", "PAA binder"],
            "PTFE": ["Polytetrafluoroethylene", "PTFE binder", "Teflon"],
            
            # Conductive additives
            "CB": ["Carbon black", "carbon black conductive additive"],
            "Super P": ["Super P carbon", "Super P conductive additive"],
            "CNT": ["Carbon nanotubes", "carbon nanotube additive"],
            "Graphene": ["Graphene", "graphene conductive additive"],
            "VGCF": ["Vapor grown carbon fiber", "vapor grown carbon fiber"]
        }
        
        # Create reverse mapping for entity resolution
        self.entity_resolution_cache = {}
        for canonical, synonyms in self.salt_synonyms.items():
            self.entity_resolution_cache[canonical.lower()] = canonical
            for synonym in synonyms:
                self.entity_resolution_cache[synonym.lower()] = canonical
                
        logger.info(f"✅ Loaded {len(self.salt_synonyms)} canonical entities")
        logger.info(f"✅ Created {len(self.entity_resolution_cache)} entity resolution mappings")
        
    def resolve_entity_name(self, entity: str) -> str:
        """Resolve entity name to canonical form using enhanced mappings."""
        if pd.isna(entity) or entity == "":
            return ""
            
        # Clean and normalize
        entity_clean = str(entity).strip()
        entity_lower = entity_clean.lower()
        
        # Direct lookup
        if entity_lower in self.entity_resolution_cache:
            return self.entity_resolution_cache[entity_lower]
            
        # Fuzzy matching for chemical formulas
        entity_normalized = re.sub(r'\s+', '', entity_clean)  # Remove spaces
        entity_normalized = re.sub(r'[₀₁₂₃₄₅₆₇₈₉]', 
                                 lambda m: str('₀₁₂₃₄₅₆₇₈₉'.index(m.group())), 
                                 entity_normalized)  # Convert subscripts
        
        for key, canonical in self.entity_resolution_cache.items():
            key_normalized = re.sub(r'\s+', '', key)
            if entity_normalized.lower() == key_normalized.lower():
                return canonical
                
        # Return original if no mapping found
        return entity_clean
        
    def extract_doi_from_text(self, text: str) -> Optional[str]:
        """Extract DOI from text using comprehensive patterns."""
        if pd.isna(text):
            return None
            
        text = str(text)
        
        # DOI patterns
        doi_patterns = [
            r'10\.\d+/[^\s\]]+',  # Standard DOI
            r'doi:\s*10\.\d+/[^\s\]]+',  # DOI with prefix
            r'DOI:\s*10\.\d+/[^\s\]]+',  # DOI with uppercase prefix
            r'https?://(?:dx\.)?doi\.org/10\.\d+/[^\s\]]+',  # DOI URLs
            r'https?://doi\.org/10\.\d+/[^\s\]]+',  # DOI URLs without dx
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doi = match.group(0)
                # Clean up the DOI
                doi = re.sub(r'^(doi:|DOI:)\s*', '', doi, flags=re.IGNORECASE)
                doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi)
                return doi.strip()
                
        return None
        
    def extract_metadata_from_row(self, row: pd.Series) -> Dict:
        """Extract comprehensive metadata from a data row."""
        metadata = {
            'source_file': self.input_file,
            'row_index': getattr(row, 'name', None),
            'doi': None,
            'publication_year': None,
            'journal': None,
            'authors': None,
            'title': None,
            'confidence_score': 1.0,
            'extraction_method': 'automated',
            'data_quality': 'high'
        }
        
        # Extract DOI from multiple possible columns
        doi_columns = ['doi', 'DOI', 'document_doi', 'paper_doi', 'reference_doi', 
                      'citation', 'source', 'reference', 'url', 'link']
        
        for col in doi_columns:
            if col in row.index and pd.notna(row[col]):
                doi = self.extract_doi_from_text(str(row[col]))
                if doi:
                    metadata['doi'] = doi
                    break
                    
        # Extract other metadata if available
        if 'year' in row.index:
            try:
                metadata['publication_year'] = int(row['year'])
            except (ValueError, TypeError):
                pass
                
        if 'journal' in row.index and pd.notna(row['journal']):
            metadata['journal'] = str(row['journal']).strip()
            
        if 'authors' in row.index and pd.notna(row['authors']):
            metadata['authors'] = str(row['authors']).strip()
            
        if 'title' in row.index and pd.notna(row['title']):
            metadata['title'] = str(row['title']).strip()
            
        # Estimate confidence based on data quality indicators
        confidence_factors = []
        if metadata['doi']:
            confidence_factors.append(0.4)
        if metadata['publication_year']:
            confidence_factors.append(0.2)
        if metadata['journal']:
            confidence_factors.append(0.2)
        if metadata['authors']:
            confidence_factors.append(0.2)
            
        metadata['confidence_score'] = min(1.0, sum(confidence_factors))
        
        return metadata
        
    def extract_relationships_from_row(self, row: pd.Series) -> List[Dict]:
        """Extract all possible subject-predicate-object relationships from a row."""
        relationships = []
        metadata = self.extract_metadata_from_row(row)
        
        # Define potential subject and object columns
        subject_columns = ['subject', 'entity1', 'compound', 'material', 'salt', 
                         'electrolyte', 'cathode', 'anode', 'component']
        object_columns = ['object', 'entity2', 'property', 'value', 'measurement', 
                        'characteristic', 'performance', 'parameter']
        relation_columns = ['relation', 'predicate', 'relationship', 'property_type', 
                          'measurement_type', 'interaction']
        
        # Extract explicit relationships if structured data
        if any(col in row.index for col in ['subject', 'predicate', 'object']):
            subject = self.resolve_entity_name(row.get('subject', ''))
            predicate = str(row.get('predicate', row.get('relation', 'RELATED_TO'))).strip().upper()
            obj = self.resolve_entity_name(row.get('object', ''))
            
            if subject and obj and predicate:
                relationships.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj,
                    'metadata': metadata.copy()
                })
                
        # Extract implicit relationships from property columns
        for subj_col in subject_columns:
            if subj_col in row.index and pd.notna(row[subj_col]):
                subject = self.resolve_entity_name(row[subj_col])
                if not subject:
                    continue
                    
                for obj_col in object_columns:
                    if obj_col in row.index and pd.notna(row[obj_col]):
                        obj = self.resolve_entity_name(row[obj_col])
                        if not obj or obj == subject:
                            continue
                            
                        # Determine predicate based on column names
                        predicate = self.infer_predicate(subj_col, obj_col, row)
                        
                        relationships.append({
                            'subject': subject,
                            'predicate': predicate,
                            'object': obj,
                            'metadata': metadata.copy()
                        })
                        
        # Extract property-value relationships
        property_patterns = {
            'conductivity': ['conductivity', 'ionic_conductivity', 'electrical_conductivity'],
            'capacity': ['capacity', 'specific_capacity', 'discharge_capacity', 'charge_capacity'],
            'voltage': ['voltage', 'potential', 'cell_voltage', 'electrode_potential'],
            'temperature': ['temperature', 'temp', 'operating_temperature'],
            'concentration': ['concentration', 'molarity', 'molality', 'conc'],
            'efficiency': ['efficiency', 'coulombic_efficiency', 'energy_efficiency'],
            'cycle_life': ['cycle_life', 'cycles', 'cycle_number', 'cycle_stability'],
            'energy_density': ['energy_density', 'specific_energy', 'energy'],
            'power_density': ['power_density', 'specific_power', 'power']
        }
        
        for prop_type, prop_columns in property_patterns.items():
            for prop_col in prop_columns:
                if prop_col in row.index and pd.notna(row[prop_col]):
                    # Find associated subject
                    subject = None
                    for subj_col in subject_columns:
                        if subj_col in row.index and pd.notna(row[subj_col]):
                            subject = self.resolve_entity_name(row[subj_col])
                            break
                            
                    if subject:
                        value = str(row[prop_col]).strip()
                        if value and value.lower() not in ['nan', 'na', 'null', '']:
                            relationships.append({
                                'subject': subject,
                                'predicate': f'HAS_{prop_type.upper()}',
                                'object': value,
                                'metadata': metadata.copy()
                            })
                            
        return relationships
        
    def infer_predicate(self, subj_col: str, obj_col: str, row: pd.Series) -> str:
        """Infer predicate based on column names and content."""
        # Predicate inference rules
        if 'property' in obj_col.lower():
            return 'HAS_PROPERTY'
        elif 'performance' in obj_col.lower():
            return 'HAS_PERFORMANCE'
        elif 'measurement' in obj_col.lower():
            return 'HAS_MEASUREMENT'
        elif 'value' in obj_col.lower():
            return 'HAS_VALUE'
        elif 'characteristic' in obj_col.lower():
            return 'HAS_CHARACTERISTIC'
        elif 'parameter' in obj_col.lower():
            return 'HAS_PARAMETER'
        elif 'component' in subj_col.lower() or 'component' in obj_col.lower():
            return 'CONTAINS_COMPONENT'
        elif 'material' in subj_col.lower() or 'material' in obj_col.lower():
            return 'COMPOSED_OF'
        elif 'electrolyte' in subj_col.lower() or 'electrolyte' in obj_col.lower():
            return 'USES_ELECTROLYTE'
        elif 'cathode' in subj_col.lower() or 'cathode' in obj_col.lower():
            return 'HAS_CATHODE'
        elif 'anode' in subj_col.lower() or 'anode' in obj_col.lower():
            return 'HAS_ANODE'
        else:
            return 'RELATED_TO'
            
    def process_chunk(self, chunk_data: Tuple[pd.DataFrame, int]) -> List[Dict]:
        """Process a chunk of data to extract relationships."""
        chunk, chunk_id = chunk_data
        relationships = []
        
        logger.info(f"🔄 Processing chunk {chunk_id} with {len(chunk)} rows")
        
        for idx, row in chunk.iterrows():
            try:
                row_relationships = self.extract_relationships_from_row(row)
                relationships.extend(row_relationships)
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
                
        logger.info(f"✅ Chunk {chunk_id} completed: {len(relationships)} relationships extracted")
        return relationships
        
    def extract_all_relationships(self, df: pd.DataFrame, n_workers: int = None) -> List[Dict]:
        """Extract all relationships using parallel processing."""
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 8)
            
        logger.info(f"🚀 Starting relationship extraction with {n_workers} workers")
        
        # Split dataframe into chunks
        chunk_size = max(100, len(df) // (n_workers * 4))
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]
        
        logger.info(f"📊 Split {len(df)} rows into {len(chunks)} chunks of ~{chunk_size} rows each")
        
        all_relationships = []
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk = {executor.submit(self.process_chunk, chunk): chunk 
                             for chunk in chunk_data}
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_relationships = future.result()
                    all_relationships.extend(chunk_relationships)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    
        logger.info(f"🎉 Extraction completed: {len(all_relationships)} total relationships")
        return all_relationships
        
    def create_knowledge_graph(self, relationships: List[Dict]) -> nx.MultiDiGraph:
        """Create NetworkX knowledge graph from relationships."""
        logger.info("🕸️ Creating knowledge graph...")
        
        G = nx.MultiDiGraph()
        
        for rel in relationships:
            subject = rel['subject']
            predicate = rel['predicate']
            obj = rel['object']
            metadata = rel['metadata']
            
            # Add nodes with attributes
            if not G.has_node(subject):
                G.add_node(subject, node_type='entity', entity_class=self.classify_entity(subject))
            if not G.has_node(obj):
                G.add_node(obj, node_type='entity', entity_class=self.classify_entity(obj))
                
            # Add edge with metadata
            G.add_edge(subject, obj, relation=predicate, **metadata)
            
        logger.info(f"✅ Knowledge graph created:")
        logger.info(f"   - Nodes: {G.number_of_nodes()}")
        logger.info(f"   - Edges: {G.number_of_edges()}")
        logger.info(f"   - Unique relations: {len(set(nx.get_edge_attributes(G, 'relation').values()))}")
        
        return G
        
    def classify_entity(self, entity: str) -> str:
        """Classify entity type based on content."""
        entity_lower = entity.lower()
        
        # Salt classifications
        if any(salt in entity_lower for salt in ['tfsi', 'pf6', 'bf4', 'clo4', 'fsi', 'otf']):
            return 'salt'
        elif any(cat in entity_lower for cat in ['li', 'na', 'k', 'mg', 'ca', 'al']):
            if any(indicator in entity_lower for indicator in ['lithium', 'sodium', 'potassium', 'magnesium']):
                return 'salt'
                
        # Battery component classifications
        if any(cathode in entity_lower for cathode in ['lifepo4', 'licoo2', 'ncm', 'nmc', 'nca', 'lmo']):
            return 'cathode'
        elif any(anode in entity_lower for anode in ['graphite', 'silicon', 'lithium metal', 'lto']):
            return 'anode'
        elif any(solvent in entity_lower for solvent in ['ec', 'pc', 'dmc', 'dec', 'emc', 'carbonate']):
            return 'solvent'
        elif any(additive in entity_lower for additive in ['vc', 'fec', 'additive']):
            return 'additive'
        elif any(sep in entity_lower for sep in ['separator', 'membrane', 'celgard']):
            return 'separator'
        elif any(binder in entity_lower for binder in ['pvdf', 'cmc', 'sbr', 'binder']):
            return 'binder'
            
        # Property classifications
        elif any(prop in entity_lower for prop in ['conductivity', 'capacity', 'voltage', 'temperature']):
            return 'property'
        elif any(unit in entity_lower for unit in ['s/cm', 'mah/g', 'v', '°c', 'molar']):
            return 'measurement'
        elif entity_lower.replace('.', '').replace('-', '').isdigit():
            return 'numerical_value'
            
        return 'general'
        
    def save_outputs(self, relationships: List[Dict], graph: nx.MultiDiGraph):
        """Save all outputs to files."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"💾 Saving outputs to {self.output_dir}/...")
        
        # Save relationships as CSV
        df_relationships = pd.DataFrame(relationships)
        relationships_file = f"{self.output_dir}/relationships_data_with_doi.csv"
        df_relationships.to_csv(relationships_file, index=False)
        logger.info(f"   ✅ Saved relationships: {relationships_file}")
        
        # Save relationships as JSON with metadata
        relationships_json_file = f"{self.output_dir}/relationships_data_with_metadata.json"
        with open(relationships_json_file, 'w') as f:
            json.dump(relationships, f, indent=2, default=str)
        logger.info(f"   ✅ Saved relationships with metadata: {relationships_json_file}")
        
        # Save graph statistics
        graph_stats = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'node_types': dict(Counter([graph.nodes[n].get('entity_class', 'unknown') 
                                      for n in graph.nodes()])),
            'relation_types': dict(Counter([d['relation'] for u, v, d in graph.edges(data=True)])),
            'most_connected_nodes': dict(sorted(dict(graph.degree()).items(), 
                                               key=lambda x: x[1], reverse=True)[:20]),
            'doi_coverage': len([r for r in relationships if r['metadata'].get('doi')]) / len(relationships)
        }
        
        stats_file = f"{self.output_dir}/graph_statistics_with_doi.json"
        with open(stats_file, 'w') as f:
            json.dump(graph_stats, f, indent=2, default=str)
        logger.info(f"   ✅ Saved graph statistics: {stats_file}")
        
        # Save entity resolution mappings
        mappings_file = f"{self.output_dir}/entity_resolution_mappings.json"
        with open(mappings_file, 'w') as f:
            json.dump(self.entity_resolution_cache, f, indent=2)
        logger.info(f"   ✅ Saved entity mappings: {mappings_file}")
        
        # Create summary report
        report_file = f"{self.output_dir}/extraction_report_with_doi.txt"
        with open(report_file, 'w') as f:
            f.write("🧪 ENHANCED SALT ONTOLOGY EXTRACTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 EXTRACTION STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total relationships extracted: {len(relationships):,}\n")
            f.write(f"Knowledge graph nodes: {graph.number_of_nodes():,}\n")
            f.write(f"Knowledge graph edges: {graph.number_of_edges():,}\n")
            f.write(f"DOI coverage: {graph_stats['doi_coverage']:.2%}\n\n")
            
            f.write("🏷️ ENTITY CLASSIFICATIONS\n")
            f.write("-" * 30 + "\n")
            for entity_type, count in graph_stats['node_types'].items():
                f.write(f"{entity_type}: {count:,}\n")
            f.write("\n")
            
            f.write("🔗 RELATION TYPES\n")
            f.write("-" * 20 + "\n")
            for relation, count in list(graph_stats['relation_types'].items())[:10]:
                f.write(f"{relation}: {count:,}\n")
            f.write("\n")
            
            f.write("⭐ MOST CONNECTED ENTITIES\n")
            f.write("-" * 30 + "\n")
            for entity, degree in list(graph_stats['most_connected_nodes'].items())[:10]:
                f.write(f"{entity}: {degree} connections\n")
                
        logger.info(f"   ✅ Saved extraction report: {report_file}")
        
    def run(self, n_workers: int = None):
        """Main execution method."""
        logger.info("🧪 STARTING ENHANCED SALT ONTOLOGY EXTRACTION WITH DOI")
        logger.info("=" * 70)
        
        # Load data
        logger.info(f"📂 Loading data from {self.input_file}...")
        try:
            # Handle both CSV and Excel files
            if self.input_file.endswith('.xlsx') or self.input_file.endswith('.xls'):
                df = pd.read_excel(self.input_file, engine='openpyxl')
                logger.info(f"✅ Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
            else:
                df = pd.read_csv(self.input_file)
                logger.info(f"✅ Loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
            
            logger.info(f"📋 Columns: {list(df.columns)}")
        except FileNotFoundError:
            logger.error(f"❌ File not found: {self.input_file}")
            logger.info("Available files in directory:")
            try:
                import glob
                dir_path = os.path.dirname(self.input_file) or '.'
                files = glob.glob(f"{dir_path}/*")
                for f in files[:10]:  # Show first 10 files
                    logger.info(f"   - {f}")
            except:
                pass
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
            
        # Extract relationships
        relationships = self.extract_all_relationships(df, n_workers)
        
        # Create knowledge graph
        graph = self.create_knowledge_graph(relationships)
        
        # Save outputs
        self.save_outputs(relationships, graph)
        
        logger.info("\n🎉 ENHANCED EXTRACTION COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Final results:")
        logger.info(f"   - Relationships: {len(relationships):,}")
        logger.info(f"   - Entities: {graph.number_of_nodes():,}")
        logger.info(f"   - Connections: {graph.number_of_edges():,}")
        logger.info(f"📁 All outputs saved to: {self.output_dir}/")


def main():
    """Main function."""
    # Configuration
    input_file = "battery_kg/t2br_data/raw_paragraphs_v2.xlsx"  # Updated to correct Excel file path
    output_dir = "salt_analysis_output"
    n_workers = multiprocessing.cpu_count()
    
    # Create and run extractor
    extractor = EnhancedSaltOntologyExtractor(input_file, output_dir)
    extractor.run(n_workers)


if __name__ == "__main__":
    main()
