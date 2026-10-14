import pandas as pd
import re
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns

def count_chm_occurrences(chm_name, kg_df):
    """Count how many times a CHM appears as either Subject or Object in the knowledge graph"""
    subject_count = (kg_df['Subject'] == chm_name).sum()
    object_count = (kg_df['Object'] == chm_name).sum()
    total_count = subject_count + object_count
    return total_count

def create_match_entry(kg_name, matched_pattern, match_type, kg_df):
    """Create a match dictionary with occurrence count"""
    occurrence_count = count_chm_occurrences(kg_name, kg_df)
    return {
        'kg_name': kg_name,
        'matched_pattern': matched_pattern,
        'match_type': match_type,
        'occurrence_count': occurrence_count
    }

def is_unwanted_match(chm_name):
    """
    Check if a CHM name should be excluded as it represents ionic liquids, 
    other metal salts, or organic cations rather than pure lithium salts
    """
    chm_lower = chm_name.lower()
    
    # Patterns for ionic liquids and organic cations
    unwanted_patterns = [
        # Ionic liquid cations
        r'\b(c\d+mim|c\d+mpyr|c\d+mp|pyr\d+|py\d+|emim|bmim|hmim|omim|c2mim|c4mim|c6mim|c8mim)\b',
        r'\b(n\d+\d+\d+\d+|p\d+\d+\d+\d+|s\d+\d+\d+)\b',  # Quaternary ammonium codes
        r'\b(pip\d+|mor\d+|aze\d+|dmpi|dmpii)\b',  # Other organic cations
        r'\b(1-ethyl-3-methyl|1-butyl-3-methyl|1-hexyl-3-methyl|tetraethyl|tetrabutyl)\b',
        r'\b(imidazolium|pyridinium|pyrrolidinium|piperidinium|morpholinium|ammonium)\b',
        r'\b(phntf2|phenyl.*ntf2)\b',  # Phenyl-containing compounds
        
        # Other metal salts (not lithium)
        r'\b(mg\(|ca\(|na\(|k\(|rb\(|cs\(|ba\(|sr\()\b',  # Other metal cations
        r'\b(sodium|potassium|magnesium|calcium|barium|strontium|cesium|rubidium)\b',
        r'\b(na[^a-z]|mg[^a-z]|ca[^a-z]|ba[^a-z]|sr[^a-z]|cs[^a-z]|rb[^a-z])\b',
        
        # Organic/polymer compounds
        r'\b(poly|polymer|oligomer|copolymer)\b',
        r'\b(pvdf|ptfe|peo|peg|pmma|pvp|pan)\b',
        r'\b(peptide|protein|dna|rna)\b',
        r'\b(methyl.*ester|ethyl.*ester|propyl.*ester|butyl.*ester)\b',
        
        # Complex organic molecules that aren't simple salts
        r'\b(benzyl|phenyl|toluyl|xylyl|biphenyl|naphthyl)\b',
        r'\b(glucose|fructose|sucrose|cellulose)\b',
        r'\b(acetone|benzene|toluene|xylene)\b'
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, chm_lower):
            return True
    
    return False

def is_valid_lithium_salt_match(chm_name, target_anion):
    """
    Check if a CHM name represents a valid lithium salt match.
    For anion-based salts, only allow lithium salts or bare anions.
    """
    chm_lower = chm_name.lower()
    
    # If it's an unwanted match, reject it
    if is_unwanted_match(chm_name):
        return False
    
    # For anion-based searches, allow:
    # 1. Names that contain "lithium" or "li " or start with "li"
    # 2. Names that are just the anion (e.g., "TFSI", "PF6", etc.)
    # 3. Names in chemical notation like "LiTFSI", "Li-TFSI", etc.
    
    lithium_indicators = [
        r'\blithium\b',
        r'^li[^a-z]',  # Starts with Li followed by non-letter
        r'\bli[\s\-]',  # Li followed by space or dash
        r'li\d*\s*\w+',  # Li possibly with number then salt name
    ]
    
    # Check if it contains lithium indicators
    has_lithium = any(re.search(pattern, chm_lower) for pattern in lithium_indicators)
    
    # Check if it's a bare anion name (no metal cation mentioned)
    bare_anion_patterns = [
        r'^(bis\(|di|tri|tetra|penta|hexa)',  # Starts with common anion prefixes
        r'^(tfsi|fsi|pf6|bf4|clo4|no3|otf|tfa|bob|dfob|beti)$',  # Common anion abbreviations only
        r'(imide|borate|phosphate|sulfate|nitrate|acetate|triflate|perchlorate)$',  # Ends with anion names
        r'^(bis\(.*\)imide|.*sulfonate|.*phosphate|.*borate)$'  # Common anion structures
    ]
    
    is_bare_anion = any(re.search(pattern, chm_lower) for pattern in bare_anion_patterns)
    
    # Also exclude if it clearly mentions other metals even if it has the anion
    other_metals = [
        r'\b(sodium|potassium|magnesium|calcium|barium|strontium|cesium|rubidium)\b',
        r'\b(na|mg|ca|ba|sr|cs|rb|zn|cu|fe|al|ni|co|mn|cr)\s',
    ]
    
    has_other_metal = any(re.search(pattern, chm_lower) for pattern in other_metals)
    
    return (has_lithium or is_bare_anion) and not has_other_metal

def find_matches_for_salt(salt_names, chm_names, strict_abbreviations=None, generic_terms=None):
    """
    Find matches between salt names and CHM names using multiple matching strategies.
    
    Args:
        salt_names: List of names/abbreviations for the salt
        chm_names: List of CHM names from the knowledge graph
        strict_abbreviations: Set of abbreviations that require word boundary matching
        generic_terms: Set of generic terms to exclude from reverse partial matching
    
    Returns:
        List of match dictionaries
    """
    if strict_abbreviations is None:
        strict_abbreviations = {'tfsi', 'fsi', 'pf6', 'bob', 'dfob', 'otf', 'tfa'}
    
    if generic_terms is None:
        generic_terms = {
            'sulfonyl', 'ethane', 'imide', 'lithium', 'bis', 'acid', 'salt', 'ion', 'li', 'lin',
            'fluoride', 'oxide', 'hydroxide', 'chloride', 'bromide', 'iodide', 'sulfate', 'nitrate',
            'carbonate', 'acetate', 'phosphate', 'borate', 'metal', 'organic', 'inorganic',
            'methyl', 'ethyl', 'propyl', 'butyl', 'phenyl', 'benzyl', 'hexyl', 'octyl'
        }
    
    # Simple unwanted patterns - focus on the most obvious cases
    unwanted_substrings = [
        'mim]', 'pyr1', 'pyr2', 'emim', 'bmim', 'hmim', 'c2mim', 'c4mim', 'c6mim',
        'methylimidazolium', 'butylimidazolium', 'ethylimidazolium',
        'mg(', 'na(', 'ca(', 'zn(', 'cu(', 'agclo4', 'nh4',
        'phntf2', 'selectfluor', 'polymer', 'pvdf', 'ptfe'
    ]
    
    def is_valid_lithium_salt_match(chm_lower, target_anion):
        """Check if this is a valid lithium salt match - only lithium compounds allowed"""
        # Must contain lithium explicitly - no bare anions allowed
        has_lithium = ('lithium' in chm_lower or 
                      chm_lower.startswith('li ') or
                      chm_lower.startswith('li-') or
                      (chm_lower.startswith('li') and len(chm_lower) > 2 and not chm_lower[2].isalpha()))
        
        # Exclude problematic generic matches
        if chm_lower in ['lif', 'li-c', 'li-o', 'li', 'lithium', 'li4', 'dilithium', 'li-s', 'li-n', 'li-p']:  # These are too generic
            return False
            
        return has_lithium
    
    matches = []
    
    # Regex patterns for broader matching (similar to jaccard_calculation.py)
    tfsi_pattern = re.compile(r'trifluoromethylsulfonyl|tfsi|tf2n|ntf2|bistriflimide', re.IGNORECASE)
    fsi_pattern = re.compile(r'fluorosulfonyl.*imide|fsi|bfsi', re.IGNORECASE)
    beti_pattern = re.compile(r'pentafluoroethylsulfonyl|beti|pesi', re.IGNORECASE)
    pf6_pattern = re.compile(r'hexafluorophosphate|pf6(?!\w)|lipf6', re.IGNORECASE)
    bf4_pattern = re.compile(r'tetrafluoroborate|bf4(?!\w)|tfb', re.IGNORECASE)  # Added word boundary
    bob_pattern = re.compile(r'bis.*oxalato.*borate|bob|oxalatoborate', re.IGNORECASE)
    dfob_pattern = re.compile(r'difluoro.*oxalato.*borate|dfob|f2bob', re.IGNORECASE)
    perchlorate_pattern = re.compile(r'perchlorate|clo4|perc(?!hlo)', re.IGNORECASE)  # Exclude perchloroethylene
    triflate_pattern = re.compile(r'trifluoromethanesulfonate|triflate|otf(?!\w)|tf(?!si|aa|v|e|p|b)(?!\w)', re.IGNORECASE)  # More restrictive
    acetate_pattern = re.compile(r'trifluoroacetic|tfa(?!\w)|tfaa', re.IGNORECASE)  # Only TFA-related, not all acetates
    phosphate_pattern = re.compile(r'lithium.*metaphosphate|metaphosphate.*lithium|condensed.*phosphate.*lithium|lipO3|trioxophosphate', re.IGNORECASE)  # Metaphosphate pattern
    
    for chm_name in chm_names:
        chm_lower = chm_name.lower().strip()
        
        # Quick filter for obvious unwanted matches
        if any(unwanted in chm_lower for unwanted in unwanted_substrings):
            continue
        
        for possible_name in salt_names:
            name_lower = possible_name.lower().strip()
            
            # Special pattern-based matching for TFSI and FSI (like jaccard_calculation.py)
            if 'tfsi' in name_lower or 'trifluoromethylsulfonyl' in name_lower:
                if tfsi_pattern.search(chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, 'tfsi'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_tfsi'
                        })
                    continue
            
            if ('fsi' in name_lower or 'fluorosulfonyl' in name_lower) and 'tfsi' not in name_lower:
                if fsi_pattern.search(chm_lower) and 'tfsi' not in chm_lower:
                    if is_valid_lithium_salt_match(chm_lower, 'fsi'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_fsi'
                        })
                    continue
            
            if 'beti' in name_lower or 'pentafluoroethylsulfonyl' in name_lower:
                if beti_pattern.search(chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, 'beti'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_beti'
                        })
                    continue
            
            if 'pf6' in name_lower or 'hexafluorophosphate' in name_lower:
                if (pf6_pattern.search(chm_lower) and 
                    not any(exclude in chm_lower for exclude in ['pyridine', 'polymer', 'vdf', 'pvdf', 'peptide'])):
                    if is_valid_lithium_salt_match(chm_lower, 'pf6'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_pf6'
                        })
                    continue
            
            if 'bf4' in name_lower or 'tetrafluoroborate' in name_lower:
                if bf4_pattern.search(chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, 'bf4'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_bf4'
                        })
                    continue
            
            # BOB pattern - only for salts specifically looking for BOB, exclude DFOB
            if ('bob' in name_lower or 'oxalatoborate' in name_lower) and 'dfob' not in name_lower:
                if bob_pattern.search(chm_lower) and 'dfob' not in chm_lower and 'difluoro' not in chm_lower:
                    if is_valid_lithium_salt_match(chm_lower, 'bob'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_bob'
                        })
                    continue
            
            # DFOB pattern - only for salts specifically looking for DFOB
            if ('dfob' in name_lower or 'difluorooxalatoborate' in name_lower):
                if dfob_pattern.search(chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, 'dfob'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_dfob'
                        })
                    continue
                    if ('lithium' in chm_lower or 
                        chm_lower.startswith('li') or 
                        ' li ' in chm_lower or
                        chm_lower in ['dfob', 'f2bob']):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_dfob'
                        })
                    continue
            
            if 'perchlorate' in name_lower or 'clo4' in name_lower:
                if perchlorate_pattern.search(chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, 'clo4'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_perchlorate'
                        })
                    continue
            
            if ('triflate' in name_lower or 'otf' in name_lower or 'trifluoromethanesulfonate' in name_lower) and 'tfsi' not in name_lower:
                if (triflate_pattern.search(chm_lower) and 
                    not any(exclude in chm_lower for exclude in ['metformin', 'ptfe', 'etfe', 'selectfluor', 'trifluoroacetic'])):
                    if is_valid_lithium_salt_match(chm_lower, 'otf'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_triflate'
                        })
                    continue
            
            if ('trifluoroacetic' in name_lower or 'tfa' in name_lower) and 'tfsi' not in name_lower:
                if acetate_pattern.search(chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, 'tfa'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_acetate'
                        })
                    continue
            
            if ('metaphosphate' in name_lower or 'lipo3' in name_lower or 'condensed' in name_lower) and 'tfsi' not in name_lower:
                if (phosphate_pattern.search(chm_lower) and 
                    not any(exclude in chm_lower for exclude in ['glycerophosphate', 'nucleotide', 'dna', 'rna', 'adenosine', 'guanosine', 'cytidine', 'thymidine', 'dihydrogen', 'h2po4', 'monobasic'])):
                    if is_valid_lithium_salt_match(chm_lower, 'lipo3'):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'pattern_metaphosphate'
                        })
                    continue
            
            # Standard exact match
            if name_lower == chm_lower:
                if is_valid_lithium_salt_match(chm_lower, name_lower):
                    # Extra check for acetate - make sure it's not matching stearate wrongly
                    if 'acetate' in name_lower and 'stear' in chm_lower:
                        continue  # Skip this incorrect match
                    matches.append({
                        'kg_name': chm_name,
                        'matched_pattern': possible_name,
                        'match_type': 'exact'
                    })
            
            # Partial match (salt name contained in KG name)
            elif name_lower in strict_abbreviations and len(name_lower) <= 4:
                pattern = r'\b' + re.escape(name_lower) + r'\b'
                if re.search(pattern, chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, name_lower):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'partial_strict_abbrev'
                        })
            elif len(name_lower) > 3 and name_lower in chm_lower:
                if is_valid_lithium_salt_match(chm_lower, name_lower):
                    matches.append({
                        'kg_name': chm_name,
                        'matched_pattern': possible_name,
                        'match_type': 'partial'
                    })
            elif len(name_lower) <= 3 and name_lower not in strict_abbreviations:
                pattern = r'\b' + re.escape(name_lower) + r'\b'
                if re.search(pattern, chm_lower):
                    if is_valid_lithium_salt_match(chm_lower, name_lower):
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'partial_word_boundary'
                        })
            
            # Reverse partial match - with validation and stricter requirements
            elif (chm_lower in name_lower and 
                  len(chm_lower) > 3 and  # Increased minimum length
                  chm_lower not in generic_terms and
                  len(chm_lower) < len(name_lower) - 2):  # CHM name should be significantly shorter than search name
                
                if is_valid_lithium_salt_match(chm_lower, chm_lower):
                    if chm_lower in strict_abbreviations:
                        pattern = r'\b' + re.escape(chm_lower) + r'\b'
                        if re.search(pattern, name_lower):
                            matches.append({
                                'kg_name': chm_name,
                                'matched_pattern': possible_name,
                                'match_type': 'reverse_partial_strict_abbrev'
                            })
                    else:
                        matches.append({
                            'kg_name': chm_name,
                            'matched_pattern': possible_name,
                            'match_type': 'reverse_partial'
                        })
    
    # Remove duplicate matches
    unique_matches = []
    seen_kg_names = set()
    for match in matches:
        if match['kg_name'] not in seen_kg_names:
            unique_matches.append(match)
            seen_kg_names.add(match['kg_name'])
    
    return unique_matches

print("="*60)
print("LITHIUM SALT EXPLORATION AND KNOWLEDGE GRAPH MATCHING")
print("="*60)

# Load the lithium salt candidates
salt_candidates = pd.read_csv('20241217_saltCandidates.csv')
print(f"Loaded {len(salt_candidates)} lithium salt candidates")

# Load the knowledge graph database
print("Loading knowledge graph database...")
try:
    kg_df = pd.read_csv('SUBRELOBJ.csv')
    print(f"Loaded knowledge graph with {len(kg_df)} triples")
except FileNotFoundError:
    print("SUBRELOBJ.csv not found, trying alternative...")
    try:
        kg_df = pd.read_csv('cleaned_fluorine_CHM_triples_v3.csv')
        print(f"Loaded cleaned knowledge graph with {len(kg_df)} triples")
    except FileNotFoundError:
        print("ERROR: No knowledge graph database found!")
        exit(1)

# Filter for CHM-related relations only
rels_to_remove = ['CHM-CMT','CHM-SMT','CHM-SPL','CMT-CHM','SMT-CHM','SPL-CHM']
kg_filtered = kg_df[~kg_df['Rel'].isin(rels_to_remove)].reset_index(drop=True)
print(f"After filtering, {len(kg_filtered)} CHM-related triples remain")

# Extract all unique chemical names from the knowledge graph
print("\nExtracting chemical names from knowledge graph...")
chm_names = set()

# Get CHM names from subjects and objects
chm_subjects = kg_filtered[kg_filtered['Rel'].str.startswith('CHM-')]['Subject'].unique()
chm_objects = kg_filtered[kg_filtered['Rel'].str.endswith('-CHM')]['Object'].unique()
chm_names.update(chm_subjects)
chm_names.update(chm_objects)

print(f"Found {len(chm_names)} unique chemical names in knowledge graph")

# Function to parse SMILES and extract chemical information
def analyze_smiles(smiles_string):
    """Extract chemical information from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None
        
        info = {
            'molecular_formula': rdMolDescriptors.CalcMolFormula(mol),
            'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'contains_fluorine': 'F' in smiles_string,
            'contains_sulfur': 'S' in smiles_string,
            'contains_nitrogen': 'N' in smiles_string,
            'contains_phosphorus': 'P' in smiles_string,
            'contains_boron': 'B' in smiles_string,
            'is_lithium_salt': '[Li+]' in smiles_string
        }
        return info
    except:
        return None

# Analyze each salt candidate
print("\nAnalyzing SMILES structures...")
salt_analysis = []

for idx, row in salt_candidates.iterrows():
    smiles = row['SMILES']
    analysis = analyze_smiles(smiles)
    
    if analysis:
        analysis['smiles'] = smiles
        analysis['salt_id'] = f"Salt_{idx+1:02d}"
        salt_analysis.append(analysis)

salt_df = pd.DataFrame(salt_analysis)
print(f"Successfully analyzed {len(salt_df)} salt structures")

# Display basic statistics
print(f"\nSALT COMPOSITION ANALYSIS:")
print(f"All contain lithium: {salt_df['is_lithium_salt'].all()}")
print(f"Contain fluorine: {salt_df['contains_fluorine'].sum()}/{len(salt_df)} ({salt_df['contains_fluorine'].mean()*100:.1f}%)")
print(f"Contain sulfur: {salt_df['contains_sulfur'].sum()}/{len(salt_df)} ({salt_df['contains_sulfur'].mean()*100:.1f}%)")
print(f"Contain nitrogen: {salt_df['contains_nitrogen'].sum()}/{len(salt_df)} ({salt_df['contains_nitrogen'].mean()*100:.1f}%)")
print(f"Contain phosphorus: {salt_df['contains_phosphorus'].sum()}/{len(salt_df)} ({salt_df['contains_phosphorus'].mean()*100:.1f}%)")
print(f"Contain boron: {salt_df['contains_boron'].sum()}/{len(salt_df)} ({salt_df['contains_boron'].mean()*100:.1f}%)")

# Function to generate possible chemical names from SMILES
def generate_possible_names(smiles, salt_id):
    """Generate possible chemical names that might appear in the knowledge graph"""
    names = []
    
    # Salt 1: LiTFSI - Lithium bis(trifluoromethyl)imide
    if 'C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F' in smiles:
        names.extend(['LiTFSI', 'Li-TFSI', 'Lithium TFSI', 'Lithium bis(trifluoromethyl)imide',
                     'bis(trifluoromethyl)imide lithium', 'TFSI', 'LiN(SO2CF3)2',
                     'lithium bis(trifluoromethanesulfonyl)imide', 'bis(trifluoromethanesulfonyl)imide',
                     'trifluoromethylsulfonyl', 'bis(trifluoromethanesulfonyl)imide', 'Tf2N',
                     'Li-NTf2', 'NTf2', 'LiNTf2', 'lithium bistriflimide', 'bistriflimide lithium',
                     'lithium bis-triflimide', 'bis-triflimide', 'LiTFSA', 'TFSA', 'Li-TFSA'])
    
    # Salt 2: LiFSI - Lithium bis(fluorosulfonyl)imide
    if '[N-](S(=O)(=O)F)S(=O)(=O)F' in smiles:
        names.extend(['LiFSI', 'Li-FSI', 'Lithium FSI', 'Lithium bis(fluorosulfonyl)imide',
                     'bis(fluorosulfonyl)imide lithium', 'FSI', 'LiN(SO2F)2',
                     'lithium fluorosulfonimide', 'fluorosulfonimide lithium', 'Li-BFSI',
                     'BFSI', 'lithium bis(fluorosulfonyl)amide', 'bis(fluorosulfonyl)amide lithium'])
    
    # Salt 3: LiPF6 - Lithium hexafluorophosphate
    if 'F[P-](F)(F)(F)(F)F' in smiles:
        names.extend(['LiPF6', 'Li-PF6', 'Lithium hexafluorophosphate', 
                     'hexafluorophosphate lithium', 'PF6', 'lithium hexafluorophosphate(V)'])
    
    # Salt 4: LiBOB - Lithium bis(oxalato)borate
    if '[B-]12(OC(=O)C(=O)O1)OC(=O)C(=O)O2' in smiles:
        names.extend(['LiBOB', 'Li-BOB', 'Lithium BOB', 'Lithium bis(oxalato)borate',
                     'bis(oxalato)borate lithium', 'BOB', 'lithium oxalatoborate',
                     'lithium dioxalatoborate', 'dioxalatoborate lithium', 'LiB(C2O4)2',
                     'lithium bis(ethanedioato)borate', 'bis(ethanedioato)borate lithium'])
    
    # Salt 5: LiDFOB - Lithium difluoro(oxalato)borate
    if 'F[B-]1(OC(C(O1)=O)=O)F' in smiles:
        names.extend(['LiDFOB', 'Li-DFOB', 'Lithium DFOB', 'Lithium difluoro(oxalato)borate',
                     'difluoro(oxalato)borate lithium', 'DFOB', 'lithium difluorooxalatoborate',
                     'difluorooxalatoborate lithium', 'LiF2BOB', 'F2BOB', 'Li-F2BOB',
                     'lithium difluoro-BOB', 'difluoro-BOB lithium'])
    
    # Salt 6: LiNO3 - Lithium nitrate
    if '[N+](=O)([O-])[O-]' in smiles:
        names.extend(['LiNO3', 'Li-NO3', 'Lithium nitrate', 'nitrate lithium',
                     'lithium nitrate(V)', 'nitric acid lithium salt'])
    
    # Salt 7: LiBF4 - Lithium tetrafluoroborate
    if '[B-](F)(F)(F)F' in smiles and '[B-]12' not in smiles:
        names.extend(['LiBF4', 'Li-BF4', 'Lithium tetrafluoroborate',
                     'tetrafluoroborate lithium', 'BF4', 'lithium tetrafluoroborate(III)',
                     'lithium tetrafluoroborane', 'tetrafluoroborane lithium', 'LiTFB', 'TFB',
                     'lithium borofluoride', 'borofluoride lithium', 'fluoborate lithium'])
    
    # Salt 8: LiClO4 - Lithium perchlorate
    if '[O-]Cl(=O)(=O)=O' in smiles:
        names.extend(['LiClO4', 'Li-ClO4', 'Lithium perchlorate',
                     'perchlorate lithium', 'ClO4', 'LiPerc', 'Perc', 'Li-Perc',
                     'lithium tetrachlorate', 'perchloric acid lithium salt', 'PCLO4'])
    
    # Salt 9: LiOTf - Lithium trifluoromethanesulfonate
    if 'C(F)(F)(F)S(=O)(=O)[O-]' in smiles and '[N-]' not in smiles:
        names.extend(['LiOTf', 'Li-OTf', 'Lithium triflate', 'Lithium trifluoromethanesulfonate',
                     'trifluoromethanesulfonate lithium', 'OTf', 'triflate', 'lithium triflate', 'LiTf',
                     'LiCF3SO3', 'CF3SO3Li', 'Li-CF3SO3', 'lithium triflic acid', 'triflic acid lithium',
                     'LiTfO', 'TfO', 'TfOLi', 'lithium trifluoromethylsulfonate'])
    
    # Salt 10: LiBETI - Lithium bis(pentafluoroethylsulfonyl)imide
    if 'C(C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(C(F)(F)F)(F)F)(F)(F)F' in smiles:
        names.extend(['LiBETI', 'Li-BETI', 'BETI', 'Lithium BETI', 
                     'Lithium bis(pentafluoroethylsulfonyl)imide',
                     'bis(pentafluoroethylsulfonyl)imide lithium',
                     'pentafluoroethylsulfonyl imide lithium',
                     'C2F5SO2-N-SO2C2F5 lithium', 'Li-C2F5-sulfonylimide',
                     'lithium bis(perfluoroethylsulfonyl)imide', 'bis(perfluoroethylsulfonyl)imide lithium',
                     'LiN(SO2C2F5)2', 'N(SO2C2F5)2-', 'Li-PESI', 'PESI'])
    
    # Salt 11: Longer perfluoroalkyl sulfonylimide variant (NOT TFSI)
    if 'C(C(C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(F)(F)F)(F)F' in smiles:
        names.extend(['Li-C4F9-sulfonylimide', 'perfluorobutyl sulfonylimide lithium',
                     'lithium perfluorobutyl bis(sulfonyl)imide',
                     'LiPFBSI', 'PFBSI', 'lithium perfluorobutyl bistriflimide',
                     'perfluorobutyl bistriflimide lithium', 'Li-PFBSI'])
    
    # Salt 12: Cyano compound
    if 'C(#N)C1=C(N=C([N-]1)C(F)(F)F)C#N' in smiles:
        names.extend(['Li-TFCN', 'lithium trifluoromethyl cyanide', 'cyano trifluoromethyl lithium',
                     'dicyano trifluoromethyl imidazole lithium', 'LiTCM', 'TCM', 'Li-TCM',
                     'lithium tricyanomethanide', 'tricyanomethanide lithium', 'lithium dicyanoimidazole',
                     'dicyanoimidazole lithium', 'LiDCI', 'DCI', 'Li-DCI',
                     'lithium 4,5-dicyano-2-trifluoromethylimidazolate', 'dicyano trifluoromethyl imidazolate lithium',
                     'Li-DCTFMI', 'DCTFMI', 'lithium trifluoromethyl dicyanoimidazole',
                     'lithium cyano trifluoromethyl compound', 'LiCN-CF3', 'lithium dicyanoimidazolate',
                     'lithium nitrile trifluoromethyl', 'trifluoromethyl dicyano lithium'])
    
    # Salt 13: Lithium metaborate
    if 'B(=O)[O-]' in smiles:
        names.extend(['LiBO2', 'Li-BO2', 'Lithium metaborate', 'metaborate lithium',
                     'lithium borate', 'BO2'])
    
    # Salt 14: Lithium trifluoroacetate
    if 'C(=O)(C(F)(F)F)[O-]' in smiles:
        names.extend(['LiTFA', 'Li-TFA', 'Lithium trifluoroacetate', 'trifluoroacetate lithium',
                     'TFA', 'lithium trifluoroacetic acid', 'LiCF3COO', 'CF3COOLi', 'Li-CF3COO',
                     'trifluoroacetic acid lithium salt', 'lithium trifluoroethanoate'])
    
    # Salt 15: Lithium iodoacetate
    if 'C(C(=O)[O-])I' in smiles:
        names.extend(['LiIA', 'Li-IA', 'Lithium iodoacetate', 'iodoacetate lithium',
                     'lithium iodoacetic acid'])
    
    # Salt 16: Lithium carbonate
    if 'C(=O)([O-])[O-]' in smiles and smiles.count('[Li+]') == 2:
        names.extend(['Li2CO3', 'Li-CO3', 'Lithium carbonate', 'carbonate lithium',
                     'CO3', 'dilithium carbonate'])
    
    # Salt 17: Lithium acetate
    if 'CC(=O)[O-]' in smiles and '[Li+]' in smiles and smiles.count('[Li+]') == 1:
        names.extend(['LiAc', 'Li-Ac', 'Lithium acetate', 'acetate lithium',
                     'lithium acetic acid', 'CH3COOLi', 'LiOAc', 'OAc', 'Li-OAc',
                     'lithium ethanoate', 'ethanoate lithium', 'acetic acid lithium salt',
                     'lithium acetate anhydrous', 'anhydrous lithium acetate'])
    
    # Salt 18: Lithium sulfate
    if '[O-]S(=O)(=O)[O-]' in smiles and smiles.count('[Li+]') == 2:
        names.extend(['Li2SO4', 'Li-SO4', 'Lithium sulfate', 'sulfate lithium',
                     'SO4', 'dilithium sulfate'])
    
    # Salt 19: Complex organic lithium salt (large molecule)
    if 'C1=CC(=O)C2=C(C1=O)C(=C3C(=C2[O-])SC4=C(C5=C(C(=O)C=CC5=O)C(=C4S3)[O-])[O-])[O-]' in smiles:
        names.extend(['Li4-organic', 'lithium organic complex', 'quinone lithium salt',
                     'tetralithium organic salt'])
    
    # Salt 20: Lithium acetylacetonate
    if 'CC(=O)CC(=O)[O-]' in smiles:
        names.extend(['LiAcAc', 'Li-AcAc', 'Lithium acetylacetonate', 'acetylacetonate lithium',
                     'lithium 2,4-pentanedione', 'Li-acac', 'acac', 'Liacac',
                     'lithium pentane-2,4-dione', 'pentane-2,4-dione lithium', 'LiPD', 'PD'])
    
    # Salt 21: Lithium metaphosphate (condensed phosphate)
    if '[O-]P(=O)=O' in smiles and smiles.count('[Li+]') == 1:
        names.extend(['LiPO3', 'Li-PO3', 'Lithium metaphosphate', 
                     'metaphosphate lithium', 'lithium condensed phosphate',
                     'condensed phosphate lithium', 'lithium trioxophosphate',
                     'trioxophosphate lithium', 'lithium phosphate(V) oxide'])
    
    # Salt 22: Perfluorobutanesulfonate
    if 'C(C(C(F)(F)S(=O)(=O)[O-])(F)F)(C(F)(F)F)(F)F' in smiles:
        names.extend(['Li-PFBS', 'LiPFBS', 'Lithium perfluorobutanesulfonate',
                     'perfluorobutanesulfonate lithium', 'PFBS', 'Li-PFBS4', 'PFBS4',
                     'lithium perfluorobutane-1-sulfonate', 'perfluorobutane-1-sulfonate lithium',
                     'lithium nonafluorobutanesulfonate', 'nonafluorobutanesulfonate lithium',
                     'lithium perfluorobutylsulfonate', 'perfluorobutylsulfonate lithium',
                     'LiC4F9SO3', 'C4F9SO3Li', 'Li-C4F9SO3', 'lithium tetrafluorobutanesulfonate',
                     'lithium perfluoro-n-butanesulfonate', 'n-perfluorobutanesulfonate lithium',
                     'lithium heptafluorobutanesulfonic acid', 'heptafluorobutanesulfonic acid lithium'])
    
    # Salt 23: Lithium oxalate
    if 'C(=O)(C(=O)[O-])[O-]' in smiles and smiles.count('[Li+]') == 2:
        names.extend(['Li2C2O4', 'Li-C2O4', 'Lithium oxalate', 'oxalate lithium',
                     'dilithium oxalate', 'ethanedioate lithium'])
    
    # Salt 24: Complex phosphate ester
    if 'CC1=CC(=C(C(=C1)C)C(=O)P(=O)(C2=CC=CC=C2)[O-])C' in smiles:
        names.extend(['Li-phosphate-ester', 'lithium phosphate ester', 'organophosphate lithium',
                     'lithium methylphenyl phosphate'])
    
    # Salt 25: Lithium phosphate (tri)
    if '[O-]P(=O)([O-])[O-]' in smiles and smiles.count('[Li+]') == 3:
        names.extend(['Li3PO4', 'Li-PO4', 'Lithium phosphate', 'phosphate lithium',
                     'trilithium phosphate', 'PO4'])
    
    # Salt 26: Lithium dodecyl sulfate
    if 'CCCCCCCCCCCCOS(=O)(=O)[O-]' in smiles:
        names.extend(['LiDS', 'Li-DS', 'Lithium dodecyl sulfate', 'dodecyl sulfate lithium',
                     'lithium lauryl sulfate', 'LLS', 'Li-LLS', 'LiSDS', 'SDS', 'Li-SDS',
                     'lithium sodium dodecyl sulfate', 'lithium n-dodecyl sulfate', 'n-dodecyl sulfate lithium'])
    
    # Salt 27: Lithium difluorophosphate
    if '[O-]P(=O)(F)F' in smiles:
        names.extend(['LiDFP', 'Li-DFP', 'Lithium difluorophosphate', 'difluorophosphate lithium',
                     'DFP', 'lithium phosphorodifluoridate', 'LiPO2F2', 'PO2F2Li', 'Li-PO2F2',
                     'lithium difluorophosphoric acid', 'difluorophosphoric acid lithium',
                     'lithium phosphate difluoride', 'phosphate difluoride lithium',
                     'lithium difluorophosphorus oxide', 'difluorophosphorus oxide lithium',
                     'lithium fluorophosphate', 'fluorophosphate lithium', 'Li-FP', 'LiFP'])
    
    # Salt 28: Lithium stearate
    if 'CCCCCCCCCCCCCCCCCC(=O)[O-]' in smiles:
        names.extend(['LiSt', 'Li-St', 'Lithium stearate', 'stearate lithium',
                     'lithium octadecanoate', 'lithium stearic acid', 'octadecanoate lithium',
                     'stearic acid lithium salt', 'Li-C18', 'C18-Li', 'lithium C18',
                     'lithium octadecanoic acid', 'octadecanoic acid lithium'])
    
    # Salt 29: Lithium enolate
    if 'C/C(=C/C(=O)C)/[O-]' in smiles:
        names.extend(['Li-enolate', 'lithium enolate', 'lithium ketone enolate',
                     'lithium butenolate', 'lithium 3-buten-2-olate', '3-buten-2-olate lithium',
                     'lithium pentan-3-en-2-olate', 'pentan-3-en-2-olate lithium',
                     'lithium vinyl acetone enolate', 'vinyl acetone enolate lithium',
                     'lithium crotonyl enolate', 'crotonyl enolate lithium',
                     'Li-C5H7O2', 'C5H7O2Li', 'lithium methylbutenolate', 'methylbutenolate lithium'])
    
    # Salt 30: Lithium bis(trimethylsilyl)amide
    if 'C[Si](C)(C)[N-][Si](C)(C)C' in smiles:
        names.extend(['LiHMDS', 'Li-HMDS', 'Lithium bis(trimethylsilyl)amide',
                     'bis(trimethylsilyl)amide lithium', 'HMDS', 'lithium hexamethyldisilazide',
                     'LiN(SiMe3)2', 'N(SiMe3)2-', 'Li-N(SiMe3)2', 'LiNSiMe3',
                     'lithium bis(trimethylsilyl)amine', 'bis(trimethylsilyl)amine lithium',
                     'lithium BTSA', 'BTSA', 'Li-BTSA', 'LiTMSA', 'TMSA'])
    
    return list(set(names))  # Remove duplicates

# Generate possible names for each salt
print("\nGenerating possible chemical names for each salt...")
salt_name_mapping = {}

for idx, row in salt_df.iterrows():
    possible_names = generate_possible_names(row['smiles'], row['salt_id'])
    salt_name_mapping[row['salt_id']] = {
        'smiles': row['smiles'],
        'formula': row['molecular_formula'],
        'possible_names': possible_names
    }

# Search for salt occurrences in the knowledge graph
print("\nSearching for salt occurrences in knowledge graph...")
matches_found = {}

def count_chm_occurrences(chm_name, kg_df):
    """Count how many times a CHM appears as either Subject or Object in the knowledge graph"""
    subject_count = (kg_df['Subject'] == chm_name).sum()
    object_count = (kg_df['Object'] == chm_name).sum()
    total_count = subject_count + object_count
    return total_count

# Define generic terms to exclude from reverse_partial matching
generic_terms = {
    'lithium', 'fluoro', 'meth', 'methyl', 'sulfur', 'phosphor', 'phosphate', 
    'borate', 'nitrate', 'acetate', 'acetic', 'carbon', 'organic', 'acid',
    'tetra', 'bis', 'tri', 'di', 'mono', 'poly', 'iso', 'neo', 'sec', 'tert',
    'sulfonyl', 'ethane', 'butane', 'propane', 'methane', 'lin', 'lib', 'lio',
    'hex', 'oct', 'dec', 'pent', 'hept', 'non', 'cyan', 'azo', 'imide',
    'ester', 'ether', 'alkyl', 'aryl', 'phenyl', 'benzyl', 'toluyl', 'ethyl',
    'fluoride', 'phosphide', 'chloride', 'bromide', 'iodide', 'oxide', 'hydroxide',
    'butyl', 'propyl', 'hexyl', 'octyl', 'decyl', 'dodecyl', 'hydrogen'
}

# Define abbreviations that should NOT match as substrings of other abbreviations
strict_abbreviations = {
    'fsi', 'tfsi', 'beti', 'bob', 'bib', 'fap', 'fep', 'dca', 'tcm', 'scn',
    'pf6', 'bf4', 'otf', 'nf2', 'fta', 'tfa', 'msi', 'esi', 'psi', 'dsi',
    'litf'  # Added to prevent LiTf from matching LiTFSI and LiTFA
}

for salt_id, salt_info in salt_name_mapping.items():
    # Use the new enhanced matching function
    matches = find_matches_for_salt(
        salt_names=salt_info['possible_names'],
        chm_names=chm_names,
        strict_abbreviations=strict_abbreviations,
        generic_terms=generic_terms
    )
    
    # Add occurrence counts to matches
    for match in matches:
        match['occurrence_count'] = count_chm_occurrences(match['kg_name'], kg_filtered)
    
    matches_found[salt_id] = matches

# Display results
print(f"\n" + "="*60)
print("SALT MATCHING RESULTS")
print("="*60)

total_salts = len(salt_name_mapping)
salts_with_matches = sum(1 for matches in matches_found.values() if matches)

print(f"Total salts analyzed: {total_salts}")
print(f"Salts with matches found: {salts_with_matches}")
print(f"Match rate: {salts_with_matches/total_salts*100:.1f}%")

# Detailed results for each salt
for salt_id, matches in matches_found.items():
    salt_info = salt_name_mapping[salt_id]
    print(f"\n{salt_id}: {salt_info['formula']}")
    print(f"SMILES: {salt_info['smiles']}")
    
    if matches:
        print(f"Found {len(matches)} match(es) in knowledge graph:")
        for match in matches:  # Show ALL matches instead of just top 5
            print(f"  - '{match['kg_name']}' (matched via '{match['matched_pattern']}', type: {match['match_type']}) ({match['occurrence_count']} occurrences)")
    else:
        print("No matches found in knowledge graph")

# Count occurrences for matched salts
print(f"\n" + "="*60)
print("OCCURRENCE ANALYSIS FOR MATCHED SALTS")
print("="*60)

occurrence_results = []

for salt_id, matches in matches_found.items():
    if not matches:
        continue
    
    salt_info = salt_name_mapping[salt_id]
    
    for match in matches[:3]:  # Analyze top 3 matches per salt
        kg_name = match['kg_name']
        
        # Count occurrences in knowledge graph
        subject_count = (kg_filtered['Subject'] == kg_name).sum()
        object_count = (kg_filtered['Object'] == kg_name).sum()
        total_count = subject_count + object_count
        
        if total_count > 0:
            occurrence_results.append({
                'salt_id': salt_id,
                'formula': salt_info['formula'],
                'kg_name': kg_name,
                'matched_pattern': match['matched_pattern'],
                'match_type': match['match_type'],
                'subject_count': subject_count,
                'object_count': object_count,
                'total_count': total_count
            })

if occurrence_results:
    occurrence_df = pd.DataFrame(occurrence_results)
    occurrence_df = occurrence_df.sort_values('total_count', ascending=False)
    
    print(f"Found actual occurrences for {len(occurrence_df)} salt-name combinations:")
    print(f"\n{'Salt ID':<10} {'Formula':<15} {'KG Name':<30} {'Count':<8} {'Pattern'}")
    print("-" * 80)
    
    for _, row in occurrence_df.head(15).iterrows():  # Show top 15
        print(f"{row['salt_id']:<10} {row['formula']:<15} {row['kg_name']:<30} {row['total_count']:<8} {row['matched_pattern']}")
    
    # Save detailed results
    occurrence_df.to_csv('output/lithium_salt_kg_matches.csv', index=False)
    print(f"\nDetailed results saved to output/lithium_salt_kg_matches.csv")
    
    # Save synonyms for each salt for use in other analyses (e.g., Jaccard calculations)
    synonyms_data = []
    for salt_id, salt_info in salt_name_mapping.items():
        # Get all matched KG names for this salt
        matched_kg_names = [match['kg_name'] for match in matches_found[salt_id]]
        
        # Get all possible names we generated
        possible_names = salt_info['possible_names']
        
        # Combine both lists and remove duplicates
        all_synonyms = list(set(matched_kg_names + possible_names))
        
        synonyms_data.append({
            'salt_id': salt_id,
            'formula': salt_info['formula'],
            'smiles': salt_info['smiles'],
            'num_possible_names': len(possible_names),
            'num_matched_kg_names': len(matched_kg_names),
            'num_total_synonyms': len(all_synonyms),
            'possible_names': '|'.join(possible_names),
            'matched_kg_names': '|'.join(matched_kg_names),
            'all_synonyms': '|'.join(all_synonyms)
        })
    
    synonyms_df = pd.DataFrame(synonyms_data)
    synonyms_df.to_csv('output/lithium_salt_synonyms.csv', index=False)
    print(f"Salt synonyms saved to output/lithium_salt_synonyms.csv")
    
    # Also save as JSON for easier programmatic access
    synonyms_dict = {}
    for salt_id, salt_info in salt_name_mapping.items():
        matched_kg_names = [match['kg_name'] for match in matches_found[salt_id]]
        possible_names = salt_info['possible_names']
        all_synonyms = list(set(matched_kg_names + possible_names))
        
        synonyms_dict[salt_id] = {
            'formula': salt_info['formula'],
            'smiles': salt_info['smiles'],
            'possible_names': possible_names,
            'matched_kg_names': matched_kg_names,
            'all_synonyms': all_synonyms,
            'match_details': [
                {
                    'kg_name': match['kg_name'],
                    'matched_pattern': match['matched_pattern'],
                    'match_type': match['match_type']
                } for match in matches_found[salt_id]
            ]
        }
    
    with open('output/lithium_salt_synonyms.json', 'w') as f:
        json.dump(synonyms_dict, f, indent=2)
    print(f"Salt synonyms saved to output/lithium_salt_synonyms.json")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total occurrences found: {occurrence_df['total_count'].sum()}")
    print(f"Average occurrences per matched salt: {occurrence_df['total_count'].mean():.1f}")
    print(f"Most common salt: {occurrence_df.iloc[0]['kg_name']} ({occurrence_df.iloc[0]['total_count']} occurrences)")
    
else:
    print("No actual occurrences found in the knowledge graph for any salt candidates")

# Create visualization
if occurrence_results:
    plt.figure(figsize=(12, 8))
    
    # Top 10 salts by occurrence
    top_10 = occurrence_df.head(10)
    
    plt.subplot(2, 2, 1)
    plt.bar(range(len(top_10)), top_10['total_count'])
    plt.xlabel('Salt Rank')
    plt.ylabel('Total Occurrences')
    plt.title('Top 10 Salts by KG Occurrences')
    plt.xticks(range(len(top_10)), [f"{row['salt_id']}" for _, row in top_10.iterrows()], rotation=45)
    
    # Element composition of all salts
    plt.subplot(2, 2, 2)
    elements = ['contains_fluorine', 'contains_sulfur', 'contains_nitrogen', 'contains_phosphorus', 'contains_boron']
    element_names = ['Fluorine', 'Sulfur', 'Nitrogen', 'Phosphorus', 'Boron']
    counts = [salt_df[elem].sum() for elem in elements]
    
    plt.bar(element_names, counts)
    plt.xlabel('Element')
    plt.ylabel('Number of Salts')
    plt.title('Element Composition of Salt Candidates')
    plt.xticks(rotation=45)
    
    # Match type distribution
    plt.subplot(2, 2, 3)
    match_types = occurrence_df['match_type'].value_counts()
    plt.pie(match_types.values, labels=match_types.index, autopct='%1.1f%%')
    plt.title('Distribution of Match Types')
    
    # Molecular weight distribution
    plt.subplot(2, 2, 4)
    plt.hist(salt_df['molecular_weight'], bins=15, alpha=0.7)
    plt.xlabel('Molecular Weight')
    plt.ylabel('Number of Salts')
    plt.title('Molecular Weight Distribution')
    
    plt.tight_layout()
    plt.savefig('output/lithium_salt_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to output/lithium_salt_analysis.png")

# Generate regex patterns for future searches
print(f"\n" + "="*60)
print("REGEX PATTERNS FOR FUTURE SEARCHES")
print("="*60)

regex_patterns = {}
for salt_id, salt_info in salt_name_mapping.items():
    if salt_id in [match['salt_id'] for match in occurrence_results[:10]]:  # Top 10 only
        # Create regex pattern from successful matches
        successful_patterns = [match['matched_pattern'] for match in matches_found[salt_id]]
        if successful_patterns:
            # Escape special characters and combine
            escaped_patterns = [re.escape(pattern) for pattern in successful_patterns]
            regex_pattern = f"(?i)({'|'.join(escaped_patterns)})"
            regex_patterns[salt_id] = {
                'formula': salt_info['formula'],
                'pattern': regex_pattern,
                'matches_found': len(matches_found[salt_id])
            }

if regex_patterns:
    print("Suggested regex patterns for top salt candidates:")
    for salt_id, pattern_info in regex_patterns.items():
        print(f"\n{salt_id} ({pattern_info['formula']}):")
        print(f"  Pattern: r'{pattern_info['pattern']}'")
        print(f"  Found {pattern_info['matches_found']} potential matches")

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("This analysis provides a foundation for:")
print("1. Identifying which lithium salts are already in your knowledge graph")
print("2. Understanding the naming conventions used for these materials")
print("3. Creating targeted searches for specific salt types")
print("4. Developing more comprehensive salt-focused analyses")
