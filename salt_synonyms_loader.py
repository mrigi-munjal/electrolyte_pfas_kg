"""
Helper functions to load and work with lithium salt synonyms
for use in Jaccard calculations and other analyses.
"""

import json
import pandas as pd
import re
from pathlib import Path

def load_salt_synonyms(file_path='output/lithium_salt_synonyms.json'):
    """
    Load salt synonyms from JSON file.
    
    Args:
        file_path: Path to the synonyms JSON file
        
    Returns:
        Dictionary with salt_id as keys and synonym data as values
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Synonyms file not found: {file_path}")
        print("Please run explore_lithium_salts.py first to generate the synonyms file.")
        return {}

def get_salt_synonyms(salt_id, file_path='output/lithium_salt_synonyms.json'):
    """
    Get all synonyms for a specific salt.
    
    Args:
        salt_id: Salt identifier (e.g., 'Salt_01')
        file_path: Path to the synonyms JSON file
        
    Returns:
        List of synonyms for the salt
    """
    synonyms_data = load_salt_synonyms(file_path)
    if salt_id in synonyms_data:
        return synonyms_data[salt_id]['all_synonyms']
    else:
        print(f"Salt {salt_id} not found in synonyms data")
        return []

def get_all_salt_synonyms(file_path='output/lithium_salt_synonyms.json'):
    """
    Get synonyms for all salts as a dictionary.
    
    Args:
        file_path: Path to the synonyms JSON file
        
    Returns:
        Dictionary with salt_id as keys and synonym lists as values
    """
    synonyms_data = load_salt_synonyms(file_path)
    return {salt_id: data['all_synonyms'] for salt_id, data in synonyms_data.items()}

def create_salt_regex_pattern(salt_id, file_path='output/lithium_salt_synonyms.json'):
    """
    Create a regex pattern for finding a salt in text.
    
    Args:
        salt_id: Salt identifier (e.g., 'Salt_01')
        file_path: Path to the synonyms JSON file
        
    Returns:
        Compiled regex pattern
    """
    synonyms = get_salt_synonyms(salt_id, file_path)
    if not synonyms:
        return None
    
    # Escape special regex characters and sort by length (longest first)
    escaped_synonyms = [re.escape(syn) for syn in synonyms]
    escaped_synonyms.sort(key=len, reverse=True)
    
    # Create case-insensitive pattern with word boundaries
    pattern = r'(?i)\b(' + '|'.join(escaped_synonyms) + r')\b'
    return re.compile(pattern)

def find_salt_in_text(text, salt_id, file_path='output/lithium_salt_synonyms.json'):
    """
    Find mentions of a salt in text.
    
    Args:
        text: Text to search in
        salt_id: Salt identifier (e.g., 'Salt_01')
        file_path: Path to the synonyms JSON file
        
    Returns:
        List of matches found
    """
    pattern = create_salt_regex_pattern(salt_id, file_path)
    if pattern is None:
        return []
    
    matches = pattern.findall(text)
    return matches

def calculate_salt_jaccard_similarity(salt_id1, salt_id2, chm_names, file_path='output/lithium_salt_synonyms.json'):
    """
    Calculate Jaccard similarity between two salts based on their synonyms found in CHM names.
    
    Args:
        salt_id1: First salt identifier
        salt_id2: Second salt identifier
        chm_names: List of CHM names to search in
        file_path: Path to the synonyms JSON file
        
    Returns:
        Jaccard similarity score (0-1)
    """
    synonyms1 = get_salt_synonyms(salt_id1, file_path)
    synonyms2 = get_salt_synonyms(salt_id2, file_path)
    
    if not synonyms1 or not synonyms2:
        return 0.0
    
    # Find which CHM names match each salt
    matches1 = set()
    matches2 = set()
    
    for chm_name in chm_names:
        chm_lower = chm_name.lower()
        
        # Check matches for salt 1
        for syn in synonyms1:
            if syn.lower() in chm_lower:
                matches1.add(chm_name)
                break
        
        # Check matches for salt 2
        for syn in synonyms2:
            if syn.lower() in chm_lower:
                matches2.add(chm_name)
                break
    
    # Calculate Jaccard similarity
    if len(matches1) == 0 and len(matches2) == 0:
        return 0.0
    
    intersection = len(matches1 & matches2)
    union = len(matches1 | matches2)
    
    return intersection / union if union > 0 else 0.0

def get_salt_formula(salt_id, file_path='output/lithium_salt_synonyms.json'):
    """
    Get the molecular formula for a salt.
    
    Args:
        salt_id: Salt identifier (e.g., 'Salt_01')
        file_path: Path to the synonyms JSON file
        
    Returns:
        Molecular formula string
    """
    synonyms_data = load_salt_synonyms(file_path)
    if salt_id in synonyms_data:
        return synonyms_data[salt_id]['formula']
    else:
        print(f"Salt {salt_id} not found in synonyms data")
        return None

def get_salt_smiles(salt_id, file_path='output/lithium_salt_synonyms.json'):
    """
    Get the SMILES string for a salt.
    
    Args:
        salt_id: Salt identifier (e.g., 'Salt_01')
        file_path: Path to the synonyms JSON file
        
    Returns:
        SMILES string
    """
    synonyms_data = load_salt_synonyms(file_path)
    if salt_id in synonyms_data:
        return synonyms_data[salt_id]['smiles']
    else:
        print(f"Salt {salt_id} not found in synonyms data")
        return None

def export_synonyms_for_jaccard(output_file='output/salts_for_jaccard.csv', file_path='output/lithium_salt_synonyms.json'):
    """
    Export synonyms in a format suitable for Jaccard calculation scripts.
    
    Args:
        output_file: Output CSV file path
        file_path: Path to the synonyms JSON file
    """
    synonyms_data = load_salt_synonyms(file_path)
    
    jaccard_data = []
    for salt_id, data in synonyms_data.items():
        # Create a pattern similar to what jaccard_calculation.py uses
        all_synonyms = data['all_synonyms']
        if all_synonyms:
            # Sort by length (longest first) for better matching
            all_synonyms.sort(key=len, reverse=True)
            pattern = '|'.join([re.escape(syn) for syn in all_synonyms])
            
            jaccard_data.append({
                'salt_id': salt_id,
                'formula': data['formula'],
                'smiles': data['smiles'],
                'num_synonyms': len(all_synonyms),
                'pattern': f"(?i){pattern}",
                'synonyms_list': '; '.join(all_synonyms[:10])  # First 10 for readability
            })
    
    df = pd.DataFrame(jaccard_data)
    df.to_csv(output_file, index=False)
    print(f"Jaccard-ready data exported to {output_file}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Load and display basic info
    synonyms = load_salt_synonyms()
    print(f"Loaded synonyms for {len(synonyms)} salts")
    
    # Example: Get synonyms for TFSI
    tfsi_synonyms = get_salt_synonyms('Salt_01')
    print(f"\nTFSI synonyms: {tfsi_synonyms[:5]}...")  # First 5
    
    # Export for Jaccard calculations
    export_synonyms_for_jaccard()
