import pandas as pd
from pathlib import Path

def load_pubchem_synonyms(file_path='PubChem_filtered_compounds_mw_250_or_below.csv'):
    """
    Load PubChem synonyms from the filtered CSV file (MW ≤ 250).
    Returns a DataFrame, filtered to MW >= 50.
    """
    try:
        df = pd.read_csv(file_path)
        # Filter out anything with molecular weight less than 50
        if 'calculated_mw' in df.columns:
            df = df[df['calculated_mw'] >= 50].reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Synonyms file not found: {file_path}")
        print("Please generate the filtered PubChem file first.")
        return pd.DataFrame()

def get_pubchem_synonyms_by_mf(mf, file_path='PubChem_filtered_compounds_mw_250_or_below.csv'):
    """
    Get all synonyms for a specific formula (mf) from the filtered PubChem file.
    Returns a list of synonyms (including cmpdname and all cmpdsynonym entries).
    """
    df = load_pubchem_synonyms(file_path)
    row = df[df['mf'] == mf]
    if row.empty:
        print(f"Formula {mf} not found in PubChem data")
        return []
    synonyms = []
    cmpdsynonym = row.iloc[0]['cmpdsynonym']
    if pd.notna(cmpdsynonym):
        synonyms.extend([s.strip() for s in str(cmpdsynonym).split('|') if s.strip()])
    cmpdname = row.iloc[0]['cmpdname']
    if pd.notna(cmpdname):
        synonyms.extend([s.strip() for s in str(cmpdname).split('/') if s.strip()])
    # Add mf and cmpdname as synonyms
    synonyms.append(str(mf))
    if pd.notna(cmpdname):
        synonyms.append(str(cmpdname))
    return list(set(synonyms))

def get_all_pubchem_synonyms(file_path='PubChem_filtered_compounds_mw_250_or_below.csv'):
    """
    Get synonyms for all formulas as a dictionary: {mf: [synonym1, synonym2, ...]}
    """
    df = load_pubchem_synonyms(file_path)
    mf_to_synonyms = {}
    for idx, row in df.iterrows():
        mf = row['mf']
        synonyms = []
        if pd.notna(row['cmpdsynonym']):
            synonyms.extend([s.strip() for s in str(row['cmpdsynonym']).split('|') if s.strip()])
        if pd.notna(row['cmpdname']):
            synonyms.extend([s.strip() for s in str(row['cmpdname']).split('/') if s.strip()])
        # Add mf and cmpdname as synonyms
        synonyms.append(str(mf))
        if pd.notna(row['cmpdname']):
            synonyms.append(str(row['cmpdname']))
        mf_to_synonyms[mf] = list(set(synonyms))
    return mf_to_synonyms

def find_pubchem_synonym_in_text_simple(text, mf, file_path='PubChem_filtered_compounds_mw_250_or_below.csv'):
    """
    Find mentions of a formula's synonyms in text using direct string matching (case-insensitive).
    Returns a list of matched synonyms found in the text.
    """
    synonyms = get_pubchem_synonyms_by_mf(mf, file_path)
    text_lower = text.lower()
    matches = [syn for syn in synonyms if syn.lower() in text_lower]
    return matches

def get_pubchem_mw(mf, file_path='PubChem_filtered_compounds_mw_250_or_below.csv'):
    """
    Get the calculated molecular weight for a formula.
    """
    df = load_pubchem_synonyms(file_path)
    row = df[df['mf'] == mf]
    if row.empty:
        print(f"Formula {mf} not found in PubChem data")
        return None
    return row.iloc[0]['calculated_mw']

def export_pubchem_synonyms_for_jaccard(output_file='output/pubchem_salts_for_jaccard.csv', file_path='PubChem_filtered_compounds_mw_250_or_below.csv'):
    """
    Export synonyms in a format suitable for Jaccard calculation scripts.
    Ensures mf and all cmpdname (split by '/') are included as unique synonyms.
    Removes synonyms that look like 'CID <number>'.
    """
    import re
    df = load_pubchem_synonyms(file_path)
    jaccard_data = []
    for idx, row in df.iterrows():
        mf = row['mf']
        all_synonyms = []
        if pd.notna(row['cmpdsynonym']):
            all_synonyms.extend([s.strip() for s in str(row['cmpdsynonym']).split('|') if s.strip()])
        # Add all cmpdname parts split by '/'
        if pd.notna(row['cmpdname']):
            cmpdname_parts = [s.strip() for s in str(row['cmpdname']).split('/') if s.strip()]
            all_synonyms.extend(cmpdname_parts)
        # Add mf and full cmpdname as synonyms
        all_synonyms.append(str(mf))
        if pd.notna(row['cmpdname']):
            all_synonyms.append(str(row['cmpdname']))
        # Remove synonyms like 'CID <number>'
        all_synonyms = [syn for syn in set(all_synonyms) if not re.fullmatch(r'CID ?\d+', syn)]
        if all_synonyms:
            all_synonyms.sort(key=len, reverse=True)
            jaccard_data.append({
                'mf': mf,
                'cmpdname': row['cmpdname'],
                'num_synonyms': len(all_synonyms),
                'synonyms_list': '; '.join(all_synonyms[:10]),
                'calculated_mw': row['calculated_mw']
            })
    out_df = pd.DataFrame(jaccard_data)
    out_df.to_csv(output_file, index=False)
    print(f"Jaccard-ready data exported to {output_file}")
    return out_df

# Example usage
def _example_usage():
    df = load_pubchem_synonyms()
    print(f"Loaded {len(df)} unique formulas from PubChem (MW ≤ 250)")
    if len(df) == 0:
        return
    example_mf = df.iloc[0]['mf']
    example_synonyms = get_pubchem_synonyms_by_mf(example_mf)
    print(f"\nSynonyms for {example_mf}: {example_synonyms[:5]}...")
    example_mw = get_pubchem_mw(example_mf)
    print(f"Molecular weight for {example_mf}: {example_mw}")
    sample_text = "This electrolyte contains Lithium Carbonate and Li2CO3."
    matches = find_pubchem_synonym_in_text_simple(sample_text, example_mf)
    print(f"\nMatches in sample text: {matches}")
    export_pubchem_synonyms_for_jaccard()

if __name__ == "__main__":
    _example_usage()
