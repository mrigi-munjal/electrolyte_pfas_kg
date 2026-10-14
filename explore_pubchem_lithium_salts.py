import pandas as pd
import numpy as np
import os
import re

def calculate_molecular_weight_fast(formula):
    """
    Fast molecular weight calculation using atomic weights dictionary
    Much faster than RDKit for simple formulas
    """
    if pd.isna(formula) or formula == '':
        return None
    
    # Common atomic weights
    atomic_weights = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011, 
        'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
        'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
        'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
        'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
        'Ga': 69.723, 'Ge': 72.64, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
        'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96,
        'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411,
        'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.905, 'Xe': 131.293
    }
    
    try:
        # Parse formula using regex to find element and count pairs
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        total_weight = 0.0
        for element, count in matches:
            if element in atomic_weights:
                count = int(count) if count else 1
                total_weight += atomic_weights[element] * count
        
        return round(total_weight, 3) if total_weight > 0 else None
    except:
        return None

def process_pubchem_data():
    """
    Main function to process PubChem data and create the two required files:
    1. PubChem_filtered_compounds_mf_deduplicated.csv
    2. PubChem_filtered_compounds_mf_deduplicated_with_mw.csv
    """
    csv_file = "PubChem_compound_smiles_substructure.csv"
    
    if not os.path.exists(csv_file):
        print(f"ERROR: File '{csv_file}' not found in current directory")
        return None, None
    
    print("="*80)
    print("PROCESSING PUBCHEM DATA FOR LITHIUM SALTS")
    print("="*80)
    
    # Get file info
    file_size = os.path.getsize(csv_file)
    print(f"Processing file: {csv_file}")
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    try:
        # Read sample to understand structure
        df_sample = pd.read_csv(csv_file, nrows=5)
        print(f"\nColumns: {list(df_sample.columns)}")
        
        # Check required columns
        required_cols = ['cmpdname', 'cmpdsynonym', 'mf']
        available_cols = df_sample.columns.tolist()
        existing_cols = [col for col in required_cols if col in available_cols]
        
        if len(existing_cols) < 3:
            print(f"ERROR: Missing required columns. Found: {existing_cols}")
            return None, None
        
        print(f"Required columns found: {existing_cols}")
        
        # Process in chunks to handle large file
        print(f"\nProcessing file in chunks...")
        chunk_size = 50000
        all_data = []
        
        for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
            # Select required columns
            filtered_chunk = chunk[existing_cols].copy()
            all_data.append(filtered_chunk)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {(i + 1) * chunk_size:,} rows...")
        
        # Combine all chunks
        print("Combining all data...")
        df_all = pd.concat(all_data, ignore_index=True)
        total_rows = len(df_all)
        print(f"Total rows loaded: {total_rows:,}")
        
        # Remove complete duplicates
        print(f"\nRemoving complete duplicate rows...")
        df_all['combined_key'] = (
            df_all['cmpdname'].fillna('').astype(str) + '|' +
            df_all['cmpdsynonym'].fillna('').astype(str) + '|' +
            df_all['mf'].fillna('').astype(str)
        )
        
        before_dedup = len(df_all)
        df_dedup = df_all.drop_duplicates(subset=['combined_key'], keep='first').copy()
        df_dedup = df_dedup.drop('combined_key', axis=1)
        after_dedup = len(df_dedup)
        
        print(f"Rows before deduplication: {before_dedup:,}")
        print(f"Rows after deduplication: {after_dedup:,}")
        print(f"Duplicates removed: {before_dedup - after_dedup:,}")
        
        # Group by molecular formula and consolidate
        print(f"\nConsolidating by molecular formula...")
        mf_grouped = df_dedup.groupby('mf').agg({
            'cmpdname': lambda x: '/'.join(x.dropna().unique()),
            'cmpdsynonym': 'first'
        }).reset_index()
        
        mf_grouped = mf_grouped.sort_values('mf').reset_index(drop=True)
        
        print(f"Unique molecular formulas: {len(mf_grouped):,}")
        
        # Save first file (without molecular weights)
        file1 = "PubChem_filtered_compounds_mf_deduplicated.csv"
        print(f"\nSaving file 1: {file1}")
        mf_grouped.to_csv(file1, index=False)
        
        size1 = os.path.getsize(file1)
        print(f"File 1 size: {size1:,} bytes ({size1/(1024*1024):.2f} MB)")
        
        # Calculate molecular weights
        print(f"\nCalculating molecular weights...")
        molecular_weights = []
        
        for i, mf in enumerate(mf_grouped['mf']):
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i:,} formulas...")
            
            mw = calculate_molecular_weight_fast(mf)
            molecular_weights.append(mw)
        
        # Add molecular weight columns
        mf_grouped['calculated_mw'] = molecular_weights
        mf_grouped['mw_calculation_method'] = ['Fast' if mw is not None else 'Failed' for mw in molecular_weights]
        
        # Statistics
        valid_mws = [mw for mw in molecular_weights if mw is not None]
        failed_count = len(molecular_weights) - len(valid_mws)
        
        print(f"\nMolecular weight calculation results:")
        print(f"  Successful: {len(valid_mws):,} ({len(valid_mws)/len(molecular_weights)*100:.1f}%)")
        print(f"  Failed: {failed_count:,} ({failed_count/len(molecular_weights)*100:.1f}%)")
        
        if valid_mws:
            print(f"  MW range: {min(valid_mws):.3f} - {max(valid_mws):.3f} g/mol")
            print(f"  Average MW: {sum(valid_mws)/len(valid_mws):.3f} g/mol")
        
        # Save second file (with molecular weights)
        file2 = "PubChem_filtered_compounds_mf_deduplicated_with_mw.csv"
        print(f"\nSaving file 2: {file2}")
        mf_grouped.to_csv(file2, index=False)
        
        size2 = os.path.getsize(file2)
        print(f"File 2 size: {size2:,} bytes ({size2/(1024*1024):.2f} MB)")
        
        # Show examples
        print(f"\nExamples from final files:")
        for idx, row in mf_grouped.head(5).iterrows():
            mw_str = f"{row['calculated_mw']:.3f}" if row['calculated_mw'] is not None else "Failed"
            compound_name = row['cmpdname'][:50] + "..." if len(row['cmpdname']) > 50 else row['cmpdname']
            print(f"  {row['mf']}: {mw_str} g/mol - {compound_name}")
        
        print(f"\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"✓ File 1: {file1}")
        print(f"  - {len(mf_grouped):,} unique molecular formulas")
        print(f"  - Columns: mf, cmpdname, cmpdsynonym")
        print(f"✓ File 2: {file2}")
        print(f"  - {len(mf_grouped):,} unique molecular formulas")
        print(f"  - Columns: mf, cmpdname, cmpdsynonym, calculated_mw, mw_calculation_method")
        print(f"  - {len(valid_mws):,} successful MW calculations")
        
        return file1, file2
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def explore_pubchem_csv():
    """
    Main function to explore and process the PubChem CSV file
    """
    csv_file = "PubChem_compound_smiles_substructure.csv"
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
    
    # Get file size
    file_size = os.path.getsize(csv_file)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    try:
        # Read just the first few rows to understand structure
        print("\nReading first 5 rows to understand structure...")
        df_sample = pd.read_csv(csv_file, nrows=5)
        
        print(f"\nColumns found: {len(df_sample.columns)}")
        print("Column names:")
        for i, col in enumerate(df_sample.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\nFirst 5 rows:")
        print(df_sample.to_string())
        
        # Try to get total number of rows (this might take a moment)
        print("\nCounting total rows...")
        total_rows = sum(1 for line in open(csv_file)) - 1  # -1 for header
        print(f"Total rows: {total_rows:,}")
        
        # Check if the required columns exist
        required_cols = ['cmpdname', 'cmpdsynonym', 'mf']
        available_cols = df_sample.columns.tolist()
        
        print(f"\nRequired columns: {required_cols}")
        existing_required_cols = [col for col in required_cols if col in available_cols]
        
        if len(existing_required_cols) > 0:
            print(f"Creating filtered CSV with columns: {existing_required_cols}")
            
            # Read the full file in chunks and save only required columns
            filtered_file = "PubChem_filtered_compounds.csv"
            chunk_size = 10000
            
            print(f"Processing file in chunks of {chunk_size:,} rows...")
            
            first_chunk = True
            total_processed = 0
            
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                # Select only the required columns
                filtered_chunk = chunk[existing_required_cols]
                
                # Write to file (append mode after first chunk)
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                
                filtered_chunk.to_csv(filtered_file, mode=mode, header=header, index=False)
                
                total_processed += len(filtered_chunk)
                if total_processed % 50000 == 0:
                    print(f"  Processed {total_processed:,} rows...")
                
                first_chunk = False
            
            print(f"Created filtered CSV: {filtered_file}")
            print(f"Total rows processed: {total_processed:,}")
            
            # Analyze the filtered CSV and create deduplicated versions
            analyze_and_deduplicate_csv(filtered_file)
            
        else:
            print("ERROR: None of the required columns were found in the CSV file")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def analyze_and_deduplicate_csv(filtered_file):
    """
    Analyze the filtered CSV and create deduplicated versions
    """
    print(f"\n" + "="*80)
    print("ANALYZING FILTERED CSV")
    print("="*80)
    
    # Load and analyze molecular formulas
    df_filtered = pd.read_csv(filtered_file)
    
    print(f"Total entries: {len(df_filtered):,}")
    
    # Check for missing MF values
    mf_missing = df_filtered['mf'].isnull().sum()
    mf_present = len(df_filtered) - mf_missing
    
    print(f"Entries with MF: {mf_present:,}")
    print(f"Entries without MF: {mf_missing:,}")
    
    if mf_present > 0:
        # Remove complete row duplicates
        print(f"\n" + "="*60)
        print("REMOVING COMPLETE ROW DUPLICATES")
        print("="*60)
        
        total_rows_before = len(df_filtered)
        df_deduplicated = df_filtered.drop_duplicates().reset_index(drop=True)
        total_rows_after = len(df_deduplicated)
        duplicate_count = total_rows_before - total_rows_after
        
        print(f"Total rows before deduplication: {total_rows_before:,}")
        print(f"Complete duplicate rows removed: {duplicate_count:,}")
        print(f"Total rows after deduplication: {total_rows_after:,}")
        
        # Save deduplicated version
        deduplicated_file = "PubChem_filtered_compounds_deduplicated.csv"
        df_deduplicated.to_csv(deduplicated_file, index=False)
        print(f"Saved deduplicated file: {deduplicated_file}")
        
        # Create molecular formula deduplicated file
        print(f"\n" + "="*60)
        print("CREATING MOLECULAR FORMULA DEDUPLICATED FILE")
        print("="*60)
        
        print("Consolidating compounds by molecular formula...")
        print("- Combining compound names with '/' separator")
        print("- Keeping only first entry's synonyms for each molecular formula")
        
        # Group by molecular formula and consolidate
        mf_grouped = df_deduplicated.groupby('mf').agg({
            'cmpdname': lambda x: '/'.join(x.dropna().unique()),  # Combine names with /
            'cmpdsynonym': 'first'  # Keep first entry's synonyms
        }).reset_index()
        
        # Sort by molecular formula for consistency
        mf_grouped = mf_grouped.sort_values('mf').reset_index(drop=True)
        
        print(f"Consolidated entries: {len(mf_grouped):,}")
        
        # Save molecular formula deduplicated file
        mf_deduplicated_file = "PubChem_filtered_compounds_mf_deduplicated.csv"
        print(f"Saving molecular formula deduplicated file: {mf_deduplicated_file}")
        mf_grouped.to_csv(mf_deduplicated_file, index=False)
        
        # Get file size
        mf_deduplicated_size = os.path.getsize(mf_deduplicated_file)
        print(f"MF deduplicated file size: {mf_deduplicated_size:,} bytes ({mf_deduplicated_size/(1024*1024):.2f} MB)")
        
        # Show some examples
        print(f"\nExamples of consolidated entries:")
        for idx, row in mf_grouped.head(5).iterrows():
            print(f"  {row['mf']}: {row['cmpdname'][:80]}{'...' if len(row['cmpdname']) > 80 else ''}")
        
        print(f"\nFiles created:")
        print(f"1. Complete deduplication: {deduplicated_file}")
        print(f"2. Molecular formula deduplication: {mf_deduplicated_file}")
        
        return mf_deduplicated_file
    
    return None

def filter_by_molecular_weight():
    """
    Filter compounds with molecular weight <= 250 g/mol from the deduplicated file with MW
    """
    input_file = "PubChem_filtered_compounds_mf_deduplicated_with_mw.csv"
    output_file = "PubChem_filtered_compounds_mw_250_or_below.csv"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found.")
        print("Please run the main processing first to create the file with molecular weights.")
        return None
    
    print(f"\n" + "="*80)
    print("FILTERING BY MOLECULAR WEIGHT (≤ 250 g/mol)")
    print("="*80)
    
    try:
        # Load the file with molecular weights
        print(f"Loading file: {input_file}")
        df = pd.read_csv(input_file)
        total_entries = len(df)
        print(f"Total entries in input file: {total_entries:,}")
        
        # Check the columns
        print(f"Columns: {list(df.columns)}")
        
        # Filter by molecular weight
        print(f"\nFiltering compounds with molecular weight ≤ 250 g/mol...")
        
        # Remove rows where molecular weight calculation failed (NaN values)
        df_valid_mw = df[df['calculated_mw'].notna()].copy()
        valid_mw_count = len(df_valid_mw)
        failed_mw_count = total_entries - valid_mw_count
        
        print(f"Entries with valid molecular weight: {valid_mw_count:,}")
        print(f"Entries with failed MW calculation: {failed_mw_count:,}")
        
        # Apply the molecular weight filter
        df_filtered = df_valid_mw[df_valid_mw['calculated_mw'] <= 250.0].copy()
        filtered_count = len(df_filtered)
        
        print(f"\nEntries with MW ≤ 250 g/mol: {filtered_count:,}")
        print(f"Percentage of valid entries: {filtered_count/valid_mw_count*100:.1f}%")
        print(f"Percentage of total entries: {filtered_count/total_entries*100:.1f}%")
        
        if filtered_count > 0:
            # Sort by molecular weight for easier analysis
            df_filtered = df_filtered.sort_values('calculated_mw').reset_index(drop=True)
            
            # Save the filtered data
            print(f"\nSaving filtered data to: {output_file}")
            df_filtered.to_csv(output_file, index=False)
            
            # Get file size
            file_size = os.path.getsize(output_file)
            print(f"Output file size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
            
            # Show statistics
            mw_values = df_filtered['calculated_mw']
            print(f"\nMolecular weight statistics for filtered data:")
            print(f"  Minimum MW: {mw_values.min():.3f} g/mol")
            print(f"  Maximum MW: {mw_values.max():.3f} g/mol")
            print(f"  Average MW: {mw_values.mean():.3f} g/mol")
            print(f"  Median MW: {mw_values.median():.3f} g/mol")
            
            # Show some examples
            print(f"\nExamples of compounds with MW ≤ 250 g/mol:")
            print(f"{'MW (g/mol)':<12} {'Formula':<15} {'Compound Name'}")
            print("-" * 80)
            
            for idx, row in df_filtered.head(10).iterrows():
                compound_name = row['cmpdname'][:50] + "..." if len(row['cmpdname']) > 50 else row['cmpdname']
                print(f"{row['calculated_mw']:<12.3f} {row['mf']:<15} {compound_name}")
            
            if filtered_count > 10:
                print(f"... and {filtered_count - 10:,} more entries")
            
            # Show molecular weight distribution
            mw_ranges = [
                (0, 50, "Very light (0-50 g/mol)"),
                (50, 100, "Light (50-100 g/mol)"),
                (100, 150, "Medium-light (100-150 g/mol)"),
                (150, 200, "Medium (150-200 g/mol)"),
                (200, 250, "Medium-heavy (200-250 g/mol)")
            ]
            
            print(f"\nMolecular weight distribution:")
            for min_mw, max_mw, label in mw_ranges:
                count = len(df_filtered[(df_filtered['calculated_mw'] > min_mw) & 
                                       (df_filtered['calculated_mw'] <= max_mw)])
                percentage = count / filtered_count * 100 if filtered_count > 0 else 0
                print(f"  {label:<25}: {count:>6,} ({percentage:>5.1f}%)")
            
            print(f"\n✓ SUCCESS: Created filtered file with {filtered_count:,} compounds (MW ≤ 250 g/mol)")
            return output_file
            
        else:
            print("\nNo compounds found with molecular weight ≤ 250 g/mol")
            return None
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# Execute the main analysis
if __name__ == "__main__":
    print("PUBCHEM LITHIUM SALT DATA PROCESSING")
    print("="*60)
    
    # Process the PubChem data and create both required files
    file1, file2 = process_pubchem_data()
    
    if file1 and file2:
        print(f"\n✓ SUCCESS: Both files created successfully!")
        print(f"  - {file1}")
        print(f"  - {file2}")
        
        # Now filter by molecular weight
        file3 = filter_by_molecular_weight()
        
        if file3:
            print(f"\n✓ SUCCESS: All three files created successfully!")
            print(f"  1. {file1}")
            print(f"  2. {file2}")
            print(f"  3. {file3}")
        else:
            print(f"\n⚠ WARNING: MW filtering failed, but first two files were created successfully")
    else:
        print(f"\n✗ FAILED: Could not create output files")
