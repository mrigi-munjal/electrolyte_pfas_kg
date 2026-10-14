#!/usr/bin/env python3
"""
Add DOI and Paragraph Information to Extracted Battery Ontology Triples

This script takes the already extracted battery ontology relationships from 
ba    # Save as CSV
    if output_format in ['csv', 'both']:
        csv_filename = f"battery_ontology_triples_with_doi_clean_{timestamp}.csv"ry_ontology_extraction_results_v2.json and adds DOI and paragraph 
information to each triple by cross-referencing with the original Excel file.

NO API CALLS are made - this only processes existing data to enhance it with provenance.
"""

import os
import json
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_excel_data(file_path):
    """Load the original Excel data with DOI and paragraph information"""
    try:
        print(f"📊 Loading original Excel data from: {file_path}")
        df = pd.read_excel(file_path)
        print(f"✅ Loaded {len(df)} rows from Excel file")
        print(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

def load_extracted_results(file_path):
    """Load the already extracted battery ontology results"""
    try:
        print(f"📥 Loading extracted results from: {file_path}")
        with open(file_path, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded {len(results)} extraction results")
        return results
    except Exception as e:
        print(f"❌ Error loading extraction results: {e}")
        return None

def identify_text_column(df):
    """Identify the text column in the DataFrame (same logic as original script)"""
    # Check for 'paragraph' column first (as seen in the data structure)
    if 'paragraph' in df.columns:
        print(f"📝 Identified text column: 'paragraph'")
        return 'paragraph'
    
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Text columns
            # Check average length of non-null entries
            non_null_entries = df[col].dropna()
            if len(non_null_entries) > 0:
                avg_length = non_null_entries.astype(str).str.len().mean()
                text_columns.append((col, avg_length))
    
    # Sort by average length and pick the longest
    text_columns.sort(key=lambda x: x[1], reverse=True)
    
    if text_columns:
        text_column = text_columns[0][0]
        print(f"📝 Identified text column: '{text_column}' (avg length: {text_columns[0][1]:.0f} chars)")
        return text_column
    else:
        print("❌ No text columns found!")
        return None

def identify_doi_column(df):
    """Identify the DOI column in the DataFrame"""
    # Look for doc_id column first (as seen in the data structure)
    if 'doc_id' in df.columns:
        print(f"📄 Identified DOI column: 'doc_id'")
        return 'doc_id'
    
    # Common DOI column names
    doi_candidates = ['doi', 'DOI', 'Doi', 'doi_url', 'DOI_URL', 'document_doi', 'paper_doi']
    
    for col in df.columns:
        col_lower = col.lower()
        if 'doi' in col_lower:
            print(f"📄 Identified DOI column: '{col}'")
            return col
    
    # If no obvious DOI column, look for columns with DOI-like patterns
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10).astype(str)
            doi_like_count = sum(1 for val in sample_values if 'doi' in val.lower() or '10.' in val)
            if doi_like_count > 5:  # If more than half look like DOIs
                print(f"📄 Identified potential DOI column: '{col}' (based on content)")
                return col
    
    print("⚠️  No DOI column identified - will use 'Unknown' for DOI values")
    return None

def create_enhanced_triples(extracted_results, excel_df, text_column, doi_column):
    """Create enhanced triples with DOI and paragraph information"""
    print("🔄 Creating enhanced triples with DOI and paragraph information...")
    
    enhanced_triples = []
    successful_mappings = 0
    total_triples = 0
    
    print(f"📚 Excel DataFrame has {len(excel_df)} rows")
    print(f"🔍 Processing {len(extracted_results)} extraction results")
    
    # Process each extraction result
    for result_idx, result in enumerate(extracted_results):
        if result_idx % 5000 == 0:
            print(f"   Processing result {result_idx + 1:,}/{len(extracted_results):,}")
        
        # Get the abstract/paragraph text from this result
        abstract_text = result.get('abstract', '').strip()
        original_index = result.get('original_index', result_idx)
        
        # Get DOI for this paragraph using original_index
        doi_value = "Unknown"
        source_paragraph = abstract_text  # Default to the abstract from extraction
        
        if doi_column and original_index is not None:
            try:
                if original_index < len(excel_df):
                    excel_row = excel_df.iloc[original_index]
                    if pd.notna(excel_row[doi_column]):
                        doi_value = str(excel_row[doi_column]).strip()
                    # Also get the original paragraph from Excel for completeness
                    if pd.notna(excel_row[text_column]):
                        source_paragraph = str(excel_row[text_column]).strip()
                        successful_mappings += 1
            except Exception as e:
                print(f"   Warning: Could not map result {result_idx} to Excel row {original_index}: {e}")
        
        # Process the relationships from this result
        parsed_analysis = result.get('parsed_analysis')
        if parsed_analysis and parsed_analysis.get('relationships'):
            relationships = parsed_analysis['relationships']
            
            for rel in relationships:
                total_triples += 1
                
                # Create enhanced triple with DOI and paragraph
                enhanced_triple = {
                    'subject': rel.get('subject', ''),
                    'relation': rel.get('relation', ''),
                    'object': rel.get('object', ''),
                    'doi': doi_value,
                    'paragraph': source_paragraph,  # Use source paragraph from Excel
                    'original_result_index': result_idx,
                    'original_excel_index': original_index,
                    'extraction_timestamp': result.get('timestamp', ''),
                    'processing_time_seconds': result.get('processing_time_seconds', 0)
                }
                
                enhanced_triples.append(enhanced_triple)
    
    print(f"✅ Enhanced triples creation complete:")
    print(f"   Total triples: {total_triples:,}")
    print(f"   Triples with DOI: {successful_mappings:,} (from {successful_mappings} unique extraction results)")
    print(f"   Extraction results without DOI mapping: {len(extracted_results) - successful_mappings:,}")
    
    return enhanced_triples

def save_enhanced_results(enhanced_triples, output_format='both'):
    """Save enhanced triples in JSON and/or CSV format"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as JSON
    if output_format in ['json', 'both']:
        json_filename = f"battery_ontology_triples_with_doi_clean_{timestamp}.json"
        try:
            with open(json_filename, 'w') as f:
                json.dump(enhanced_triples, f, indent=2)
            print(f"💾 Saved {len(enhanced_triples):,} enhanced triples to: {json_filename}")
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
    
    # Save as CSV
    if output_format in ['csv', 'both']:
        csv_filename = f"battery_ontology_triples_with_doi_clean_{timestamp}.csv"
        try:
            df = pd.DataFrame(enhanced_triples)
            df.to_csv(csv_filename, index=False)
            print(f"💾 Saved {len(enhanced_triples):,} enhanced triples to: {csv_filename}")
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
    
    return json_filename if output_format in ['json', 'both'] else csv_filename

def create_summary_statistics(enhanced_triples):
    """Create summary statistics of the enhanced triples"""
    print("\n📊 ENHANCED TRIPLES SUMMARY STATISTICS")
    print("=" * 50)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(enhanced_triples)
    
    # Basic counts
    print(f"Total enhanced triples: {len(df):,}")
    print(f"Unique subjects: {df['subject'].nunique():,}")
    print(f"Unique relations: {df['relation'].nunique():,}")
    print(f"Unique objects: {df['object'].nunique():,}")
    print(f"Unique DOIs: {df['doi'].nunique():,}")
    
    # DOI distribution
    doi_counts = df['doi'].value_counts()
    known_dois = doi_counts[doi_counts.index != 'Unknown']
    print(f"\nDOI Information:")
    print(f"  Triples with known DOI: {known_dois.sum():,} ({known_dois.sum()/len(df)*100:.1f}%)")
    print(f"  Triples with unknown DOI: {doi_counts.get('Unknown', 0):,}")
    print(f"  Most prolific DOIs:")
    for doi, count in known_dois.head(5).items():
        print(f"    {doi}: {count} triples")
    
    # Relation type distribution
    print(f"\nTop 10 relation types:")
    for relation, count in df['relation'].value_counts().head(10).items():
        print(f"  {relation}: {count:,} triples")
    
    # Subject type analysis (try to infer types)
    print(f"\nTop 10 subjects:")
    for subject, count in df['subject'].value_counts().head(10).items():
        print(f"  {subject}: {count:,} triples")
    
    return df

def main():
    """Main function to add DOI information to extracted triples"""
    print("🔗 ADDING DOI INFORMATION TO EXTRACTED BATTERY ONTOLOGY TRIPLES")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define file paths
    excel_file_paths = [
        "battery_kg/t2br_data/raw_paragraphs_v2.xlsx",  # If running from main directory
        "t2br_data/raw_paragraphs_v2.xlsx"              # If running from battery_kg directory
    ]
    
    extraction_results_file = "battery_ontology_extraction_results_v2.json"
    
    # Load original Excel data
    excel_df = None
    for excel_file_path in excel_file_paths:
        if os.path.exists(excel_file_path):
            excel_df = load_excel_data(excel_file_path)
            break
    
    if excel_df is None:
        print("❌ Excel file not found in any of the expected locations:")
        for path in excel_file_paths:
            print(f"  - {path}")
        return
    
    # Load extracted results
    if not os.path.exists(extraction_results_file):
        print(f"❌ Extraction results file not found: {extraction_results_file}")
        print("   Please make sure you have run the battery ontology extraction first.")
        return
    
    extracted_results = load_extracted_results(extraction_results_file)
    if not extracted_results:
        return
    
    # Identify text and DOI columns
    text_column = identify_text_column(excel_df)
    if not text_column:
        return
    
    doi_column = identify_doi_column(excel_df)
    
    # Create enhanced triples
    enhanced_triples = create_enhanced_triples(extracted_results, excel_df, text_column, doi_column)
    
    if not enhanced_triples:
        print("❌ No enhanced triples created!")
        return
    
    # Save enhanced results
    output_file = save_enhanced_results(enhanced_triples, output_format='both')
    
    # Create summary statistics
    summary_df = create_summary_statistics(enhanced_triples)
    
    # Save summary report
    summary_filename = f"enhanced_triples_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(summary_filename, 'w') as f:
            f.write("ENHANCED BATTERY ONTOLOGY TRIPLES SUMMARY REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total enhanced triples: {len(enhanced_triples):,}\n")
            f.write(f"Unique DOIs: {summary_df['doi'].nunique():,}\n")
            f.write(f"Unique relations: {summary_df['relation'].nunique():,}\n")
            f.write(f"Source extraction file: {extraction_results_file}\n")
            f.write(f"Enhanced output file: {output_file}\n")
        print(f"📄 Summary report saved to: {summary_filename}")
    except Exception as e:
        print(f"⚠️  Could not save summary report: {e}")
    
    print(f"\n{'='*70}")
    print("✅ DOI ENHANCEMENT COMPLETE!")
    print("=" * 70)
    print(f"Enhanced {len(enhanced_triples):,} triples with DOI and paragraph information")
    print(f"Output files created:")
    print(f"  - JSON: battery_ontology_triples_with_doi_clean_*.json")
    print(f"  - CSV: battery_ontology_triples_with_doi_clean_*.csv")
    print(f"  - Summary: {summary_filename}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
