#!/usr/bin/env python3
"""
Create Evaluation Worksheet for Battery Ontology Triples

This script samples 300 random triples from the canonicalized battery ontology dataset
and creates an Excel worksheet for manual evaluation with an empty "correctness" column
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_canonicalized_data(file_path):
    """Load the canonicalized relationships data with DOI information"""
    try:
        print(f"📊 Loading canonicalized data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df):,} canonicalized relationships")
        
        # Display basic info
        print(f"   Columns: {list(df.columns)}")
        print(f"   Unique subjects: {df['subject'].nunique():,}")
        print(f"   Unique relations: {df['relation'].nunique():,}")
        print(f"   Unique objects: {df['object'].nunique():,}")
        
        # Check DOI coverage
        if 'doi' in df.columns:
            known_dois = len(df[df['doi'] != 'Unknown'])
            print(f"   Triples with known DOI: {known_dois:,} ({known_dois/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def sample_representative_triples(df, n_samples=300, random_seed=42):
    """Sample representative triples from the dataset"""
    print(f"\n🎲 Sampling {n_samples} representative triples...")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Strategy: Sample from different strata to ensure diversity
    
    # 1. Sample by relation type (ensure coverage of different relations)
    relation_counts = df['relation'].value_counts()
    print(f"   Found {len(relation_counts)} unique relation types")
    
    # Calculate samples per relation (proportional to frequency, with minimum 1)
    samples_per_relation = {}
    total_relations = len(relation_counts)
    min_samples_per_relation = max(1, n_samples // (total_relations * 2))  # At least some samples per relation
    
    remaining_samples = n_samples
    sampled_dfs = []
    
    # For top relations, sample proportionally
    top_relations = relation_counts.head(20)  # Focus on top 20 relations
    
    for relation, count in top_relations.items():
        # Proportional sampling with minimum
        proportion = count / len(df)
        relation_samples = max(min_samples_per_relation, int(n_samples * proportion))
        relation_samples = min(relation_samples, remaining_samples, count)  # Don't exceed available
        
        if relation_samples > 0 and remaining_samples > 0:
            relation_df = df[df['relation'] == relation].sample(n=relation_samples, random_state=random_seed)
            sampled_dfs.append(relation_df)
            remaining_samples -= relation_samples
            samples_per_relation[relation] = relation_samples
            
            print(f"   {relation}: {relation_samples} samples")
    
    # If we still have remaining samples, fill randomly from the rest
    if remaining_samples > 0:
        # Get relations not yet sampled or under-sampled
        sampled_indices = pd.concat(sampled_dfs).index if sampled_dfs else pd.Index([])
        remaining_df = df.drop(sampled_indices)
        
        if len(remaining_df) > 0:
            additional_samples = min(remaining_samples, len(remaining_df))
            additional_df = remaining_df.sample(n=additional_samples, random_state=random_seed)
            sampled_dfs.append(additional_df)
            print(f"   Additional random samples: {additional_samples}")
    
    # Combine all samples
    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the final sample
    final_sample = final_sample.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"✅ Successfully sampled {len(final_sample)} triples")
    
    # Show sampling statistics
    print(f"\n📈 Sampling statistics:")
    sample_relation_counts = final_sample['relation'].value_counts()
    for relation, count in sample_relation_counts.head(10).items():
        print(f"   {relation}: {count} samples")
    
    return final_sample

def create_evaluation_worksheet(sampled_df, output_path):
    """Create an Excel worksheet for manual evaluation"""
    print(f"\n📝 Creating evaluation worksheet...")
    
    # Prepare the evaluation DataFrame
    eval_df = sampled_df.copy()
    
    # Add evaluation columns
    eval_df['correctness'] = ''  # Empty column for 1/0 evaluation
    eval_df['comments'] = ''     # Empty column for optional comments
    eval_df['evaluator'] = ''    # Empty column for evaluator name
    eval_df['evaluation_date'] = ''  # Empty column for date
    
    # Add a row number for easy reference
    eval_df['row_id'] = range(1, len(eval_df) + 1)
    
    # Reorder columns for better evaluation workflow
    core_columns = ['row_id', 'subject', 'relation', 'object']
    evaluation_columns = ['correctness', 'comments', 'evaluator', 'evaluation_date']
    metadata_columns = [col for col in eval_df.columns 
                       if col not in core_columns + evaluation_columns]
    
    # Final column order: core, evaluation, then metadata
    final_columns = core_columns + evaluation_columns + metadata_columns
    eval_df = eval_df[final_columns]
    
    # Create Excel file with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write the main evaluation sheet
        eval_df.to_excel(writer, sheet_name='Triple_Evaluation', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Triple_Evaluation']
        
        # Format the worksheet
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        
        # Header formatting
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        
        # Apply header formatting
        for col_num in range(1, len(final_columns) + 1):
            cell = worksheet[f'{get_column_letter(col_num)}1']
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Highlight evaluation columns
        eval_fill = PatternFill(start_color='FFF2CC', end_color='FFF2CC', fill_type='solid')
        correctness_fill = PatternFill(start_color='E7F3FF', end_color='E7F3FF', fill_type='solid')
        
        # Find column indices for evaluation columns
        correctness_col = final_columns.index('correctness') + 1
        comments_col = final_columns.index('comments') + 1
        evaluator_col = final_columns.index('evaluator') + 1
        date_col = final_columns.index('evaluation_date') + 1
        
        # Format evaluation columns
        for row in range(2, len(eval_df) + 2):  # Start from row 2 (after header)
            # Correctness column - special highlighting
            worksheet[f'{get_column_letter(correctness_col)}{row}'].fill = correctness_fill
            
            # Other evaluation columns
            for col in [comments_col, evaluator_col, date_col]:
                worksheet[f'{get_column_letter(col)}{row}'].fill = eval_fill
        
        # Adjust column widths
        column_widths = {
            'row_id': 8,
            'subject': 25,
            'relation': 30,
            'object': 25,
            'correctness': 12,
            'comments': 40,
            'evaluator': 15,
            'evaluation_date': 15,
            'doi': 20,
            'paragraph': 50
        }
        
        for col_name, width in column_widths.items():
            if col_name in final_columns:
                col_idx = final_columns.index(col_name) + 1
                worksheet.column_dimensions[get_column_letter(col_idx)].width = width
        
        # Freeze panes (freeze first row and first 4 columns)
        worksheet.freeze_panes = 'E2'
        
        # Add instructions sheet
        instructions = [
            ["BATTERY ONTOLOGY TRIPLE EVALUATION INSTRUCTIONS", ""],
            ["", ""],
            ["Purpose:", "Evaluate the correctness of extracted battery ontology relationships"],
            ["", ""],
            ["Evaluation Criteria:", ""],
            ["1. Subject-Relation-Object Accuracy:", "Is the relationship factually correct?"],
            ["2. Semantic Consistency:", "Do the terms make scientific sense together?"],
            ["3. Extraction Quality:", "Was the relationship properly extracted from the source?"],
            ["", ""],
            ["Scoring:", ""],
            ["1 = Correct", "The triple accurately represents a valid battery relationship"],
            ["0 = Incorrect", "The triple contains errors, is nonsensical, or poorly extracted"],
            ["", ""],
            ["Required Fields:", ""],
            ["- correctness", "Enter 1 for correct, 0 for incorrect"],
            ["- evaluator", "Enter your name or initials"],
            ["- evaluation_date", "Enter the date of evaluation (YYYY-MM-DD)"],
            ["", ""],
            ["Optional Field:", ""],
            ["- comments", "Add any notes about the evaluation decision"],
            ["", ""],
            ["Tips:", ""],
            ["- Use the DOI to check the original paper if needed"],
            ["- Read the paragraph context to understand the extraction"],
            ["- Consider if the relationship makes scientific sense"],
            ["- Look for common extraction errors (wrong entity boundaries, etc.)"],
            ["", ""],
            ["Examples:", ""],
            ["CORRECT:", "LiPF6 | DISSOLVED IN | ethylene carbonate"],
            ["INCORRECT:", "LiPF6 | HAS PROPERTY | dissolved in (wrong relation type)"],
            ["INCORRECT:", "Li | STABLE IN WINDOW | battery (incomplete entity name)"],
        ]
        
        # Create instructions sheet
        inst_df = pd.DataFrame(instructions, columns=['Instruction', 'Description'])
        inst_df.to_excel(writer, sheet_name='Instructions', index=False)
        
        # Format instructions sheet
        inst_worksheet = writer.sheets['Instructions']
        
        # Make first row bold and larger
        inst_worksheet['A1'].font = Font(bold=True, size=14)
        inst_worksheet.merge_cells('A1:B1')
        
        # Adjust column widths for instructions
        inst_worksheet.column_dimensions['A'].width = 30
        inst_worksheet.column_dimensions['B'].width = 60
        
        # Create summary sheet
        summary_data = [
            ["EVALUATION SUMMARY", ""],
            ["", ""],
            ["Dataset Information:", ""],
            ["Total triples in dataset:", f"{len(sampled_df.index):,}"],
            ["Sample size:", f"{len(eval_df):,}"],
            ["Sampling method:", "Stratified by relation type + random"],
            ["Random seed:", "42"],
            ["", ""],
            ["Evaluation Progress:", ""],
            ["Total to evaluate:", f"{len(eval_df)}"],
            ["Completed:", "=COUNTA(Triple_Evaluation.E:E)-1"],
            ["Remaining:", f"=300-COUNTA(Triple_Evaluation.E:E)+1"],
            ["Progress %:", "=(COUNTA(Triple_Evaluation.E:E)-1)/300*100"],
            ["", ""],
            ["Results (automatically calculated):", ""],
            ["Correct (1s):", "=COUNTIF(Triple_Evaluation.E:E,1)"],
            ["Incorrect (0s):", "=COUNTIF(Triple_Evaluation.E:E,0)"],
            ["Accuracy:", "=COUNTIF(Triple_Evaluation.E:E,1)/(COUNTIF(Triple_Evaluation.E:E,1)+COUNTIF(Triple_Evaluation.E:E,0))"],
            ["", ""],
            ["File Information:", ""],
            ["Created:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Source file:", "relationships_data_canonical_with_doi_20250820_140958.csv"],
        ]
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        summary_worksheet = writer.sheets['Summary']
        summary_worksheet['A1'].font = Font(bold=True, size=14)
        summary_worksheet.merge_cells('A1:B1')
        summary_worksheet.column_dimensions['A'].width = 25
        summary_worksheet.column_dimensions['B'].width = 40
    
    print(f"✅ Created evaluation worksheet: {output_path}")
    return output_path

def main():
    """Main function to create the evaluation worksheet"""
    print("📋 CREATING BATTERY ONTOLOGY TRIPLE EVALUATION WORKSHEET")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # File paths
    input_file = 'battery_ontology_analysis/relationships_data_canonical_with_doi_20250820_140958.csv'
    output_file = f'battery_ontology_evaluation_worksheet_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    
    # Load the canonicalized data
    df = load_canonicalized_data(input_file)
    if df is None:
        return
    
    # Sample representative triples
    sampled_df = sample_representative_triples(df, n_samples=300)
    
    # Create evaluation worksheet
    worksheet_path = create_evaluation_worksheet(sampled_df, output_file)
    
    print(f"\n{'='*60}")
    print("✅ EVALUATION WORKSHEET CREATED!")
    print("=" * 60)
    print(f"Output file: {worksheet_path}")
    print()
    print("📋 Worksheet contains:")
    print("  - 300 randomly sampled triples")
    print("  - Empty 'correctness' column for 1/0 evaluation")
    print("  - Comments column for notes")
    print("  - Full provenance (DOI, paragraph)")
    print("  - Instructions sheet with evaluation guidelines")
    print("  - Summary sheet with automatic progress tracking")
    print()
    print("🔍 Evaluation process:")
    print("  1. Open the Excel file")
    print("  2. Read the Instructions sheet")
    print("  3. Evaluate each triple in the Triple_Evaluation sheet")
    print("  4. Enter 1 (correct) or 0 (incorrect) in the 'correctness' column")
    print("  5. Add your name and date")
    print("  6. Optionally add comments")
    print("  7. Check the Summary sheet for progress")
    print()
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
