#!/usr/bin/env python3
"""
Analyze LLM-as-Judge Evaluation Results by Dataset

This script analyzes the evaluation results from llm_triple_evaluation.py
and calculates per-dataset statistics (Battery KG and Salt KG separately).

Input: evaluation_combined_{timestamp}.csv from evaluation_output/
Output: Per-dataset statistics including PASS/FAIL rates, agreement, and Cohen's Kappa

Author: AI Assistant
Date: 2026-04-06
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ============================================================
# CONFIGURATION
# ============================================================

# Repo-relative so the script survives a move or a change of username.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "evaluation_output"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def interpret_kappa(kappa):
    """Interpret Cohen's Kappa value."""
    if np.isnan(kappa):
        return "Unable to compute"
    elif kappa < 0:
        return "Poor (less than chance agreement)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def find_latest_evaluation_file():
    """Find the most recent evaluation_combined CSV file."""
    files = list(OUTPUT_DIR.glob("evaluation_combined_*.csv"))
    if not files:
        print("ERROR: No evaluation_combined_*.csv files found in evaluation_output/")
        sys.exit(1)
    
    # Sort by modification time, get latest
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return latest


def calculate_dataset_stats(df: pd.DataFrame, dataset_name: str) -> dict:
    """Calculate statistics for a single dataset."""
    total = len(df)
    
    if total == 0:
        return None
    
    # Mistral stats
    mistral_pass = (df['mistral_correctness'] == 'PASS').sum()
    mistral_fail = (df['mistral_correctness'] == 'FAIL').sum()
    mistral_error = (df['mistral_correctness'] == 'ERROR').sum()
    
    # LLaMA stats
    llama_pass = (df['llama_correctness'] == 'PASS').sum()
    llama_fail = (df['llama_correctness'] == 'FAIL').sum()
    llama_error = (df['llama_correctness'] == 'ERROR').sum()
    
    # Agreement stats
    agree = (df['agreement'] == 'AGREE').sum()
    disagree = (df['agreement'] == 'DISAGREE').sum()
    
    # Cohen's Kappa (only for non-ERROR results)
    valid_df = df[(df['mistral_correctness'] != 'ERROR') & (df['llama_correctness'] != 'ERROR')]
    kappa = 0.0
    if len(valid_df) > 1:
        mistral_labels = [1 if c == 'PASS' else 0 for c in valid_df['mistral_correctness']]
        llama_labels = [1 if c == 'PASS' else 0 for c in valid_df['llama_correctness']]
        try:
            kappa = cohen_kappa_score(mistral_labels, llama_labels)
        except:
            kappa = 0.0
    
    return {
        'dataset': dataset_name,
        'total': total,
        'mistral': {
            'pass': mistral_pass,
            'pass_rate': mistral_pass / total * 100,
            'fail': mistral_fail,
            'fail_rate': mistral_fail / total * 100,
            'error': mistral_error,
            'error_rate': mistral_error / total * 100
        },
        'llama': {
            'pass': llama_pass,
            'pass_rate': llama_pass / total * 100,
            'fail': llama_fail,
            'fail_rate': llama_fail / total * 100,
            'error': llama_error,
            'error_rate': llama_error / total * 100
        },
        'agreement': {
            'agree': agree,
            'agree_rate': agree / total * 100,
            'disagree': disagree,
            'disagree_rate': disagree / total * 100
        },
        'kappa': kappa,
        'kappa_interpretation': interpret_kappa(kappa),
        'valid_comparisons': len(valid_df)
    }


def generate_report(all_stats: dict, combined_stats: dict, input_file: str, timestamp: str) -> str:
    """Generate a comprehensive report."""
    
    report = f"""
================================================================================
LLM-AS-JUDGE EVALUATION RESULTS - PER-DATASET ANALYSIS
Generated: {timestamp}
Input File: {input_file}
================================================================================

"""
    
    # Per-dataset sections
    for dataset_name, stats in all_stats.items():
        if stats is None:
            continue
            
        report += f"""
================================================================================
DATASET: {dataset_name.upper()}
================================================================================

Total Triples Evaluated: {stats['total']}
Valid Comparisons: {stats['valid_comparisons']}

MISTRAL-LARGE-3-675B
--------------------
PASS:  {stats['mistral']['pass']:>4} ({stats['mistral']['pass_rate']:.1f}%)
FAIL:  {stats['mistral']['fail']:>4} ({stats['mistral']['fail_rate']:.1f}%)
ERROR: {stats['mistral']['error']:>4} ({stats['mistral']['error_rate']:.1f}%)

LLAMA-3.1-405B
--------------
PASS:  {stats['llama']['pass']:>4} ({stats['llama']['pass_rate']:.1f}%)
FAIL:  {stats['llama']['fail']:>4} ({stats['llama']['fail_rate']:.1f}%)
ERROR: {stats['llama']['error']:>4} ({stats['llama']['error_rate']:.1f}%)

INTER-ANNOTATOR AGREEMENT
-------------------------
Agree:    {stats['agreement']['agree']:>4} ({stats['agreement']['agree_rate']:.1f}%)
Disagree: {stats['agreement']['disagree']:>4} ({stats['agreement']['disagree_rate']:.1f}%)

Cohen's Kappa: {stats['kappa']:.3f} ({stats['kappa_interpretation']})

"""

    # Combined summary
    report += f"""
================================================================================
OVERALL COMBINED RESULTS (ALL DATASETS)
================================================================================

Total Triples Evaluated: {combined_stats['total']}
Valid Comparisons: {combined_stats['valid_comparisons']}

MISTRAL-LARGE-3-675B
--------------------
PASS:  {combined_stats['mistral']['pass']:>4} ({combined_stats['mistral']['pass_rate']:.1f}%)
FAIL:  {combined_stats['mistral']['fail']:>4} ({combined_stats['mistral']['fail_rate']:.1f}%)
ERROR: {combined_stats['mistral']['error']:>4} ({combined_stats['mistral']['error_rate']:.1f}%)

LLAMA-3.1-405B
--------------
PASS:  {combined_stats['llama']['pass']:>4} ({combined_stats['llama']['pass_rate']:.1f}%)
FAIL:  {combined_stats['llama']['fail']:>4} ({combined_stats['llama']['fail_rate']:.1f}%)
ERROR: {combined_stats['llama']['error']:>4} ({combined_stats['llama']['error_rate']:.1f}%)

INTER-ANNOTATOR AGREEMENT
-------------------------
Agree:    {combined_stats['agreement']['agree']:>4} ({combined_stats['agreement']['agree_rate']:.1f}%)
Disagree: {combined_stats['agreement']['disagree']:>4} ({combined_stats['agreement']['disagree_rate']:.1f}%)

Cohen's Kappa: {combined_stats['kappa']:.3f} ({combined_stats['kappa_interpretation']})

================================================================================
COMPARISON SUMMARY TABLE
================================================================================

Dataset         | Mistral PASS | LLaMA PASS | Agreement | Cohen's Kappa
----------------|--------------|------------|-----------|---------------
"""
    
    for dataset_name, stats in all_stats.items():
        if stats is None:
            continue
        report += f"{dataset_name:<15} | {stats['mistral']['pass_rate']:>10.1f}% | {stats['llama']['pass_rate']:>8.1f}% | {stats['agreement']['agree_rate']:>7.1f}%  | {stats['kappa']:.3f}\n"
    
    report += f"{'OVERALL':<15} | {combined_stats['mistral']['pass_rate']:>10.1f}% | {combined_stats['llama']['pass_rate']:>8.1f}% | {combined_stats['agreement']['agree_rate']:>7.1f}%  | {combined_stats['kappa']:.3f}\n"
    
    report += """
================================================================================
Interpretation of Cohen's Kappa:
- < 0.00: Poor agreement (less than chance)
- 0.00 - 0.20: Slight agreement
- 0.21 - 0.40: Fair agreement
- 0.41 - 0.60: Moderate agreement
- 0.61 - 0.80: Substantial agreement
- 0.81 - 1.00: Almost perfect agreement
================================================================================
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze LLM evaluation results by dataset')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to evaluation_combined CSV file (default: latest)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: evaluation_output/per_dataset_analysis_TIMESTAMP.txt)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION RESULTS - PER-DATASET ANALYSIS")
    print("=" * 70)
    
    # Find input file
    if args.input:
        input_file = Path(args.input)
        if not input_file.exists():
            print(f"ERROR: Input file not found: {input_file}")
            sys.exit(1)
    else:
        input_file = find_latest_evaluation_file()
    
    print(f"\nInput file: {input_file}")
    
    # Load data
    print("Loading evaluation results...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} evaluation records")
    
    # Get unique datasets
    datasets = df['dataset'].unique()
    print(f"Datasets found: {list(datasets)}")
    
    # Calculate per-dataset stats
    all_stats = {}
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        stats = calculate_dataset_stats(dataset_df, dataset)
        all_stats[dataset] = stats
        print(f"\n{dataset}: {stats['total']} triples")
    
    # Calculate combined stats
    combined_stats = calculate_dataset_stats(df, 'combined')
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = generate_report(all_stats, combined_stats, str(input_file), timestamp)
    
    # Print report to console
    print(report)
    
    # Save report
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = OUTPUT_DIR / f"per_dataset_analysis_{timestamp}.txt"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    
    # Also save as JSON for programmatic access
    json_output = OUTPUT_DIR / f"per_dataset_analysis_{timestamp}.json"
    json_data = {
        'timestamp': timestamp,
        'input_file': str(input_file),
        'per_dataset': all_stats,
        'combined': combined_stats
    }
    with open(json_output, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"JSON data saved to: {json_output}")


if __name__ == "__main__":
    main()
