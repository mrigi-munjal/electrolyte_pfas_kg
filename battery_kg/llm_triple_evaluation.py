#!/usr/bin/env python3
"""
LLM-as-Judge Triple Evaluation with Inter-Annotator Agreement
Uses Mistral-Large-3-675B and LLaMA-3.1-405B via RITS API
Evaluates triples from canonicalized Battery KG and Salt KG datasets
Calculates Cohen's Kappa for inter-annotator agreement
Now with parallel API calls for faster execution
"""

import os
import sys
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
NUM_TRIPLES_PER_DATASET = 300

# Number of parallel workers (from environment or default)
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 8))

# RITS API Configuration
RITS_API_KEY = os.environ.get("RITS_API_KEY")

# Model configurations for RITS
MODELS = {
    "mistral_large": {
        "name": "Mistral-Large-3-675B",
        "base_url": os.environ.get("RITS_BASE_URL", "").rstrip("/") + "/mistral-large-3-675b-2512-fp4/v1",
        "model_id": "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4"
    },
    "llama_405b": {
        "name": "LLaMA-3.1-405B",
        "base_url": os.environ.get("RITS_BASE_URL", "").rstrip("/") + "/llama-3-1-405b-instruct-fp8/v1",
        "model_id": "meta-llama/llama-3-1-405b-instruct-fp8"
    }
}

# Repo-relative so the paths survive a move or a change of username.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DATASETS = {
    "battery_kg": str(REPO_ROOT / "battery_ontology_analysis" / "relationships_data_canonical_with_doi_20250820_140958.csv"),
    "salt_kg": str(REPO_ROOT / "salt_analysis_output" / "canonical" / "salt_relationships_canonical_with_source.csv"),
}

# Output directory
OUTPUT_DIR = REPO_ROOT / "evaluation_output"

# Thread-safe result storage
results_lock = Lock()

# ============================================================
# EVALUATION PROMPT
# ============================================================

EVALUATION_PROMPT = """You are an expert judge evaluating the quality of knowledge graph triples extracted from scientific text about batteries and electrolytes.

Given:
- A source text (abstract/paragraph from a scientific paper)
- An extracted triple in the form (Subject, Relation, Object)

Evaluate the triple based on Correctness: 

Is the triple consistent with the source text? It does not need to be a verbatim extraction — paraphrasing, reasonable inference, and standard domain knowledge are acceptable. Mark FAIL only if the triple clearly contradicts or cannot be supported by the source text. 

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"correctness": "PASS", "reasoning": "Brief explanation"}}
or
{{"correctness": "FAIL", "reasoning": "Brief explanation"}}

SOURCE TEXT:
{source_text}

EXTRACTED TRIPLE:
Subject: {subject}
Relation: {relation}
Object: {object}

Your evaluation (JSON only):"""

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(dataset_name: str, file_path: str) -> pd.DataFrame:
    """Load a canonical dataset."""
    logger.info(f"Loading {dataset_name} from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded {len(df)} triples from {dataset_name}")
    return df

def extract_triples_from_df(df: pd.DataFrame, dataset_name: str) -> List[Dict]:
    """Extract triples from a dataframe with their source text."""
    triples = []
    
    # Determine column names based on dataset
    if 'subject' in df.columns:
        subj_col, rel_col, obj_col = 'subject', 'relation', 'object'
    elif 'Subject' in df.columns:
        subj_col, rel_col, obj_col = 'Subject', 'Relation', 'Object'
    else:
        logger.error(f"Could not find triple columns in {dataset_name}")
        return triples
    
    # Find source text column
    source_col = None
    for col in ['source_text', 'abstract', 'paragraph', 'text', 'Abstract']:
        if col in df.columns:
            source_col = col
            break
    
    for idx, row in df.iterrows():
        triple = {
            'subject': str(row[subj_col]),
            'relation': str(row[rel_col]),
            'object': str(row[obj_col]),
            'source_text': str(row[source_col]) if source_col else "Source text not available",
            'dataset': dataset_name,
            'original_index': idx
        }
        # Add DOI if available
        if 'doc_id' in df.columns:
            triple['doc_id'] = str(row['doc_id'])
        elif 'doi' in df.columns:
            triple['doc_id'] = str(row['doi'])
        
        triples.append(triple)
    
    logger.info(f"Extracted {len(triples)} triples from {dataset_name}")
    return triples

def sample_triples(triples: List[Dict], n: int, seed: int) -> List[Dict]:
    """Randomly sample n triples with a fixed seed for reproducibility."""
    random.seed(seed)
    sampled = random.sample(triples, min(n, len(triples)))
    logger.info(f"Sampled {len(sampled)} triples (seed={seed})")
    return sampled

def create_client(model_key: str) -> OpenAI:
    """Create an OpenAI client for a specific RITS model."""
    model_config = MODELS[model_key]
    return OpenAI(
        api_key=RITS_API_KEY,
        base_url=model_config["base_url"],
        default_headers={'RITS_API_KEY': RITS_API_KEY}
    )

def evaluate_single_triple(
    client: OpenAI,
    model_key: str,
    triple: Dict,
    triple_idx: int,
    total: int
) -> Dict:
    """Evaluate a single triple using the LLM."""
    model_config = MODELS[model_key]
    
    prompt = EVALUATION_PROMPT.format(
        source_text=triple['source_text'],  # Full source text (no truncation)
        subject=triple['subject'],
        relation=triple['relation'],
        object=triple['object']
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
                timeout=120  # 2 minute timeout
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Clean up response if needed
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                
                evaluation = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract correctness from text
                if 'PASS' in response_text.upper() and 'FAIL' not in response_text.upper():
                    correctness = 'PASS'
                elif 'FAIL' in response_text.upper():
                    correctness = 'FAIL'
                else:
                    correctness = 'ERROR'
                evaluation = {
                    'correctness': correctness,
                    'reasoning': response_text[:200]
                }
            
            return {
                **triple,
                'model': model_key,
                'model_name': model_config['name'],
                'correctness': evaluation.get('correctness', 'ERROR'),
                'reasoning': evaluation.get('reasoning', ''),
                'raw_response': response_text
            }
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed for triple {triple_idx}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error(f"Error evaluating triple {triple_idx}: {str(e)}")
                return {
                    **triple,
                    'model': model_key,
                    'model_name': model_config['name'],
                    'correctness': 'ERROR',
                    'reasoning': f"Error: {str(e)}",
                    'raw_response': ''
                }

def evaluate_triples_parallel(
    model_key: str,
    triples: List[Dict],
    num_workers: int = 8
) -> List[Dict]:
    """Evaluate triples in parallel using ThreadPoolExecutor."""
    model_name = MODELS[model_key]['name']
    logger.info(f"Starting parallel evaluation with {model_name} using {num_workers} workers")
    
    results = [None] * len(triples)
    completed = [0]  # Use list for mutable counter in closure
    
    print(f"\n{'='*60}")
    print(f"Evaluating with: {model_name} ({num_workers} parallel workers)")
    print(f"{'='*60}")
    
    def evaluate_with_index(args):
        idx, triple = args
        client = create_client(model_key)  # Create client per thread for thread safety
        result = evaluate_single_triple(client, model_key, triple, idx, len(triples))
        return idx, result
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(evaluate_with_index, (idx, triple)): idx 
            for idx, triple in enumerate(triples)
        }
        
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
                
                with results_lock:
                    completed[0] += 1
                    current = completed[0]
                
                # Print progress
                correctness = result.get('correctness', 'ERROR')
                subj = result['subject'][:30]
                rel = result['relation'][:20]
                print(f"  [{model_key}] {current}/{len(triples)}: ({subj}..., {rel}...) -> {correctness}")
                
                # Save intermediate results every 50 triples
                if current % 50 == 0:
                    with results_lock:
                        save_intermediate_results(model_key, [r for r in results if r is not None])
                        
            except Exception as e:
                logger.error(f"Future failed: {e}")
    
    # Filter out None results
    results = [r for r in results if r is not None]
    logger.info(f"Completed {len(results)} evaluations with {model_name}")
    
    return results

def save_intermediate_results(model_key: str, results: List[Dict]):
    """Save intermediate results to a file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"intermediate_{model_key}_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"[{model_key}] Saved intermediate results ({len(results)} evaluations)")

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

def calculate_cohens_kappa(results1: List[Dict], results2: List[Dict]) -> Dict:
    """Calculate Cohen's Kappa for inter-annotator agreement."""
    # Match results by triple
    matched = []
    for r1 in results1:
        for r2 in results2:
            if (r1['subject'] == r2['subject'] and 
                r1['relation'] == r2['relation'] and 
                r1['object'] == r2['object']):
                matched.append((r1, r2))
                break
    
    if not matched:
        return {
            'kappa_correctness': 0.0,
            'agreement_rate': 0.0,
            'n_matched': 0
        }
    
    # Extract labels
    correctness1 = [1 if r[0]['correctness'] == 'PASS' else 0 for r in matched if r[0]['correctness'] != 'ERROR' and r[1]['correctness'] != 'ERROR']
    correctness2 = [1 if r[1]['correctness'] == 'PASS' else 0 for r in matched if r[0]['correctness'] != 'ERROR' and r[1]['correctness'] != 'ERROR']
    
    # Calculate kappa
    try:
        kappa_correctness = cohen_kappa_score(correctness1, correctness2) if len(correctness1) > 1 else 0.0
    except:
        kappa_correctness = 0.0
    
    # Calculate agreement rate
    agreement = sum(1 for c1, c2 in zip(correctness1, correctness2) if c1 == c2)
    agreement_rate = agreement / len(correctness1) if correctness1 else 0.0
    
    return {
        'kappa_correctness': kappa_correctness,
        'agreement_rate': agreement_rate,
        'n_matched': len(matched),
        'n_valid': len(correctness1)
    }

def save_results(
    results_mistral: List[Dict],
    results_llama: List[Dict],
    agreement: Dict,
    timestamp: str
):
    """Save all results to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save individual model results
    with open(OUTPUT_DIR / f"evaluation_mistral_large_{timestamp}.json", 'w') as f:
        json.dump(results_mistral, f, indent=2, default=str)
    
    with open(OUTPUT_DIR / f"evaluation_llama_405b_{timestamp}.json", 'w') as f:
        json.dump(results_llama, f, indent=2, default=str)
    
    # Save inter-annotator agreement
    with open(OUTPUT_DIR / f"inter_annotator_agreement_{timestamp}.json", 'w') as f:
        json.dump(agreement, f, indent=2)
    
    # Create combined CSV
    combined = []
    for r1 in results_mistral:
        row = {
            'subject': r1['subject'],
            'relation': r1['relation'],
            'object': r1['object'],
            'dataset': r1['dataset'],
            'source_text': r1.get('source_text', ''),  # Full source text (no truncation)
            'mistral_correctness': r1['correctness'],
        }
        # Find matching llama result
        for r2 in results_llama:
            if (r1['subject'] == r2['subject'] and 
                r1['relation'] == r2['relation'] and 
                r1['object'] == r2['object']):
                row['llama_correctness'] = r2['correctness']
                row['agreement'] = 'AGREE' if r1['correctness'] == r2['correctness'] else 'DISAGREE'
                break
        combined.append(row)
    
    df = pd.DataFrame(combined)
    df.to_csv(OUTPUT_DIR / f"evaluation_combined_{timestamp}.csv", index=False)
    
    # Generate report
    report = generate_report(results_mistral, results_llama, agreement, timestamp)
    with open(OUTPUT_DIR / f"evaluation_report_{timestamp}.txt", 'w') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print("Results saved to:")
    print(f"  evaluation_mistral_large_{timestamp}.json")
    print(f"  evaluation_llama_405b_{timestamp}.json")
    print(f"  evaluation_combined_{timestamp}.csv")
    print(f"  inter_annotator_agreement_{timestamp}.json")
    print(f"  evaluation_report_{timestamp}.txt")
    print(f"{'='*60}")

def generate_report(
    results_mistral: List[Dict],
    results_llama: List[Dict],
    agreement: Dict,
    timestamp: str
) -> str:
    """Generate a human-readable report."""
    
    def calc_stats(results):
        total = len(results)
        pass_count = sum(1 for r in results if r['correctness'] == 'PASS')
        fail_count = sum(1 for r in results if r['correctness'] == 'FAIL')
        error_count = sum(1 for r in results if r['correctness'] == 'ERROR')
        return {
            'total': total,
            'pass': pass_count,
            'fail': fail_count,
            'error': error_count,
            'pass_rate': pass_count / total if total > 0 else 0
        }
    
    mistral_stats = calc_stats(results_mistral)
    llama_stats = calc_stats(results_llama)
    
    report = f"""
================================================================================
LLM-AS-JUDGE TRIPLE EVALUATION REPORT
Generated: {timestamp}
================================================================================

CONFIGURATION
-------------
Random Seed: {RANDOM_SEED}
Triples per Dataset: {NUM_TRIPLES_PER_DATASET}
Total Triples Evaluated: {len(results_mistral)}
Parallel Workers: {NUM_WORKERS}

MODELS USED
-----------
1. Mistral-Large-3-675B (mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4)
2. LLaMA-3.1-405B (meta-llama/llama-3-1-405b-instruct-fp8)

================================================================================
RESULTS SUMMARY
================================================================================

MISTRAL-LARGE-3-675B
--------------------
Total Evaluated: {mistral_stats['total']}
PASS: {mistral_stats['pass']} ({mistral_stats['pass_rate']*100:.1f}%)
FAIL: {mistral_stats['fail']} ({mistral_stats['fail']/mistral_stats['total']*100 if mistral_stats['total'] > 0 else 0:.1f}%)
ERROR: {mistral_stats['error']} ({mistral_stats['error']/mistral_stats['total']*100 if mistral_stats['total'] > 0 else 0:.1f}%)

LLAMA-3.1-405B
--------------
Total Evaluated: {llama_stats['total']}
PASS: {llama_stats['pass']} ({llama_stats['pass_rate']*100:.1f}%)
FAIL: {llama_stats['fail']} ({llama_stats['fail']/llama_stats['total']*100 if llama_stats['total'] > 0 else 0:.1f}%)
ERROR: {llama_stats['error']} ({llama_stats['error']/llama_stats['total']*100 if llama_stats['total'] > 0 else 0:.1f}%)

================================================================================
INTER-ANNOTATOR AGREEMENT
================================================================================

Matched Triples: {agreement['n_matched']}
Valid Comparisons: {agreement.get('n_valid', agreement['n_matched'])}
Agreement Rate: {agreement['agreement_rate']*100:.1f}%

Cohen's Kappa Score:
- Correctness: {agreement['kappa_correctness']:.3f} ({interpret_kappa(agreement['kappa_correctness'])})

Interpretation of Cohen's Kappa:
- < 0.00: Poor agreement
- 0.00 - 0.20: Slight agreement
- 0.21 - 0.40: Fair agreement
- 0.41 - 0.60: Moderate agreement
- 0.61 - 0.80: Substantial agreement
- 0.81 - 1.00: Almost perfect agreement

================================================================================
"""
    return report

# ============================================================
# MAIN
# ============================================================

def main():
    print(f"""
LLM-AS-JUDGE TRIPLE EVALUATION WITH INTER-ANNOTATOR AGREEMENT
================================================================================
Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Random seed: {RANDOM_SEED}
Number of triples per dataset: {NUM_TRIPLES_PER_DATASET}
Parallel workers: {NUM_WORKERS}
================================================================================
""")
    
    # Check API key
    if not RITS_API_KEY:
        print("ERROR: RITS_API_KEY environment variable not found!")
        sys.exit(1)
    print("RITS API key loaded")
    
    # Print model info
    print("\nModels for evaluation:")
    for key, config in MODELS.items():
        print(f"  - {config['name']}")
    
    # Print dataset info
    print("\nDatasets to evaluate:")
    for name, path in DATASETS.items():
        print(f"  - {name}: {path}")
    
    # Load and sample triples from each dataset
    all_triples = []
    for dataset_name, file_path in DATASETS.items():
        df = load_dataset(dataset_name, file_path)
        triples = extract_triples_from_df(df, dataset_name)
        sampled = sample_triples(triples, NUM_TRIPLES_PER_DATASET, RANDOM_SEED)
        all_triples.extend(sampled)
        print(f"  Sampled {len(sampled)} triples from {dataset_name}")
    
    print(f"\nTotal triples to evaluate: {len(all_triples)}")
    
    # Run evaluations
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS (PARALLEL)")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate with Mistral-Large
    logger.info("Starting evaluation with Mistral-Large-3-675B")
    results_mistral = evaluate_triples_parallel("mistral_large", all_triples, NUM_WORKERS)
    
    # Evaluate with LLaMA-405B
    logger.info("Starting evaluation with LLaMA-3.1-405B")
    results_llama = evaluate_triples_parallel("llama_405b", all_triples, NUM_WORKERS)
    
    # Calculate inter-annotator agreement
    print("\n" + "="*80)
    print("CALCULATING INTER-ANNOTATOR AGREEMENT")
    print("="*80)
    
    agreement = calculate_cohens_kappa(results_mistral, results_llama)
    
    print(f"\nCohen's Kappa (Correctness): {agreement['kappa_correctness']:.3f}")
    print(f"Agreement Rate: {agreement['agreement_rate']*100:.1f}%")
    
    # Save results
    save_results(results_mistral, results_llama, agreement, timestamp)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
