#!/usr/bin/env python3
"""
LLM Querying of Candidate Salts from Jaccard Analysis

This script queries the knowledge graph triples for top-performing salts
from the Jaccard similarity analysis and uses RITS API with Mistral to
analyze their potential advantages/limitations compared to LiTFSI.
"""

import os
import json
import pandas as pd
from openai import OpenAI
import time
from datetime import datetime
from collections import defaultdict
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

warnings.filterwarnings('ignore')

# Load the RITS API key from environment variable
RITS_API_KEY = os.environ.get("RITS_API_KEY")

# Model configuration for Mistral Small
MODEL_CONFIG = {
    "base_url": os.environ.get("RITS_BASE_URL", "").rstrip("/") + "/mistral-small-3-1-24b-2503/v1",
    "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
}

def load_jaccard_results(file_path='salt_analysis_output/jaccard_similarities_LiTFSI_top50.csv'):
    """Load the Jaccard similarity results"""
    try:
        print(f"Loading Jaccard results from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} salt candidates")
        return df
    except Exception as e:
        print(f"Error loading Jaccard results: {e}")
        return None

def load_knowledge_graph_triples(file_path='salt_analysis_output/salt_knowledge_graph_triples.json'):
    """Load the knowledge graph triples"""
    try:
        print(f"Loading knowledge graph triples from: {file_path}")
        with open(file_path, 'r') as f:
            triples = json.load(f)
        print(f"Successfully loaded {len(triples):,} triples")
        return triples
    except Exception as e:
        print(f"Error loading triples: {e}")
        return None

def get_salt_knowledge_profile(salt_name, triples):
    """Extract all knowledge about a specific salt from the triples"""
    salt_profile = {
        'relationships': [],
        'properties': [],
        'applications': [],
        'interactions': [],
        'performance_metrics': []
    }
    
    for triple in triples:
        if triple['subject'] == salt_name:
            relationship = {
                'predicate': triple['predicate'],
                'object': triple['object']
            }
            salt_profile['relationships'].append(relationship)
            
            # Categorize relationships
            predicate = triple['predicate'].upper()
            obj = triple['object']
            
            if 'PROPERTY' in predicate:
                salt_profile['properties'].append(obj)
            elif 'USED IN' in predicate or 'APPLICATION' in predicate:
                salt_profile['applications'].append(obj)
            elif 'CO OCCURS WITH' in predicate or 'DISSOLVED IN' in predicate:
                salt_profile['interactions'].append(obj)
            elif 'CONCENTRATION' in predicate or 'TEMP' in predicate or 'VOLTAGE' in predicate:
                salt_profile['performance_metrics'].append(f"{predicate}: {obj}")
    
    return salt_profile

def create_salt_comparison_prompt(candidate_salt, reference_salt, candidate_profile, reference_profile):
    """Create a prompt for comparing candidate salt to LiTFSI"""
    
    # Format the knowledge profiles
    def format_profile(profile, salt_name):
        formatted = f"\n**{salt_name} Knowledge Profile:**\n"
        
        if profile['properties']:
            formatted += f"- Properties: {', '.join(set(profile['properties']))}\n"
        
        if profile['applications']:
            formatted += f"- Applications: {', '.join(set(profile['applications']))}\n"
            
        if profile['interactions']:
            formatted += f"- Interactions: {', '.join(set(profile['interactions']))}\n"
            
        if profile['performance_metrics']:
            formatted += f"- Performance Metrics: {', '.join(set(profile['performance_metrics']))}\n"
        
        # Show all relationships
        formatted += f"- All Relationships:\n"
        for rel in profile['relationships'][:15]:  # Limit to first 15 for readability
            formatted += f"  • {rel['predicate']}: {rel['object']}\n"
        
        if len(profile['relationships']) > 15:
            formatted += f"  ... and {len(profile['relationships']) - 15} more relationships\n"
        
        return formatted
    
    candidate_formatted = format_profile(candidate_profile, candidate_salt)
    reference_formatted = format_profile(reference_profile, reference_salt)
    
    prompt = f"""
You are an expert in battery electrochemistry and electrolyte materials. I want you to analyze and compare two lithium salts used in battery electrolytes.

**TASK**: Compare {candidate_salt} to {reference_salt} (the reference standard) and provide insights about the potential advantages and limitations of using {candidate_salt} as a replacement for {reference_salt} in lithium-ion battery electrolytes.

**KNOWLEDGE BASE**:
{reference_formatted}
{candidate_formatted}

**ANALYSIS REQUIREMENTS**:
Please provide a comprehensive analysis in clear text format with bullet points covering the following aspects:

**1. SIMILARITY ASSESSMENT**
• Brief assessment of how similar {candidate_salt} and {reference_salt} are

**2. KEY DIFFERENCES**  
• List 3-5 main differences between the two salts

**3. ADVANTAGES OF {candidate_salt} OVER {reference_salt}**
• Electrochemical advantages:
• Performance advantages:
• Safety advantages:
• Cost/Manufacturing advantages:

**4. LIMITATIONS OF {candidate_salt} COMPARED TO {reference_salt}**
• Electrochemical limitations:
• Performance limitations:
• Safety limitations:
• Cost/Manufacturing limitations:

**5. APPLICATION SUITABILITY**
• Best applications for {candidate_salt}:
• Challenging applications for {candidate_salt}:
• Reasoning for application suitability:

**6. REPLACEMENT FEASIBILITY**
• Feasibility score (1-10 scale):
• Drop-in replacement potential (yes/no/partial):
• Required modifications for implementation:
• Compatibility concerns:

**7. RESEARCH GAPS**
• Areas needing further research:

**8. OVERALL ASSESSMENT**
• Summary of whether {candidate_salt} is a viable alternative to {reference_salt} and under what conditions

**IMPORTANT GUIDELINES**:
- Base your analysis on electrochemical principles and battery science
- Consider ionic conductivity, electrochemical stability, thermal stability, safety, and cost
- Be specific about application contexts (high voltage, high temperature, fast charging, etc.)
- If information is limited, indicate uncertainty and suggest areas for further research
- Focus on practical implications for battery performance and manufacturing
- Use clear bullet points and avoid JSON formatting
"""
    
    return prompt

def generate_rits_response(prompt, temperature=0.1, max_tokens=3072):
    """Generate response using RITS API with Mistral model"""
    try:
        
        client = OpenAI(
            api_key=RITS_API_KEY,
            base_url=MODEL_CONFIG["base_url"],
            default_headers={'RITS_API_KEY': RITS_API_KEY},
        )
        
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_CONFIG["model_id"],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"🚫 Error generating RITS response: {e}")
        return f"Error: {str(e)}"

def parse_text_response(response):
    """Process the text response from the model"""
    try:
        # Clean the response - remove any markdown formatting if present
        cleaned_response = response.strip()
        
        # Remove potential markdown code blocks
        if cleaned_response.startswith('```'):
            lines = cleaned_response.split('\n')
            cleaned_response = '\n'.join(lines[1:-1])
        
        # Return the cleaned text response
        return {
            "analysis_text": cleaned_response,
            "parsed_successfully": True
        }
            
    except Exception as e:
        print(f"Error processing response: {e}")
        return {
            "analysis_text": response,
            "parsed_successfully": False,
            "error": str(e)
        }

def save_analysis_results(results, output_dir='salt_analysis_output'):
    """Save the analysis results to JSON and text files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results as JSON
    json_path = os.path.join(output_dir, 'llm_candidate_salt_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create a readable text report
    txt_path = os.path.join(output_dir, 'llm_candidate_salt_analysis_report.txt')
    with open(txt_path, 'w') as f:
        f.write("LLM ANALYSIS OF CANDIDATE SALTS\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, result in enumerate(results):
            f.write(f"\n{'='*60}\n")
            f.write(f"CANDIDATE {i+1}: {result['candidate_salt']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Jaccard Similarity to LiTFSI: {result['jaccard_similarity']:.4f}\n")
            f.write(f"Knowledge Relationships: {result['knowledge_relationships']}\n")
            f.write(f"Processing Time: {result['processing_time_seconds']:.2f} seconds\n\n")
            
            if result.get('parsed_analysis') and result['parsed_analysis'].get('parsed_successfully'):
                f.write("ANALYSIS:\n")
                f.write("-"*20 + "\n")
                f.write(result['parsed_analysis']['analysis_text'])
                f.write("\n\n")
            else:
                f.write("ANALYSIS FAILED OR INCOMPLETE\n")
                f.write("-"*30 + "\n")
                f.write("Raw response preview:\n")
                f.write(result['raw_response'][:500] + "...\n\n")
    
    # Create simple summary CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'candidate_salt': result['candidate_salt'],
            'jaccard_similarity': result['jaccard_similarity'],
            'knowledge_relationships': result['knowledge_relationships'],
            'analysis_successful': result.get('parsed_analysis', {}).get('parsed_successfully', False),
            'processing_time_seconds': result['processing_time_seconds'],
            'analysis_timestamp': result['timestamp']
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'llm_candidate_salt_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to:")
    print(f"  - Full analysis (JSON): {json_path}")
    print(f"  - Readable report: {txt_path}")
    print(f"  - Summary (CSV): {csv_path}")
    
    return json_path, txt_path, csv_path

def process_candidate_batch(batch_data):
    """Process a batch of candidate salts in a separate process"""
    batch_idx, candidates_batch, reference_salt, triples, total_batches = batch_data
    
    print(f"🚀 STARTING BATCH {batch_idx + 1}/{total_batches} with {len(candidates_batch)} candidates")
    print(f"📊 Batch {batch_idx + 1} candidates: {[c['salt'] for c in candidates_batch]}")
    
    batch_results = []
    
    # Get reference salt profile
    reference_profile = get_salt_knowledge_profile(reference_salt, triples)
    
    for i, candidate_data in enumerate(candidates_batch):
        try:
            candidate_salt = candidate_data['salt']
            jaccard_score = candidate_data['jaccard_similarity']
            
            print(f"⚗️  BATCH {batch_idx + 1} - Processing {i+1}/{len(candidates_batch)}: {candidate_salt} (Jaccard: {jaccard_score:.4f})")
            
            # Get candidate salt profile
            candidate_profile = get_salt_knowledge_profile(candidate_salt, triples)
            
            if len(candidate_profile['relationships']) == 0:
                print(f"⚠️  WARNING: No knowledge found for {candidate_salt}, skipping...")
                continue
            
            print(f"📚 {candidate_salt}: Found {len(candidate_profile['relationships'])} knowledge relationships")
            
            # Create comparison prompt
            prompt = create_salt_comparison_prompt(candidate_salt, reference_salt, 
                                                 candidate_profile, reference_profile)
            
            # Generate LLM response
            print(f"🤖 Querying LLM for {candidate_salt}...")
            start_time = time.time()
            response = generate_rits_response(prompt)
            end_time = time.time()
            
            print(f"⏱️  LLM response for {candidate_salt} generated in {end_time - start_time:.2f} seconds")
            
            # Parse response
            parsed_analysis = parse_text_response(response)
            
            # Store result
            result_entry = {
                "candidate_salt": candidate_salt,
                "jaccard_similarity": jaccard_score,
                "knowledge_relationships": len(candidate_profile['relationships']),
                "raw_response": response,
                "parsed_analysis": parsed_analysis,
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": end_time - start_time,
                "batch_id": batch_idx
            }
            
            batch_results.append(result_entry)
            
            if parsed_analysis and parsed_analysis.get('parsed_successfully'):
                print(f"✅ {candidate_salt}: Successfully analyzed ({len(parsed_analysis['analysis_text'])} chars)")
                # Show brief preview of analysis
                preview = parsed_analysis['analysis_text'][:150].replace('\n', ' ').strip()
                print(f"📄 Preview: {preview}...")
            else:
                print(f"❌ {candidate_salt}: Analysis failed")
            
            # Add delay between API calls within batch to respect rate limits
            if i < len(candidates_batch) - 1:  # Don't delay after last item
                print(f"⏸️  Pausing 2 seconds before next candidate...")
                time.sleep(2)  # 2 second delay between candidates in same batch
            
        except Exception as e:
            print(f"💥 ERROR processing {candidate_salt} in batch {batch_idx}: {e}")
            # Create error entry
            error_result = {
                "candidate_salt": candidate_salt,
                "jaccard_similarity": jaccard_score,
                "knowledge_relationships": 0,
                "raw_response": f"Error: {str(e)}",
                "parsed_analysis": {"analysis_text": f"Processing error: {str(e)}", "parsed_successfully": False},
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": 0,
                "batch_id": batch_idx,
                "error": True
            }
            batch_results.append(error_result)
            continue
    
    print(f"🏁 COMPLETED BATCH {batch_idx + 1}/{total_batches} - Processed {len(batch_results)} candidates")
    return batch_results

def save_results_incrementally(results, output_file):
    """Save results to JSON file incrementally with proper error handling"""
    try:
        # Create backup of existing file first
        if os.path.exists(output_file):
            backup_file = output_file.replace('.json', '_backup.json')
            import shutil
            shutil.copy2(output_file, backup_file)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_file} ({len(results)} entries)")
        
    except Exception as e:
        print(f"💥 Error saving results: {e}")
        # Try to save to a different filename
        try:
            emergency_file = f"emergency_backup_{int(time.time())}.json"
            with open(emergency_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"🚨 Emergency backup saved to: {emergency_file}")
        except Exception as e2:
            print(f"🆘 Critical error - could not save results anywhere: {e2}")

def main():
    """Main function to run the LLM analysis of candidate salts using parallel processing"""
    print("🚀 LLM Analysis of Candidate Salts from Jaccard Similarity (Multi-Process)")
    print("="*70)
    
    # Check if API key is available
    if not RITS_API_KEY:
        print("🚫 ERROR: RITS_API_KEY environment variable not found!")
        print("Please set it using: export RITS_API_KEY='your_api_key'")
        return
    
    print(f"🔑 API Key found: {RITS_API_KEY[:10]}...")
    print(f"🤖 Using model: {MODEL_CONFIG['model_id']}")
    
    # Determine number of workers
    num_workers = int(os.environ.get('NUM_WORKERS', '4'))  # Default to 4 workers
    print(f"👥 Using {num_workers} parallel workers")
    
    # Ensure we don't use more workers than candidates
    max_candidates = int(os.environ.get('MAX_CANDIDATES', '20'))
    if num_workers > max_candidates:
        num_workers = max(1, max_candidates // 2)  # Use at most half the candidates as workers
        print(f"⚖️  Adjusted to {num_workers} workers to match candidate count")
    
    # Load Jaccard results
    print(f"\n📊 Loading data...")
    jaccard_df = load_jaccard_results()
    if jaccard_df is None:
        return
    
    # Load knowledge graph triples
    triples = load_knowledge_graph_triples()
    if triples is None:
        return
    
    # Get reference salt
    reference_salt = "LiTFSI"
    
    # Limit candidates to manage costs and time (can be increased)
    top_candidates = jaccard_df.head(max_candidates)
    print(f"\n🎯 Analyzing top {len(top_candidates)} candidates with {num_workers} workers...")
    
    # Show candidates overview
    print(f"📋 Candidates to analyze:")
    for i, row in top_candidates.iterrows():
        print(f"  {i+1:2d}. {row['salt']:15s} (Jaccard: {row['jaccard_unweighted']:.4f})")
    
    # Prepare candidate data for parallel processing
    candidates_to_process = []
    for idx, row in top_candidates.iterrows():
        candidate_data = {
            'salt': row['salt'],
            'jaccard_similarity': row['jaccard_unweighted']
        }
        candidates_to_process.append(candidate_data)
    
    # Split candidates into batches for parallel processing
    # Ensure we have reasonable batch sizes (at least 1 candidate per batch, but not too small)
    min_batch_size = 1
    ideal_batch_size = max(min_batch_size, len(candidates_to_process) // num_workers)
    batch_size = max(min_batch_size, ideal_batch_size)
    
    batches = []
    
    for i in range(0, len(candidates_to_process), batch_size):
        batch = candidates_to_process[i:i + batch_size]
        batch_data = (len(batches), batch, reference_salt, triples, 
                     math.ceil(len(candidates_to_process) / batch_size))
        batches.append(batch_data)
    
    print(f"Split into {len(batches)} batches of ~{batch_size} candidates each")
    print(f"Batch sizes: {[len(batch[1]) for batch in batches]}")
    
    # Adjust num_workers to not exceed number of batches
    actual_workers = min(num_workers, len(batches))
    print(f"Using {actual_workers} actual workers for {len(batches)} batches")
    
    # Initialize results storage
    output_file = "salt_analysis_output/llm_candidate_salt_analysis_parallel.json"
    all_results = []
    
    # Process batches in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_candidate_batch, batch_data): batch_data[0] 
                          for batch_data in batches}
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                
                print(f"🎯 BATCH {batch_idx + 1} COMPLETE! Total results so far: {len(all_results)}")
                
                # Save results incrementally
                save_results_incrementally(all_results, output_file)
                
                # Show progress summary
                successful_so_far = sum(1 for r in all_results if r.get('parsed_analysis', {}).get('parsed_successfully', False))
                print(f"📈 Progress: {len(all_results)} processed, {successful_so_far} successful ({successful_so_far/len(all_results)*100:.1f}%)")
                
            except Exception as e:
                print(f"💥 BATCH {batch_idx + 1} FAILED with exception: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final save with all output formats
    json_path, txt_path, csv_path = save_analysis_results(all_results)
    
    # Print summary
    print(f"\n{'='*70}")
    print("🎉 PARALLEL ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Total candidates processed: {len(all_results)}")
    print(f"⏰ Total processing time: {total_time/60:.2f} minutes")
    print(f"⚡ Average time per candidate: {total_time/len(all_results):.2f} seconds")
    successful_analyses = sum(1 for r in all_results if r.get('parsed_analysis', {}).get('parsed_successfully', False))
    print(f"✅ Successfully processed: {successful_analyses}/{len(all_results)} ({successful_analyses/len(all_results)*100:.1f}%)")
    print(f"📁 Results saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - Report: {txt_path}")
    print(f"  - Summary: {csv_path}")
    
    # Show quick preview
    print(f"\n📋 Processed candidates:")
    for i, result in enumerate(all_results):
        status = "✅" if result.get('parsed_analysis', {}).get('parsed_successfully', False) else "❌"
        salt = result['candidate_salt']
        jaccard = result['jaccard_similarity']
        batch_id = result.get('batch_id', 0)
        print(f"{i+1:2d}. {status} {salt:15s} (Jaccard: {jaccard:.4f}, Batch: {batch_id+1})")
    
    # Show top candidates summary
    if successful_analyses > 0:
        print(f"\n🏆 Top 3 successfully analyzed candidates:")
        successful_results = [r for r in all_results if r.get('parsed_analysis', {}).get('parsed_successfully', False)]
        successful_results.sort(key=lambda x: x['jaccard_similarity'], reverse=True)
        
        for i, result in enumerate(successful_results[:3]):
            salt = result['candidate_salt']
            jaccard = result['jaccard_similarity']
            analysis_preview = result['parsed_analysis']['analysis_text'][:150].replace('\n', ' ')
            print(f"{i+1}. {salt} (Jaccard: {jaccard:.4f})")
            print(f"   📄 Preview: {analysis_preview}...")
            print()

if __name__ == "__main__":
    main()
