import os
import json
import pandas as pd
from openai import OpenAI
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# Load the RITS API key from environment variable
RITS_API_KEY = os.environ.get("RITS_API_KEY")

# Model configuration for Mistral Small
MODEL_CONFIG = {
    "base_url": os.environ.get("RITS_BASE_URL", "").rstrip("/") + "/mistral-small-3-1-24b-2503/v1",
    "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
}

def create_battery_analysis_prompt(abstract):
    """Create a structured prompt for battery component analysis"""
    prompt = f"""
You are an expert in battery materials. Analyze the following paragraph and extract information about electrolyte salts (like LiPF6, LiTFSI, LiBF4, LiClO4, LiFSI, LiAsF6 etc.). Here is a list of relationships I want you to extract:
("SALT", "DISSOLVED IN", "SOLVENT")
("SALT", "CO OCCURS WITH", "SOLVENT")
("SALT", "CO OCCURS WITH", "ADDITIVE")
("SALT", "CO OCCURS WITH", "SALT")
("SALT", "USED IN", "BATTERY_CELL")
("SALT", "STABLE IN WINDOW", "VOLTAGE_WINDOW")
("SALT", "OPERATING TEMP RANGE", "TEMPERATURE_RANGE")
("SALT", "HAS PROPERTY", "PROPERTY")
("SALT", "AFFECTS PROPERTY OF", "PROPERTY")
("SALT", "HAS CHEMICAL STRUCTURE", "CHEMICAL STRUCTURE")
("SALT", "HAS CONCENTRATION IN ELECTROLYTE", "VALUE")
("SALT", "HAS ROLE", "ROLE") 


IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any explanation or additional text. Not all relationships will be present in every paragraph, so only include those that are relevant to the salts mentioned. Some paragraphs may mention multiple salts, so ensure each salt is listed with its relationships.

Here is an example of the JSON format I expect:
{{
    "salts": [
        {{
            "salt": "LiPF6",
            "relationships": [
                {{"relation": "DISSOLVED IN", "object": "EC/DMC"}},
                {{"relation": "CO OCCURS WITH", "object": "LiTFSI"}},
                {{"relation": "USED IN", "object": "Lithium-ion battery"}},
                {{"relation": "HAS CONCENTRATION", "object": "1M"}},
                {{"relation": "HAS PROPERTY", "object": "high conductivity"}}
            ]
        }}
    ]
}}

If NO electrolyte salts are mentioned, respond with:
{{
    "salts": []
}}

Don't extract generic salt names like "Li", "Li-ions", "lithium salt", "lithium salts", "lithium-based salt", "lithium-based salts", "lithium ion salt", "lithium ion salts", "lithium-ion salt", "lithium-ion salts", "lithium containing salt", "lithium containing salts", "lithium-containing salt", or "lithium-containing salts", "salt" or others like this. 
For "AFFECTS PROPERTY OF" and "HAS PROPERTY" relations, try to extract whether the corelation is negative or positive. For example, instead of just "interface impedance", say "increases interface impedance", "reduces interface impedance" or "decreases interface impedance" if that information is present.

Paragraph to analyze:
{abstract}

Remember: Respond ONLY with valid JSON. No explanations, no additional text. 
"""
    return prompt

def generate_rits_response(prompt, temperature=0.0, max_tokens=2048):
    """Generate response using RITS API with Mistral model"""
    try:
        print("Connecting to RITS API...")
        
        client = OpenAI(
            api_key=RITS_API_KEY,
            base_url=MODEL_CONFIG["base_url"],
            default_headers={'RITS_API_KEY': RITS_API_KEY},
        )
        
        print("Sending request to Mistral model...")
        
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_CONFIG["model_id"],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}"

def parse_json_response(response):
    """Try to parse JSON from the model response with better error handling"""
    try:
        # Clean the response - remove any markdown formatting
        cleaned_response = response.strip()
        
        # Remove potential markdown code blocks
        if cleaned_response.startswith('```'):
            lines = cleaned_response.split('\n')
            cleaned_response = '\n'.join(lines[1:-1])
        
        # Try to find JSON in the response
        start_idx = cleaned_response.find('{')
        end_idx = cleaned_response.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = cleaned_response[start_idx:end_idx]
            parsed = json.loads(json_str)
            return parsed
        else:
            print("No JSON found in response")
            print(f"Response preview: {response[:200]}...")
            return None
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic response: {response[:500]}...")
        return None
    except Exception as e:
        print(f"Unexpected error parsing response: {e}")
        return None

def load_excel_data(file_path):
    """Load data from Excel file"""
    try:
        print(f"Loading data from: {file_path}")
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} rows from Excel file")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

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
        print(f"Results saved to: {output_file} ({len(results)} entries)")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        # Try to save to a different filename
        try:
            emergency_file = f"emergency_backup_{int(time.time())}.json"
            with open(emergency_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Emergency backup saved to: {emergency_file}")
        except Exception as e2:
            print(f"Critical error - could not save results anywhere: {e2}")

def process_batch_of_abstracts(batch_data):
    """Process a batch of abstracts in a separate process"""
    batch_idx, abstracts_batch, batch_size, total_batches = batch_data
    
    print(f"Starting batch {batch_idx + 1}/{total_batches} with {len(abstracts_batch)} entries")
    
    batch_results = []
    
    for i, (original_idx, abstract_text) in enumerate(abstracts_batch):
        try:
            result = process_abstract(abstract_text, i, len(abstracts_batch), batch_idx)
            result['original_index'] = original_idx  # Keep track of original index
            batch_results.append(result)
            
            # Add delay between API calls within batch
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing entry {original_idx} in batch {batch_idx}: {e}")
            # Create error entry
            error_result = {
                "original_index": original_idx,
                "index": i,
                "batch_id": batch_idx,
                "abstract": abstract_text,
                "raw_response": f"Error: {str(e)}",
                "parsed_analysis": None,
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": 0,
                "error": True
            }
            batch_results.append(error_result)
            continue
    
    print(f"Completed batch {batch_idx + 1}/{total_batches}")
    return batch_results

def process_abstract(abstract_text, index, total, batch_id=0):
    """Process a single abstract and return the analysis result"""
    print(f"Batch {batch_id} - Processing abstract {index + 1}/{total}")
    print(f"Abstract preview: {abstract_text[:100]}...")
    
    # Create the prompt
    prompt = create_battery_analysis_prompt(abstract_text)
    
    # Generate response
    start_time = time.time()
    response = generate_rits_response(prompt)
    end_time = time.time()
    
    print(f"Response generated in {end_time - start_time:.2f} seconds")
    
    # Parse the response
    parsed_result = parse_json_response(response)
    
    # Create result entry
    result_entry = {
        "index": index,
        "batch_id": batch_id,
        "abstract": abstract_text,
        "raw_response": response,
        "parsed_analysis": parsed_result,
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": end_time - start_time
    }
    
    if parsed_result:
        salts = parsed_result.get('salts', [])
        if salts:
            salt_names = [salt.get('salt', 'Unknown') for salt in salts]
            total_relationships = sum(len(salt.get('relationships', [])) for salt in salts)
            print(f"✅ Successfully parsed {len(salts)} salt(s): {salt_names} with {total_relationships} total relationships")
        else:
            print("✅ Successfully parsed - No salts found in this paragraph")
    else:
        print("❌ Failed to parse JSON response")
    
    return result_entry

def main():
    """Main function to process abstracts from Excel file using multiple processes"""
    print("Battery Salt Ontology Extraction from Excel Data (Multi-GPU)")
    print("=" * 60)
    
    # Check if API key is available
    if not RITS_API_KEY:
        print("ERROR: RITS_API_KEY environment variable not found!")
        print("Please set it using: export RITS_API_KEY='your_api_key'")
        return
    
    print(f"API Key found: {RITS_API_KEY[:10]}...")
    print(f"Using model: {MODEL_CONFIG['model_id']}")
    
    # Determine number of workers (GPUs)
    num_gpus = int(os.environ.get('NUM_GPUS', '4'))  # Default to 4 GPUs
    print(f"Using {num_gpus} parallel workers (GPUs)")
    print()
    
    # Load Excel data - handle different working directories
    excel_file_paths = [
        "battery_kg/t2br_data/raw_paragraphs_v2.xlsx",  # If running from main directory
        "t2br_data/raw_paragraphs_v2.xlsx"              # If running from battery_kg directory
    ]
    
    df = None
    for excel_file_path in excel_file_paths:
        if os.path.exists(excel_file_path):
            df = load_excel_data(excel_file_path)
            break
    
    if df is None:
        print("Excel file not found in any of the expected locations:")
        for path in excel_file_paths:
            print(f"  - {path}")
        print("Available files in current directory:")
        try:
            print(f"Current directory: {os.getcwd()}")
            print(f"Files: {os.listdir('.')}")
            if os.path.exists("battery_kg"):
                print(f"Files in battery_kg/: {os.listdir('battery_kg')}")
                if os.path.exists("battery_kg/t2br_data"):
                    print(f"Files in battery_kg/t2br_data/: {os.listdir('battery_kg/t2br_data')}")
        except Exception as e:
            print(f"Error listing files: {e}")
        return
    
    # Check if there's a column that looks like abstracts/paragraphs
    print(f"\nDataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Try to identify the text column (usually the longest text column)
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
    
    if not text_columns:
        print("No text columns found in the Excel file!")
        return
    
    text_column = text_columns[0][0]
    print(f"\nUsing column '{text_column}' as abstract/text source (avg length: {text_columns[0][1]:.0f} chars)")
    
    # Filter out null/empty entries
    df_filtered = df[df[text_column].notna()].copy()
    df_filtered = df_filtered[df_filtered[text_column].astype(str).str.len() > 50]  # Filter very short entries
    
    print(f"Processing {len(df_filtered)} entries after filtering")
    
    # Initialize results storage
    output_file = "salt_ontology_extraction_results_v2.json"
    all_results = []
    
    # Load existing results if file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing results")
        except:
            print("Could not load existing results, starting fresh")
            all_results = []
    
    start_index = len(all_results)
    remaining_data = df_filtered.iloc[start_index:]
    
    if len(remaining_data) == 0:
        print("All entries have already been processed!")
        return
    
    print(f"Processing {len(remaining_data)} remaining entries")
    
    # Prepare data for parallel processing
    abstracts_to_process = []
    for idx, row in remaining_data.iterrows():
        abstract_text = str(row[text_column])
        
        # Skip if too short or too long
        if len(abstract_text) < 100:
            print(f"Skipping entry {idx}: too short ({len(abstract_text)} chars)")
            continue
        if len(abstract_text) > 12000:  # Increased limit for better salt extraction
            print(f"Truncating entry {idx}: too long ({len(abstract_text)} chars)")
            # Smart truncation: try to end at a sentence boundary
            truncated = abstract_text[:12000]
            last_period = truncated.rfind('.')
            if last_period > 10000:  # Only use sentence boundary if it's not too early
                abstract_text = truncated[:last_period + 1]
            else:
                abstract_text = truncated + "..."
        
        abstracts_to_process.append((idx, abstract_text))
    
    print(f"Total valid abstracts to process: {len(abstracts_to_process)}")
    
    # Split data into batches for parallel processing
    batch_size = max(1, len(abstracts_to_process) // num_gpus)
    batches = []
    
    for i in range(0, len(abstracts_to_process), batch_size):
        batch = abstracts_to_process[i:i + batch_size]
        batches.append((len(batches), batch, batch_size, math.ceil(len(abstracts_to_process) / batch_size)))
    
    print(f"Split into {len(batches)} batches of ~{batch_size} entries each")
    
    # Process batches in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_batch_of_abstracts, batch_data): batch_data[0] 
                          for batch_data in batches}
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                
                print(f"Completed batch {batch_idx + 1}, total results: {len(all_results)}")
                
                # Save results incrementally
                save_results_incrementally(all_results, output_file)
                
            except Exception as e:
                print(f"Batch {batch_idx + 1} generated an exception: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final save
    save_results_incrementally(all_results, output_file)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total entries processed: {len(all_results)}")
    print(f"Total processing time: {total_time/3600:.2f} hours")
    print(f"Average time per entry: {total_time/len(all_results):.2f} seconds")
    print(f"Results saved to: {output_file}")
    
    # Count successful parses
    successful_parses = sum(1 for result in all_results if result.get('parsed_analysis') is not None)
    print(f"Successfully parsed: {successful_parses}/{len(all_results)} ({successful_parses/len(all_results)*100:.1f}%)")
    
    # Show summary stats for salt relationships
    if successful_parses > 0:
        all_salts = set()
        all_relationships = {}
        
        for result in all_results:
            if result.get('parsed_analysis') and result['parsed_analysis'].get('salts'):
                salts = result['parsed_analysis']['salts']
                for salt_entry in salts:
                    salt_name = salt_entry.get('salt')
                    if salt_name:
                        all_salts.add(salt_name)
                        relationships = salt_entry.get('relationships', [])
                        for rel in relationships:
                            rel_type = rel.get('relation', 'Unknown')
                            if rel_type not in all_relationships:
                                all_relationships[rel_type] = 0
                            all_relationships[rel_type] += 1
        
        print(f"\nExtracted unique salts: {len(all_salts)}")
        if all_salts:
            print(f"Salt types found: {sorted(list(all_salts))}")
        print(f"Relationship types found:")
        for rel_type, count in sorted(all_relationships.items(), key=lambda x: x[1], reverse=True):
            print(f"  {rel_type}: {count} instances")

if __name__ == "__main__":
    main()
