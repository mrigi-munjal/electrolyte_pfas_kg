#!/usr/bin/env python3
"""
Enhanced Salt Ontology Extraction with DOI Integration and Property Value Extraction

This script extracts subject-predicate-object triples from battery/salt knowledge graphs
with DOI tracking and enhanced property relation extraction including values and directions.

Author: AI Assistant
Date: 2025-08-15
"""

import os
import json
import pandas as pd
from openai import OpenAI
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the RITS API key from environment variable
RITS_API_KEY = os.environ.get("RITS_API_KEY")

# Model configuration for Mistral Small
MODEL_CONFIG = {
    "base_url": os.environ.get("RITS_BASE_URL", "").rstrip("/") + "/mistral-small-3-1-24b-2503/v1",
    "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
}

def create_enhanced_battery_analysis_prompt(paragraph, doc_id):
    """Create an enhanced structured prompt for battery component analysis with DOI tracking"""
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

IMPORTANT INSTRUCTIONS FOR PROPERTY RELATIONS:
For "AFFECTS PROPERTY OF" and "HAS PROPERTY" relations, try to extract whether the correlation is negative or positive or if there is a value available for it. For example, instead of just "interface impedance", say "increases interface impedance", "reduces interface impedance" or "decreases interface impedance" or "<1 Ω·cm²" or "10–100 Ω·cm²" etc. if that information is present.

Examples of enhanced property extraction:
- "increases ionic conductivity"
- "reduces interfacial resistance"
- "decreases battery capacity"
- "conductivity of 2.3 × 10⁻⁴ S cm⁻¹"
- "resistance < 50 Ω"
- "capacity retention > 90%"
- "voltage window 0-5V"

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any explanation or additional text. Not all relationships will be present in every paragraph, so only include those that are relevant to the salts mentioned. Some paragraphs may mention multiple salts, so ensure each salt is listed with its relationships.

Here is an example of the JSON format I expect:
{{
    "doc_id": "{doc_id}",
    "salts": [
        {{
            "salt": "LiPF6",
            "relationships": [
                {{"relation": "DISSOLVED IN", "object": "EC/DMC"}},
                {{"relation": "CO OCCURS WITH", "object": "LiTFSI"}},
                {{"relation": "USED IN", "object": "lithium-ion battery"}},
                {{"relation": "HAS CONCENTRATION IN ELECTROLYTE", "object": "1M"}},
                {{"relation": "HAS PROPERTY", "object": "ionic conductivity of 1.2 mS cm⁻¹"}},
                {{"relation": "AFFECTS PROPERTY OF", "object": "increases cycling stability"}}
            ]
        }}
    ]
}}

If NO electrolyte salts are mentioned, respond with:
{{
    "doc_id": "{doc_id}",
    "salts": []
}}

Don't extract generic salt names like "Li", "Li-ions", "lithium salt", "lithium salts", "lithium-based salt", "lithium-based salts", "lithium ion salt", "lithium ion salts", "lithium-ion salt", "lithium-ion salts", "lithium containing salt", "lithium containing salts", "lithium-containing salt", or "lithium-containing salts", "salt" or others like this. 

Paragraph to analyze:
{paragraph}

Remember: Respond ONLY with valid JSON. No explanations, no additional text. Make sure to ONLY use the paragraph provided and don't make up any details that are not present.
"""
    return prompt

def generate_rits_response(prompt, temperature=0.0, max_tokens=2048):
    """Generate response using RITS API with Mistral model"""
    try:
        logger.debug("Connecting to RITS API...")
        
        client = OpenAI(
            api_key=RITS_API_KEY,
            base_url=MODEL_CONFIG["base_url"],
            default_headers={'RITS_API_KEY': RITS_API_KEY},
        )
        
        logger.debug("Sending request to Mistral model...")
        
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_CONFIG["model_id"],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error: {str(e)}"

def parse_json_response(response):
    """Try to parse JSON from the model response with better error handling"""
    try:
        # Clean the response - remove any markdown formatting
        response = response.replace("```json", "").replace("```", "").strip()
        
        # Try to parse as JSON
        parsed = json.loads(response)
        return parsed, None
    except json.JSONDecodeError as e:
        # Try to find JSON within the response
        try:
            # Look for JSON object starting with { and ending with }
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                parsed = json.loads(json_str)
                return parsed, None
        except:
            pass
        
        return None, f"JSON parsing error: {str(e)}"

def process_single_paragraph(args):
    """Process a single paragraph with enhanced DOI tracking"""
    idx, row, max_retries = args
    doc_id = row['doc_id']
    paragraph = row['paragraph']
    
    logger.info(f"Processing paragraph {idx + 1} from DOI: {doc_id[:50]}...")
    
    prompt = create_enhanced_battery_analysis_prompt(paragraph, doc_id)
    
    for attempt in range(max_retries):
        try:
            response = generate_rits_response(prompt)
            
            if response.startswith("Error:"):
                logger.warning(f"API error on attempt {attempt + 1}: {response}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {
                        'doc_id': doc_id,
                        'paragraph_index': idx,
                        'paragraph': paragraph,
                        'error': response,
                        'salts': []
                    }
            
            parsed_response, error = parse_json_response(response)
            
            if error:
                logger.warning(f"JSON parsing error on attempt {attempt + 1}: {error}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return {
                        'doc_id': doc_id,
                        'paragraph_index': idx,
                        'paragraph': paragraph,
                        'error': f"Failed to parse JSON after {max_retries} attempts: {error}",
                        'raw_response': response,
                        'salts': []
                    }
            
            # Ensure doc_id is included in the response
            if 'doc_id' not in parsed_response:
                parsed_response['doc_id'] = doc_id
            
            # Add metadata
            parsed_response['paragraph_index'] = idx
            parsed_response['paragraph'] = paragraph
            parsed_response['processed_at'] = datetime.now().isoformat()
            
            logger.info(f"✅ Successfully processed paragraph {idx + 1} with {len(parsed_response.get('salts', []))} salt(s)")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Unexpected error processing paragraph {idx + 1}, attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    'doc_id': doc_id,
                    'paragraph_index': idx,
                    'paragraph': paragraph,
                    'error': f"Unexpected error: {str(e)}",
                    'salts': []
                }
    
    return {
        'doc_id': doc_id,
        'paragraph_index': idx,
        'paragraph': paragraph,
        'error': f"Failed after {max_retries} attempts",
        'salts': []
    }

def load_data_with_doi():
    """Load paragraph data with DOI information"""
    logger.info("📊 Loading paragraph data with DOI information...")
    
    input_file = "battery_kg/t2br_data/raw_paragraphs_v2.xlsx"
    
    if not os.path.exists(input_file):
        logger.error(f"❌ Input file not found: {input_file}")
        return None
    
    try:
        # Load the Excel file
        df = pd.read_excel(input_file, engine='openpyxl')
        logger.info(f"✅ Loaded {len(df)} paragraphs from {input_file}")
        
        # Check required columns
        required_columns = ['doc_id', 'paragraph']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Remove rows with missing data
        initial_count = len(df)
        df = df.dropna(subset=required_columns)
        final_count = len(df)
        
        if initial_count != final_count:
            logger.info(f"🧹 Removed {initial_count - final_count} rows with missing data")
        
        logger.info(f"📈 Ready to process {final_count} paragraphs with DOI information")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None

def save_enhanced_results(results, output_dir="salt_analysis_output"):
    """Save results with enhanced DOI tracking and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    raw_output_file = os.path.join(output_dir, f"enhanced_salt_extraction_raw_{timestamp}.json")
    with open(raw_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved raw results to: {raw_output_file}")
    
    # Create enhanced triples with DOI information
    enhanced_triples = []
    
    for result in results:
        doc_id = result.get('doc_id', 'unknown')
        paragraph_index = result.get('paragraph_index', -1)
        
        if 'salts' in result and result['salts']:
            for salt_data in result['salts']:
                salt_name = salt_data.get('salt', 'unknown')
                relationships = salt_data.get('relationships', [])
                
                for rel in relationships:
                    triple = {
                        'Subject': salt_name,
                        'Predicate': rel.get('relation', 'unknown'),
                        'Object': rel.get('object', 'unknown'),
                        'DOI': doc_id,
                        'ParagraphIndex': paragraph_index,
                        'Source': 'Enhanced LLM Extraction',
                        'ExtractedAt': result.get('processed_at', datetime.now().isoformat())
                    }
                    enhanced_triples.append(triple)
    
    # Save enhanced triples
    triples_output_file = os.path.join(output_dir, f"enhanced_salt_knowledge_graph_triples_{timestamp}.json")
    with open(triples_output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_triples, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved enhanced triples to: {triples_output_file}")
    
    # Create summary statistics
    total_paragraphs = len(results)
    successful_extractions = len([r for r in results if 'salts' in r and r['salts']])
    total_triples = len(enhanced_triples)
    unique_salts = len(set([t['Subject'] for t in enhanced_triples]))
    unique_dois = len(set([r.get('doc_id', 'unknown') for r in results]))
    
    # Count property relations with values/directions
    property_relations = [t for t in enhanced_triples if t['Predicate'] in ['HAS PROPERTY', 'AFFECTS PROPERTY OF']]
    enhanced_properties = [t for t in property_relations if any(keyword in t['Object'].lower() 
                          for keyword in ['increases', 'decreases', 'reduces', 'improves', 'enhances', 
                                        'mS', 'Ω', '%', 'V', 'cm⁻¹', 'S cm', '<', '>', '=', '±'])]
    
    summary = {
        'extraction_summary': {
            'total_paragraphs_processed': total_paragraphs,
            'successful_extractions': successful_extractions,
            'success_rate': f"{(successful_extractions/total_paragraphs)*100:.1f}%" if total_paragraphs > 0 else "0%",
            'total_triples_extracted': total_triples,
            'unique_salts_found': unique_salts,
            'unique_dois_processed': unique_dois,
            'property_relations_total': len(property_relations),
            'enhanced_property_relations': len(enhanced_properties),
            'enhanced_property_rate': f"{(len(enhanced_properties)/len(property_relations))*100:.1f}%" if property_relations else "0%"
        },
        'processing_metadata': {
            'processed_at': datetime.now().isoformat(),
            'model_used': MODEL_CONFIG["model_id"],
            'input_file': "battery_kg/t2br_data/raw_paragraphs_v2.xlsx"
        }
    }
    
    summary_file = os.path.join(output_dir, f"enhanced_extraction_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"📈 Saved summary to: {summary_file}")
    
    return {
        'raw_file': raw_output_file,
        'triples_file': triples_output_file,
        'summary_file': summary_file,
        'summary': summary
    }

def main():
    """Main execution function with enhanced DOI tracking"""
    logger.info("🚀 Starting Enhanced Salt Ontology Extraction with DOI Integration")
    logger.info("=" * 70)
    
    # Check API key
    if not RITS_API_KEY:
        logger.error("❌ RITS_API_KEY environment variable not set!")
        return
    
    # Load data
    df = load_data_with_doi()
    if df is None:
        logger.error("❌ Failed to load input data")
        return
    
    # Configuration
    num_workers = min(8, mp.cpu_count())  # Limit to 8 workers to avoid overwhelming the API
    max_retries = 3
    
    logger.info(f"🔧 Configuration:")
    logger.info(f"   - Workers: {num_workers}")
    logger.info(f"   - Max retries per paragraph: {max_retries}")
    logger.info(f"   - Model: {MODEL_CONFIG['model_id']}")
    
    # Prepare arguments for parallel processing
    args_list = [(idx, row, max_retries) for idx, row in df.iterrows()]
    
    # Process paragraphs in parallel
    results = []
    start_time = time.time()
    
    logger.info(f"🔄 Processing {len(args_list)} paragraphs with enhanced property extraction...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_paragraph, args): args for args in args_list}
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = len(args_list) - completed
                    eta = remaining / rate if rate > 0 else 0
                    logger.info(f"📊 Progress: {completed}/{len(args_list)} ({(completed/len(args_list))*100:.1f}%) | "
                               f"Rate: {rate:.1f}/min | ETA: {eta/60:.1f} min")
                
            except Exception as e:
                logger.error(f"Error processing future: {e}")
                # Add error result
                args = future_to_args[future]
                results.append({
                    'paragraph_index': args[0],
                    'doc_id': args[1]['doc_id'] if 'doc_id' in args[1] else 'unknown',
                    'paragraph': args[1]['paragraph'] if 'paragraph' in args[1] else '',
                    'error': f"Future processing error: {str(e)}",
                    'salts': []
                })
    
    # Save results
    logger.info("💾 Saving enhanced results with DOI tracking...")
    output_info = save_enhanced_results(results)
    
    # Print final summary
    summary = output_info['summary']['extraction_summary']
    logger.info("\n" + "=" * 70)
    logger.info("✅ ENHANCED SALT ONTOLOGY EXTRACTION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"📊 SUMMARY:")
    logger.info(f"   - Total paragraphs processed: {summary['total_paragraphs_processed']:,}")
    logger.info(f"   - Successful extractions: {summary['successful_extractions']:,}")
    logger.info(f"   - Success rate: {summary['success_rate']}")
    logger.info(f"   - Total triples extracted: {summary['total_triples_extracted']:,}")
    logger.info(f"   - Unique salts found: {summary['unique_salts_found']:,}")
    logger.info(f"   - Unique DOIs processed: {summary['unique_dois_processed']:,}")
    logger.info(f"   - Property relations: {summary['property_relations_total']:,}")
    logger.info(f"   - Enhanced property relations: {summary['enhanced_property_relations']:,} ({summary['enhanced_property_rate']})")
    logger.info(f"\n📁 Output files:")
    logger.info(f"   - Raw results: {output_info['raw_file']}")
    logger.info(f"   - Enhanced triples: {output_info['triples_file']}")
    logger.info(f"   - Summary: {output_info['summary_file']}")
    
    total_time = time.time() - start_time
    logger.info(f"\n⏱️  Total processing time: {total_time/60:.1f} minutes")
    logger.info(f"🚀 Average rate: {len(results)/(total_time/60):.1f} paragraphs/minute")

if __name__ == "__main__":
    main()
