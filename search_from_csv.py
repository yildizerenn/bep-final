#!/usr/bin/env python3
"""
Script to search for LinkedIn profiles from a CSV file of names and analyze them.
Enhanced with incremental saving, progress tracking, and robust error handling.

Usage:
    python search_from_csv.py --input names.csv [--no_gdpr] [--force]
"""

import argparse
import sys
import logging
import pandas as pd
import os
import json
import tempfile
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from search_by_name import search_specific_person
from scraper.profile_scraper import create_linkedin_dataset
from analysis.nationality_classifier import get_or_train_classifier, classify_surnames
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_csv_names(
    csv_file,
    output_dir,
    first_name_col='First Name',
    last_name_col='Last Name',
    gdpr_compliant=True,
    force=False
):
    """
    Process names from a CSV file, find LinkedIn profiles, and analyze them.
    Enhanced with incremental saving and progress tracking.
    
    Args:
        csv_file: Path to CSV file with names
        output_dir: Directory to save results
        first_name_col: Column name for first names
        last_name_col: Column name for last names
        gdpr_compliant: Whether to remove personal info for GDPR compliance
        force: Whether to force reprocessing of existing results
    
    Returns:
        Path to the output file with combined results
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load CSV file
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} names from {csv_file}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return None
    
    # Check if required columns exist
    if first_name_col not in df.columns or last_name_col not in df.columns:
        logger.error(f"CSV file must contain '{first_name_col}' and '{last_name_col}' columns")
        return None
    
    # Create full names
    df['Full Name'] = df[first_name_col] + ' ' + df[last_name_col]
    
    # Progress tracking files
    progress_file = output_path / "batch_progress.json"
    partial_results_file = output_path / "batch_analyzed_partial.csv"
    final_output = output_path / "batch_analyzed.csv"
    
    # Load existing progress
    processed_names = set()
    partial_results = []
    
    if progress_file.exists() and not force:
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_names = set(progress_data.get('processed_names', []))
                logger.info(f"Resuming: {len(processed_names)} names already processed")
        except:
            logger.warning("Could not load progress file, starting fresh")
    
    # Load partial results if they exist
    if partial_results_file.exists() and not force:
        try:
            existing_df = pd.read_csv(partial_results_file)
            partial_results = existing_df.to_dict('records')
            logger.info(f"Loaded {len(partial_results)} existing results")
        except:
            logger.warning("Could not load partial results, starting fresh")
    
    # Load classifier once for all processing
    try:
        classifier = get_or_train_classifier(
            model_path=str(Path(config.MODELS_DIR) / "dutch_surname_classifier.pkl"),
            training_data_path=str(config.SURNAME_DATASET_FILE)
        )
        logger.info("Nationality classifier loaded successfully")
    except Exception as e:
        logger.error(f"Error loading classifier: {str(e)}")
        return None
    
    # Process each name
    logger.info("Starting batch processing...")
    successful_count = 0
    api_limit_reached = False
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing names"):
        name = row['Full Name']
        
        # Skip if already processed (unless force is True)
        if name in processed_names and not force:
            continue
        
        # If API limits were reached, stop processing new names
        if api_limit_reached:
            logger.info("API limits reached. Stopping processing.")
            break
            
        try:
            logger.info(f"Processing: {name}")
            
            # Search for LinkedIn profile
            try:
                linkedin_url = search_specific_person(name)
                if not linkedin_url:
                    logger.info(f"No LinkedIn profile found for {name}")
                    processed_names.add(name)
                    continue
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ['credit', 'limit', 'quota', 'exceeded']):
                    logger.warning(f"SerpAPI credits exhausted at {name}")
                    api_limit_reached = True
                    break
                else:
                    logger.warning(f"Error searching for {name}: {str(e)}")
                    continue
            
            # Create temporary file for single URL
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(f"{linkedin_url}\n")
                temp_url_file = temp_file.name
            
            try:
                # Process single profile
                dataset = create_linkedin_dataset(
                    url_file_path=temp_url_file,
                    api_key=config.PROXYCURL_API_KEY,
                    delay=1,  # Add small delay between requests
                    output_file=None  # Don't save raw data
                )
                
                if dataset.empty:
                    logger.info(f"{name}: Not a TU/e graduate, skipping")
                    processed_names.add(name)
                    continue
                
                # Classify nationality
                analyzed_df = classify_surnames(dataset, classifier)
                
                # Apply GDPR compliance if requested
                if gdpr_compliant:
                    columns_to_remove = [
                        'full_name',
                        'city', 
                        'surname',
                        'is_dutch_surname',
                        'index'
                    ]
                    
                    for col in columns_to_remove:
                        if col in analyzed_df.columns:
                            analyzed_df = analyzed_df.drop(columns=[col])
                
                # Add original name information for tracking
                result_dict = analyzed_df.iloc[0].to_dict()
                result_dict['original_first_name'] = row[first_name_col]
                result_dict['original_last_name'] = row[last_name_col]
                
                partial_results.append(result_dict)
                successful_count += 1
                
                logger.info(f"✅ {name}: {result_dict['is_dutch']} ({result_dict['dutch_probability']:.2%})")
                
                # Mark as processed
                processed_names.add(name)
                
                # Save incremental progress
                save_incremental_progress(
                    partial_results, 
                    processed_names, 
                    partial_results_file, 
                    progress_file
                )
                
            finally:
                # Clean up temp file
                os.unlink(temp_url_file)
                
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ['credit', 'limit', 'quota', 'exceeded']):
                logger.warning(f"Proxycurl credits exhausted at {name}")
                api_limit_reached = True
                break
            else:
                logger.warning(f"Error processing {name}: {str(e)}")
                continue
    
    # Create final results
    if partial_results:
        final_df = pd.DataFrame(partial_results)
        final_df.to_csv(final_output, index=False)
        
        logger.info(f"Batch processing complete!")
        logger.info(f"Successfully processed: {successful_count}/{len(df)} names")
        logger.info(f"Final results saved to: {final_output}")
        
        # Clean up progress files
        if progress_file.exists():
            progress_file.unlink()
        if partial_results_file.exists():
            partial_results_file.unlink()
        
        # Print summary statistics
        dutch_count = (final_df['is_dutch'] == 'Dutch').sum()
        international_count = (final_df['is_dutch'] == 'International').sum()
        
        logger.info("=== BATCH ANALYSIS SUMMARY ===")
        logger.info(f"Total profiles analyzed: {len(final_df)}")
        logger.info(f"Dutch graduates: {dutch_count}")
        logger.info(f"International graduates: {international_count}")
        logger.info("==============================")
        
        return final_output
    else:
        logger.warning("No profiles were successfully processed")
        return None

def save_incremental_progress(partial_results, processed_names, partial_file, progress_file):
    """Save incremental progress to files."""
    try:
        # Save partial results
        if partial_results:
            partial_df = pd.DataFrame(partial_results)
            partial_df.to_csv(partial_file, index=False)
        
        # Save progress tracking
        progress_data = {'processed_names': list(processed_names)}
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
            
    except Exception as e:
        logger.warning(f"Error saving progress: {str(e)}")

def main():
    """Run the CSV processing as a standalone script."""
    parser = argparse.ArgumentParser(description="Process LinkedIn profiles from CSV file of names")
    parser.add_argument('--input', type=str, required=True,
                        help='CSV file containing names')
    parser.add_argument('--output_dir', type=str, default='data/batch_searches',
                        help='Directory to save output files')
    parser.add_argument('--first_name_col', type=str, default='First Name',
                        help='Column name for first names')
    parser.add_argument('--last_name_col', type=str, default='Last Name',
                        help='Column name for last names')
    parser.add_argument('--no_gdpr', action='store_true',
                        help='Do not remove personal identifying information')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of all names (ignore existing results)')
    
    args = parser.parse_args()
    
    result = process_csv_names(
        csv_file=args.input,
        output_dir=args.output_dir,
        first_name_col=args.first_name_col,
        last_name_col=args.last_name_col,
        gdpr_compliant=not args.no_gdpr,
        force=args.force
    )
    
    if result:
        print(f"\n✅ Success! Results saved to: {result}")
    else:
        print(f"\n❌ Failed to process CSV file")
        sys.exit(1)

if __name__ == "__main__":
    main()