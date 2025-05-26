#!/usr/bin/env python3
"""
Script to search for LinkedIn profiles from a CSV file of names.

Usage:
    python search_from_csv.py --input names.csv [--gdpr_compliant]
"""

import argparse
import sys
import logging
import pandas as pd
import os
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
    gdpr_compliant=True
):
    """
    Process names from a CSV file, find LinkedIn profiles, and analyze them.
    
    Args:
        csv_file: Path to CSV file with names
        output_dir: Directory to save results
        first_name_col: Column name for first names
        last_name_col: Column name for last names
        gdpr_compliant: Whether to remove personal info for GDPR compliance
    
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
    
    # Create full names and search for LinkedIn profiles
    df['Full Name'] = df[first_name_col] + ' ' + df[last_name_col]
    
    # Create temporary file to store LinkedIn URLs
    temp_url_file = output_path / "temp_urls.txt"
    
    # Track which names were successfully found
    found_profiles = []
    
    # Search for each name
    logger.info("Searching for LinkedIn profiles...")
    with open(temp_url_file, "w") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            name = row['Full Name']
            linkedin_url = search_specific_person(name)
            
            if linkedin_url:
                f.write(f"{linkedin_url}\n")
                found_profiles.append(idx)
    
    if not found_profiles:
        logger.warning("No LinkedIn profiles found for any names in the CSV")
        if temp_url_file.exists():
            temp_url_file.unlink()
        return None
    
    # Process the profiles
    raw_output = output_path / "batch_raw.csv"
    final_output = output_path / "batch_analyzed.csv"
    
    try:
        # Collect profile data
        dataset = create_linkedin_dataset(
            url_file_path=str(temp_url_file),
            api_key=config.PROXYCURL_API_KEY,
            delay=1,  # Add small delay between requests
            output_file=str(raw_output)
        )
        
        if dataset.empty:
            logger.warning("No profile data retrieved")
            return None
        
        # Get or train classifier
        classifier = get_or_train_classifier(
            model_path=str(Path(config.MODELS_DIR) / "dutch_surname_classifier.pkl"),
            training_data_path=str(config.SURNAME_DATASET_FILE)
        )
        
        # Classify surnames
        analyzed_df = classify_surnames(dataset, classifier)
        
        # Apply GDPR compliance if requested
        if gdpr_compliant:
            logger.info("Removing personal identifying information for GDPR compliance")
            columns_to_remove = [
                'full_name',
                'city',
                'surname',
                'is_dutch_surname',
                'index'
            ]
            
            # Remove columns that exist in the DataFrame
            for col in columns_to_remove:
                if col in analyzed_df.columns:
                    analyzed_df = analyzed_df.drop(columns=[col])
        
        # Save final results
        analyzed_df.to_csv(final_output, index=False)
        logger.info(f"Results saved to {final_output}")
        
        return final_output
    
    except Exception as e:
        logger.error(f"Error processing profiles: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if temp_url_file.exists():
            temp_url_file.unlink()

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
    
    args = parser.parse_args()
    
    process_csv_names(
        csv_file=args.input,
        output_dir=args.output_dir,
        first_name_col=args.first_name_col,
        last_name_col=args.last_name_col,
        gdpr_compliant=not args.no_gdpr
    )

if __name__ == "__main__":
    main()