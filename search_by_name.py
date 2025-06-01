#!/usr/bin/env python3
"""
Script to search for a specific person's LinkedIn profile and analyze it.
Enhanced with robust error handling and automatic skip for existing results.

Usage:
    python search_by_name.py --name "John Smith" [--no_gdpr] [--force]
"""

import argparse
import sys
import logging
import tempfile
import os
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serpapi import GoogleSearch
from scraper.profile_scraper import create_linkedin_dataset
from analysis.nationality_classifier import get_or_train_classifier, classify_surnames
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def search_specific_person(name, api_key=None):
    """
    Search for a specific person's LinkedIn profile using optimized SerpAPI.
    Now uses the optimized search from link_scraper.
    
    Args:
        name: Full name of the person to search
        api_key: SerpAPI API key
    
    Returns:
        LinkedIn profile URL or None if not found
    """
    # Import the optimized function
    from scraper.link_scraper import search_specific_person_optimized
    
    return search_specific_person_optimized(name, api_key, max_results=5)

def process_individual(name, output_dir, gdpr_compliant=True, force=False):
    """
    Process an individual by name, find their LinkedIn profile, and analyze it.
    Enhanced with result checking and robust error handling.
    
    Args:
        name: Person's full name
        output_dir: Directory to save results
        gdpr_compliant: Whether to remove personal info for GDPR compliance
        force: Whether to force reprocessing even if result exists
    
    Returns:
        Path to the output file or None if processing failed
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Check if result already exists
    final_output = output_path / f"{name.replace(' ', '_')}_analyzed.csv"
    
    if final_output.exists() and not force:
        logger.info(f"Result already exists: {final_output}")
        logger.info("Use --force to reprocess this person")
        return final_output
    
    # Search for LinkedIn profile with robust error handling
    try:
        linkedin_url = search_specific_person(name)
        if not linkedin_url:
            logger.warning(f"No LinkedIn profile found for {name}")
            return None
    except Exception as e:
        if any(keyword in str(e).lower() for keyword in ['credit', 'limit', 'quota', 'exceeded']):
            logger.warning(f"SerpAPI credits exhausted while searching for {name}")
            logger.info("Try again later when credits are renewed")
            return None
        else:
            logger.error(f"Error searching for {name}: {str(e)}")
            return None
    
    # Create temporary URL file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(f"{linkedin_url}\n")
        temp_url_file = temp_file.name
    
    try:
        # Collect profile data without saving raw file
        logger.info(f"Collecting profile data for {name}...")
        dataset = create_linkedin_dataset(
            url_file_path=temp_url_file,
            api_key=config.PROXYCURL_API_KEY,
            delay=0,  # No delay needed for single profile
            output_file=None  # Don't save raw data
        )
        
        if dataset.empty:
            logger.warning(f"No profile data retrieved for {name} (may not be a TU/e graduate)")
            return None
        
        # Get or train classifier
        logger.info("Loading nationality classifier...")
        classifier = get_or_train_classifier(
            model_path=str(Path(config.MODELS_DIR) / "dutch_surname_classifier.pkl"),
            training_data_path=str(config.SURNAME_DATASET_FILE)
        )
        
        # Classify surname
        logger.info("Analyzing nationality...")
        analyzed_df = classify_surnames(dataset, classifier)
        
        # Apply GDPR compliance if requested
        if gdpr_compliant:
            logger.info("Applying GDPR compliance (removing personal identifying information)")
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
        logger.info(f"Analysis complete! Results saved to {final_output}")
        
        # Print summary
        nationality = analyzed_df.iloc[0]['is_dutch']
        probability = analyzed_df.iloc[0]['dutch_probability']
        logger.info(f"Classification: {nationality} (confidence: {probability:.2%})")
        
        return final_output
    
    except Exception as e:
        if any(keyword in str(e).lower() for keyword in ['credit', 'limit', 'quota', 'exceeded']):
            logger.warning(f"Proxycurl credits exhausted while processing {name}")
            logger.info("Try again later when credits are renewed")
            return None
        else:
            logger.error(f"Error processing {name}: {str(e)}")
            return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_url_file):
            os.unlink(temp_url_file)

def main():
    """Run the name search as a standalone script."""
    parser = argparse.ArgumentParser(description="Search for specific person's LinkedIn profile")
    parser.add_argument('--name', type=str, required=True,
                        help='Full name of the person to search')
    parser.add_argument('--output_dir', type=str, default='data/individual_searches',
                        help='Directory to save output files')
    parser.add_argument('--no_gdpr', action='store_true',
                        help='Do not remove personal identifying information')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if result already exists')
    
    args = parser.parse_args()
    
    # Process the individual
    result = process_individual(
        name=args.name,
        output_dir=args.output_dir,
        gdpr_compliant=not args.no_gdpr,
        force=args.force
    )
    
    if result:
        print(f"\n✅ Success! Results saved to: {result}")
    else:
        print(f"\n❌ Failed to process {args.name}")
        sys.exit(1)

if __name__ == "__main__":
    main()