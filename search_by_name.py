#!/usr/bin/env python3
"""
Script to search for a specific person's LinkedIn profile and analyze it.

Usage:
    python search_by_name.py --name "John Smith" [--gdpr_compliant]
"""

import argparse
import sys
import logging
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
    Search for a specific person's LinkedIn profile using SerpAPI.
    
    Args:
        name: Full name of the person to search
        api_key: SerpAPI API key
    
    Returns:
        LinkedIn profile URL or None if not found
    """
    if api_key is None:
        api_key = config.SERPAPI_API_KEY
    
    # Construct search query
    query = f'"{name}" site:linkedin.com/in "Eindhoven University of Technology"'
    
    params = {
        "engine": "google",
        "q": query,
        "num": 1,  # Only need the first result
        "api_key": api_key
    }
    
    try:
        # Execute search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract URL from results
        organic_results = results.get("organic_results", [])
        if organic_results:
            url = organic_results[0].get("link")
            # Verify it's a LinkedIn profile URL
            if url and "linkedin.com/in/" in url:
                logger.info(f"Found LinkedIn profile: {url}")
                return url
        
        logger.warning(f"No LinkedIn profile found for '{name}'")
        return None
    
    except Exception as e:
        logger.error(f"Error searching for {name}: {str(e)}")
        return None

def process_individual(name, output_dir, gdpr_compliant=True):
    """
    Process an individual by name, find their LinkedIn profile, and analyze it.
    
    Args:
        name: Person's full name
        output_dir: Directory to save results
        gdpr_compliant: Whether to remove personal info for GDPR compliance
    
    Returns:
        Path to the output file or None if processing failed
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Search for LinkedIn profile
    linkedin_url = search_specific_person(name)
    if not linkedin_url:
        return None
    
    # Save URL to temporary file
    temp_url_file = output_path / f"{name.replace(' ', '_')}_url.txt"
    with open(temp_url_file, "w") as f:
        f.write(f"{linkedin_url}\n")
    
    # Process the profile
    raw_output = output_path / f"{name.replace(' ', '_')}_raw.csv"
    final_output = output_path / f"{name.replace(' ', '_')}_analyzed.csv"
    
    try:
        # Collect profile data
        dataset = create_linkedin_dataset(
            url_file_path=str(temp_url_file),
            api_key=config.PROXYCURL_API_KEY,
            delay=0,  # No delay needed for single profile
            output_file=str(raw_output)
        )
        
        if dataset.empty:
            logger.warning(f"No profile data retrieved for {name}")
            return None
        
        # Get or train classifier
        classifier = get_or_train_classifier(
            model_path=str(Path(config.MODELS_DIR) / "dutch_surname_classifier.pkl"),
            training_data_path=str(config.SURNAME_DATASET_FILE)
        )
        
        # Classify surname
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
        logger.error(f"Error processing {name}: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if temp_url_file.exists():
            temp_url_file.unlink()

def main():
    """Run the name search as a standalone script."""
    parser = argparse.ArgumentParser(description="Search for specific person's LinkedIn profile")
    parser.add_argument('--name', type=str, required=True,
                        help='Full name of the person to search')
    parser.add_argument('--output_dir', type=str, default='data/individual_searches',
                        help='Directory to save output files')
    parser.add_argument('--no_gdpr', action='store_true',
                        help='Do not remove personal identifying information')
    
    args = parser.parse_args()
    
    process_individual(
        name=args.name,
        output_dir=args.output_dir,
        gdpr_compliant=not args.no_gdpr
    )

if __name__ == "__main__":
    main()