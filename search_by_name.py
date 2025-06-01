#!/usr/bin/env python3
"""
Script to search for a specific person's LinkedIn profile and analyze it.

Usage:
    python search_by_name.py --name "John Smith" [--no_gdpr]
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
    Search for a specific person's LinkedIn profile using SerpAPI.
    
    Args:
        name: Full name of the person to search
        api_key: SerpAPI API key
    
    Returns:
        LinkedIn profile URL or None if not found
    """
    if api_key is None:
        api_key = config.SERPAPI_API_KEY
    
    # Split name into parts for more flexible searching
    name_parts = name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[-1] if len(name_parts) > 1 else ""
    
    # Try multiple search strategies
    search_queries = [
        # Strategy 1: Exact full name with TU/e
        f'"{name}" site:linkedin.com/in "Eindhoven University of Technology"',
        
        # Strategy 2: First and last name with TU/e (less strict)
        f'"{first_name}" "{last_name}" site:linkedin.com/in "Eindhoven University of Technology"',
        
        # Strategy 3: Try with TU/e variations
        f'"{name}" site:linkedin.com/in ("Eindhoven University of Technology" OR "TU/e" OR "Technische Universiteit Eindhoven")',
        
        # Strategy 4: Just the name on LinkedIn (broadest)
        f'"{name}" site:linkedin.com/in'
    ]
    
    for i, query in enumerate(search_queries):
        logger.info(f"Trying search strategy {i+1}: {query}")
        
        params = {
            "engine": "google",
            "q": query,
            "num": 5,  # Get more results to check for exact matches
            "api_key": api_key
        }
        
        try:
            # Execute search
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Extract URLs from results
            organic_results = results.get("organic_results", [])
            
            for result in organic_results:
                url = result.get("link", "")
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()
                
                logger.info(f"Checking result: {title} - {url}")
                
                # Verify it's a LinkedIn profile URL
                if url and "linkedin.com/in/" in url:
                    # Check if the title or snippet contains the person's name
                    name_lower = name.lower()
                    first_lower = first_name.lower()
                    last_lower = last_name.lower()
                    
                    # More flexible name matching
                    name_matches = (
                        name_lower in title or
                        (first_lower in title and last_lower in title) or
                        name_lower in snippet or
                        (first_lower in snippet and last_lower in snippet)
                    )
                    
                    # Check for TU/e connection (except for strategy 4)
                    tue_connection = (
                        i == 3 or  # Strategy 4 doesn't require TU/e check
                        "eindhoven" in title or
                        "eindhoven" in snippet or
                        "tu/e" in title or
                        "tu/e" in snippet
                    )
                    
                    if name_matches and tue_connection:
                        logger.info(f"Found matching LinkedIn profile: {url}")
                        return url
                    else:
                        logger.info(f"Name match: {name_matches}, TU/e connection: {tue_connection}")
            
            # If we found results but no matches, try next strategy
            if organic_results:
                logger.info(f"Found {len(organic_results)} results but no exact matches for strategy {i+1}")
            else:
                logger.info(f"No results found for strategy {i+1}")
                
        except Exception as e:
            logger.error(f"Error with search strategy {i+1}: {str(e)}")
            continue
    
    logger.warning(f"No LinkedIn profile found for '{name}' after trying all strategies")
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
    
    # Create temporary URL file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(f"{linkedin_url}\n")
        temp_url_file = temp_file.name
    
    try:
        # Collect profile data without saving raw file
        dataset = create_linkedin_dataset(
            url_file_path=temp_url_file,
            api_key=config.PROXYCURL_API_KEY,
            delay=0,  # No delay needed for single profile
            output_file=None  # Don't save raw data
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
        
        # Save final results (only analyzed, no raw)
        final_output = output_path / f"{name.replace(' ', '_')}_analyzed.csv"
        analyzed_df.to_csv(final_output, index=False)
        logger.info(f"Results saved to {final_output}")
        
        return final_output
    
    except Exception as e:
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
    
    args = parser.parse_args()
    
    process_individual(
        name=args.name,
        output_dir=args.output_dir,
        gdpr_compliant=not args.no_gdpr
    )

if __name__ == "__main__":
    main()