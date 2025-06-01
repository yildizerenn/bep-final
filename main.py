#!/usr/bin/env python3
"""
Main script for the TU/e LinkedIn Graduate Analyzer pipeline.

This script orchestrates the entire process:
1. Scrapes LinkedIn URLs using Google Search
2. Collects profile data using the Proxycurl API
3. Analyzes profiles and classifies nationality based on surnames
4. Offers individual name search and CSV batch processing options

Usage:
    python main.py [--skip_scraping] [--skip_collection] [--skip_analysis]
    python main.py --search_name "John Smith" [--no_gdpr]
    python main.py --csv_input names.csv [--no_gdpr]
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path to allow relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scraper.link_scraper import search_tue_graduates
from scraper.profile_scraper import create_linkedin_dataset
from analysis.profile_analyzer import analyze_profiles
from analysis.nationality_classifier import get_or_train_classifier, classify_surnames
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.DATA_DIR, "pipeline.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TU/e LinkedIn Graduate Analyzer")
    
    # Standard pipeline options
    parser.add_argument('--skip_scraping', action='store_true', 
                        help='Skip the URL scraping step')
    parser.add_argument('--skip_collection', action='store_true', 
                        help='Skip the profile data collection step')
    parser.add_argument('--skip_analysis', action='store_true', 
                        help='Skip the profile analysis step')
    parser.add_argument('--url_file', type=str, 
                        default=str(config.DEFAULT_URLS_FILE),
                        help='Path to the file containing LinkedIn URLs')
    parser.add_argument('--profiles_file', type=str, 
                        default=str(config.DEFAULT_PROFILES_FILE),
                        help='Path to save/load the profiles dataset')
    parser.add_argument('--analysis_file', type=str, 
                        default=str(config.DEFAULT_ANALYSIS_FILE),
                        help='Path to save the analysis results')
    parser.add_argument('--model_path', type=str,
                        default=str(Path(config.MODELS_DIR) / "dutch_surname_classifier.pkl"),
                        help='Path to save/load the trained model')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining of the model even if it exists')
    
    # New name search options
    parser.add_argument('--search_name', type=str, 
                        help='Search for a specific person by name')
    parser.add_argument('--csv_input', type=str,
                        help='Process names from a CSV file')
    parser.add_argument('--first_name_col', type=str, default='First Name',
                        help='Column name for first names in CSV')
    parser.add_argument('--last_name_col', type=str, default='Last Name',
                        help='Column name for last names in CSV')
    
    # GDPR option (used by all modes)
    parser.add_argument('--no_gdpr', action='store_true',
                        help='Do not remove personal identifying information')
    
    return parser.parse_args()

def process_individual_search(name, gdpr_compliant=True):
    """Process individual name search without creating raw dataset."""
    from search_by_name import search_specific_person
    
    logger.info(f"Searching for individual: {name}")
    
    # Search for LinkedIn profile
    linkedin_url = search_specific_person(name)
    if not linkedin_url:
        logger.warning(f"No results found for {name}")
        return None
    
    # Create temporary URL file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(f"{linkedin_url}\n")
        temp_url_file = temp_file.name
    
    try:
        # Get profile data without saving raw file
        dataset = create_linkedin_dataset(
            url_file_path=temp_url_file,
            api_key=config.PROXYCURL_API_KEY,
            delay=0,
            output_file=None  # Don't save raw data
        )
        
        if dataset.empty:
            logger.warning(f"No profile data retrieved for {name}")
            return None
        
        # Get classifier and analyze
        classifier = get_or_train_classifier(
            model_path=str(Path(config.MODELS_DIR) / "dutch_surname_classifier.pkl"),
            training_data_path=str(config.SURNAME_DATASET_FILE)
        )
        
        analyzed_df = classify_surnames(dataset, classifier)
        
        # Apply GDPR compliance if requested
        if gdpr_compliant:
            columns_to_remove = ['full_name', 'city', 'surname', 'is_dutch_surname', 'index']
            for col in columns_to_remove:
                if col in analyzed_df.columns:
                    analyzed_df = analyzed_df.drop(columns=[col])
        
        # Save results
        output_dir = Path('data/individual_searches')
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{name.replace(' ', '_')}_analyzed.csv"
        analyzed_df.to_csv(output_file, index=False)
        
        logger.info(f"Individual search complete. Results saved to {output_file}")
        return output_file
        
    finally:
        # Clean up temporary file
        os.unlink(temp_url_file)

def process_csv_batch(csv_file, first_name_col, last_name_col, gdpr_compliant=True):
    """Process CSV batch without creating raw dataset."""
    from search_from_csv import process_csv_names
    
    logger.info(f"Processing names from CSV: {csv_file}")
    
    result = process_csv_names(
        csv_file=csv_file,
        output_dir='data/batch_searches',
        first_name_col=first_name_col,
        last_name_col=last_name_col,
        gdpr_compliant=gdpr_compliant
    )
    
    if result:
        logger.info(f"CSV processing complete. Results saved to {result}")
    else:
        logger.warning(f"No results found for names in {csv_file}")
    
    return result

def main():
    """Run the complete pipeline or specific name searches."""
    args = parse_args()
    
    logger.info("Starting TU/e LinkedIn Graduate Analyzer")
    
    # Handle specific name search if requested
    if args.search_name:
        process_individual_search(args.search_name, gdpr_compliant=not args.no_gdpr)
        return
    
    # Handle CSV batch processing if requested
    if args.csv_input:
        process_csv_batch(
            csv_file=args.csv_input,
            first_name_col=args.first_name_col,
            last_name_col=args.last_name_col,
            gdpr_compliant=not args.no_gdpr
        )
        return
    
    # If no specific search mode is selected, run the standard pipeline
    logger.info("Running standard pipeline")
    
    # Step 1: Scrape LinkedIn URLs
    if not args.skip_scraping:
        logger.info("Step 1: Scraping LinkedIn URLs")
        try:
            profiles = search_tue_graduates(
                api_key=config.SERPAPI_API_KEY,
                queries=config.LINKEDIN_SEARCH_QUERIES,
                output_file=args.url_file
            )
            logger.info(f"Found {len(profiles)} unique LinkedIn profiles")
        except Exception as e:
            logger.error(f"Error during URL scraping: {str(e)}")
            return
    else:
        logger.info("Skipping URL scraping step")
    
    # Step 2: Collect profile data
    if not args.skip_collection:
        logger.info("Step 2: Collecting profile data")
        try:
            dataset = create_linkedin_dataset(
                url_file_path=args.url_file,
                api_key=config.PROXYCURL_API_KEY,
                delay=config.API_REQUEST_DELAY,
                output_file=args.profiles_file  # Still save for standard pipeline
            )
            logger.info(f"Collected data for {len(dataset)} profiles")
        except Exception as e:
            logger.error(f"Error during profile data collection: {str(e)}")
            return
    else:
        logger.info("Skipping profile data collection step")
    
    # Step 3: Analyze profiles
    if not args.skip_analysis:
        logger.info("Step 3: Analyzing profiles")
        try:
            analysis_results = analyze_profiles(
                profiles_file=args.profiles_file,
                surname_dataset_file=config.SURNAME_DATASET_FILE,
                model_path=args.model_path,
                output_file=args.analysis_file,
                force_retrain=args.force_retrain,
                gdpr_compliant=not args.no_gdpr
            )
            logger.info(f"Analysis complete. Results saved to {args.analysis_file}")
        except Exception as e:
            logger.error(f"Error during profile analysis: {str(e)}")
            return
    else:
        logger.info("Skipping profile analysis step")
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()