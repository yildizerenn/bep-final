#!/usr/bin/env python3
"""
Test script for the entire TU/e LinkedIn Graduate Analyzer pipeline.

This script tests the complete pipeline with a small number of LinkedIn profiles:
1. Uses a small subset of LinkedIn URLs
2. Collects profile data for these URLs
3. Analyzes the profiles and classifies nationality
4. Prints summary statistics

This is useful for verifying that the entire pipeline works end-to-end.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.link_scraper import search_tue_graduates
from scraper.profile_scraper import create_linkedin_dataset
from analysis.profile_analyzer import analyze_profiles, generate_summary_stats
from utils.data_utils import get_random_urls, save_urls_to_file
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline(
    url_count: int = 5,
    serpapi_key: str = None,
    proxycurl_key: str = None,
    test_dir: str = None,
    delay: int = 2
):
    """
    Test the complete pipeline with a small number of profiles.
    
    Args:
        url_count: Number of LinkedIn URLs to use for testing
        serpapi_key: SerpAPI key for Google search
        proxycurl_key: Proxycurl API key for LinkedIn data
        test_dir: Directory to store test outputs
        delay: Delay between API requests in seconds
    
    Returns:
        None
    """
    # Setup test directory
    if test_dir is None:
        test_dir = Path(config.DATA_DIR) / "pipeline_test"
    else:
        test_dir = Path(test_dir)
    
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Define test file paths
    test_urls_file = test_dir / "test_urls.txt"
    test_profiles_file = test_dir / "test_profiles.csv"
    test_analysis_file = test_dir / "test_analysis.csv"
    
    # Use default API keys if not provided
    if serpapi_key is None:
        serpapi_key = config.SERPAPI_API_KEY
    
    if proxycurl_key is None:
        proxycurl_key = config.PROXYCURL_API_KEY
    
    try:
        # Step 1: Get LinkedIn URLs (use only first query for testing)
        logger.info("Step 1: Getting LinkedIn URLs")
        test_query = config.LINKEDIN_SEARCH_QUERIES[0]
        profiles = search_tue_graduates(
            api_key=serpapi_key,
            queries=[test_query],
            output_file=str(test_urls_file)
        )
        
        if not profiles:
            logger.error("Failed to get LinkedIn URLs")
            return
        
        # Limit to specified count
        if len(profiles) > url_count:
            random_urls = get_random_urls(str(test_urls_file), url_count)
            save_urls_to_file(random_urls, str(test_urls_file))
            logger.info(f"Limited testing to {url_count} random URLs")
        
        # Step 2: Collect profile data
        logger.info("Step 2: Collecting profile data")
        dataset = create_linkedin_dataset(
            url_file_path=str(test_urls_file),
            api_key=proxycurl_key,
            delay=delay,
            output_file=str(test_profiles_file)
        )
        
        if dataset.empty:
            logger.error("Failed to collect profile data")
            return
        
        # Step 3: Analyze profiles
        logger.info("Step 3: Analyzing profiles")
        analyzed_df = analyze_profiles(
            profiles_file=str(test_profiles_file),
            surname_dataset_file=str(config.SURNAME_DATASET_FILE),
            output_file=str(test_analysis_file)
        )
        
        if analyzed_df.empty:
            logger.error("Failed to analyze profiles")
            return
        
        # Step 4: Print summary stats
        logger.info("Step 4: Generating summary statistics")
        stats = generate_summary_stats(analyzed_df)
        
        print("\n========== PIPELINE TEST RESULTS ==========")
        print(f"Total profiles analyzed: {stats['total_profiles']}")
        print(f"Dutch graduates: {stats['dutch_count']} ({stats['dutch_percentage']}%)")
        print(f"International graduates: {stats['international_count']}")
        print(f"Graduates with Bachelor's degree: {stats['bachelor_count']}")
        print(f"Graduates with Master's degree: {stats['master_count']}")
        print(f"Currently working in academia: {stats['academic_count']}")
        print(f"Currently working in industry: {stats['industry_count']}")
        print(f"Currently students: {stats['student_count']}")
        print(f"Currently affiliated with TU/e: {stats['currently_at_tue']}")
        print("=========================================\n")
        
        logger.info(f"Pipeline test completed successfully. Results saved in {test_dir}")
        
    except Exception as e:
        logger.error(f"Error during pipeline test: {str(e)}")

def main():
    """Run the test script with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the complete TU/e LinkedIn Graduate Analyzer pipeline"
    )
    parser.add_argument('--url_count', type=int, default=5,
                        help='Number of LinkedIn URLs to use for testing')
    parser.add_argument('--serpapi_key', type=str, default=config.SERPAPI_API_KEY,
                        help='SerpAPI key for Google search')
    parser.add_argument('--proxycurl_key', type=str, default=config.PROXYCURL_API_KEY,
                        help='Proxycurl API key for LinkedIn data')
    parser.add_argument('--test_dir', type=str, 
                        default=str(Path(config.DATA_DIR) / "pipeline_test"),
                        help='Directory to store test outputs')
    parser.add_argument('--delay', type=int, default=config.API_REQUEST_DELAY,
                        help='Delay between API requests in seconds')
    
    args = parser.parse_args()
    
    # Run the test
    test_pipeline(
        url_count=args.url_count,
        serpapi_key=args.serpapi_key,
        proxycurl_key=args.proxycurl_key,
        test_dir=args.test_dir,
        delay=args.delay
    )

if __name__ == "__main__":
    main()