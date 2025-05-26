"""
LinkedIn URL scraper using Google Search.

This module uses SerpAPI to search Google for LinkedIn profiles of TU/e graduates.
"""

import os
import logging
from typing import List, Dict, Set, Optional
from serpapi import GoogleSearch
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def search_tue_graduates(
    api_key: str = None,
    queries: List[str] = None,
    output_file: str = "tue_graduate_urls.txt"
) -> List[Dict]:
    """
    Search for LinkedIn profiles of TU/e graduates using Google Search via SerpAPI.
    
    Args:
        api_key: SerpAPI API key
        queries: List of search queries to use
        output_file: Path to save the URLs to
    
    Returns:
        List of dictionaries containing profile information
    """
    # Get API key from environment variable if not provided
    if api_key is None:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("No SerpAPI API key provided. Set SERPAPI_API_KEY environment variable.")
    
    # Use default queries if none provided
    if queries is None:
        queries = [
            'site:linkedin.com/in "Eindhoven University of Technology" "Bachelor"',
            'site:linkedin.com/in "Eindhoven University of Technology" "Master"',
            'site:linkedin.com/in "Eindhoven University of Technology" "BSc"',
            'site:linkedin.com/in "Eindhoven University of Technology" "MSc"',
        ]
    
    # Track all profiles
    all_profiles = []
    seen_urls = set()  # To avoid duplicates
    
    for query in queries:
        logger.info(f"Searching with query: {query}")
        start = 0
        has_more_results = True
        
        while has_more_results:
            params = {
                "engine": "google",
                "q": query,
                "num": 100,  # Maximum number of results per page
                "start": start,  # Pagination parameter
                "api_key": api_key
            }
            
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # Get the organic results
                organic_results = results.get("organic_results", [])
                
                if not organic_results:
                    # No more results to process
                    logger.info("  No more results found for this query.")
                    has_more_results = False
                    break
                    
                # Process results
                new_profiles = 0
                for result in organic_results:
                    url = result["link"]
                    
                    # Add any new LinkedIn profile URL
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_profiles.append({
                            "url": url,
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", "")
                        })
                        new_profiles += 1
                
                logger.info(f"  Found {len(organic_results)} results, {new_profiles} new profiles")
                
                # Check if there are more pages
                if "serpapi_pagination" in results and "next" in results["serpapi_pagination"]:
                    # Move to the next page
                    start += len(organic_results)
                else:
                    has_more_results = False
                    
            except Exception as e:
                logger.error(f"Error performing search: {str(e)}")
                has_more_results = False
    
    # Save all URLs to the output file
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for profile in all_profiles:
            f.write(f"{profile['url']}\n")
    
    logger.info(f"Found {len(all_profiles)} unique LinkedIn profiles of TU/e graduates")
    logger.info(f"All URLs saved to {output_file}")
    
    return all_profiles

def main():
    """Run the LinkedIn URL scraper as a standalone script."""
    import argparse
    import sys
    
    # Add parent directory to path to allow relative imports when run as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from config import SERPAPI_API_KEY, LINKEDIN_SEARCH_QUERIES, DEFAULT_URLS_FILE
    
    parser = argparse.ArgumentParser(description="LinkedIn URL Scraper for TU/e Graduates")
    parser.add_argument('--api_key', type=str, default=SERPAPI_API_KEY,
                        help='SerpAPI API key')
    parser.add_argument('--output', type=str, default=str(DEFAULT_URLS_FILE),
                        help='Output file to save LinkedIn URLs')
    
    args = parser.parse_args()
    
    # Configure logging for standalone use
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the scraper
    search_tue_graduates(
        api_key=args.api_key,
        queries=LINKEDIN_SEARCH_QUERIES,
        output_file=args.output
    )

if __name__ == "__main__":
    main()