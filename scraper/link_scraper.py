"""
LinkedIn URL scraper using Google Search.
Optimized for reduced credit consumption and improved accuracy.

This module uses SerpAPI to search Google for LinkedIn profiles of TU/e graduates
with smart filtering and flexible matching.
"""

import os
import logging
import re
import unicodedata
from typing import List, Dict, Set, Optional
from serpapi import GoogleSearch
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def normalize_name(name):
    """
    Normalize name by removing accents, special characters, and converting to lowercase.
    Handles international characters like Turkish ö, ğ, etc.
    
    Args:
        name: Name to normalize
        
    Returns:
        Normalized name string
    """
    if not name:
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove accents and special characters
    name = unicodedata.normalize('NFD', name)
    name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
    
    # Replace common character mappings
    replacements = {
        'ö': 'o', 'ü': 'u', 'ğ': 'g', 'ç': 'c', 'ş': 's', 'ı': 'i',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ø': 'o', 'ű': 'u'
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove extra spaces and non-alphanumeric characters except spaces and hyphens
    name = re.sub(r'[^a-z0-9\s\-]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def extract_name_from_linkedin_url(url):
    """
    Extract name from LinkedIn URL.
    
    Args:
        url: LinkedIn profile URL
        
    Returns:
        Extracted name or None
    """
    if not url or 'linkedin.com/in/' not in url:
        return None
    
    try:
        # Extract the username part after /in/
        path_part = url.split('linkedin.com/in/')[-1]
        # Remove query parameters and fragments
        username = path_part.split('?')[0].split('#')[0].split('/')[0]
        
        # Convert LinkedIn username to readable name
        # Remove trailing numbers and common suffixes
        username = re.sub(r'-?\d+[a-z]*$', '', username)
        username = re.sub(r'-(jr|sr|phd|md|ii|iii)$', '', username)
        
        # Replace hyphens with spaces
        name = username.replace('-', ' ').strip()
        
        return normalize_name(name) if name else None
        
    except Exception as e:
        logger.debug(f"Error extracting name from URL {url}: {str(e)}")
        return None

def flexible_name_match(search_name, profile_title, profile_snippet, profile_url):
    """
    Flexible name matching with multiple strategies.
    
    Args:
        search_name: Name being searched for
        profile_title: LinkedIn profile title from search results
        profile_snippet: Profile snippet from search results  
        profile_url: LinkedIn profile URL
        
    Returns:
        Tuple of (is_match, confidence_score, match_method)
    """
    if not search_name:
        return False, 0, "no_search_name"
    
    # Normalize search name
    normalized_search = normalize_name(search_name)
    search_parts = normalized_search.split()
    
    # Strategy 1: Exact name in title or snippet
    title_norm = normalize_name(profile_title or "")
    snippet_norm = normalize_name(profile_snippet or "")
    
    if normalized_search in title_norm or normalized_search in snippet_norm:
        return True, 95, "exact_match_text"
    
    # Strategy 2: All name parts present in title or snippet
    combined_text = f"{title_norm} {snippet_norm}"
    if len(search_parts) >= 2 and all(part in combined_text for part in search_parts):
        return True, 85, "all_parts_text"
    
    # Strategy 3: Name extracted from LinkedIn URL
    url_name = extract_name_from_linkedin_url(profile_url)
    if url_name:
        url_parts = url_name.split()
        
        # Exact match with URL name
        if normalized_search == url_name:
            return True, 90, "exact_match_url"
        
        # All search name parts in URL name
        if len(search_parts) >= 2 and all(part in url_parts for part in search_parts):
            return True, 80, "all_parts_url"
        
        # At least first and last name match
        if len(search_parts) >= 2 and len(url_parts) >= 2:
            if search_parts[0] in url_parts and search_parts[-1] in url_parts:
                return True, 75, "first_last_url"
    
    # Strategy 4: Partial match - at least 2 name parts in any text
    if len(search_parts) >= 2:
        matches_in_text = sum(1 for part in search_parts if part in combined_text)
        if matches_in_text >= 2:
            confidence = min(70, matches_in_text * 25)
            return True, confidence, "partial_match"
    
    return False, 0, "no_match"

def flexible_tue_detection(profile_title, profile_snippet, profile_url):
    """
    Flexible TU/e connection detection.
    
    Args:
        profile_title: LinkedIn profile title
        profile_snippet: Profile snippet  
        profile_url: LinkedIn profile URL
        
    Returns:
        Tuple of (has_tue_connection, confidence_score, detection_method)
    """
    # TU/e keywords (expanded)
    tue_keywords = [
        "eindhoven university of technology",
        "eindhoven university", 
        "tu/e", "tue", "tu eindhoven",
        "technische universiteit eindhoven",
        "eindhoven tech"
    ]
    
    # Combine all available text
    combined_text = normalize_name(f"{profile_title or ''} {profile_snippet or ''}")
    
    # Strategy 1: Direct keyword match
    for keyword in tue_keywords:
        if keyword in combined_text:
            return True, 95, f"keyword_{keyword}"
    
    # Strategy 2: Partial university name
    if "eindhoven" in combined_text and "university" in combined_text:
        return True, 85, "partial_university"
    
    if "eindhoven" in combined_text and ("tech" in combined_text or "technisch" in combined_text):
        return True, 80, "partial_tech"
    
    # Strategy 3: Just "eindhoven" (lower confidence)
    if "eindhoven" in combined_text:
        return True, 60, "eindhoven_only"
    
    # Strategy 4: Netherlands-based LinkedIn profile (very low confidence)
    if "netherlands" in combined_text or "nl.linkedin.com" in (profile_url or ""):
        return True, 30, "netherlands_hint"
    
    return False, 0, "no_tue_connection"

def search_tue_graduates(
    api_key: str = None,
    queries: List[str] = None,
    output_file: str = "tue_graduate_urls.txt",
    max_results: int = 10
) -> List[Dict]:
    """
    Search for LinkedIn profiles of TU/e graduates using optimized Google Search.
    
    Args:
        api_key: SerpAPI API key
        queries: List of search queries to use
        output_file: Path to save the URLs to
        max_results: Maximum results per query (credit saving)
    
    Returns:
        List of dictionaries containing profile information
    """
    # Get API key from environment variable if not provided
    if api_key is None:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("No SerpAPI API key provided. Set SERPAPI_API_KEY environment variable.")
    
    # Use optimized default queries if none provided
    if queries is None:
        queries = [
            'site:linkedin.com/in ("Eindhoven University of Technology" OR "TU/e" OR "Technische Universiteit Eindhoven" OR "TU Eindhoven")'
        ]
    
    # Track all profiles
    all_profiles = []
    seen_urls = set()  # To avoid duplicates
    
    logger.info(f"Starting optimized search with {len(queries)} queries, max {max_results} results each")
    
    for query_idx, query in enumerate(queries):
        logger.info(f"Query {query_idx + 1}/{len(queries)}: {query}")
        
        params = {
            "engine": "google",
            "q": query,
            "num": max_results,  # Limit results to save credits
            "start": 0,
            "api_key": api_key
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Check for API errors
            if 'error' in results:
                error_msg = results.get('error', 'Unknown error')
                logger.error(f"SerpAPI error: {error_msg}")
                if any(keyword in error_msg.lower() for keyword in ['credit', 'limit', 'quota']):
                    logger.warning("API credits exhausted")
                    break
                continue
            
            # Get the organic results
            organic_results = results.get("organic_results", [])
            
            if not organic_results:
                logger.info("  No results found for this query")
                continue
                
            # Process results with smart filtering
            new_profiles = 0
            for result in organic_results:
                url = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                # Skip if not a LinkedIn profile URL
                if not url or 'linkedin.com/in/' not in url:
                    continue
                
                # Skip duplicates
                if url in seen_urls:
                    continue
                
                # For general search (not name-specific), we accept all LinkedIn profiles
                # TU/e verification will happen during profile scraping
                seen_urls.add(url)
                all_profiles.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "query_used": query
                })
                new_profiles += 1
                
                logger.debug(f"  Added: {title[:50]}...")
            
            logger.info(f"  Found {len(organic_results)} results, added {new_profiles} new profiles")
            
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ['credit', 'limit', 'quota']):
                logger.warning(f"SerpAPI credits exhausted: {str(e)}")
                break
            else:
                logger.error(f"Error performing search: {str(e)}")
                continue
    
    # Save all URLs to the output file
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for profile in all_profiles:
            f.write(f"{profile['url']}\n")
    
    logger.info(f"Found {len(all_profiles)} unique LinkedIn profiles")
    logger.info(f"All URLs saved to {output_file}")
    
    return all_profiles

def search_specific_person_optimized(name, api_key=None, max_results=5):
    """
    Optimized search for a specific person with flexible matching.
    
    Args:
        name: Full name of the person to search
        api_key: SerpAPI API key
        max_results: Maximum results to check (credit saving)
    
    Returns:
        LinkedIn profile URL or None if not found
    """
    if api_key is None:
        api_key = os.environ.get("SERPAPI_API_KEY")
    
    # Split name into parts for flexible searching
    name_parts = name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[-1] if len(name_parts) > 1 else ""
    
    # Optimized search strategies (fewer queries to save credits)
    search_queries = [
        # Strategy 1: Name with TU/e (most specific)
        f'"{name}" site:linkedin.com/in ("Eindhoven University" OR "TU/e")',
        
        # Strategy 2: First and last name with TU/e (broader)
        f'"{first_name}" "{last_name}" site:linkedin.com/in "Eindhoven"',
        
        # Strategy 3: Just the name on LinkedIn (broadest, for backup)
        f'"{name}" site:linkedin.com/in'
    ]
    
    for i, query in enumerate(search_queries):
        logger.info(f"Strategy {i+1}: {query}")
        
        params = {
            "engine": "google",
            "q": query,
            "num": max_results,  # Limit to save credits
            "api_key": api_key
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Check for API errors
            if 'error' in results:
                error_msg = results.get('error', 'Unknown error')
                if any(keyword in error_msg.lower() for keyword in ['credit', 'limit', 'quota']):
                    logger.warning(f"SerpAPI credits exhausted: {error_msg}")
                    raise Exception(f"API limit reached: {error_msg}")
                else:
                    logger.error(f"API error: {error_msg}")
                    continue
            
            organic_results = results.get("organic_results", [])
            
            for result in organic_results:
                url = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                # Verify it's a LinkedIn profile URL
                if not url or "linkedin.com/in/" not in url:
                    continue
                
                # Flexible name matching
                name_match, name_confidence, name_method = flexible_name_match(
                    name, title, snippet, url
                )
                
                # Flexible TU/e detection
                tue_match, tue_confidence, tue_method = flexible_tue_detection(
                    title, snippet, url
                )
                
                # Scoring system
                total_score = 0
                if name_match:
                    total_score += name_confidence
                if tue_match:
                    total_score += tue_confidence * 0.5  # TU/e is less critical for individual search
                
                # Accept if good enough match
                threshold = 80 if i < 2 else 70  # Lower threshold for broader searches
                
                if name_match and total_score >= threshold:
                    logger.info(f"✅ Found match: {url}")
                    logger.info(f"   Name: {name_method} (confidence: {name_confidence}%)")
                    logger.info(f"   TU/e: {tue_method} (confidence: {tue_confidence}%)")
                    logger.info(f"   Total score: {total_score:.1f}")
                    return url
                else:
                    logger.debug(f"❌ Rejected: {url} (score: {total_score:.1f}, threshold: {threshold})")
                    logger.debug(f"   Name: {name_method} ({name_confidence}%), TU/e: {tue_method} ({tue_confidence}%)")
        
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ['credit', 'limit', 'quota']):
                raise  # Re-raise API limit errors
            else:
                logger.error(f"Error with search strategy {i+1}: {str(e)}")
                continue
    
    logger.warning(f"No suitable LinkedIn profile found for '{name}'")
    return None

def main():
    """Run the LinkedIn URL scraper as a standalone script."""
    import argparse
    import sys
    
    # Add parent directory to path to allow relative imports when run as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    import config
    
    parser = argparse.ArgumentParser(description="Optimized LinkedIn URL Scraper for TU/e Graduates")
    parser.add_argument('--api_key', type=str, default=config.SERPAPI_API_KEY,
                        help='SerpAPI API key')
    parser.add_argument('--output', type=str, default=str(config.DEFAULT_URLS_FILE),
                        help='Output file to save LinkedIn URLs')
    parser.add_argument('--max_results', type=int, default=config.MAX_RESULTS_PER_QUERY,
                        help='Maximum results per query (credit saving)')
    
    args = parser.parse_args()
    
    # Configure logging for standalone use
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the optimized scraper
    search_tue_graduates(
        api_key=args.api_key,
        queries=config.LINKEDIN_SEARCH_QUERIES,
        output_file=args.output,
        max_results=args.max_results
    )

if __name__ == "__main__":
    main()