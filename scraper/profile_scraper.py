"""
LinkedIn profile data scraper using Proxycurl API.

This module collects detailed profile data from LinkedIn using the Proxycurl API.
"""

import requests
import pandas as pd
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

def extract_profile_info(profile_data: Dict) -> Optional[Dict]:
    """
    Extract relevant information from a LinkedIn profile.
    
    Args:
        profile_data: Raw profile data from Proxycurl API
        
    Returns:
        Dictionary with extracted profile information or None if not a TU/e graduate
    """
    # First check if the person studied at Eindhoven University of Technology
    studied_at_tue = False
    tue_variations = [
        "eindhoven university of technology", 
        "tu/e", 
        "technische universiteit eindhoven",
        "tu eindhoven"
    ]
    
    if 'education' in profile_data and profile_data['education']:
        for edu in profile_data['education']:
            school = edu.get('school', '').lower()
            for variation in tue_variations:
                if variation in school:
                    studied_at_tue = True
                    break
            if studied_at_tue:
                break
    
    # If the person didn't study at TU/e, return None to skip this profile
    if not studied_at_tue:
        return None
    
    # Initialize dictionary with personal information
    profile_info = {
        # Personal information
        'full_name': profile_data.get('full_name', None),
        'country': profile_data.get('country_full_name', None),
        'city': profile_data.get('city', None),
        
        # Initialize education fields
        'bachelor_degree': None,
        'bachelor_field': None, 
        'bachelor_end_year': None,
        'bachelor_university': None,  
        'master_degree': None,
        'master_field': None,
        'master_end_year': None,
        'master_university': None,  
        
        # Initialize latest experience fields
        'latest_job_title': None,
        'latest_company': None,
        'latest_job_start_date': None,
        'latest_job_end_date': None
    }
    
    # Helper function to check if a school is TU/e
    def is_tue(school_name):
        school_lower = school_name.lower()
        for variation in tue_variations:
            if variation in school_lower:
                return True
        return False
    
    # Continue with extraction only for TU/e graduates
    # Extract education information
    if 'education' in profile_data and profile_data['education']:
        # Separate TU/e and non-TU/e education
        tue_education = []
        other_education = []
        
        for edu in profile_data['education']:
            if is_tue(edu.get('school', '')):
                tue_education.append(edu)
            else:
                other_education.append(edu)
        
        # Process TU/e education first, then other education
        all_education = tue_education + other_education
        
        for edu in all_education:
            degree_name = edu.get('degree_name', '')
            
            # Skip if no degree name
            if not degree_name:
                continue
            
            degree_lower = degree_name.lower()
            
            # Check if it's a bachelor's degree
            if not profile_info['bachelor_degree'] and (
                re.search(r'bachelor|bsc|b\.sc|b\s*sc|bs\b|b\.s\.|b\s+s\b', degree_lower) or 
                (degree_lower.startswith('b') and 'applied science' in degree_lower) or
                'undergraduate' in degree_lower
            ):
                profile_info['bachelor_degree'] = degree_name
                profile_info['bachelor_field'] = edu.get('field_of_study', 'N/A')
                profile_info['bachelor_university'] = edu.get('school', 'N/A')
                if edu.get('ends_at') and edu['ends_at'].get('year'):
                    profile_info['bachelor_end_year'] = int(edu['ends_at']['year'])
            
            # Check if it's a master's degree or graduate degree
            elif not profile_info['master_degree'] and (
                re.search(r'master|msc|m\.sc|m\s*sc|ms\b|m\.s\.|m\s+s\b|mba|m\.b\.a', degree_lower) or 
                re.search(r'graduate|postgraduate|post-graduate', degree_lower) or
                ('data science' in degree_lower and 'bachelor' not in degree_lower) or
                'erasmus mundus' in degree_lower.lower()  # Common joint master program
            ):
                profile_info['master_degree'] = degree_name
                profile_info['master_field'] = edu.get('field_of_study', 'N/A')
                profile_info['master_university'] = edu.get('school', 'N/A')
                if edu.get('ends_at') and edu['ends_at'].get('year'):
                    profile_info['master_end_year'] = int(edu['ends_at']['year'])
    
    # Extract latest experience information
    if 'experiences' in profile_data and profile_data['experiences'] and len(profile_data['experiences']) > 0:
        latest_exp = profile_data['experiences'][0]
        
        profile_info['latest_job_title'] = latest_exp.get('title')
        profile_info['latest_company'] = latest_exp.get('company')
        
        # Format start date
        start_month = latest_exp.get('starts_at', {}).get('month')
        start_year = latest_exp.get('starts_at', {}).get('year')
        if start_month and start_year:
            profile_info['latest_job_start_date'] = f"{start_month}/{start_year}"
        elif start_year:
            profile_info['latest_job_start_date'] = f"{start_year}"
        
        # Format end date (if exists)
        end_info = latest_exp.get('ends_at')
        if end_info:
            end_month = end_info.get('month')
            end_year = end_info.get('year')
            if end_month and end_year:
                profile_info['latest_job_end_date'] = f"{end_month}/{end_year}"
            elif end_year:
                profile_info['latest_job_end_date'] = f"{end_year}"
        else:
            profile_info['latest_job_end_date'] = "Present"
    
    return profile_info

def create_linkedin_dataset(
    url_file_path: str,
    api_key: str,
    delay: int = 1,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a dataset from LinkedIn profiles listed in a text file
    
    Args:
        url_file_path: Path to text file containing LinkedIn URLs
        api_key: Proxycurl API key
        delay: Delay between API calls to avoid rate limiting
        output_file: Path to save the dataset as a CSV file (optional)
    
    Returns:
        pandas.DataFrame: Dataset with extracted profile information
    """
    # Initialize API endpoint and headers
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    headers = {'Authorization': 'Bearer ' + api_key}
    
    # Initialize empty list to store profile data
    all_profiles = []
    
    # Read LinkedIn URLs from file
    with open(url_file_path, 'r') as file:
        linkedin_urls = [line.strip() for line in file if line.strip()]
    
    logger.info(f"Processing {len(linkedin_urls)} LinkedIn URLs")
    
    # Process each URL
    for i, url in enumerate(linkedin_urls):
        try:
            logger.info(f"Processing {i+1}/{len(linkedin_urls)}: {url}")
            
            # Make API request
            response = requests.get(
                api_endpoint,
                params={'url': url, 'skills': 'include'},
                headers=headers
            )
            
            # Check if request was successful
            if response.status_code == 200:
                profile_data = response.json()
                
                # Extract profile information
                profile_info = extract_profile_info(profile_data)
                
                # Check if profile is a TU/e graduate
                if profile_info is not None:
                    # Add profile URL and index to the info
                    profile_info['linkedin_url'] = url
                    profile_info['index'] = i + 1
                    
                    # Append to list of profiles
                    all_profiles.append(profile_info)
                    logger.info(f"Successfully processed profile: {profile_info.get('full_name')}")
                else:
                    logger.info(f"Skipping profile: Not a TU/e graduate")
            else:
                logger.error(f"Error processing URL: {url}")
                logger.error(f"Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # Add delay to avoid rate limiting
            if i < len(linkedin_urls) - 1:  # No need to delay after the last request
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
    
    # Create DataFrame from all profiles
    if all_profiles:
        df = pd.DataFrame(all_profiles)
        
        # Save to CSV if output file is specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Dataset saved to {output_file}")
        
        return df
    else:
        logger.warning("No profiles were successfully processed.")
        return pd.DataFrame()

def main():
    """Run the LinkedIn profile scraper as a standalone script."""
    import argparse
    import sys
    
    # Add parent directory to path to allow relative imports when run as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from config import PROXYCURL_API_KEY, API_REQUEST_DELAY, DEFAULT_URLS_FILE, DEFAULT_PROFILES_FILE
    
    parser = argparse.ArgumentParser(description="LinkedIn Profile Scraper")
    parser.add_argument('--url_file', type=str, default=str(DEFAULT_URLS_FILE),
                        help='Text file containing LinkedIn URLs')
    parser.add_argument('--api_key', type=str, default=PROXYCURL_API_KEY,
                        help='Proxycurl API key')
    parser.add_argument('--delay', type=int, default=API_REQUEST_DELAY,
                        help='Delay between API requests in seconds')
    parser.add_argument('--output', type=str, default=str(DEFAULT_PROFILES_FILE),
                        help='Output CSV file to save the dataset')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit the number of profiles to process (0 for all)')
    
    args = parser.parse_args()
    
    # Configure logging for standalone use
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Limit the number of URLs to process if specified
    linkedin_urls = []
    with open(args.url_file, 'r') as file:
        linkedin_urls = [line.strip() for line in file if line.strip()]
    
    if args.limit > 0 and args.limit < len(linkedin_urls):
        logger.info(f"Limiting to {args.limit} URLs out of {len(linkedin_urls)}")
        linkedin_urls = linkedin_urls[:args.limit]
    
    # Write the limited URLs to a temporary file
    if args.limit > 0:
        temp_file = args.url_file + ".temp"
        with open(temp_file, 'w') as file:
            for url in linkedin_urls:
                file.write(f"{url}\n")
        url_file_to_use = temp_file
    else:
        url_file_to_use = args.url_file
    
    # Run the scraper
    create_linkedin_dataset(
        url_file_path=url_file_to_use,
        api_key=args.api_key,
        delay=args.delay,
        output_file=args.output
    )
    
    # Clean up temporary file if created
    if args.limit > 0:
        Path(temp_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()