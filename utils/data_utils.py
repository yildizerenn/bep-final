"""
Data utility functions for the TU/e LinkedIn Graduate Analyzer.

This module contains helper functions for data loading, processing, and cleaning.
"""

import pandas as pd
import random
import logging
from pathlib import Path
from typing import List, Optional

# Set up logging
logger = logging.getLogger(__name__)

def load_profile_data(file_path: str) -> pd.DataFrame:
    """
    Load profile data from a CSV file with error handling.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with profile data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} profiles from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return pd.DataFrame()

def save_profile_data(df: pd.DataFrame, file_path: str) -> bool:
    """
    Save profile data to a CSV file with error handling.
    
    Args:
        df: DataFrame with profile data
        file_path: Path to save the CSV file
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        output_path = Path(file_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved {len(df)} profiles to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        return False

def get_random_urls(url_file: str, count: int = 10) -> List[str]:
    """
    Get a random sample of URLs from a file.
    
    Args:
        url_file: Path to the file containing URLs
        count: Number of random URLs to return
        
    Returns:
        List of randomly selected URLs
    """
    try:
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        if count >= len(urls):
            logger.warning(f"Requested {count} URLs but only {len(urls)} are available")
            return urls
        
        random_urls = random.sample(urls, count)
        logger.info(f"Selected {len(random_urls)} random URLs from {url_file}")
        return random_urls
    except Exception as e:
        logger.error(f"Error getting random URLs from {url_file}: {str(e)}")
        return []

def save_urls_to_file(urls: List[str], output_file: str) -> bool:
    """
    Save a list of URLs to a file, one URL per line.
    
    Args:
        urls: List of URLs to save
        output_file: Path to save the URLs
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_file, 'w') as f:
            for url in urls:
                f.write(f"{url}\n")
        
        logger.info(f"Successfully saved {len(urls)} URLs to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving URLs to {output_file}: {str(e)}")
        return False

def clean_profile_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize profile data.
    
    Args:
        df: DataFrame with profile data
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Handle missing values for specific columns
    for col in ['full_name', 'country', 'city']:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna('Unknown')
    
    # Standardize date formats
    for col in ['bachelor_end_year', 'master_end_year']:
        if col in clean_df.columns:
            # Convert to numeric, coerce errors to NaN
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    # Fix 'Present' string in job dates
    if 'latest_job_end_date' in clean_df.columns:
        clean_df['is_current_job'] = clean_df['latest_job_end_date'].apply(
            lambda x: 1 if x == 'Present' else 0
        )
    
    logger.info(f"Cleaned {len(clean_df)} profile records")
    return clean_df