"""
Test script for the LinkedIn profile scraper with random profiles,
including nationality analysis based on surnames.

This script selects random LinkedIn profile URLs, processes them,
and then performs nationality analysis on the resulting profiles.
Modified to skip creating raw dataset.
"""

import os
import sys
import logging
import random
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.profile_scraper import create_linkedin_dataset
from analysis.nationality_classifier import get_or_train_classifier, classify_surnames
from utils.data_utils import get_random_urls, save_urls_to_file
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_with_random_profiles(
    url_file: str,
    final_output_file: str,
    surname_dataset_file: str,
    model_path: str,
    count: int = 10,
    api_key: str = None,
    delay: int = 2,
    force_retrain: bool = False,
    gdpr_compliant: bool = True
):
    """
    Test the profile scraper with random LinkedIn profiles and analyze nationality.
    Modified to skip creating raw dataset.
    
    Args:
        url_file: Path to the file containing all LinkedIn URLs
        final_output_file: Path to save the final analyzed profiles
        surname_dataset_file: Path to the surname nationality training dataset
        model_path: Path to save/load the trained model
        count: Number of random profiles to test
        api_key: Proxycurl API key
        delay: Delay between API requests in seconds
        force_retrain: Whether to force model retraining
        gdpr_compliant: Whether to remove personal identifying information
    
    Returns:
        None
    """
    logger.info(f"Testing with {count} random LinkedIn profiles")
    
    # Get random URLs
    random_urls = get_random_urls(url_file, count)
    if not random_urls:
        logger.error("Failed to get random URLs")
        return
    
    # Save random URLs to a temporary file
    temp_url_file = str(Path(url_file).with_suffix(".random.txt"))
    if not save_urls_to_file(random_urls, temp_url_file):
        logger.error("Failed to save random URLs to temporary file")
        return
    
    # Use default API key if not provided
    if api_key is None:
        api_key = config.PROXYCURL_API_KEY
    
    try:
        # Run the profile scraper on the random URLs (without saving raw data)
        logger.info(f"Running profile scraper on {len(random_urls)} random URLs")
        dataset = create_linkedin_dataset(
            url_file_path=temp_url_file,
            api_key=api_key,
            delay=delay,
            output_file=None  # Don't save raw data
        )
        
        # Check results from scraping
        if dataset.empty:
            logger.warning("No profiles were successfully processed")
            return
        else:
            logger.info(f"Successfully processed {len(dataset)} out of {len(random_urls)} profiles")
            
            # Print the names of the profiles that were successfully processed
            if 'full_name' in dataset.columns:
                names = dataset['full_name'].tolist()
                logger.info(f"Successfully processed profiles: {', '.join(names)}")
        
        # Now perform nationality analysis on the scraped profiles
        logger.info("Performing nationality analysis on the profiles")
        
        # Get or train the Dutch nationality classifier
        if force_retrain:
            logger.info("Forcing retraining of model")
            from analysis.nationality_classifier import train_dutch_classifier
            classifier = train_dutch_classifier(surname_dataset_file, model_path)
        else:
            classifier = get_or_train_classifier(
                model_path=model_path,
                training_data_path=surname_dataset_file
            )
        
        # Classify the surnames
        analyzed_df = classify_surnames(dataset, classifier)
        
        # GDPR COMPLIANCE - Remove personal identifying information if requested
        if gdpr_compliant:
            logger.info("Removing personal identifying information for GDPR compliance")
            columns_to_remove = [
                'full_name',        # Personal identifier
                'city',             # Location data
                'surname',          # Personal identifier
                'is_dutch_surname', # Could be used to infer surname
                'index'             # Arbitrary numbering
            ]
            
            # Remove columns that exist in the DataFrame
            existing_columns = [col for col in columns_to_remove if col in analyzed_df.columns]
            if existing_columns:
                logger.info(f"Removing columns: {', '.join(existing_columns)}")
                analyzed_df = analyzed_df.drop(columns=existing_columns)
                logger.info(f"Remaining columns: {', '.join(analyzed_df.columns)}")
            else:
                logger.info("No columns to remove for GDPR compliance")
        
        # Save the final analyzed dataset only
        analyzed_df.to_csv(final_output_file, index=False)
        logger.info(f"Analysis complete. Results saved to {final_output_file}")
        
        # Print nationality classification results
        dutch_count = (analyzed_df['is_dutch'] == 'Dutch').sum()
        international_count = (analyzed_df['is_dutch'] == 'International').sum()
        logger.info(f"Nationality classification results:")
        logger.info(f"  Dutch: {dutch_count}")
        logger.info(f"  International: {international_count}")
        
        # Print summary statistics
        logger.info("\n========== TEST SUMMARY ==========")
        logger.info(f"Total profiles tested: {count}")
        logger.info(f"Successfully processed: {len(analyzed_df)}")
        logger.info(f"Dutch graduates: {dutch_count}")
        logger.info(f"International graduates: {international_count}")
        logger.info("==================================")
            
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
    finally:
        # Clean up temporary file
        try:
            os.remove(temp_url_file)
            logger.info(f"Cleaned up temporary file: {temp_url_file}")
        except:
            logger.warning(f"Could not remove temporary file: {temp_url_file}")

def main():
    """Run the test script with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test LinkedIn profile scraper with random profiles and perform nationality analysis"
    )
    parser.add_argument('--url_file', type=str, default=str(config.DEFAULT_URLS_FILE),
                        help='Text file containing LinkedIn URLs')
    parser.add_argument('--output', type=str, 
                        default=str(Path(config.DATA_DIR) / "test_profiles_analyzed.csv"),
                        help='Output CSV file to save the analyzed results')
    parser.add_argument('--surname_dataset', type=str, default=str(config.SURNAME_DATASET_FILE),
                        help='Training dataset for surname nationality classification')
    parser.add_argument('--model_path', type=str, 
                        default=str(Path(config.MODELS_DIR) / "dutch_surname_classifier.pkl"),
                        help='Path to save/load the trained model')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of random profiles to test')
    parser.add_argument('--api_key', type=str, default=config.PROXYCURL_API_KEY,
                        help='Proxycurl API key')
    parser.add_argument('--delay', type=int, default=config.API_REQUEST_DELAY,
                        help='Delay between API requests in seconds')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining of the model even if it exists')
    parser.add_argument('--no_gdpr', action='store_true',
                        help='Do not remove personal identifying information')
    
    args = parser.parse_args()
    
    # Run the test
    test_with_random_profiles(
        url_file=args.url_file,
        final_output_file=args.output,
        surname_dataset_file=args.surname_dataset,
        model_path=args.model_path,
        count=args.count,
        api_key=args.api_key,
        delay=args.delay,
        force_retrain=args.force_retrain,
        gdpr_compliant=not args.no_gdpr
    )

if __name__ == "__main__":
    main()