#!/usr/bin/env python3
"""
Main script for the TU/e LinkedIn Graduate Analyzer pipeline.

This script orchestrates the entire process with robust error handling:
1. Scrapes LinkedIn URLs using Google Search (incremental saving)
2. Collects profile data using the Proxycurl API (incremental saving)
3. Analyzes profiles and classifies nationality based on surnames
4. Offers individual name search and CSV batch processing options
5. Automatically continues when API credits are exhausted

Usage:
    python main.py [--skip_scraping] [--skip_collection] [--skip_analysis]
    python main.py --search_name "John Smith" [--no_gdpr]
    python main.py --csv_input names.csv [--no_gdpr]
"""

import argparse
import logging
import sys
import os
import json
import tempfile
import pandas as pd
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

def robust_url_collection(api_key, queries, output_file):
    """
    Collect URLs with automatic continuation when API limits are reached.
    """
    logger.info("Starting robust URL collection...")
    
    # Check if we have existing URLs
    existing_urls = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_urls = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(existing_urls)} existing URLs")
    
    try:
        # Try to collect more URLs
        all_profiles = search_tue_graduates(
            api_key=api_key,
            queries=queries,
            output_file=output_file
        )
        logger.info(f"Successfully collected {len(all_profiles)} total URLs")
        return all_profiles
        
    except Exception as e:
        if "credit" in str(e).lower() or "limit" in str(e).lower() or "quota" in str(e).lower():
            logger.warning(f"SerpAPI credits/limits reached: {str(e)}")
            logger.info(f"Continuing with {len(existing_urls)} existing URLs...")
            return [{"url": url} for url in existing_urls]
        else:
            logger.error(f"Error during URL collection: {str(e)}")
            if existing_urls:
                logger.info(f"Using {len(existing_urls)} existing URLs due to error")
                return [{"url": url} for url in existing_urls]
            else:
                raise

def robust_profile_collection(url_file, api_key, delay):
    """
    Collect profiles incrementally with automatic continuation.
    """
    logger.info("Starting robust profile collection...")
    
    # Load URLs to process
    try:
        with open(url_file, 'r') as f:
            all_urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"URL file not found: {url_file}")
        return pd.DataFrame()
    
    if not all_urls:
        logger.error("No URLs found to process")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(all_urls)} URLs...")
    
    # Create temporary progress file
    progress_file = Path(config.DATA_DIR) / "profile_collection_progress.json"
    
    # Load existing progress
    processed_urls = set()
    all_profiles = []
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_urls = set(progress_data.get('processed_urls', []))
                logger.info(f"Resuming: {len(processed_urls)} URLs already processed")
        except:
            logger.warning("Could not load progress file, starting fresh")
    
    # Process URLs one by one
    for i, url in enumerate(all_urls):
        if url in processed_urls:
            continue
            
        try:
            logger.info(f"Processing {i+1}/{len(all_urls)}: {url}")
            
            # Create temporary file for single URL
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(f"{url}\n")
                temp_url_file = temp_file.name
            
            try:
                # Process single profile
                dataset = create_linkedin_dataset(
                    url_file_path=temp_url_file,
                    api_key=api_key,
                    delay=delay,
                    output_file=None  # Don't save raw data
                )
                
                if not dataset.empty:
                    all_profiles.append(dataset.iloc[0].to_dict())
                    logger.info(f"Successfully processed: {dataset.iloc[0].get('full_name', 'Unknown')}")
                else:
                    logger.info("Profile skipped (not a TU/e graduate)")
                
                # Mark as processed
                processed_urls.add(url)
                
                # Save progress
                progress_data = {'processed_urls': list(processed_urls)}
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f)
                    
            finally:
                # Clean up temp file
                os.unlink(temp_url_file)
                
        except Exception as e:
            if "credit" in str(e).lower() or "limit" in str(e).lower() or "quota" in str(e).lower():
                logger.warning(f"Proxycurl credits/limits reached: {str(e)}")
                logger.info(f"Successfully processed {len(all_profiles)} profiles before limit")
                break
            else:
                logger.warning(f"Error processing {url}: {str(e)}")
                continue
    
    # Create final DataFrame
    if all_profiles:
        final_df = pd.DataFrame(all_profiles)
        logger.info(f"Profile collection complete: {len(final_df)} profiles collected")
        
        # Clean up progress file
        if progress_file.exists():
            progress_file.unlink()
            
        return final_df
    else:
        logger.warning("No profiles were successfully collected")
        return pd.DataFrame()

def process_individual_search(name, gdpr_compliant=True):
    """Process individual name search without creating raw dataset."""
    from search_by_name import search_specific_person
    
    logger.info(f"Searching for individual: {name}")
    
    # Check if result already exists
    output_dir = Path('data/individual_searches')
    output_file = output_dir / f"{name.replace(' ', '_')}_analyzed.csv"
    
    if output_file.exists():
        logger.info(f"Result already exists: {output_file}")
        return output_file
    
    # Search for LinkedIn profile
    try:
        linkedin_url = search_specific_person(name)
        if not linkedin_url:
            logger.warning(f"No results found for {name}")
            return None
    except Exception as e:
        if "credit" in str(e).lower() or "limit" in str(e).lower():
            logger.warning(f"SerpAPI credits exhausted while searching for {name}")
            return None
        else:
            raise
    
    # Create temporary URL file
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
        output_dir.mkdir(exist_ok=True, parents=True)
        analyzed_df.to_csv(output_file, index=False)
        
        logger.info(f"Individual search complete. Results saved to {output_file}")
        return output_file
        
    except Exception as e:
        if "credit" in str(e).lower() or "limit" in str(e).lower():
            logger.warning(f"Proxycurl credits exhausted while processing {name}")
            return None
        else:
            raise
    finally:
        # Clean up temporary file
        os.unlink(temp_url_file)

def process_csv_batch(csv_file, first_name_col, last_name_col, gdpr_compliant=True):
    """Process CSV batch without creating raw dataset."""
    from search_from_csv import process_csv_names
    
    logger.info(f"Processing names from CSV: {csv_file}")
    
    try:
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
        
    except Exception as e:
        if "credit" in str(e).lower() or "limit" in str(e).lower():
            logger.warning("API credits exhausted during CSV processing")
            logger.info("Some names may have been processed. Check output directory for partial results.")
            return None
        else:
            raise

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
    
    # Standard pipeline with robust error handling
    logger.info("Running standard pipeline with robust error handling")
    
    collected_profiles = []
    
    # Step 1: Robust URL Collection
    if not args.skip_scraping:
        logger.info("Step 1: Robust URL scraping")
        try:
            profiles = robust_url_collection(
                api_key=config.SERPAPI_API_KEY,
                queries=config.LINKEDIN_SEARCH_QUERIES,
                output_file=args.url_file
            )
            logger.info(f"URL collection complete: {len(profiles)} URLs available")
            
            if len(profiles) < 5:  # Minimum threshold
                logger.warning("Very few URLs collected. Consider checking API keys or search queries.")
                
        except Exception as e:
            logger.error(f"Error during URL scraping: {str(e)}")
            # Check if we have existing URLs to continue with
            if os.path.exists(args.url_file):
                with open(args.url_file, 'r') as f:
                    existing_urls = [line.strip() for line in f if line.strip()]
                if existing_urls:
                    logger.info(f"Continuing with {len(existing_urls)} existing URLs")
                else:
                    logger.error("No URLs available to process")
                    return
            else:
                logger.error("No URLs available to process")
                return
    else:
        logger.info("Skipping URL scraping step")
    
    # Step 2: Robust Profile Collection
    if not args.skip_collection:
        logger.info("Step 2: Robust profile data collection")
        try:
            dataset = robust_profile_collection(
                url_file=args.url_file,
                api_key=config.PROXYCURL_API_KEY,
                delay=config.API_REQUEST_DELAY
            )
            
            if dataset.empty:
                logger.error("No profile data collected")
                return
            else:
                logger.info(f"Profile collection complete: {len(dataset)} profiles collected")
                collected_profiles = dataset
                
        except Exception as e:
            logger.error(f"Error during profile data collection: {str(e)}")
            return
    else:
        logger.info("Skipping profile data collection step")
        # Try to load existing analysis file for analysis step
        if os.path.exists(args.analysis_file):
            try:
                collected_profiles = pd.read_csv(args.analysis_file)
                logger.info(f"Loaded {len(collected_profiles)} profiles from existing analysis file")
            except:
                logger.error("Could not load existing profiles for analysis")
                return
        else:
            logger.error("No existing profiles found for analysis")
            return
    
    # Step 3: Profile Analysis
    if not args.skip_analysis and not collected_profiles.empty:
        logger.info("Step 3: Profile analysis")
        try:
            # Get or train classifier
            classifier = get_or_train_classifier(
                model_path=args.model_path,
                training_data_path=str(config.SURNAME_DATASET_FILE)
            )
            
            # Classify surnames
            analyzed_df = classify_surnames(collected_profiles, classifier)
            
            # Apply GDPR compliance if requested
            if not args.no_gdpr:
                logger.info("Applying GDPR compliance (removing personal identifying information)")
                columns_to_remove = ['full_name', 'city', 'surname', 'is_dutch_surname', 'index']
                for col in columns_to_remove:
                    if col in analyzed_df.columns:
                        analyzed_df = analyzed_df.drop(columns=[col])
            
            # Save final analysis
            Path(args.analysis_file).parent.mkdir(exist_ok=True, parents=True)
            analyzed_df.to_csv(args.analysis_file, index=False)
            
            logger.info(f"Analysis complete. Results saved to {args.analysis_file}")
            
            # Print summary statistics
            dutch_count = (analyzed_df['is_dutch'] == 'Dutch').sum()
            international_count = (analyzed_df['is_dutch'] == 'International').sum()
            
            logger.info("=== ANALYSIS SUMMARY ===")
            logger.info(f"Total profiles analyzed: {len(analyzed_df)}")
            logger.info(f"Dutch graduates: {dutch_count}")
            logger.info(f"International graduates: {international_count}")
            logger.info("========================")
            
        except Exception as e:
            logger.error(f"Error during profile analysis: {str(e)}")
            return
    else:
        logger.info("Skipping profile analysis step")
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()