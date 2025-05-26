"""
LinkedIn profile analyzer.

This module analyzes the LinkedIn profile dataset, combining nationality
classification with education and work experience analysis.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .nationality_classifier import get_or_train_classifier, classify_surnames

# Set up logging
logger = logging.getLogger(__name__)

def analyze_profiles(
    profiles_file: str,
    surname_dataset_file: str,
    model_path: str,
    output_file: Optional[str] = None,
    force_retrain: bool = False,
    gdpr_compliant: bool = True  # New parameter for GDPR compliance
) -> pd.DataFrame:
    """
    Analyze LinkedIn profiles, classifying nationality and extracting key metrics.
    
    Args:
        profiles_file: Path to CSV file with LinkedIn profile data
        surname_dataset_file: Path to CSV file with surname nationality training data
        model_path: Path to save/load the trained nationality classifier model
        output_file: Path to save analysis results
        force_retrain: Whether to force retraining of the model even if it exists
        gdpr_compliant: Whether to remove personal identifying information
        
    Returns:
        DataFrame with analysis results
    """
    logger.info(f"Loading profiles from {profiles_file}")
    profiles_df = pd.read_csv(profiles_file)
    
    # Get or train nationality classifier
    logger.info("Getting nationality classifier model")
    classifier = get_or_train_classifier(model_path, surname_dataset_file)
    
    # Classify nationalities
    logger.info("Classifying nationalities based on surnames")
    profiles_df = classify_surnames(profiles_df, classifier)
    
    # Additional analysis
    logger.info("Performing additional analysis")
    
    # Calculate the TU/e graduation years
    profiles_df['tue_bachelor_grad'] = profiles_df.apply(
        lambda row: row['bachelor_end_year'] if 'eindhoven university of technology' in str(row['bachelor_university']).lower() else None, 
        axis=1
    )
    
    profiles_df['tue_master_grad'] = profiles_df.apply(
        lambda row: row['master_end_year'] if 'eindhoven university of technology' in str(row['master_university']).lower() else None, 
        axis=1
    )
    
    # Determine if currently working at TU/e
    profiles_df['works_at_tue'] = profiles_df['latest_company'].apply(
        lambda x: 1 if x and any(name in str(x).lower() for name in ['eindhoven university', 'tu/e', 'tue']) else 0
    )
    
    # Determine current career status
    profiles_df['career_status'] = profiles_df.apply(
        lambda row: 'Student' if 'student' in str(row['latest_job_title']).lower() 
                    else 'Academic' if any(title in str(row['latest_job_title']).lower() 
                                           for title in ['professor', 'lecturer', 'researcher', 'phd'])
                    else 'Industry', 
        axis=1
    )
    
    # Remove personally identifiable information for GDPR compliance if requested
    if gdpr_compliant:
        logger.info("Removing personal identifiable information for GDPR compliance")
        columns_to_remove = [
            'full_name',    # Personal identifier
            'city',         # Location data
            'surname',      # Personal identifier
            'is_dutch_surname',  # Could be used to infer surname
            'index'         # Arbitrary numbering
        ]
        
        # Remove columns that exist in the DataFrame
        for col in columns_to_remove:
            if col in profiles_df.columns:
                profiles_df = profiles_df.drop(columns=[col])
    
    # Save results if output_file is provided
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        profiles_df.to_csv(output_file, index=False)
        logger.info(f"Analysis results saved to {output_file}")
    
    return profiles_df

def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics from the analyzed profiles.
    
    Args:
        df: DataFrame with analyzed profile data
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {}
    
    # Total profiles
    stats['total_profiles'] = len(df)
    
    # Nationality breakdown
    nationality_counts = df['is_dutch'].value_counts()
    stats['dutch_count'] = nationality_counts.get('Dutch', 0)
    stats['international_count'] = nationality_counts.get('International', 0)
    stats['dutch_percentage'] = round(stats['dutch_count'] / stats['total_profiles'] * 100, 1) if stats['total_profiles'] > 0 else 0
    
    # Education levels
    stats['bachelor_count'] = df['bachelor_degree'].notna().sum()
    stats['master_count'] = df['master_degree'].notna().sum()
    
    # Career paths
    career_counts = df['career_status'].value_counts() if 'career_status' in df.columns else {}
    stats['academic_count'] = career_counts.get('Academic', 0)
    stats['industry_count'] = career_counts.get('Industry', 0)
    stats['student_count'] = career_counts.get('Student', 0)
    
    # TU/e affiliation
    stats['currently_at_tue'] = df['works_at_tue'].sum() if 'works_at_tue' in df.columns else 0
    
    return stats

def main():
    """Run the profile analyzer as a standalone script."""
    import argparse
    import sys
    
    # Add parent directory to path to allow relative imports when run as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from config import DEFAULT_PROFILES_FILE, DEFAULT_ANALYSIS_FILE, SURNAME_DATASET_FILE, MODELS_DIR
    
    parser = argparse.ArgumentParser(description="LinkedIn Profile Analyzer")
    parser.add_argument('--profiles_file', type=str, default=str(DEFAULT_PROFILES_FILE),
                        help='CSV file with LinkedIn profile data')
    parser.add_argument('--surname_dataset', type=str, default=str(SURNAME_DATASET_FILE),
                        help='CSV file with surname nationality training data')
    parser.add_argument('--model_path', type=str, 
                        default=str(Path(MODELS_DIR) / "dutch_surname_classifier.pkl"),
                        help='Path to save/load the trained model')
    parser.add_argument('--output', type=str, default=str(DEFAULT_ANALYSIS_FILE),
                        help='Output CSV file to save analysis results')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics after analysis')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining of the model even if it exists')
    parser.add_argument('--no_gdpr', action='store_true',
                        help='Do not remove personal identifying information')
    
    args = parser.parse_args()
    
    # Configure logging for standalone use
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    analyzed_df = analyze_profiles(
        profiles_file=args.profiles_file,
        surname_dataset_file=args.surname_dataset,
        model_path=args.model_path,
        output_file=args.output,
        force_retrain=args.force_retrain,
        gdpr_compliant=not args.no_gdpr
    )
    
    # Print summary stats if requested
    if args.summary:
        stats = generate_summary_stats(analyzed_df)
        print("\n========== SUMMARY STATISTICS ==========")
        print(f"Total profiles analyzed: {stats['total_profiles']}")
        print(f"Dutch graduates: {stats['dutch_count']} ({stats['dutch_percentage']}%)")
        print(f"International graduates: {stats['international_count']}")
        print(f"Graduates with Bachelor's degree: {stats['bachelor_count']}")
        print(f"Graduates with Master's degree: {stats['master_count']}")
        print(f"Currently working in academia: {stats['academic_count']}")
        print(f"Currently working in industry: {stats['industry_count']}")
        print(f"Currently students: {stats['student_count']}")
        print(f"Currently affiliated with TU/e: {stats['currently_at_tue']}")
        print("========================================\n")

if __name__ == "__main__":
    main()