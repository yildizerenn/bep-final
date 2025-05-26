"""
Data processing utilities for LinkedIn profile data.
Handles data cleaning, validation, and transformation.
"""
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_dataset(df):
    """
    Clean and prepare the dataset for analysis.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    logger.info("Cleaning dataset...")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Convert year columns to numeric
    year_columns = [
        'bachelor_end_year', 
        'master_end_year'
    ]
    
    for col in year_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    # Fill missing values with appropriate placeholders
    result_df = result_df.fillna({
        'bachelor_degree': 'None',
        'bachelor_field': 'None',
        'bachelor_university': 'None',
        'master_degree': 'None',
        'master_field': 'None',
        'master_university': 'None',
        'latest_job_title': 'None',
        'latest_company': 'None',
        'latest_job_start_date': 'None',
        'latest_job_end_date': 'None'
    })
    
    # Create education_level column based on degrees
    result_df['education_level'] = 'Unknown'
    
    # Bachelor only
    bachelor_mask = (
        (result_df['bachelor_degree'] != 'None') & 
        (result_df['master_degree'] == 'None')
    )
    result_df.loc[bachelor_mask, 'education_level'] = 'Bachelor'
    
    # Master (with or without Bachelor)
    master_mask = result_df['master_degree'] != 'None'
    result_df.loc[master_mask, 'education_level'] = 'Master'
    
    # Check for incomplete records (missing crucial information)
    incomplete_mask = (
        (result_df['full_name'].isna()) |
        (
            (result_df['bachelor_degree'] == 'None') & 
            (result_df['master_degree'] == 'None')
        )
    )
    
    if incomplete_mask.any():
        logger.warning(f"Found {incomplete_mask.sum()} incomplete records")
    
    # Add any additional cleaning steps here
    
    logger.info(f"Dataset cleaning complete. Final shape: {result_df.shape}")
    return result_df

def generate_analysis_metrics(df):
    """
    Generate analysis metrics from the processed dataset.
    
    Args:
        df (pandas.DataFrame): Processed DataFrame
        
    Returns:
        dict: Dictionary of analysis metrics
    """
    logger.info("Generating analysis metrics...")
    
    metrics = {}
    
    # Total number of graduates
    metrics['total_graduates'] = len(df)
    
    # Education level counts
    education_counts = df['education_level'].value_counts().to_dict()
    metrics['education_counts'] = education_counts
    
    # Nationality distribution
    if 'is_dutch' in df.columns:
        nationality_counts = df['is_dutch'].value_counts().to_dict()
        metrics['nationality_counts'] = nationality_counts
    
    # Most common fields of study for Bachelor's
    if 'bachelor_field' in df.columns:
        bachelor_fields = df[df['bachelor_field'] != 'None']['bachelor_field'].value_counts().head(10).to_dict()
        metrics['top_bachelor_fields'] = bachelor_fields
    
    # Most common fields of study for Master's
    if 'master_field' in df.columns:
        master_fields = df[df['master_field'] != 'None']['master_field'].value_counts().head(10).to_dict()
        metrics['top_master_fields'] = master_fields
    
    # Most common companies
    if 'latest_company' in df.columns:
        companies = df[df['latest_company'] != 'None']['latest_company'].value_counts().head(10).to_dict()
        metrics['top_companies'] = companies
    
    # Graduation year distribution
    if 'bachelor_end_year' in df.columns:
        year_counts = df[df['bachelor_end_year'].notna()]['bachelor_end_year'].astype(int).value_counts().sort_index().to_dict()
        metrics['graduation_year_counts'] = year_counts
    
    logger.info("Analysis metrics generation complete")
    return metrics

def run_data_processor(input_path, output_path):
    """
    Main function to run the data processor
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save the processed CSV file
        
    Returns:
        pandas.DataFrame: Processed DataFrame
        dict: Analysis metrics
    """
    logger.info("Starting data processor...")
    
    try:
        # Load the dataset
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} profiles from {input_path}")
        
        # Clean the dataset
        processed_df = clean_dataset(df)
        
        # Generate analysis metrics
        metrics = generate_analysis_metrics(processed_df)
        
        # Save the processed dataset
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        # Print key metrics
        logger.info(f"Total graduates: {metrics['total_graduates']}")
        if 'nationality_counts' in metrics:
            dutch_count = metrics['nationality_counts'].get('Dutch', 0)
            international_count = metrics['nationality_counts'].get('International', 0)
            logger.info(f"Dutch graduates: {dutch_count}, International graduates: {international_count}")
        
        return processed_df, metrics
    
    except Exception as e:
        logger.error(f"Error in data processor: {e}")
        return None, None

if __name__ == "__main__":
    from config import FINAL_DATA_PATH, PROCESSED_DATA_DIR
    import os
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "tue_graduates_processed.csv")
    run_data_processor(FINAL_DATA_PATH, output_path)