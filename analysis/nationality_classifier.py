"""
Dutch surname nationality classifier.

This module provides functionality to classify surnames as Dutch or international
using machine learning on character-level features.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Optimal threshold determined through model evaluation
OPTIMAL_THRESHOLD = 0.5028

def extract_surname(full_name: Optional[str]) -> Optional[str]:
    """
    Extract surname from a full name, handling Dutch prefixes.
    
    Args:
        full_name: Full name to extract surname from
        
    Returns:
        Extracted surname or None if input is invalid
    """
    if pd.isna(full_name):
        return None
        
    name_parts = str(full_name).split()
    if not name_parts:
        return None
        
    # Dutch prefixes that are part of surnames
    dutch_prefixes = ['van', 'de', 'den', 'der', 'ten', 'ter', 'te', 'op', "'t", 'tot']
    
    # Check for compound Dutch surnames (with prefixes)
    for i in range(len(name_parts)-1, 0, -1):
        if name_parts[i-1].lower() in dutch_prefixes:
            # Return the prefix + surname as a compound surname
            return ' '.join(name_parts[i-1:]).lower()
    
    # If no prefix is found, return just the last part
    return name_parts[-1].lower()

def train_dutch_classifier(training_data_path: str, model_save_path: Optional[str] = None) -> Pipeline:
    """
    Train a classifier for Dutch surnames.
    
    Args:
        training_data_path: Path to the training dataset CSV
        model_save_path: Path to save the trained model (optional)
        
    Returns:
        Trained scikit-learn Pipeline classifier
    """
    logger.info("Training Dutch surname classifier...")
    
    # Load training data
    training_data = pd.read_csv(training_data_path)
    
    # Check if the Nationality column uses numeric values
    # If values are numeric (1, 0), convert to 'Dutch' and 'International'
    if 'Nationality' in training_data.columns:
        if training_data['Nationality'].dtype in [np.int64, np.float64, int, float]:
            logger.info("Converting numeric nationality values to string labels")
            training_data['NationalityLabel'] = training_data['Nationality'].map(
                {1: 'Dutch', 0: 'International', '1': 'Dutch', '0': 'International'}
            )
            # Handle any NaN or missing values
            if training_data['NationalityLabel'].isna().any():
                logger.warning(f"Found {training_data['NationalityLabel'].isna().sum()} rows with missing nationality")
                training_data = training_data.dropna(subset=['NationalityLabel'])
            
            nationality_column = 'NationalityLabel'
        else:
            nationality_column = 'Nationality'
    else:
        logger.error("Training data does not have a 'Nationality' column")
        raise ValueError("Training data must have a 'Nationality' column")
    
    # Feature extraction: Character n-grams (best performing approach)
    feature_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3)))
    ])

    # Classifier: Logistic Regression (best performing model)
    surname_classifier = Pipeline([
        ('features', feature_pipeline),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    # Train the model
    X = training_data['Surname'].str.lower()
    y = training_data[nationality_column]
    
    # Log the distribution of nationality labels
    logger.info(f"Nationality distribution in training data: {y.value_counts().to_dict()}")
    
    surname_classifier.fit(X, y)
    
    # Save the model if a save path is provided
    if model_save_path:
        save_dir = os.path.dirname(model_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(surname_classifier, f)
        logger.info(f"Model saved to {model_save_path}")
    
    logger.info("Model training complete")
    return surname_classifier

def load_dutch_classifier(model_path: str) -> Pipeline:
    """
    Load a previously trained classifier from a file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded scikit-learn Pipeline classifier
    """
    logger.info(f"Loading Dutch surname classifier from {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        logger.info("Model loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def get_or_train_classifier(model_path: str, training_data_path: str) -> Pipeline:
    """
    Get a trained classifier, either by loading from file or training a new one.
    
    Args:
        model_path: Path to the saved model file
        training_data_path: Path to the training dataset CSV (used if model doesn't exist)
        
    Returns:
        Scikit-learn Pipeline classifier
    """
    # Check if model file exists
    if os.path.exists(model_path):
        return load_dutch_classifier(model_path)
    else:
        # Train new model and save it
        logger.info(f"Model not found at {model_path}, training new model")
        return train_dutch_classifier(training_data_path, model_path)

def classify_surnames(
    df: pd.DataFrame,
    surname_classifier: Pipeline,
    threshold: float = OPTIMAL_THRESHOLD
) -> pd.DataFrame:
    """
    Classify surnames in a dataframe as Dutch or international.
    
    Args:
        df: DataFrame containing a 'full_name' column
        surname_classifier: Trained surname classifier
        threshold: Probability threshold for Dutch classification
        
    Returns:
        DataFrame with added nationality classification columns
    """
    logger.info("Extracting surnames from full names...")
    # Extract surnames
    if 'surname' not in df.columns:
        df['surname'] = df['full_name'].apply(extract_surname)

    # Apply the classifier
    logger.info("Classifying surnames...")
    valid_mask = df['surname'].notna()
    predictions = np.zeros(len(df))
    probabilities = np.zeros(len(df))

    if valid_mask.any():
        valid_surnames = df.loc[valid_mask, 'surname']
        # Get class names from the classifier
        class_names = surname_classifier.classes_
        dutch_index = np.where(class_names == 'Dutch')[0][0] if 'Dutch' in class_names else 0
        
        # Get probabilities
        valid_probabilities = surname_classifier.predict_proba(valid_surnames)[:, dutch_index]
        
        # Using provided threshold
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        
        predictions[valid_mask] = valid_predictions
        probabilities[valid_mask] = valid_probabilities

    # Add classification results to the dataframe
    df['is_dutch_surname'] = predictions.astype(int)
    df['dutch_probability'] = probabilities
    df['is_dutch'] = df['is_dutch_surname'].map({1: 'Dutch', 0: 'International'})
    
    logger.info("Surname classification complete")
    return df

def main():
    """Run the nationality classifier as a standalone script."""
    import argparse
    import sys
    
    # Add parent directory to path to allow relative imports when run as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from config import SURNAME_DATASET_FILE, MODELS_DIR
    
    parser = argparse.ArgumentParser(description="Dutch Surname Nationality Classifier")
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with full_name column')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file to save the results')
    parser.add_argument('--training_data', type=str, default=str(SURNAME_DATASET_FILE),
                        help='Training dataset for the classifier')
    parser.add_argument('--model_path', type=str, default=str(Path(MODELS_DIR) / "dutch_surname_classifier.pkl"),
                        help='Path to save/load the trained model')
    parser.add_argument('--threshold', type=float, default=OPTIMAL_THRESHOLD,
                        help='Probability threshold for Dutch classification')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining even if model exists')
    
    args = parser.parse_args()
    
    # Configure logging for standalone use
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load input data
    df = pd.read_csv(args.input)
    
    # Get or train classifier
    if args.force_retrain:
        logger.info("Forcing retraining of model")
        classifier = train_dutch_classifier(args.training_data, args.model_path)
    else:
        classifier = get_or_train_classifier(args.model_path, args.training_data)
    
    # Classify surnames
    result_df = classify_surnames(df, classifier, args.threshold)
    
    # Save results
    result_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()