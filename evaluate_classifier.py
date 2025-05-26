#!/usr/bin/env python3
"""
Evaluation script for the Dutch surname nationality classifier.

This script evaluates the performance of the nationality classifier using
cross-validation and generates metrics and visualizations for reporting.

Usage:
    python evaluate_classifier.py --dataset sample_data/final_surname_dataset.csv
"""

import argparse
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.nationality_classifier import extract_surname, OPTIMAL_THRESHOLD
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_classifier(dataset_path, output_dir=None, cv_folds=5):
    """
    Evaluate the Dutch surname nationality classifier using cross-validation.
    
    Args:
        dataset_path: Path to the surname dataset CSV
        output_dir: Directory to save evaluation results and plots
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Loading surname dataset from {dataset_path}")
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} surnames from dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None
    
    # Check if required columns exist
    if 'Surname' not in df.columns or 'Nationality' not in df.columns:
        logger.error("Dataset must contain 'Surname' and 'Nationality' columns")
        return None
    
    # Convert numeric nationality to labels if needed
    if df['Nationality'].dtype in [np.int64, np.float64, int, float]:
        logger.info("Converting numeric nationality values to string labels")
        df['NationalityLabel'] = df['Nationality'].map({
            1: 'Dutch', 0: 'International', 
            '1': 'Dutch', '0': 'International'
        })
        nationality_column = 'NationalityLabel'
    else:
        nationality_column = 'Nationality'
    
    # Prepare data
    X = df['Surname'].str.lower()
    y = df[nationality_column]
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize classifier pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    
    # Perform cross-validation
    logger.info(f"Performing {cv_folds}-fold cross-validation")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=cv_folds), scoring='accuracy')
    
    # Train model on full training set
    pipeline.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    # Determine which column corresponds to 'Dutch'
    dutch_index = np.where(pipeline.classes_ == 'Dutch')[0][0]
    y_prob_dutch = y_prob[:, dutch_index]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create evaluation metrics dictionary
    metrics = {
        'accuracy': {
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'scores': cv_scores
        },
        'classification_report': report,
        'confusion_matrix': cm,
        'threshold': OPTIMAL_THRESHOLD
    }
    
    # Print results
    logger.info(f"Cross-validation accuracy: {metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}")
    logger.info(f"Test set accuracy: {report['accuracy']:.4f}")
    logger.info(f"Dutch precision: {report['Dutch']['precision']:.4f}")
    logger.info(f"Dutch recall: {report['Dutch']['recall']:.4f}")
    logger.info(f"Dutch F1-score: {report['Dutch']['f1-score']:.4f}")
    logger.info(f"Sample size: {len(df)} surnames")
    logger.info(f"Dutch samples: {sum(y == 'Dutch')} ({sum(y == 'Dutch')/len(y)*100:.1f}%)")
    logger.info(f"International samples: {sum(y == 'International')} ({sum(y == 'International')/len(y)*100:.1f}%)")
    
    # Generate plots if output directory is specified
    if output_dir:
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png')
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test == 'Dutch', y_prob_dutch)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curve.png')
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test == 'Dutch', y_prob_dutch)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.axhline(y=sum(y_test == 'Dutch') / len(y_test), color='red', linestyle='--', 
                    label=f'Baseline (Dutch ratio = {sum(y_test == "Dutch") / len(y_test):.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_path / 'precision_recall_curve.png')
        
        # Feature importance
        if hasattr(pipeline.named_steps['classifier'], 'coef_'):
            # Get feature names and coefficients
            vectorizer = pipeline.named_steps['vectorizer']
            coefficients = pipeline.named_steps['classifier'].coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(coefficients)
            }).sort_values('Importance', ascending=False)
            
            # Save top 20 features
            top_features = feature_importance.head(20)
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title('Top 20 Important Features (Character N-grams)')
            plt.tight_layout()
            plt.savefig(output_path / 'feature_importance.png')
            
            # Save feature importance to CSV
            feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)
        
        # Save metrics to text file
        with open(output_path / 'evaluation_metrics.txt', 'w') as f:
            f.write(f"Model Evaluation Metrics\n")
            f.write(f"=======================\n\n")
            f.write(f"Cross-validation accuracy: {metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}\n")
            f.write(f"Test set accuracy: {report['accuracy']:.4f}\n\n")
            f.write(f"Classification Report:\n")
            f.write(f"Dutch precision: {report['Dutch']['precision']:.4f}\n")
            f.write(f"Dutch recall: {report['Dutch']['recall']:.4f}\n")
            f.write(f"Dutch F1-score: {report['Dutch']['f1-score']:.4f}\n\n")
            f.write(f"International precision: {report['International']['precision']:.4f}\n")
            f.write(f"International recall: {report['International']['recall']:.4f}\n")
            f.write(f"International F1-score: {report['International']['f1-score']:.4f}\n\n")
            f.write(f"Dataset Information:\n")
            f.write(f"Sample size: {len(df)} surnames\n")
            f.write(f"Dutch samples: {sum(y == 'Dutch')} ({sum(y == 'Dutch')/len(y)*100:.1f}%)\n")
            f.write(f"International samples: {sum(y == 'International')} ({sum(y == 'International')/len(y)*100:.1f}%)\n")
    
    return metrics

def main():
    """Run the evaluation as a standalone script."""
    parser = argparse.ArgumentParser(description="Evaluate Dutch surname nationality classifier")
    parser.add_argument('--dataset', type=str, default=str(Path(config.SURNAME_DATASET_FILE)),
                      help='Path to surname dataset CSV')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--cv_folds', type=int, default=5,
                      help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    evaluate_classifier(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        cv_folds=args.cv_folds
    )

if __name__ == "__main__":
    main()