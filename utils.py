import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

def save_results_to_csv(results, filename=None):
    """
    Save evaluation results to CSV file
    
    Args:
        results: Dictionary with evaluation results
        filename: Output filename (optional)
    
    Returns:
        Filename of saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_classification_results_{timestamp}.csv"
    
    # Convert results to DataFrame
    data = []
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1_Score': metrics.get('f1_score', 0)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    return filename

def load_custom_dataset(filepath):
    """
    Load custom dataset from various file formats
    
    Args:
        filepath: Path to the dataset file
    
    Returns:
        DataFrame with text and emotion columns
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(filepath)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif file_extension == '.json':
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Validate required columns
        if 'text' not in df.columns or 'emotion' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'emotion' columns")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def validate_dataset(df):
    """
    Validate dataset format and content
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check required columns
    if 'text' not in df.columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Missing 'text' column")
    
    if 'emotion' not in df.columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Missing 'emotion' column")
    
    if not validation_results['is_valid']:
        return validation_results
    
    # Check for empty or null values
    null_texts = df['text'].isnull().sum()
    null_emotions = df['emotion'].isnull().sum()
    
    if null_texts > 0:
        validation_results['warnings'].append(f"{null_texts} null values in 'text' column")
    
    if null_emotions > 0:
        validation_results['warnings'].append(f"{null_emotions} null values in 'emotion' column")
    
    # Check for empty strings
    empty_texts = (df['text'].str.strip() == '').sum()
    if empty_texts > 0:
        validation_results['warnings'].append(f"{empty_texts} empty text strings")
    
    # Basic statistics
    validation_results['statistics'] = {
        'total_samples': len(df),
        'unique_emotions': df['emotion'].nunique(),
        'emotion_distribution': df['emotion'].value_counts().to_dict(),
        'avg_text_length': df['text'].str.len().mean(),
        'min_samples_per_emotion': df['emotion'].value_counts().min()
    }
    
    # Check for class imbalance
    min_samples = validation_results['statistics']['min_samples_per_emotion']
    max_samples = df['emotion'].value_counts().max()
    
    if max_samples / min_samples > 10:  # Significant imbalance
        validation_results['warnings'].append(
            f"Significant class imbalance detected (ratio: {max_samples/min_samples:.1f}:1)"
        )
    
    return validation_results

def generate_sample_predictions():
    """
    Generate sample predictions for demonstration
    
    Returns:
        List of sample texts with expected emotions
    """
    samples = [
        {"text": "I'm so excited about this new opportunity!", "expected": "Happy"},
        {"text": "This is really frustrating and annoying", "expected": "Angry"},
        {"text": "I feel so sad and lonely today", "expected": "Sad"},
        {"text": "The meeting is scheduled for tomorrow", "expected": "Neutral"},
        {"text": "I'm terrified of what might happen", "expected": "Fear"},
        {"text": "Wow, I can't believe this happened!", "expected": "Surprise"}
    ]
    
    return samples

def export_model_summary(classifier, filename=None):
    """
    Export model summary and configuration
    
    Args:
        classifier: Trained EmotionClassifier instance
        filename: Output filename (optional)
    
    Returns:
        Dictionary with model summary
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_summary_{timestamp}.json"
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': list(classifier.models.keys()),
        'emotion_labels': classifier.emotion_labels,
        'vectorizer_params': classifier.vectorizer.get_params() if classifier.vectorizer else None,
        'model_info': classifier.get_model_info()
    }
    
    # Save to JSON file
    import json
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary

def calculate_confidence_threshold(probabilities, threshold=0.8):
    """
    Determine if prediction confidence is above threshold
    
    Args:
        probabilities: Dictionary of emotion probabilities
        threshold: Confidence threshold (0-1)
    
    Returns:
        Boolean indicating if prediction is confident enough
    """
    max_prob = max(probabilities.values())
    return max_prob >= threshold

def get_emotion_color_map():
    """
    Get color mapping for emotions for consistent visualization
    
    Returns:
        Dictionary mapping emotions to colors
    """
    color_map = {
        'Happy': '#FFD700',      # Gold
        'Sad': '#4169E1',        # Royal Blue
        'Angry': '#FF4500',      # Orange Red
        'Neutral': '#808080',    # Gray
        'Fear': '#800080',       # Purple
        'Surprise': '#FF69B4'    # Hot Pink
    }
    
    return color_map

def format_metrics_for_display(metrics):
    """
    Format metrics dictionary for better display
    
    Args:
        metrics: Dictionary with metric values
    
    Returns:
        Formatted string representation
    """
    formatted = []
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{metric.title()}: {value:.4f}")
        else:
            formatted.append(f"{metric.title()}: {value}")
    
    return "\n".join(formatted)

def check_system_requirements():
    """
    Check if all required packages are available
    
    Returns:
        Dictionary with availability status
    """
    requirements = {
        'pandas': False,
        'numpy': False,
        'sklearn': False,
        'nltk': False,
        'streamlit': False,
        'plotly': False
    }
    
    try:
        import pandas
        requirements['pandas'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        requirements['numpy'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        requirements['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import nltk
        requirements['nltk'] = True
    except ImportError:
        pass
    
    try:
        import streamlit
        requirements['streamlit'] = True
    except ImportError:
        pass
    
    try:
        import plotly
        requirements['plotly'] = True
    except ImportError:
        pass
    
    return requirements
