import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import time
from data_processor import DataProcessor

class EmotionClassifier:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.data_processor = DataProcessor()
        self.X_test = None
        self.y_test = None
        self.emotion_labels = None
        
    def train_models(self, df, test_size=0.2, random_state=42, max_features=5000, 
                    ngram_range=(1, 2), models_to_train=None):
        """
        Train multiple emotion classification models
        
        Args:
            df: DataFrame with 'text' and 'emotion' columns
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            max_features: Maximum features for TF-IDF
            ngram_range: N-gram range for TF-IDF
            models_to_train: List of model names to train
        
        Returns:
            Dictionary with training results for each model
        """
        if models_to_train is None:
            models_to_train = ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression"]
        
        # Preprocess data
        df_processed = self.data_processor.preprocess_data(df.copy())
        
        # Prepare features and labels
        X = df_processed['text']
        y = df_processed['emotion']
        
        # Store emotion labels
        self.emotion_labels = sorted(y.unique())
        
        # Ensure minimum samples per class for proper training
        emotion_counts = y.value_counts()
        min_samples = emotion_counts.min()
        
        if min_samples < 4:
            raise ValueError(f"Not enough samples for training. Minimum samples per emotion: {min_samples}. Need at least 4 samples per emotion.")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Ensure we have at least one sample per class in both train and test
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        
        if train_counts.min() < 1 or test_counts.min() < 1:
            # Adjust test_size if needed
            test_size = max(0.1, min(0.3, test_size))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize TF-IDF vectorizer with improved parameters
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        # Vectorize training data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Define models with improved parameters
        model_configs = {
            "Naive Bayes": MultinomialNB(alpha=0.1),
            "SVM": SVC(kernel='linear', C=1.0, probability=True, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=random_state),
            "Logistic Regression": LogisticRegression(C=1.0, random_state=random_state, max_iter=2000, solver='liblinear')
        }
        
        results = {}
        
        # Train selected models
        for model_name in models_to_train:
            if model_name in model_configs:
                print(f"Training {model_name}...")
                
                start_time = time.time()
                
                # Train model
                model = model_configs[model_name]
                model.fit(X_train_vectorized, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_vectorized)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time
                
                # Store model and results
                self.models[model_name] = model
                results[model_name] = {
                    'accuracy': accuracy,
                    'training_time': training_time
                }
                
                print(f"{model_name} - Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")
        
        return results
    
    def predict_single(self, text, model_name):
        """
        Predict emotion for a single text
        
        Args:
            text: Input text string
            model_name: Name of the model to use
        
        Returns:
            Tuple of (predicted_emotion, probabilities_dict)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        if self.vectorizer is None:
            raise ValueError("Models not trained yet. Please train models first.")
        
        # Preprocess text
        processed_text = self.data_processor.preprocess_text(text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Get model
        model = self.models[model_name]
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            prob_dict = dict(zip(self.emotion_labels, probabilities))
        else:
            # For models without probability prediction, create dummy probabilities
            prob_dict = {emotion: 0.0 for emotion in self.emotion_labels}
            prob_dict[prediction] = 1.0
        
        return prediction, prob_dict
    
    def predict_batch(self, texts, model_name):
        """
        Predict emotions for a batch of texts
        
        Args:
            texts: List of text strings
            model_name: Name of the model to use
        
        Returns:
            List of predicted emotions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        if self.vectorizer is None:
            raise ValueError("Models not trained yet. Please train models first.")
        
        # Preprocess texts
        processed_texts = [self.data_processor.preprocess_text(text) for text in texts]
        
        # Vectorize
        texts_vectorized = self.vectorizer.transform(processed_texts)
        
        # Get model and predict
        model = self.models[model_name]
        predictions = model.predict(texts_vectorized)
        
        return predictions.tolist()
    
    def get_model_info(self):
        """Get information about trained models"""
        if not self.models:
            return "No models trained yet."
        
        info = {}
        for model_name, model in self.models.items():
            info[model_name] = {
                'type': type(model).__name__,
                'parameters': model.get_params()
            }
        
        return info
    
    def save_models(self, filepath):
        """Save trained models to file"""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'emotion_labels': self.emotion_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_models(self, filepath):
        """Load trained models from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.emotion_labels = model_data['emotion_labels']
