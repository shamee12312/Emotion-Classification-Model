import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

class DataProcessor:
    def __init__(self):
        self.download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
            for item in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{item}')
                except LookupError:
                    try:
                        nltk.download(item, quiet=True)
                    except:
                        pass
        except:
            pass
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text: Input text string
        
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (keep the text part)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional - you might want to keep them)
        text = re.sub(r'\d+', '', text)
        
        # Keep emotion-related punctuation patterns and remove others
        # Preserve exclamation marks and question marks as they carry emotional weight
        text = re.sub(r'[^\w\s!?]', ' ', text)
        
        # Normalize multiple exclamation/question marks
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        
        Args:
            text: Input text string
        
        Returns:
            Processed text string
        """
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token.lower() not in self.stop_words and len(token) > 2:
                    try:
                        lemmatized = self.lemmatizer.lemmatize(token.lower())
                        processed_tokens.append(lemmatized)
                    except:
                        processed_tokens.append(token.lower())
            
            return ' '.join(processed_tokens)
        except:
            # Fallback if NLTK processing fails
            words = text.split()
            filtered_words = [word for word in words if len(word) > 2]
            return ' '.join(filtered_words)
    
    def preprocess_text(self, text):
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text string
        
        Returns:
            Fully processed text string
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and lemmatize
        processed_text = self.tokenize_and_lemmatize(cleaned_text)
        
        return processed_text
    
    def preprocess_data(self, df):
        """
        Preprocess entire dataframe
        
        Args:
            df: DataFrame with 'text' and 'emotion' columns
        
        Returns:
            DataFrame with preprocessed text
        """
        if 'text' not in df.columns or 'emotion' not in df.columns:
            raise ValueError("DataFrame must contain 'text' and 'emotion' columns")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Remove rows with missing values
        df_processed = df_processed.dropna(subset=['text', 'emotion'])
        
        # Remove empty strings
        df_processed = df_processed[df_processed['text'].str.strip() != '']
        
        # Preprocess text
        df_processed['text'] = df_processed['text'].apply(self.preprocess_text)
        
        # Remove rows where preprocessing resulted in empty strings
        df_processed = df_processed[df_processed['text'].str.strip() != '']
        
        # Normalize emotion labels (capitalize first letter)
        df_processed['emotion'] = df_processed['emotion'].str.title()
        
        return df_processed
    
    def get_text_statistics(self, df):
        """
        Get basic statistics about the text data
        
        Args:
            df: DataFrame with 'text' column
        
        Returns:
            Dictionary with text statistics
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain 'text' column")
        
        stats = {}
        
        # Text length statistics
        text_lengths = df['text'].str.len()
        stats['avg_length'] = text_lengths.mean()
        stats['median_length'] = text_lengths.median()
        stats['min_length'] = text_lengths.min()
        stats['max_length'] = text_lengths.max()
        
        # Word count statistics
        word_counts = df['text'].str.split().str.len()
        stats['avg_words'] = word_counts.mean()
        stats['median_words'] = word_counts.median()
        stats['min_words'] = word_counts.min()
        stats['max_words'] = word_counts.max()
        
        # Unique words
        all_words = ' '.join(df['text']).split()
        stats['unique_words'] = len(set(all_words))
        stats['total_words'] = len(all_words)
        
        return stats
    
    def get_emotion_distribution(self, df):
        """
        Get distribution of emotions in the dataset
        
        Args:
            df: DataFrame with 'emotion' column
        
        Returns:
            Dictionary with emotion counts and percentages
        """
        if 'emotion' not in df.columns:
            raise ValueError("DataFrame must contain 'emotion' column")
        
        emotion_counts = df['emotion'].value_counts()
        emotion_percentages = df['emotion'].value_counts(normalize=True) * 100
        
        distribution = {}
        for emotion in emotion_counts.index:
            distribution[emotion] = {
                'count': emotion_counts[emotion],
                'percentage': emotion_percentages[emotion]
            }
        
        return distribution
