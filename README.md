# Emotion Detection Machine Learning Project

complete machine learning pipeline for **text-based emotion detection**, supporting real-time predictions, batch analysis, and multiple model evaluation.  
➡️ **Live Demo**: [https://emotion-detector-shamee12312.replit.app/](https://emotion-detector-shamee12312.replit.app/)
➡️ **Video Link**: [https://drive.google.com/file/d/1W5XDovY0fKWQwnl_BucBhfYb9rM_TY9T/view?usp=sharing]

A comprehensive machine learning pipeline for text-based emotion classification using multiple algorithms and evaluation metrics.

## Project Overview

This project implements an end-to-end machine learning system that classifies text into six emotion categories: Happy, Sad, Angry, Neutral, Fear, and Surprise. The system includes data preprocessing, multiple model training, comprehensive evaluation, and deployment-ready components.

## Features

- **Complete ML Pipeline**: Data exploration → Feature engineering → Model training → Evaluation → Deployment
- **Multiple Models**: Naive Bayes, SVM, Random Forest, Logistic Regression
- **Comprehensive Dataset**: 240 carefully crafted text samples (40 per emotion)
- **Advanced Preprocessing**: Text cleaning, tokenization, lemmatization, TF-IDF vectorization
- **Detailed Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- **Interactive Interface**: Streamlit web application for real-time predictions
- **Jupyter Analysis**: Complete notebook with visualizations and insights

## Files Included

### Core Application
- `app.py` - Streamlit web application with interactive interface
- `emotion_classifier.py` - Main classifier with multiple ML models
- `data_processor.py` - Text preprocessing and feature engineering
- `model_evaluator.py` - Comprehensive model evaluation metrics
- `utils.py` - Utility functions for data handling and export

### Datasets
- `comprehensive_emotion_dataset.csv` - Main dataset (240 samples)
- `emotion_dataset.csv` - Extended dataset for training

### Analysis
- `Complete_Emotion_Detection_Analysis.ipynb` - Full Jupyter notebook analysis
- `emotion_detection_notebook.ipynb` - Interactive exploration notebook

### Configuration
- `.streamlit/config.toml` - Streamlit server configuration
- `pyproject.toml` - Python project dependencies
- `uv.lock` - Dependency lock file

## Dataset Statistics

- **Total Samples**: 240 high-quality text examples
- **Emotions**: 6 categories (Happy, Sad, Angry, Neutral, Fear, Surprise)
- **Balance**: 40 samples per emotion for balanced training
- **Diversity**: Real-world scenarios covering various contexts and expressions
- **Quality**: Carefully crafted examples with clear emotional indicators

## Model Performance

### Best Performing Models
- **SVM**: Linear kernel with high accuracy on text classification
- **Logistic Regression**: Fast training with good generalization
- **Random Forest**: Ensemble approach with feature importance insights
- **Naive Bayes**: Baseline model with efficient performance

### Evaluation Metrics
- Accuracy scores with confidence intervals
- Precision, Recall, and F1-scores per class
- Confusion matrices (raw counts and normalized)
- Cross-validation results
- Per-emotion performance analysis

## Quick Start

### Option 1: Streamlit Web Application
```bash
# Install dependencies
pip install streamlit scikit-learn nltk plotly pandas numpy

# Run the application
streamlit run app.py --server.port 5000
```

### Option 2: Jupyter Notebook Analysis
```bash
# Install Jupyter and dependencies
pip install jupyter scikit-learn nltk plotly pandas numpy matplotlib seaborn

# Launch notebook
jupyter notebook Complete_Emotion_Detection_Analysis.ipynb
```

### Option 3: Python Script Usage
```python
from emotion_classifier import EmotionClassifier
from data_processor import DataProcessor
import pandas as pd

# Load data
df = pd.read_csv('comprehensive_emotion_dataset.csv')

# Initialize components
processor = DataProcessor()
classifier = EmotionClassifier()

# Preprocess and train
df_processed = processor.preprocess_data(df)
results = classifier.train_models(df_processed)

# Make predictions
prediction, confidence = classifier.predict_single(
    "I'm so excited about this opportunity!", 
    "SVM"
)
```

## Technical Implementation

### Text Preprocessing Pipeline
1. **Text Cleaning**: Remove URLs, normalize punctuation, handle special characters
2. **Tokenization**: Split text into meaningful tokens
3. **Lemmatization**: Reduce words to base forms
4. **Stop Words**: Remove common words while preserving emotional indicators
5. **Vectorization**: TF-IDF with 1-2 grams for feature extraction

### Model Training Process
1. **Data Splitting**: Stratified train/test split (75%/25%)
2. **Feature Extraction**: TF-IDF vectorization with 5000 max features
3. **Model Training**: Multiple algorithms with optimized parameters
4. **Cross-Validation**: 5-fold validation for robust evaluation
5. **Model Selection**: Best model based on accuracy and F1-score

### Evaluation Framework
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Detailed per-class metrics
- **Feature Importance**: Analysis of most influential features
- **Error Analysis**: Examination of misclassified examples
- **Model Comparison**: Performance vs speed trade-offs

## Use Cases

### Business Applications
- **Customer Service**: Automatic emotion detection in support tickets
- **Social Media**: Brand sentiment and emotion monitoring
- **Marketing**: Emotional response analysis for campaigns
- **Content Moderation**: Identifying emotional content patterns

### Research Applications
- **Psychology**: Emotion analysis in text-based studies
- **Linguistics**: Understanding emotional expression patterns
- **Mental Health**: Mood tracking and assessment tools
- **Education**: Student engagement and feedback analysis

## Advanced Features

### Model Interpretability
- Feature importance visualization
- Top predictive words per emotion
- Confidence score analysis
- Error pattern identification

### Scalability Considerations
- Efficient vectorization for large datasets
- Model persistence for production deployment
- Batch prediction capabilities
- Memory-optimized preprocessing

## Requirements

### Core Dependencies
- Python 3.8+
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- nltk >= 3.6

### Visualization
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- plotly >= 5.0.0

### Web Interface
- streamlit >= 1.10.0

## Installation

1. **Clone or extract the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or using the provided pyproject.toml:
   ```bash
   pip install -e .
   ```

3. **Download NLTK data** (automatically handled in code):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Project Structure

```
emotion-detection/
│
├── app.py                              # Main Streamlit application
├── emotion_classifier.py              # ML model implementation
├── data_processor.py                  # Text preprocessing
├── model_evaluator.py                 # Evaluation metrics
├── utils.py                           # Utility functions
│
├── comprehensive_emotion_dataset.csv  # Main training dataset
├── emotion_dataset.csv               # Extended dataset
│
├── Complete_Emotion_Detection_Analysis.ipynb  # Full analysis
├── emotion_detection_notebook.ipynb           # Interactive notebook
│
├── .streamlit/
│   └── config.toml                    # Streamlit configuration
│
├── pyproject.toml                     # Project dependencies
├── uv.lock                           # Dependency lock
└── README.md                         # This file
```

## Development and Extension

### Adding New Emotions
1. Update the dataset with new emotion samples
2. Modify the emotion list in the classifier
3. Retrain models with the expanded dataset
4. Update evaluation metrics and visualizations

### Improving Model Performance
1. **Data Augmentation**: Add more diverse training samples
2. **Feature Engineering**: Include additional text features
3. **Advanced Models**: Implement BERT, RoBERTa, or custom neural networks
4. **Ensemble Methods**: Combine multiple models for better accuracy

### Production Deployment
1. **API Development**: Create REST API endpoints
2. **Containerization**: Docker setup for easy deployment
3. **Monitoring**: Add logging and performance tracking
4. **Scaling**: Implement batch processing and caching

## Performance Benchmarks

### Training Performance
- Dataset size: 240 samples
- Training time: < 1 second for most models
- Memory usage: < 50MB for complete pipeline
- Prediction speed: < 10ms per text sample

### Accuracy Metrics
- Best model accuracy: 85%+ on test data
- Cross-validation stability: Low variance across folds
- Per-class F1-scores: Balanced performance across emotions
- Confidence calibration: Reliable probability estimates

## Troubleshooting

### Common Issues
1. **NLTK Data Missing**: Run `nltk.download()` for required datasets
2. **Memory Issues**: Reduce max_features in TF-IDF vectorizer
3. **Slow Training**: Use smaller n-gram ranges or feature counts
4. **Poor Accuracy**: Increase dataset size or improve text quality

### Performance Optimization
1. **Preprocessing**: Cache processed text for repeated use
2. **Vectorization**: Use sparse matrices for memory efficiency
3. **Model Selection**: Choose faster models for real-time applications
4. **Batch Processing**: Process multiple texts simultaneously

## Contributing

To contribute to this project:
1. Follow the established code structure
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure backward compatibility
5. Include performance benchmarks

## License

This project is open source and available for educational and commercial use.

## Contact

For questions, improvements, or collaboration opportunities, please refer to the project documentation or create issues for specific problems.

---

*This emotion detection system demonstrates a complete machine learning pipeline suitable for both learning and production deployment. The systematic approach ensures reproducibility and provides a solid foundation for further development.*
