import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def evaluate_model(self, model_name):
        """
        Comprehensive evaluation of a trained model
        
        Args:
            model_name: Name of the model to evaluate
        
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.classifier.models:
            raise ValueError(f"Model {model_name} not found")
        
        if self.classifier.X_test is None or self.classifier.y_test is None:
            raise ValueError("No test data available. Please train models first.")
        
        # Get model and test data
        model = self.classifier.models[model_name]
        X_test_vectorized = self.classifier.vectorizer.transform(self.classifier.X_test)
        y_test = self.classifier.y_test
        
        # Make predictions
        y_pred = model.predict(X_test_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.classifier.emotion_labels)
        
        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'labels': self.classifier.emotion_labels,
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }
        
        return results
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models
        
        Returns:
            Dictionary with evaluation results for all models
        """
        if not self.classifier.models:
            raise ValueError("No models trained yet")
        
        all_results = {}
        
        for model_name in self.classifier.models.keys():
            all_results[model_name] = self.evaluate_model(model_name)
        
        return all_results
    
    def compare_models(self):
        """
        Compare performance of all trained models
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.classifier.models:
            raise ValueError("No models trained yet")
        
        comparison_data = []
        
        for model_name in self.classifier.models.keys():
            results = self.evaluate_model(model_name)
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def get_detailed_classification_report(self, model_name):
        """
        Get detailed classification report for a specific model
        
        Args:
            model_name: Name of the model
        
        Returns:
            Formatted DataFrame with classification metrics
        """
        results = self.evaluate_model(model_name)
        report = results['classification_report']
        
        # Convert to DataFrame for better visualization
        report_df = pd.DataFrame(report).transpose()
        
        # Remove macro and weighted avg rows for cleaner display
        emotions_only = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        
        return emotions_only
    
    def get_confusion_matrix_normalized(self, model_name):
        """
        Get normalized confusion matrix
        
        Args:
            model_name: Name of the model
        
        Returns:
            Normalized confusion matrix
        """
        results = self.evaluate_model(model_name)
        cm = results['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm_normalized
    
    def get_model_performance_summary(self):
        """
        Get summary of all model performances
        
        Returns:
            Dictionary with performance summary
        """
        if not self.classifier.models:
            return {"error": "No models trained yet"}
        
        summary = {}
        best_accuracy = 0
        best_model = None
        
        for model_name in self.classifier.models.keys():
            results = self.evaluate_model(model_name)
            
            summary[model_name] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
            
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_model = model_name
        
        summary['best_model'] = {
            'name': best_model,
            'accuracy': best_accuracy
        }
        
        return summary
    
    def get_per_emotion_performance(self, model_name):
        """
        Get performance metrics for each emotion class
        
        Args:
            model_name: Name of the model
        
        Returns:
            DataFrame with per-emotion metrics
        """
        results = self.evaluate_model(model_name)
        report = results['classification_report']
        
        emotion_metrics = []
        
        for emotion in self.classifier.emotion_labels:
            if emotion in report:
                emotion_metrics.append({
                    'Emotion': emotion,
                    'Precision': report[emotion]['precision'],
                    'Recall': report[emotion]['recall'],
                    'F1-Score': report[emotion]['f1-score'],
                    'Support': report[emotion]['support']
                })
        
        return pd.DataFrame(emotion_metrics)
    
    def get_misclassified_examples(self, model_name, num_examples=10):
        """
        Get examples of misclassified texts
        
        Args:
            model_name: Name of the model
            num_examples: Number of examples to return
        
        Returns:
            DataFrame with misclassified examples
        """
        results = self.evaluate_model(model_name)
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Get misclassified indices
        misclassified_indices = []
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                misclassified_indices.append(i)
        
        # Limit to requested number of examples
        misclassified_indices = misclassified_indices[:num_examples]
        
        # Get corresponding texts and predictions
        misclassified_examples = []
        X_test_list = self.classifier.X_test.tolist()
        
        for idx in misclassified_indices:
            misclassified_examples.append({
                'Text': X_test_list[idx],
                'True_Emotion': y_true[idx],
                'Predicted_Emotion': y_pred[idx]
            })
        
        return pd.DataFrame(misclassified_examples)
