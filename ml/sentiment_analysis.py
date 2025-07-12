"""
sentiment_analysis.py

This module defines the SentimentAnalyzer class. This class is responsible for loading a pre-trained
sentiment analysis model from disk. 
TF-IDF + Logistic Regression pipeline using joblib.
"""
import joblib
import os

class SentimentAnalyzer:
    """
    A sentiment analysis utility that loads and provides access to a trained model.
    
    The model is a TF-IDF vectorizer combined with a classifier serialized using joblib and is stored at
    the path that was specified.
    """
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(base_dir, "embeddings", "tfidf_pipeline.py")
        self.model = None
    
    def load_sentiment_model(self):
        if not self.model:
            self.model = joblib.load(self.model_path)
        return self.model