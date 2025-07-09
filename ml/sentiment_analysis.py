import joblib
import os

class SentimentAnalyzer:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(base_dir, "embeddings", "tfidf_model.pk1")
        self.model = None
    
    def load_sentiment_model(self):
        if not self.model:
            self.model = joblib.load(self.model_path)
        return self.model