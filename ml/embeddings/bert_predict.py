"""
This module provides inference only mood prediction using a previously trained BERT model.
It should not be used for training purposes. The model and tokenizer are loaded from saved files,
and predictions are made on input text data.
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertTokenizerFast
from ml.embeddings.bert_pipeline import BERTClassifier
import os
import joblib

# Path setup for model and tokenizer files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_BIN = os.path.join(MODEL_DIR, "pytorch_model.bin")
TOKENIZER_DIR = os.path.join("tokenizer")
MLB_PATH = os.path.join(MODEL_DIR, "label_binarizer.pk1")

# Loading tokenizer, label binarizer, and determining number of labels
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
mlb = joblib.load(MLB_PATH)
num_labels = len(mlb.classes_)

# Device setup for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model reconstruction using pretrained BERT and custom classifier head
bert = BertModel.from_pretrained("bert-base-uncased")
model = BERTClassifier(bert, num_labels=num_labels).to(device)
state = torch.load(MODEL_BIN, map_location=device)
model.load_state_dict(state)

# Set model to evaluation mode to disable dropout and others training-specific layers
model.eval()

def predict_mood(text):
    """
    Predicts the mood of the given input text using the pretrained BERT classifier.

    Args:
        text (str): The input text string for which mood prediction is to be made.

    Returns:
        int: The predicted mood class index.
    """
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        mood = torch.argmax(predictions, dim=1).item()
        
    return mood