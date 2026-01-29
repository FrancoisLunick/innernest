import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertTokenizerFast
from ml.embeddings.bert_pipeline import BERTClassifier
import os
import joblib

# Load the saved model and tokenizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_BIN = os.path.join(MODEL_DIR, "pytorch_model.bin")
TOKENIZER_DIR = os.path.join("tokenizer")
MLB_PATH = os.path.join(MODEL_DIR, "label_binarizer.pk1")

tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
mlb = joblib.load(MLB_PATH)
num_labels = len(mlb.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = BertModel.from_pretrained("bert-base-uncased")
model = BERTClassifier(bert, num_labels=num_labels).to(device)
state = torch.load(MODEL_BIN, map_location=device)
model.load_state_dict(state)
model.eval()

def predict_mood(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        mood = torch.argmax(predictions, dim=1).item()
        
    return mood