import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model and tokenizer
MODEL_PATH = "saved_model"
TOKENIZER_PATH = "saved_model/tokenizer"

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_mood(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        mood = torch.argmax(predictions, dim=1).item()
        
    return mood