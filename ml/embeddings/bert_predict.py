import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model and tokenizer
MODEL_PATH = "saved_model/pytorch_model.bin"
TOKENIZER_PATH = "saved_model/tokenizer"

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval