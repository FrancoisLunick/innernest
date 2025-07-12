from transformers import BertTokenizerFast
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ml.ml_data.training_data import training_data

bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def load_data():
    texts = [entry[0] for entry in training_data]
    labels = [entry[1] for entry in training_data]
    
    print([(texts, labels)])
    
    return (texts, labels)
