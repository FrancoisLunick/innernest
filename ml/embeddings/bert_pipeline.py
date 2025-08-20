from transformers import BertTokenizerFast, BertModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ml.ml_data.training_data import training_data
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
import joblib
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score

bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def load_data():
    texts = [entry[0] for entry in training_data]
    labels = [entry[1] for entry in training_data]
    
    # print([(texts, labels)])
    
    return (texts, labels)

class JournalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len = 512) -> None:
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index) -> any:
        
        text = self.texts[index]
        label = torch.tensor(self.labels[index], dtype=torch.float)
        
        encoding = self.tokenizer(text, padding = "max_length", truncation = True, max_length = self.max_len, return_tensors = "pt")
        
        return {
             "input_ids": encoding["input_ids"].squeeze(0),
             "attention_mask": encoding["attention_mask"].squeeze(0),
             "labels": label
        }
