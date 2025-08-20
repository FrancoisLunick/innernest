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
    """
    Loads texts and their corresponding emotion labels from the training data.

    Returns:
        tuple: Tuple containing a list of journal entry texts and corresponding multilabel emotion annotations.
    """
    texts = [entry[0] for entry in training_data]
    labels = [entry[1] for entry in training_data]
    
    # print([(texts, labels)])
    
    return (texts, labels)

class JournalDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for tokenized journal entries and multilabel emotions.

    """
    
    def __init__(self, texts, labels, tokenizer, max_len = 512) -> None:
        """
        Initializes the custom dataset with texts and corresponding multi-labels.

        Args:
            texts (List[str]): List of journal entry texts.
            labels (List[List[int]]): Encoded emotion labels.
            tokenizer (_type_): Tokenizer to process input text.
            max_len (int, optional): Maximum token length for padding/truncation. Defaults to 512.
        """
        
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index) -> any:
        """
        Retrieves the tokenized representation and label for a given index.

        Args:
            index (int): Index of the data sample.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels tensors.
        """
        
        # Get the text and label for the given index
        text = self.texts[index]
        label = torch.tensor(self.labels[index], dtype=torch.float)
        
        # Tokenize the text with padding and truncation to max_len
        encoding = self.tokenizer(text, padding = "max_length", truncation = True, max_length = self.max_len, return_tensors = "pt")
        
        return {
             "input_ids": encoding["input_ids"].squeeze(0),
             "attention_mask": encoding["attention_mask"].squeeze(0),
             "labels": label
        }
    
class BERTClassifier(torch.nn.Module):
    """
    A multilabel emotion classifier built on top of a pretrained BERT model.

    """
    
    def __init__(self, bert, num_labels, dropout_prob = 0.3) -> None:
        """
        Initializes the BERT-based classifier.

        Args:
            bert (BertModel): Pretrained BERT model.
            num_labels (int): Number of output emotion classes.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.3.
        """
        
        super(BERTClassifier, self).__init__()
        
        self.bert = bert
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token ids tensor
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Logits tensor for each label.
        """
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        
        return logits

def train_model(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The BERT classifier model.
        dataloader (DataLoader): DataLoader providing training batches.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run the training on.

    Returns:
        float: Average training loss over the epoch.
    """
    
    model.train()
    total_loss = 0.0
    
    # Iterate over batches
    for batch in dataloader:
        
        # Move inputs and labels to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels.float())
        # Backpropagation 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
    
def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a validation or test set.

    Args:
        model (torch.nn.Module): The BERT classifier model.
        dataloader (DataLoader): DataLoader providing evaluation batches.
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Average evaluation loss.
    """
    
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        # Iterate over batches without gradient computation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            
            # Apply sigmoid and threshold at 0.5 for multilabel predictions
            preds = torch.sigmoid(outputs) > 0.5
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            
    avg_loss = total_loss / len(dataloader)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    
    # Compute evaluation metrics
    ham_loss = hamming_loss(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    print(f"Loss: {avg_loss:.4f}")
    print(f"Hamming Loss: {ham_loss:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return avg_loss

def save_model(model, tokenizer, output_dir = "saved_model"):
    """
    Saves the trained model and tokenizer.

    Args:
        model (torch.nn.Module): Trained model to save.
        tokenizer (PreTrainedTokenizer): Tokenizer to save.
        output_dir (str, optional): Dir to save the model and tokenizer. Defaults to "saved_model".
    """
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    
    # Save model state dict and tokenizer
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"Model Saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

def main():
    """
    Main function to load data, train and evaluate the BERT classifier, and also save the model.
    """
    
    # Load texts and raw labels
    texts, raw_labels = load_data()

    # Initialize MultiLabelBinarizer and encode labels
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(raw_labels)
    
    # Create dataset and dataloader
    dataset = JournalDataset(texts, labels, bert_tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained BERT model
    bert = BertModel.from_pretrained("bert-base-uncased")
    
    # Initialize classifier model
    model = BERTClassifier(bert, num_labels=len(mlb.classes_)).to(device)

    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(3):
        loss = train_model(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # Evaluate the trained model
    evaluate(model, dataloader, criterion, device)

    # Save model and tokenizer
    save_model(model, bert_tokenizer)
    
    # Save the label binarizer for future use
    joblib.dump(mlb, os.path.join("saved_model", "label_binarizer.pkl"))

if __name__ == "__main__":
    main()