from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.model_selection import iterative_train_test_split
import numpy
import joblib
import string
import nltk
from nltk.corpus import stopwords
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ml.ml_data.training_data import training_data

nltk.download('stopwords')

def text_preprocessing():
    
    clean_sentences = []
    labels = []
    
    for (sentence, label) in training_data:
        
        sentence = sentence.lower()
        translator = str.maketrans('', '', string.punctuation)
        
        clean_sentence = sentence.translate(translator)
        
        words = clean_sentence.split()
        
        eng_stopwords = set(stopwords.words('english'))
        
        filtered_words = [word for word in words if word not in eng_stopwords]
        
        clean_sentences.append((" ".join(filtered_words)))
        labels.append(label)
    
    return (clean_sentences, labels)

def encode_and_train():
    
    X, y = text_preprocessing()
    
    print(X)
    print(y)
    
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(y)
    
    X_numpy = numpy.array(X).reshape(-1, 1)
    y_numpy = numpy.array(encoded_labels)
    
    X_train, y_train, X_test, y_test = iterative_train_test_split(X_numpy, y_numpy, test_size=0.2)
    
    X_train = X_train.ravel()
    X_test = X_test.ravel()
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    c_report = classification_report(y_test, y_pred, target_names=mlb.classes_)
    
    print(c_report)
    
    joblib.dump(model, 'sentiment_model.pk1')
    
encode_and_train()