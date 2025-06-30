from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

# Base class for the model
base = declarative_base()

class Sentiment(base):
    __tablename__ = "sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = (Integer)
    user_text_input = Column(String)
    sentiment_label = Column(String)