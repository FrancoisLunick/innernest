from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

# Base class for the model
base = declarative_base()

class Sentiment(base):
    __tablename__ = "sentiments"