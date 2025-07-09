"""
This module contains Pydantic models used to validate and serialize data for sentiment analysis,
including base structure for user input, creation schema for requests, and output schema for responses.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# Base class for sentiment data shared by both request and response models.
class SentimentBase(BaseModel):
    """
    Defines the base structure for sentiment-related data.

    Args:
        user_input (str): The input text from the user to be analyzed.
        sentiment_label (Optional[str]): The predicted sentiment label.
        confidence_score (Optional[float]): The confidence level of the sentiment prediction. 
    """
    user_input: str = Field(description="The input text from the user.")
    sentiment_label: Optional[str] = Field(None, description="The predicted sentiment label.")
    confidence_score: Optional[float] = Field(None, description="The confidence score of the prediction")

# Used for creating new sentiment entries 
class SentimentCreation(SentimentBase):
    """
    Inherits from SentimentBase.
    Used for validating input when a new sentiment record is submitted.
    """
    pass

# Used for returning sentiment records from the database
class SentimentOutput(SentimentBase):
    """
    Extends SentimentBase to include fields that are returned from the database.
    
    Args:
        id (int): Unique id for the sentiment entry.
        user_id (int): Id for the user who submitted the input.
        created_time (datetime): Timestamp of when the sentiment analysis was recorded.
    """
    id: int = Field(description="The unique ID of the sentiment record.")
    user_id: int = Field(description="The ID of the user who submitted the input.")
    created_time: datetime = Field(description="Timestamp when the sentiment was recorded.")
    
    # Enables ORM compatibility for data conversion from SQLAlchemy models.
    class Config:
        orm_mode = True