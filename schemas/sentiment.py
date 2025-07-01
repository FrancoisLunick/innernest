from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SentimentBase(BaseModel):
    user_input: str = Field(description="The input text from the user.")
    sentiment_label: Optional[str] = Field(None, description="The predicted sentiment label.")
    confidence_score: Optional[float] = Field(None, description="The confidence score of the prediction")
    
class SentimentCreation(SentimentBase):
    pass

class SentimentOutput(SentimentBase):
    id: int = Field(description="The unique ID of the sentiment record.")
    user_id: int = Field(description="The ID of the user who submitted the input.")
    created_time: datetime = Field(description="Timestamp when the sentiment was recorded.")
    
    class Config:
        orm_mode = True