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
    id: int
    user_id: int
    created_time: datetime
    
    class Config:
        orm_mode = True