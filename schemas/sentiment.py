from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SentimentBase(BaseModel):
    user_input: str
    sentiment_label: Optional[str] = None
    confidence_score: Optional[float] = None
    
class SentimentCreation(SentimentBase):
    pass