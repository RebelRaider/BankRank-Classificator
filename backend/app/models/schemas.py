from pydantic import BaseModel, validator
from typing import Optional, Dict, List
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str
    file_type: str
    file_size: int

class DocumentCreate(DocumentBase):
    content: str

class Document(DocumentBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class ToxicityResult(BaseModel):
    toxicity_score: float
    is_toxic: bool
    confidence: float

class RatingResult(BaseModel):
    category: str
    confidence: float
    probabilities: Dict[str, float]

class ClassificationResponse(BaseModel):
    document_id: int
    toxicity_result: ToxicityResult
    rating_result: RatingResult
    processing_time: float
    
class FileUploadResponse(BaseModel):
    document_id: int
    filename: str
    file_size: int
    status: str

class AnalyticsData(BaseModel):
    total_documents: int
    avg_processing_time: float
    toxicity_distribution: Dict[str, int]
    rating_distribution: Dict[str, int]
    daily_statistics: List[Dict]
