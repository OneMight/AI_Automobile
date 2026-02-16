from pydantic import BaseModel
from typing import List

class SimilarModel(BaseModel):
    model: str
    mark: str
    confidence: float

class RecognitionResponse(BaseModel):
    model: str
    mark: str
    yearManufacture: str
    recognizedTime: float 
    confidence: float
    similarModels: List[SimilarModel]