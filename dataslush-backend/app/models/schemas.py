from typing import Optional, Dict
from pydantic import BaseModel, Field

class BasicRecommendRequest(BaseModel):
    job_description: str
    top_k: int = Field(5, ge=1)

class WeightedRecommendRequest(BaseModel):
    job_description: str
    top_k: int = Field(5, ge=1)
    weights: Optional[Dict[str, float]] = {
        "bio": 0.5,
        "skills": 0.2,
        "software": 0.2,
        "location": 0.1
    }
