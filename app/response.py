from pydantic import BaseModel, Field
from typing import List

class RAGResponse(BaseModel):
    response: str = Field(description="Response content")
    source: List[str] = Field(description="Array of Sources")
