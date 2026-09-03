from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    response: str = Field(description="Response content")
    source: str = Field(description="Source file")
