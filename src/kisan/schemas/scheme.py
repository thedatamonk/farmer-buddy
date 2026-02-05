"""Government scheme related Pydantic schemas."""

from pydantic import BaseModel, Field


class SchemeDocument(BaseModel):
    """A retrieved document chunk about a government scheme."""

    content: str = Field(..., description="Text content of the chunk")
    source: str = Field(..., description="Source document/PDF name")
    page_number: int | None = Field(default=None, description="Page number in source document")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class SchemeResult(BaseModel):
    """Result of scheme information retrieval."""

    query: str = Field(..., description="Original query")
    documents: list[SchemeDocument] = Field(default_factory=list, description="Retrieved docs")
    answer: str | None = Field(default=None, description="Synthesized answer from documents")
    schemes_mentioned: list[str] = Field(default_factory=list, description="Schemes mentioned")


class SchemeInfo(BaseModel):
    """Detailed information about a specific scheme."""

    name: str = Field(..., description="Name of the scheme")
    description: str = Field(..., description="Brief description")
    eligibility: list[str] = Field(default_factory=list, description="Eligibility criteria")
    benefits: list[str] = Field(default_factory=list, description="Benefits provided")
    how_to_apply: str | None = Field(default=None, description="Application process")
    documents_required: list[str] = Field(default_factory=list, description="Required documents")
    website: str | None = Field(default=None, description="Official website")


class IndexingResult(BaseModel):
    """Result of document indexing operation."""

    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    collection_name: str = Field(..., description="Name of the vector collection")
    success: bool = Field(..., description="Whether indexing was successful")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
