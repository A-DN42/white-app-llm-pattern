"""
Data models for the RAG application.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    """Request model for adding documents."""
    documents: List[str] = Field(..., description="List of document contents")
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    "This is the first document content.",
                    "This is the second document content."
                ]
            }
        }


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    text: str = Field(..., description="The question to ask")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "What is the main topic of the documents?"
            }
        }


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")


class QueryResponse(BaseModel):
    """Response model for query operations."""
    success: bool = Field(..., description="Whether the query was successful")
    response: Optional[str] = Field(None, description="The answer from the RAG system")
    question: Optional[str] = Field(None, description="The original question")
    message: Optional[str] = Field(None, description="Error message if unsuccessful")



class StatsResponse(BaseModel):
    """Response model for statistics."""
    success: bool = Field(..., description="Whether the operation was successful")
    document_count: Optional[int] = Field(None, description="Number of documents in vector store")
    model_name: Optional[str] = Field(None, description="Name of the LLM model")
    embedding_model: Optional[str] = Field(None, description="Name of the embedding model")
    persist_directory: Optional[str] = Field(None, description="Vector store persistence directory")
    chunk_size: Optional[int] = Field(None, description="Document chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Document chunk overlap")
    sample_sources: Optional[List[str]] = Field(None, description="Sample of document sources")
    message: Optional[str] = Field(None, description="Error message if unsuccessful")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    version: str = Field(default="1.0.0", description="Application version")


class FileUploadResponse(BaseModel):
    """Response model for file upload operations."""
    success: bool = Field(..., description="Whether the file upload was successful")
    message: str = Field(..., description="Status message")
    filename: Optional[str] = Field(None, description="Name of the uploaded file")
    file_type: Optional[str] = Field(None, description="Type of the uploaded file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="File metadata")


class SupportedFormatsResponse(BaseModel):
    """Response model for supported file formats."""
    supported_formats: List[str] = Field(..., description="List of supported file extensions")
    descriptions: Dict[str, str] = Field(..., description="Description of each format") 