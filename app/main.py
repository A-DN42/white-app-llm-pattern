"""
FastAPI application for RAG system.
"""

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    DocumentRequest, QueryRequest, DocumentResponse, 
    QueryResponse, StatsResponse, HealthResponse,
    FileUploadResponse, SupportedFormatsResponse
)
from pydantic import BaseModel
from typing import Optional, List
from .rag_service import get_rag_service


class QueryWithModelRequest(BaseModel):
    text: str
    model_name: Optional[str] = None


class AvailableModelsResponse(BaseModel):
    success: bool = True
    available_models: List[str]
    current_default: str
    message: Optional[str] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Architecture Patterns API",
    description="Demonstration of multiple AI architecture patterns: RAG (Retrieval-Augmented Generation) and Direct LLM interaction using Ollama and Chroma",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.get("/supported-formats", response_model=SupportedFormatsResponse)
async def get_supported_formats():
    """Get information about supported file formats."""
    try:
        rag_service = get_rag_service()
        result = rag_service.get_supported_formats()
        
        return SupportedFormatsResponse(
            supported_formats=result["supported_formats"],
            descriptions=result["descriptions"]
        )
        
    except Exception as e:
        logger.error(f"Error getting supported formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents", response_model=DocumentResponse)
async def add_documents(request: DocumentRequest):
    """Add documents to the vector store."""
    try:
        rag_service = get_rag_service()
        result = rag_service.add_documents(request.documents)
        
        if result["success"]:
            return DocumentResponse(success=True, message=result["message"])
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        logger.error(f"Error in add_documents endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file to add to the vector store."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process the file
        rag_service = get_rag_service()
        result = rag_service.add_file(file_content, file.filename)
        
        if result["success"]:
            return FileUploadResponse(
                success=True,
                message=result["message"],
                filename=result.get("filename"),
                file_type=result.get("file_type"),
                metadata=result.get("metadata")
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        logger.error(f"Error in upload_file endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available-models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get list of available Ollama models."""
    try:
        rag_service = get_rag_service()
        result = rag_service.get_available_models()
        
        if result["success"]:
            return AvailableModelsResponse(
                available_models=result["available_models"],
                current_default=result["current_default"]
            )
        else:
            return AvailableModelsResponse(
                success=False,
                message=result["message"],
                available_models=[],
                current_default="mistral"
            )
            
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system with default model."""
    try:
        rag_service = get_rag_service()
        result = rag_service.query(request.text)
        
        if result["success"]:
            return QueryResponse(
                success=True,
                response=result["response"],
                question=result["question"]
            )
        else:
            return QueryResponse(
                success=False,
                message=result["message"]
            )
            
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-with-model", response_model=QueryResponse)
async def query_with_model(request: QueryWithModelRequest):
    """Query the RAG system with specific model."""
    try:
        rag_service = get_rag_service()
        result = rag_service.query(request.text, request.model_name)
        
        if result["success"]:
            return QueryResponse(
                success=True,
                response=result["response"],
                question=result["question"]
            )
        else:
            return QueryResponse(
                success=False,
                message=result["message"]
            )
            
    except Exception as e:
        logger.error(f"Error in query with model endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm-query", response_model=QueryResponse)
async def direct_llm_query(request: QueryRequest):
    """Query the LLM directly without RAG (no document retrieval) with default model."""
    try:
        rag_service = get_rag_service()
        result = rag_service.direct_llm_query(request.text)
        
        if result["success"]:
            return QueryResponse(
                success=True,
                response=result["response"],
                question=result["question"]
            )
        else:
            return QueryResponse(
                success=False,
                message=result["message"]
            )
            
    except Exception as e:
        logger.error(f"Error in direct LLM query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm-query-with-model", response_model=QueryResponse)
async def direct_llm_query_with_model(request: QueryWithModelRequest):
    """Query the LLM directly without RAG with specific model."""
    try:
        rag_service = get_rag_service()
        result = rag_service.direct_llm_query(request.text, request.model_name)
        
        if result["success"]:
            return QueryResponse(
                success=True,
                response=result["response"],
                question=result["question"]
            )
        else:
            return QueryResponse(
                success=False,
                message=result["message"]
            )
            
    except Exception as e:
        logger.error(f"Error in direct LLM query with model endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the vector store."""
    try:
        rag_service = get_rag_service()
        result = rag_service.get_stats()
        
        if result["success"]:
            return StatsResponse(
                success=True,
                document_count=result.get("document_count"),
                model_name=result.get("model_name"),
                embedding_model=result.get("embedding_model"),
                persist_directory=result.get("persist_directory"),
                chunk_size=result.get("chunk_size"),
                chunk_overlap=result.get("chunk_overlap"),
                sample_sources=result.get("sample_sources")
            )
        else:
            return StatsResponse(
                success=False,
                message=result["message"]
            )
            
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents", response_model=DocumentResponse)
async def clear_documents():
    """Clear all documents from the vector store."""
    try:
        rag_service = get_rag_service()
        result = rag_service.clear_documents()
        
        if result["success"]:
            return DocumentResponse(success=True, message=result["message"])
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        logger.error(f"Error in clear_documents endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 