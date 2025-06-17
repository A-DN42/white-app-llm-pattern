"""
RAG Service Layer
Handles all RAG operations including document processing, vector storage, and querying.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.schema import Document

from .file_processor import get_file_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    """
    Service class for RAG operations.
    Handles document processing, vector storage, and querying.
    """
    
    def __init__(
        self,
        model_name: str = "mistral",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "data/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self._llm: Optional[Ollama] = None
        self._current_llm_model: Optional[str] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[Chroma] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        
        # Available Ollama models
        self.available_models = [
            "mistral",
            "llama2",
            "codellama",
            "phi",
            "neural-chat",
            "starling-lm",
            "orca-mini",
            "vicuna"
        ]
        
        # Ensure data directory exists
        Path(persist_directory).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("RAG Service initialized")
    
    def get_llm(self, model_name: Optional[str] = None) -> Ollama:
        """Get LLM instance, optionally switching models."""
        target_model = model_name or self.model_name
        
        # Check if we need to reinitialize the LLM
        if self._llm is None or self._current_llm_model != target_model:
            try:
                self._llm = Ollama(model=target_model)
                self._current_llm_model = target_model
                logger.info(f"LLM initialized/switched to model: {target_model}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM with model {target_model}: {e}")
                raise
        return self._llm
    
    @property
    def llm(self) -> Ollama:
        """Lazy initialization of LLM with default model."""
        return self.get_llm()
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy initialization of embeddings."""
        if self._embeddings is None:
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model
                )
                logger.info(f"Embeddings initialized with model: {self.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                raise
        return self._embeddings
    
    @property
    def vector_store(self) -> Chroma:
        """Lazy initialization of vector store."""
        if self._vector_store is None:
            try:
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Vector store initialized at: {self.persist_directory}")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise
        return self._vector_store
    
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazy initialization of text splitter."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        return self._text_splitter
    
    def add_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document content strings
            
        Returns:
            Dict with success status and message
        """
        try:
            if not documents:
                return {"success": False, "message": "No documents provided"}
            
            # Create Document objects
            docs = [Document(page_content=doc) for doc in documents if doc.strip()]
            
            if not docs:
                return {"success": False, "message": "No valid documents found"}
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(docs)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
            
            logger.info(f"Added {len(docs)} documents ({len(chunks)} chunks) to vector store")
            
            return {
                "success": True,
                "message": f"Successfully added {len(docs)} documents ({len(chunks)} chunks)"
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"success": False, "message": f"Error adding documents: {str(e)}"}
    
    def add_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process and add a file to the vector store.
        
        Args:
            file_content: The file content as bytes
            filename: The original filename
            
        Returns:
            Dict with success status, message, and metadata
        """
        try:
            # Process the file
            file_processor = get_file_processor()
            
            # Check if file type is supported
            if not file_processor.is_supported_file(filename):
                return {
                    "success": False,
                    "message": f"Unsupported file type. Supported formats: {', '.join(file_processor.SUPPORTED_EXTENSIONS)}"
                }
            
            # Process the file content
            result = file_processor.process_file(file_content, filename)
            
            if not result["success"]:
                return result
            
            # Add the extracted text to vector store
            extracted_text = result["text"]
            doc_result = self.add_documents([extracted_text])
            
            if doc_result["success"]:
                return {
                    "success": True,
                    "message": f"Successfully processed and added file: {filename}",
                    "filename": filename,
                    "file_type": result["metadata"]["file_type"],
                    "metadata": result["metadata"]
                }
            else:
                return {
                    "success": False,
                    "message": f"File processed but failed to add to vector store: {doc_result['message']}"
                }
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {"success": False, "message": f"Error processing file: {str(e)}"}
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported file formats."""
        file_processor = get_file_processor()
        
        descriptions = {
            ".csv": "Comma-separated values - tabular data",
            ".xlsx": "Microsoft Excel spreadsheet",
            ".xls": "Microsoft Excel spreadsheet (legacy)",
            ".pdf": "Portable Document Format",
            ".docx": "Microsoft Word document",
            ".txt": "Plain text file"
        }
        
        return {
            "success": True,
            "supported_formats": list(file_processor.SUPPORTED_EXTENSIONS),
            "descriptions": descriptions
        }
    
    def query(self, question: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            model_name: Optional model name to use (defaults to service default)
            
        Returns:
            Dict with response and metadata
        """
        try:
            if not question.strip():
                return {"success": False, "message": "No question provided"}
            
            # Check if vector store has documents
            try:
                collection = self.vector_store._collection
                if collection.count() == 0:
                    return {
                        "success": False,
                        "message": "No documents in vector store. Please add documents first."
                    }
            except Exception:
                return {
                    "success": False,
                    "message": "Vector store not properly initialized. Please add documents first."
                }
            
            # Get LLM instance (potentially switching models)
            llm_instance = self.get_llm(model_name)
            used_model = model_name or self.model_name
            
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_instance,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            # Get response
            response = qa_chain.run(question)
            
            logger.info(f"RAG query processed successfully with {used_model}: {question[:50]}...")
            
            return {
                "success": True,
                "response": response,
                "question": question,
                "model_used": used_model
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"success": False, "message": f"Error processing query: {str(e)}"}
    
    def direct_llm_query(self, question: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the LLM directly without RAG (no document retrieval).
        
        Args:
            question: The question to ask the LLM
            model_name: Optional model name to use (defaults to service default)
            
        Returns:
            Dict with response and metadata
        """
        try:
            if not question.strip():
                return {"success": False, "message": "No question provided"}
            
            # Get LLM instance (potentially switching models)
            llm_instance = self.get_llm(model_name)
            used_model = model_name or self.model_name
            
            # Query LLM directly without document context
            response = llm_instance.invoke(question)
            
            logger.info(f"Direct LLM query processed successfully with {used_model}: {question[:50]}...")
            
            return {
                "success": True,
                "response": response,
                "question": question,
                "mode": "direct_llm",
                "model_used": used_model
            }
            
        except Exception as e:
            logger.error(f"Error processing direct LLM query: {e}")
            return {"success": False, "message": f"Error processing direct LLM query: {str(e)}"}
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available Ollama models."""
        try:
            return {
                "success": True,
                "available_models": self.available_models,
                "current_default": self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return {"success": False, "message": f"Error getting available models: {str(e)}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get additional metadata if documents exist
            metadata_info = {}
            if count > 0:
                try:
                    # Get a sample of documents to understand the data
                    sample_docs = collection.get(limit=5)
                    if sample_docs and sample_docs.get('metadatas'):
                        # Extract unique sources or document types from metadata
                        sources = set()
                        for metadata in sample_docs['metadatas']:
                            if metadata and 'source' in metadata:
                                sources.add(metadata['source'])
                        if sources:
                            metadata_info['sample_sources'] = list(sources)[:5]  # Limit to 5 sources
                except Exception as meta_error:
                    logger.warning(f"Could not extract metadata info: {meta_error}")
            
            return {
                "success": True,
                "document_count": count,
                "model_name": self.model_name,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                **metadata_info
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"success": False, "message": f"Error getting stats: {str(e)}"}
    
    def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the vector store."""
        try:
            # Get the collection and delete all documents
            collection = self.vector_store._collection
            
            # Get all document IDs
            all_docs = collection.get()
            if all_docs['ids']:
                # Delete all documents by their IDs
                collection.delete(ids=all_docs['ids'])
                logger.info(f"Cleared {len(all_docs['ids'])} documents from vector store")
            else:
                logger.info("Vector store was already empty")
            
            # Persist the changes
            self.vector_store.persist()
            
            return {"success": True, "message": "All documents cleared successfully"}
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            # Fallback: if the above method fails, try the directory removal approach
            try:
                # Reset vector store instance
                self._vector_store = None
                
                # Remove persist directory
                import shutil
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                
                logger.info("Vector store cleared using directory removal fallback")
                return {"success": True, "message": "All documents cleared (using fallback method)"}
                
            except Exception as fallback_error:
                logger.error(f"Error in fallback clear method: {fallback_error}")
                return {"success": False, "message": f"Error clearing documents: {str(e)}"}


# Global service instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service 