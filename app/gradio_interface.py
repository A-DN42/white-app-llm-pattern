"""
Gradio interface for the RAG application.
"""

import gradio as gr
import logging
from typing import Tuple, Optional

from .rag_service import get_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_documents(doc_text: str) -> str:
    """Process documents and add them to the vector store."""
    try:
        if not doc_text.strip():
            return "❌ No documents provided"
        
        # Split the input text into documents (one per line)
        documents = [line.strip() for line in doc_text.split('\n') if line.strip()]
        
        if not documents:
            return "❌ No valid documents found"
        
        # Add documents to the vector store
        rag_service = get_rag_service()
        result = rag_service.add_documents(documents)
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return f"❌ Error processing documents: {str(e)}"


def process_file_upload(file) -> str:
    """Process uploaded file and add to vector store."""
    try:
        if file is None:
            return "❌ No file uploaded"
        
        # Read file content
        with open(file.name, 'rb') as f:
            file_content = f.read()
        
        # Get filename
        filename = file.name.split('/')[-1]  # Get just the filename
        
        # Process the file
        rag_service = get_rag_service()
        result = rag_service.add_file(file_content, filename)
        
        if result["success"]:
            metadata = result.get("metadata", {})
            file_type = result.get("file_type", "Unknown")
            
            # Create detailed success message
            details = []
            if "rows" in metadata:
                details.append(f"Rows: {metadata['rows']}")
            if "columns" in metadata:
                details.append(f"Columns: {metadata['columns']}")
            if "pages" in metadata:
                details.append(f"Pages: {metadata['pages']}")
            if "paragraphs" in metadata:
                details.append(f"Paragraphs: {metadata['paragraphs']}")
            if "sheets" in metadata:
                details.append(f"Sheets: {metadata['sheets']}")
            
            detail_str = f" ({', '.join(details)})" if details else ""
            
            return f"✅ Successfully processed {file_type} file: {filename}{detail_str}"
        else:
            return f"❌ {result['message']}"
            
    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        return f"❌ Error processing file: {str(e)}"


def get_supported_formats() -> str:
    """Get information about supported file formats."""
    try:
        rag_service = get_rag_service()
        result = rag_service.get_supported_formats()
        
        if result["success"]:
            formats_info = []
            for fmt in result["supported_formats"]:
                desc = result["descriptions"].get(fmt, "")
                formats_info.append(f"• **{fmt}**: {desc}")
            
            return f"""📋 **Supported File Formats:**

{chr(10).join(formats_info)}

Upload any of these file types to automatically extract and add their content to the vector store."""
        else:
            return f"❌ Error getting supported formats"
            
    except Exception as e:
        logger.error(f"Error getting supported formats: {e}")
        return f"❌ Error getting supported formats: {str(e)}"


def process_query(question: str, model_name: str) -> str:
    """Process a query and return the response."""
    try:
        if not question.strip():
            return "❌ No question provided"
        
        # Get response from the RAG system
        rag_service = get_rag_service()
        result = rag_service.query(question, model_name if model_name != "Default" else None)
        
        if result["success"]:
            model_used = result.get("model_used", "unknown")
            return f"**Model Used:** {model_used}\n\n{result['response']}"
        else:
            return f"❌ {result['message']}"
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"❌ Error processing query: {str(e)}"


def process_direct_llm_query(question: str, model_name: str) -> str:
    """Process a direct LLM query without RAG."""
    try:
        if not question.strip():
            return "❌ No question provided"
        
        # Get response from the LLM directly (no RAG)
        rag_service = get_rag_service()
        result = rag_service.direct_llm_query(question, model_name if model_name != "Default" else None)
        
        if result["success"]:
            model_used = result.get("model_used", "unknown")
            return f"**Model Used:** {model_used}\n\n{result['response']}"
        else:
            return f"❌ {result['message']}"
            
    except Exception as e:
        logger.error(f"Error processing direct LLM query: {e}")
        return f"❌ Error processing direct LLM query: {str(e)}"


def get_stats() -> str:
    """Get statistics about the vector store."""
    try:
        rag_service = get_rag_service()
        result = rag_service.get_stats()
        
        if result["success"]:
            stats_text = f"""📊 **Vector Store Statistics**

**Documents & Storage:**
- Documents in vector store: {result['document_count']}
- Persistence directory: {result.get('persist_directory', 'N/A')}

**Model Configuration:**
- LLM Model: {result['model_name']}
- Embedding Model: {result['embedding_model']}

**Processing Settings:**
- Chunk size: {result.get('chunk_size', 'N/A')} characters
- Chunk overlap: {result.get('chunk_overlap', 'N/A')} characters"""

            # Add sample sources if available
            if result.get('sample_sources'):
                stats_text += f"\n\n**Sample Document Sources:**\n"
                for source in result['sample_sources']:
                    stats_text += f"- {source}\n"
            
            return stats_text
        else:
            return f"❌ {result['message']}"
            
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return f"❌ Error getting stats: {str(e)}"


def clear_documents() -> Tuple[str, str, Optional[str]]:
    """Clear all documents from the vector store."""
    try:
        rag_service = get_rag_service()
        
        # Get current count before clearing
        stats_before = rag_service.get_stats()
        doc_count_before = stats_before.get('document_count', 0) if stats_before.get('success') else 0
        
        # Clear documents
        result = rag_service.clear_documents()
        
        if result["success"]:
            # Get count after clearing to verify
            stats_after = rag_service.get_stats()
            doc_count_after = stats_after.get('document_count', 0) if stats_after.get('success') else 0
            
            success_msg = f"✅ Successfully cleared {doc_count_before} documents from vector store"
            if doc_count_after == 0:
                success_msg += f"\n📊 Vector store is now empty (verified: {doc_count_after} documents remaining)"
            else:
                success_msg += f"\n⚠️ Warning: {doc_count_after} documents may still remain"
            
            return success_msg, "", None
        else:
            return f"❌ {result['message']}", "", None
            
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        return f"❌ Error clearing documents: {str(e)}", "", None


def get_available_models() -> list:
    """Get list of available models for dropdowns."""
    try:
        rag_service = get_rag_service()
        result = rag_service.get_available_models()
        
        if result["success"]:
            models = ["Default"] + result["available_models"]
            return models
        else:
            return ["Default", "mistral"]  # Fallback
            
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return ["Default", "mistral"]  # Fallback


# Create the Gradio interface
def create_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(
        title="AI Architecture Patterns Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .file-upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🏗️ AI Architecture Patterns Demo
            
            **Multiple AI Architecture Implementations** - Explore different patterns for AI applications:
            - 📚 **RAG Pattern**: Retrieval-Augmented Generation with document context
            - 🤖 **Direct LLM**: Pure language model interaction
            - 🔧 **Local Processing**: All running locally with Ollama and Chroma
            """
        )
        
        with gr.Tab("📚 RAG Pattern - Document Management"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Text Documents")
                    doc_input = gr.Textbox(
                        label="Enter Documents",
                        placeholder="Enter your documents here (one per line)...\n\nExample:\nDocument 1 content here\nDocument 2 content here",
                        lines=10,
                        max_lines=15
                    )
                    
                    upload_text_btn = gr.Button("📤 Upload Text Documents", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📁 File Upload")
                    file_input = gr.File(
                        label="Upload File",
                        file_types=[".csv", ".xlsx", ".xls", ".pdf", ".docx", ".txt"],
                        type="filepath"
                    )
                    
                    upload_file_btn = gr.Button("📤 Upload File", variant="primary")
                    
                    gr.Markdown("### 📋 Supported Formats")
                    formats_btn = gr.Button("📋 Show Supported Formats", variant="secondary")
                
                with gr.Column(scale=1):
                    doc_output = gr.Textbox(
                        label="📊 Status",
                        lines=15,
                        interactive=False
                    )
                    
                    with gr.Row():
                        stats_btn = gr.Button("📈 Get Statistics", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
            
            # Event handlers for document management
            upload_text_btn.click(
                fn=process_documents,
                inputs=doc_input,
                outputs=doc_output
            )
            
            upload_file_btn.click(
                fn=process_file_upload,
                inputs=file_input,
                outputs=doc_output
            )
            
            formats_btn.click(
                fn=get_supported_formats,
                outputs=doc_output
            )
            
            clear_btn.click(
                fn=clear_documents,
                outputs=[doc_output, doc_input, file_input]
            )
            
            stats_btn.click(
                fn=get_stats,
                outputs=doc_output
            )
        
        with gr.Tab("📚 RAG Pattern - Query System"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="🔍 Your Question",
                        placeholder="Ask a question about your documents...\n\nExample: What are the main topics discussed in the documents?",
                        lines=4
                    )
                    
                    rag_model_dropdown = gr.Dropdown(
                        label="🤖 Select Model",
                        choices=get_available_models(),
                        value="Default",
                        info="Choose which Ollama model to use for RAG queries"
                    )
                    
                    query_btn = gr.Button("🚀 Get Answer", variant="primary")
                
                with gr.Column(scale=3):
                    query_output = gr.Textbox(
                        label="💬 Answer",
                        lines=15,
                        interactive=False
                    )
            
            # Event handler for queries
            query_btn.click(
                fn=process_query,
                inputs=[query_input, rag_model_dropdown],
                outputs=query_output
            )
            
            # Allow Enter key to submit query
            query_input.submit(
                fn=process_query,
                inputs=[query_input, rag_model_dropdown],
                outputs=query_output
            )
        
        with gr.Tab("🤖 Direct LLM Pattern"):
            gr.Markdown(
                """
                ### 🤖 Direct LLM Architecture Pattern
                
                This demonstrates the **Direct LLM Pattern** - pure language model interaction without retrieval augmentation.
                Perfect for general questions, creative writing, coding help, or any topic not requiring specific document context.
                
                **Architecture**: User Query → LLM → Response (No document retrieval or vector search)
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    llm_input = gr.Textbox(
                        label="💭 Your Message",
                        placeholder="Ask me anything! Choose your model below...\n\nExamples:\n- Explain quantum computing\n- Write a Python function\n- Tell me a joke\n- Help me with creative writing",
                        lines=6
                    )
                    
                    llm_model_dropdown = gr.Dropdown(
                        label="🤖 Select Model",
                        choices=get_available_models(),
                        value="Default",
                        info="Choose which Ollama model to use for direct LLM queries"
                    )
                    
                    llm_btn = gr.Button("🚀 Send Message", variant="primary")
                    
                    gr.Markdown(
                        """
                        **🏗️ Pattern Benefits:**
                        - Simple, direct architecture
                        - Fast response times
                        - General-purpose capabilities
                        - No document preprocessing required
                        - Ideal for conversational AI applications
                        - **Model Comparison**: Try different models to compare responses!
                        """
                    )
                
                with gr.Column(scale=3):
                    llm_output = gr.Textbox(
                        label="🤖 LLM Response",
                        lines=20,
                        interactive=False,
                        placeholder="LLM response will appear here..."
                    )
            
            # Event handlers for direct LLM chat
            llm_btn.click(
                fn=process_direct_llm_query,
                inputs=[llm_input, llm_model_dropdown],
                outputs=llm_output
            )
            
            # Allow Enter key to submit LLM query
            llm_input.submit(
                fn=process_direct_llm_query,
                inputs=[llm_input, llm_model_dropdown],
                outputs=llm_output
            )
        
        with gr.Tab("ℹ️ About Architecture Patterns"):
            gr.Markdown(
                """
                ## About This Demo Application
                
                This application demonstrates **multiple AI architecture patterns** for different use cases:
                
                ### 📚 RAG Pattern (Retrieval-Augmented Generation)
                1. **Upload Documents**: Add text documents or files to the knowledge base
                2. **File Support**: Process CSV, Excel, PDF, Word, and text files
                3. **Ask Questions**: Query the system about your documents
                4. **Get Intelligent Answers**: Receive contextual responses based on your documents
                
                ### 🤖 Direct LLM Pattern
                1. **Direct Interaction**: Chat directly with the language model
                2. **No Context**: Pure LLM capabilities without document retrieval
                3. **General Purpose**: Suitable for any topic or creative task
                
                ### Supported File Formats
                
                - **📊 CSV (.csv)**: Comma-separated values - tabular data
                - **📈 Excel (.xlsx, .xls)**: Microsoft Excel spreadsheets
                - **📄 PDF (.pdf)**: Portable Document Format files
                - **📝 Word (.docx)**: Microsoft Word documents
                - **📃 Text (.txt)**: Plain text files
                
                ### Technology Stack
                - **LLM**: Ollama with Mistral model (local inference)
                - **Vector Store**: Chroma (local storage)
                - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
                - **File Processing**: Pandas, PyPDF2, python-docx
                - **Frontend**: Gradio
                - **Backend**: FastAPI
                
                ### Architecture Patterns Explained
                
                **RAG Pattern Flow:**
                1. Documents/files are processed and converted to text
                2. Text is split into chunks and converted to embeddings
                3. Embeddings are stored in a local vector database
                4. When you ask a question, relevant chunks are retrieved
                5. The LLM generates an answer based on the retrieved context
                
                **Direct LLM Pattern Flow:**
                1. User submits a question directly to the LLM
                2. LLM processes the query using its training knowledge
                3. Response is generated without any document context
                
                ### Privacy & Security
                - All processing happens locally on your machine
                - No data is sent to external services
                - Your documents remain private and secure
                - Files are processed in memory and not permanently stored
                """
            )
    
    return demo


def launch_interface():
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    launch_interface() 