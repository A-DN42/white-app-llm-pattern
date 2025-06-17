#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting RAG Application Setup${NC}"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama is not installed. Please install it first:${NC}"
    echo -e "${YELLOW}Visit https://ollama.ai/download and follow the installation instructions.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Ollama found${NC}"

# Check if Mistral model is downloaded
if ! ollama list | grep -q "mistral"; then
    echo -e "${YELLOW}📥 Downloading Mistral model...${NC}"
    ollama pull mistral
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Mistral model downloaded successfully${NC}"
    else
        echo -e "${RED}❌ Failed to download Mistral model${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Mistral model already available${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}🔧 Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Clean install of requirements
echo -e "${YELLOW}📦 Installing requirements...${NC}"
pip install --upgrade pip
pip uninstall -y sentence-transformers huggingface-hub langchain langchain-community langchain-core chromadb
pip install --no-cache-dir -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Requirements installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install requirements${NC}"
    exit 1
fi

# Verify installations
echo -e "${YELLOW}🔍 Verifying installations...${NC}"

python -c "from sentence_transformers import SentenceTransformer; print('✅ Sentence transformers OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Sentence transformers verification failed${NC}"
    exit 1
fi

python -c "import huggingface_hub; print('✅ Huggingface hub OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Huggingface hub verification failed${NC}"
    exit 1
fi

python -c "from langchain.chains import RetrievalQA; print('✅ LangChain OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ LangChain verification failed${NC}"
    exit 1
fi

python -c "import chromadb; print('✅ ChromaDB OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ChromaDB verification failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All verifications passed${NC}"

# Create data directory
mkdir -p data

# Clean up old vector store if it exists
if [ -d "local_chroma_db" ]; then
    echo -e "${YELLOW}🧹 Cleaning up old vector store...${NC}"
    rm -rf local_chroma_db
fi

# Run both FastAPI and Gradio interfaces
echo -e "${GREEN}🚀 Starting the application...${NC}"
echo -e "${YELLOW}FastAPI will be available at: http://localhost:8000${NC}"
echo -e "${YELLOW}Gradio interface will be available at: http://localhost:7860${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

# Start FastAPI in background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Start Gradio interface in background
python -c "from app.gradio_interface import launch_interface; launch_interface()" &
GRADIO_PID=$!

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
    kill $FASTAPI_PID 2>/dev/null
    kill $GRADIO_PID 2>/dev/null
    echo -e "${GREEN}✅ Services stopped${NC}"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait 