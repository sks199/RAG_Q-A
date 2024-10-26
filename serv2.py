import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
from tqdm import tqdm
import torch
import gc
import time
import hashlib
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Medical Q&A Bot")

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

class ServiceStatus(BaseModel):
    status: str
    message: str

# Global variables
embeddings = None
vector_store = None
is_ready = False

# Constants
CACHE_DIR = Path("cache")
VECTOR_STORE_PATH = CACHE_DIR / "vector_store.faiss"
CHUNKS_CACHE_PATH = CACHE_DIR / "chunks.pkl"
DOCUMENT_HASH_PATH = CACHE_DIR / "document_hash.txt"

def compute_documents_hash(documents: List[str]) -> str:
    """Compute a hash of the documents to detect changes."""
    combined_content = "".join(documents)
    return hashlib.md5(combined_content.encode()).hexdigest()

def should_rebuild_vector_store(documents: List[str]) -> bool:
    """Check if we need to rebuild the vector store."""
    if not CACHE_DIR.exists():
        return True
    
    if not VECTOR_STORE_PATH.exists() or not DOCUMENT_HASH_PATH.exists():
        return True
    
    current_hash = compute_documents_hash(documents)
    try:
        with open(DOCUMENT_HASH_PATH, 'r') as f:
            stored_hash = f.read().strip()
        return current_hash != stored_hash
    except:
        return True

# def save_vector_store(vector_store, documents: List[str]):
#     """Save vector store and document hash to disk."""
#     CACHE_DIR.mkdir(exist_ok=True)
    
#     # Save vector store
#     vector_store.save_local(str(VECTOR_STORE_PATH))
    
#     # Save document hash
#     with open(DOCUMENT_HASH_PATH, 'w') as f:
#         f.write(compute_documents_hash(documents))
    
#     logger.info("Vector store and hash saved to disk")

# def load_cached_vector_store():
#     """Load vector store from disk if it exists."""
#     if VECTOR_STORE_PATH.exists():
#         logger.info("Loading vector store from cache...")
#         return FAISS.load_local(str(VECTOR_STORE_PATH), embeddings)
#     return None

def initialize_embeddings():
    """Initialize the embedding model with optimized settings."""
    global embeddings
    try:
        # Set torch to use CPU to avoid CUDA memory issues
        torch.set_num_threads(4)  # Limit CPU threads
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Embeddings model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        raise

def load_documents(file_paths: List[str]) -> List[str]:
    """Load documents from provided file paths with progress bar."""
    documents = []
    logger.info("Loading documents...")
    try:
        for file_path in tqdm(file_paths, desc="Loading documents"):
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {str(e)}")
                continue
        return documents
    except Exception as e:
        logger.error(f"Error in document loading: {str(e)}")
        raise

def preprocess_documents(documents: List[str]) -> List[str]:
    """Preprocess and split documents into chunks."""
    logger.info("Preprocessing documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    try:
        for doc in tqdm(documents, desc="Preprocessing documents"):
            chunks.extend(text_splitter.split_text(doc))
        return chunks
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def load_cached_vector_store():
    """Load vector store from disk if it exists."""
    if VECTOR_STORE_PATH.exists():
        logger.info("Loading vector store from cache...")
        return FAISS.load_local(
            str(VECTOR_STORE_PATH), 
            embeddings,
            allow_dangerous_deserialization=True  # Add this flag
        )
    return None

def save_vector_store(vector_store, documents: List[str]):
    """Save vector store and document hash to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Save vector store with the safety flag
    vector_store.save_local(
        str(VECTOR_STORE_PATH),
        allow_dangerous_deserialization=True  # Add this flag
    )
    
    # Save document hash
    with open(DOCUMENT_HASH_PATH, 'w') as f:
        f.write(compute_documents_hash(documents))
    
    logger.info("Vector store and hash saved to disk")

def initialize_vector_store(documents: List[str], chunks: List[str]):
    """Initialize or load FAISS vector store."""
    global vector_store
    try:
        # Check if we can use cached vector store
        if not should_rebuild_vector_store(documents):
            vector_store = load_cached_vector_store()
            if vector_store is not None:
                logger.info("Using cached vector store")
                return

        logger.info("Building new vector store...")
        # Process in batches
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size), desc="Building vector store"):
            batch = chunks[i:i + batch_size]
            if i == 0:
                vector_store = FAISS.from_texts(
                    batch, 
                    embeddings,
                    allow_dangerous_deserialization=True  # Add this flag
                )
            else:
                batch_vectorstore = FAISS.from_texts(
                    batch, 
                    embeddings,
                    allow_dangerous_deserialization=True  # Add this flag
                )
                vector_store.merge_from(batch_vectorstore)

        # Save to disk for future use
        save_vector_store(vector_store, documents)
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("Vector store initialization completed")
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

def setup_qa_chain():
    """Set up the QA chain with Groq LLM."""
    try:
        llm = ChatGroq(
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.2-3b-preview"
        )
        
        prompt_template = """You are a medical information assistant. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return chain
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    global is_ready
    try:
        logger.info("Starting service initialization...")
        
        # Initialize embeddings first
        initialize_embeddings()
        
        # Load documents
        documents = load_documents(["MSD -  Health_topics_data.xlsx - Sheet1.csv"])
        if not documents:
            logger.warning("No documents were loaded successfully")
            return
        
        # Preprocess documents
        chunks = preprocess_documents(documents)
        if not chunks:
            logger.warning("No chunks were created during preprocessing")
            return
        
        # Initialize vector store
        initialize_vector_store(documents, chunks)
        
        is_ready = True
        logger.info("Service initialized successfully and ready to accept requests")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        is_ready = False
        raise

@app.get("/status", response_model=ServiceStatus)
async def get_status():
    """Check the current status of the service."""
    if is_ready:
        return ServiceStatus(status="ready", message="Service is ready to accept requests")
    return ServiceStatus(status="initializing", message="Service is still initializing")

@app.post("/answer", response_model=Answer)
async def get_answer(question: Question):
    """Generate answer for the given question."""
    # First check if the question is empty
    if not question.question or not question.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    # Then check if service is ready
    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again in a few moments."
        )
        
    try:
        logger.info(f"Received question: {question.question}")
        
        qa_chain = setup_qa_chain()
        if not qa_chain:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize QA chain"
            )
            
        result = qa_chain({"query": question.question})
        if not result or "result" not in result:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate answer"
            )
        
        logger.info("Answer generated successfully")
        return Answer(answer=result["result"])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")