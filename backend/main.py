import os
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
from contextlib import asynccontextmanager # <--- 1. IMPORT THIS

# Import the logic from our other file
from rag_logic import build_database_if_needed, create_rag_chain

# --- STATE ---
# This will hold the chain once it's loaded.
rag_chain = None

# --- 2. CREATE THE NEW LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup, build the database from source files if it doesn't exist,
    then load the RAG chain into memory.
    """
    global rag_chain
    print("Server starting up...")
    retriever = build_database_if_needed()
    if retriever:
        rag_chain = create_rag_chain(retriever)
        print("✅ RAG chain is loaded and ready to answer questions.")
    else:
        print("❌ RAG chain could not be initialized. Check logs for errors.")
    
    yield
    
    # Code below yield runs on shutdown (optional)
    print("Server shutting down...")


# --- 3. INITIALIZE FASTAPI APP WITH THE LIFESPAN ---
app = FastAPI(title="Multi-Modal RAG API", lifespan=lifespan)


# --- DATA MODELS ---
class Query(BaseModel):
    question: str

# --- API ENDPOINTS (No changes needed here) ---
@app.post("/query/")
async def query_endpoint(query: Query):
    """Endpoint to ask a question to the RAG chain."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain is not ready. Please check the server logs.")
    
    try:
        result = await rag_chain.ainvoke(query.question)
        response_content = {
            "answer": result.get("response", "No answer generated."),
            "texts": [doc for doc in result.get("context", {}).get("texts", [])],
            "images": [img for img in result.get("context", {}).get("images", [])]
        }
        return JSONResponse(content=response_content)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during query: {e}")