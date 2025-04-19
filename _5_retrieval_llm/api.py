import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _5_retrieval_llm.main import query_supabase, call_openai_llm

llm_app = FastAPI(title="Retrieval and LLM API")

class QueryRequest(BaseModel):
    user_query: str
    chat_history: list = []

# Convert anything that FastAPI can't serialize
def sanitize(obj):
    if isinstance(obj, np.generic): 
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize(i) for i in obj)
    return obj

@llm_app.get("/")
def root():
    return {"message": "Document Retrieval and LLM API is running."}

@llm_app.post("/query")
def query_documents(request: QueryRequest):
    try:
        retrieved_chunks = query_supabase(request.user_query)
        sanitized_chunks = sanitize(retrieved_chunks)
        return {"retrieved_chunks": sanitized_chunks}
    except Exception as e:
        print("Error in /query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@llm_app.post("/chat")
def chat_with_llm(request: QueryRequest):
    try:
        retrieved_chunks = query_supabase(request.user_query)
        answer, chat_history = call_openai_llm(
            request.user_query, retrieved_chunks, request.chat_history
        )
        return {"answer": answer, "chat_history": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8400)

app = llm_app