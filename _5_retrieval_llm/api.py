import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import sys
import os
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _5_retrieval_llm.main import query_supabase, call_openai_llm
from auth.dependencies import get_current_user

# llm_app = FastAPI(title="Retrieval and LLM API")

router = APIRouter()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class QueryRequest(BaseModel):
    user_query: str
    chat_history: list = []

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


@router.get("/")
def root():
    return {"message": "Document Retrieval and LLM API is running."}

@router.post("/session/{session_id}/query")
def query_documents(
    session_id: str,
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):

    user_id = current_user

    try:
        retrieved_chunks = query_supabase(request.user_query, current_user, session_id)
        sanitized_chunks = sanitize(retrieved_chunks)
        return {"retrieved_chunks": sanitized_chunks}
    except Exception as e:
        print("Error in /query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/chat")
def chat_with_llm(
    session_id: str,
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):

    user_id = current_user

    try:
        retrieved_chunks = query_supabase(request.user_query, current_user, session_id)
        answer, chat_history = call_openai_llm(
            request.user_query, retrieved_chunks, request.chat_history
        )
        return {"answer": answer, "chat_history": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8400)

