import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
import shutil
import uuid
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _5_retrieval_llm.main import query_supabase, call_openai_llm
from auth.dependencies import get_current_user

# llm_app = FastAPI(title="Retrieval and LLM API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_CONNECTION = os.getenv("DB_CONNECTION")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

def get_user_uuid_by_email(email: str) -> str:
    """Function to get the UUID for a user by email from the users table in Supabase."""
    response = supabase.table("users").select("id").eq("email", email).execute()
    if response.data:
        return response.data[0]["id"]
    else:
        raise ValueError(f"User with email {email} not found.")

def save_chat_message(user_id, session_id, role, message):
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()  # Convert to string
    }).execute()

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
    email = current_user["email"]
    user_uuid = get_user_uuid_by_email(email)

    try:
        retrieved_chunks = query_supabase(request.user_query, user_uuid, session_id)
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

    email = current_user["email"]
    user_uuid = get_user_uuid_by_email(email)

    try:
        retrieved_chunks = query_supabase(request.user_query, user_uuid, session_id)
        answer, chat_history = call_openai_llm(
            request.user_query, retrieved_chunks, request.chat_history
        )

        save_chat_message(user_uuid, session_id, "user", request.user_query)
        save_chat_message(user_uuid, session_id, "assistant", answer)

        return {"answer": answer, "chat_history": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/history")
def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):

    user_id = current_user
    user_uuid = get_user_uuid_by_email(current_user["email"])

    try:

        response = supabase.table("chat_history") \
            .select("role, message, timestamp") \
            .eq("user_id", user_uuid) \
            .eq("session_id", session_id) \
            .order("timestamp", desc=False) \
            .execute()

        history = response.data
        return {"chat_history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8400)

