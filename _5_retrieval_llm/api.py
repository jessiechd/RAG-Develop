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
from typing import List, Optional


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _5_retrieval_llm.main import query_supabase, call_openai_llm, get_accessible_session_ids
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
        "timestamp": datetime.utcnow().isoformat()  
    }).execute()

def validate_user_access_to_session(user_uuid: str, session_id: str) -> None:
    session = supabase.table("sessions").select("*").eq("id", session_id).single().execute().data
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    is_owner = session["created_by"] == user_uuid
    is_public = session.get("is_public", False)
    allowed_roles = session.get("allowed_roles", [])
    user_role_resp = supabase.table("users").select("user_role").eq("id", user_uuid).single().execute()
    user_role = user_role_resp.data["user_role"] if user_role_resp.data else None

    if not (is_owner or is_public or user_role in allowed_roles):
        raise HTTPException(status_code=403, detail="You do not have access to this session's chat history")


def get_user_role_by_id(user_id: str) -> str:
    user = supabase.table("users").select("user_role").eq("id", user_id).single().execute()
    return user.data["user_role"] if user.data else None

def is_valid_uuid(value):
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False

def are_valid_uuids(session_ids):
    try:
        return all(uuid.UUID(sid) for sid in session_ids)
    except:
        return False



class QueryRequest(BaseModel):
    user_query: str
    chat_history: list = []
    session_ids: Optional[List[str]] = None 


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

@router.post("/query")
def query_documents(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    email = current_user["email"]
    user_uuid = get_user_uuid_by_email(email)

    try:
        session_ids = request.session_ids or get_accessible_session_ids(supabase, user_uuid)

        retrieved_chunks = query_supabase(request.user_query, user_uuid, session_ids)
        sanitized_chunks = sanitize(retrieved_chunks)

        return {"retrieved_chunks": sanitized_chunks}

    except Exception as e:
        print("Error in /query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
def chat_with_llm(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    email = current_user["email"]
    user_uuid = get_user_uuid_by_email(email)

    try:
        session_ids = request.session_ids or get_accessible_session_ids(supabase, user_uuid)

        retrieved_chunks = query_supabase(request.user_query, user_uuid, session_ids)

        answer, chat_history = call_openai_llm(
            request.user_query, retrieved_chunks, request.chat_history
        )

        if session_ids and len(session_ids) == 1 and is_valid_uuid(session_ids[0]):
            save_chat_message(user_uuid, session_ids[0], "user", request.user_query)
            save_chat_message(user_uuid, session_ids[0], "assistant", answer)

        return {
            "answer": answer,
            "chat_history": chat_history
        }

    except Exception as e:
        print("Error in /chat:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session/{session_id}/history")
def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_uuid = get_user_uuid_by_email(current_user["email"])

    validate_user_access_to_session(user_uuid, session_id)

    try:
        response = supabase.table("chat_history") \
            .select("role, message, timestamp") \
            .eq("user_id", user_uuid) \
            .eq("session_id", session_id) \
            .order("timestamp", desc=False) \
            .execute()

        return {"chat_history": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8400)

