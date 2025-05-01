from fastapi import APIRouter, Depends, HTTPException
import os
import json
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _3_chunking.main import process_markdown, list_markdown_files_from_supabase
from auth.dependencies import get_current_user 
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_DIR = TEMP_DIR / "img"
OUTPUT_DIR = TEMP_DIR / "chunking"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



@router.get("/session/{session_id}/chunking")
def process_all_markdown_files(
    session_id: str,
    current_user: str = Depends(get_current_user)  
):
    """
    Endpoint untuk memproses file markdown dari Supabase dan mengunggah hasilnya ke Supabase.
    """
    user_id = current_user
    try:
        files = list_markdown_files_from_supabase(user_id, session_id)
        if not files:
            raise HTTPException(status_code=404, detail="No markdown files found in Supabase.")

        results = []
        for file_name in files:
            url = process_markdown(file_name, user_id, session_id)
            results.append({
                "file_name": file_name,
                "json_url": url
            })

        return {"message": "All files processed.", "results": results}

    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def home():
    return {"message": "API is running!"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8200)
