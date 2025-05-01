from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import json
import os
import uvicorn
import sys
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import logging
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _4_embedding_store.main import store_chunks_in_supabase, download_file_from_supabase, list_chunked_json_files_from_supabase
from auth.dependencies import get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_DIR = TEMP_DIR / "embedding"

os.makedirs(INPUT_DIR, exist_ok=True)


@router.get("/sessions/{session_id}/embedding")
def process_json_files(
    session_id: str,
    current_user: str = Depends(get_current_user)  
):
    
    user_id = current_user
    try:
        chunked_files = list_chunked_json_files_from_supabase(user_id, session_id)
        if not chunked_files:
            raise HTTPException(status_code=404, detail="No chunked JSON files found in Supabase.")

        results = []
        for file_name in chunked_files:
            file_path = download_file_from_supabase(file_name, user_id, session_id)
            with open(file_path, "r", encoding="utf-8") as json_file:
                chunks = json.load(json_file)

            store_chunks_in_supabase(chunks)

            results.append({
                "file_name": file_name,
                "status": "stored"
            })

        return {"message": "All chunks embedded.", "results": results}

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


    finally:
        if TEMP_DIR.exists():
            try:
                shutil.rmtree(TEMP_DIR)
                logger.info(f"Temporary directory {TEMP_DIR} removed.")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {TEMP_DIR}: {e}")


@router.get("/")
def home():
    return {"message": "API is running!"}


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8300)


