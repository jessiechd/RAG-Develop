from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import shutil
import json
import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from _1_parsing.main import convert_and_save, extract_nodes, download_file_from_supabase
from _2_image.main import process_markdown_files
from _3_chunking.main import process_markdown, list_markdown_files_from_supabase
from _4_embedding_store.main import store_chunks_in_supabase, list_chunked_json_files_from_supabase
from auth.dependencies import get_current_user
import logging
from dotenv import load_dotenv
from supabase import Client, create_client
from pydantic import BaseModel

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

INPUT_DIR_parsing = TEMP_DIR / "input_pdfs"
OUTPUT_DIR_parsing = TEMP_DIR / "parsed"
os.makedirs(INPUT_DIR_parsing, exist_ok=True)
os.makedirs(OUTPUT_DIR_parsing, exist_ok=True)

INPUT_DIR_img = TEMP_DIR / "parsed"
OUTPUT_DIR_img = TEMP_DIR / "img"
os.makedirs(INPUT_DIR_img, exist_ok=True)
os.makedirs(OUTPUT_DIR_img, exist_ok=True)

INPUT_DIR_chunking = TEMP_DIR / "img"
OUTPUT_DIR_chunking = TEMP_DIR / "chunking"
os.makedirs(INPUT_DIR_chunking, exist_ok=True)
os.makedirs(OUTPUT_DIR_chunking, exist_ok=True)

INPUT_DIR_embedding = TEMP_DIR / "embedding"
os.makedirs(INPUT_DIR_embedding, exist_ok=True)

class SessionCreate(BaseModel):
    session_name: str | None = None  

@router.post("/session")
def create_session(data: SessionCreate, current_user: str = Depends(get_current_user)):
    user = supabase.table("users").select("id").eq("email", current_user).single().execute()
    user_id = user.data["id"]
    session_id = str(uuid4())
    session_name = data.session_name or f"Session-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    now_str = datetime.utcnow().isoformat()

    supabase.table("sessions").insert({
        "id": session_id,
        "user_id": user_id,
        "session_name": session_name,
        "created_at": now_str,
        "last_used": now_str
    }).execute()

    return {"session_id": session_id, "session_name": session_name}

@router.post("/session/{session_id}/run_pipeline")
async def run_pipeline(
    session_id: str,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    user_id = current_user
    file_location_parsing = f"user_{user_id}/{session_id}/input_pdfs/{file.filename}"
    
    # Upload file for parsing
    file_content = await file.read()
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(file_location_parsing, file_content)
        logger.info(f"File uploaded to Supabase Storage at {file_location_parsing}")
    except Exception as e:
        logger.error(f"Error uploading file to Supabase: {e}")
        return {"message": "Failed to upload file to Supabase Storage."}

    # Process parsing
    downloaded_file = download_file_from_supabase(file.filename, current_user=user_id, session_id=session_id)
    if not downloaded_file:
        return {"message": "File download failed."}
    
    convert_and_save(user_id=user_id, session_id=session_id)
    md_filename = Path(file.filename).with_suffix(".md")
    md_file_path = OUTPUT_DIR_parsing / f"{md_filename.stem}.md" 

    if md_file_path.exists():
        extract_nodes(md_file_path, user_id, session_id)  

        with open(md_file_path, "r", encoding="utf-8") as md_file:
            md_content = md_file.read()

        md_file_location = f"user_{user_id}/{session_id}/parsed/{md_filename.name}"
        supabase.storage.from_(SUPABASE_BUCKET).upload(md_file_location, md_content.encode('utf-8'))
        logger.info(f"Markdown file uploaded to Supabase Storage at {md_file_location}")

        # Upload artifacts (images)
        artifact_dir = OUTPUT_DIR_parsing / f"{md_filename.stem}_artifacts"
        if artifact_dir.exists():
            for image_path in artifact_dir.glob("*"):
                with open(image_path, "rb") as img_file:
                    img_content = img_file.read()
                image_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_artifacts/{image_path.name}"
                supabase.storage.from_(SUPABASE_BUCKET).upload(image_location, img_content)
                logger.info(f"Artifact image uploaded to Supabase Storage at {image_location}")

        # Upload JSON nodes file
        json_file_path = OUTPUT_DIR_parsing / f"{md_filename.stem}_nodes.json"
        if json_file_path.exists():
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                json_content = json_file.read()
            json_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_nodes.json"
            supabase.storage.from_(SUPABASE_BUCKET).upload(json_location, json_content.encode('utf-8'))
            logger.info(f"JSON nodes file uploaded to Supabase Storage at {json_location}")
        else:
            logger.error(f"JSON file not found: {json_file_path}")

    # Proceed with image processing
    result = process_markdown_files(input_folder=INPUT_DIR_img, output_folder=OUTPUT_DIR_img, current_user=current_user, session_id=session_id)
    if "Error" in result.get("message", ""):
        raise HTTPException(status_code=500, detail=result["message"])

    # Proceed with chunking
    try:
        files = list_markdown_files_from_supabase(user_id, session_id)
        if not files:
            raise HTTPException(status_code=404, detail="No markdown files found in Supabase.")

        results = []
        for file_name in files:
            url = process_markdown(file_name, user_id, session_id)
            results.append({"file_name": file_name, "json_url": url})

        return {"message": "All files processed.", "results": results}
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Proceed with embedding
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
            results.append({"file_name": file_name, "status": "stored"})

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
