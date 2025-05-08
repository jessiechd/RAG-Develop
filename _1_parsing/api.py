from fastapi import APIRouter, UploadFile, File, Depends
from pathlib import Path
import logging
import sys
import os
from supabase import create_client, Client
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _1_parsing.main import convert_and_save, extract_nodes, download_file_from_supabase_parsing
from auth.dependencies import get_current_user  


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
# INPUT_DIR = BASE_DIR / "input_pdfs"
# OUTPUT_DIR = BASE_DIR.parent / "_2_image" / "input_md"

TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_DIR = TEMP_DIR / "input_pdfs"
OUTPUT_DIR = TEMP_DIR / "parsed"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


@router.post("/session/{session_id}/upload")
async def upload_file_to_session(
    session_id: str,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    user_id = current_user  
    file_location = f"user_{user_id}/{session_id}/input_pdfs/{file.filename}"
    file_path = Path(file.filename)

    file_content = await file.read()

    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(str(file_location), file_content)
        logger.info(f"File uploaded to Supabase Storage at {file_location}")
    except Exception as e:
        logger.error(f"Error uploading file to Supabase: {e}")
        return {"message": "Failed to upload file to Supabase Storage."}

    file_name = file.filename
    file_path = file_name
    downloaded_file = download_file_from_supabase_parsing(file_path, current_user=user_id, session_id=session_id)
    if not downloaded_file:
        return {"message": "File download failed."}

    convert_and_save(user_id=user_id, session_id=session_id)

    md_filename = Path(file.filename).with_suffix(".md")
    md_file_path = OUTPUT_DIR / f"{md_filename.stem}.md" 

    if md_file_path.exists():
        extract_nodes(md_file_path, user_id, session_id)  

        with open(md_file_path, "r", encoding="utf-8") as md_file:
            md_content = md_file.read()

        md_file_location = f"user_{user_id}/{session_id}/parsed/{md_filename.name}"
        supabase.storage.from_(SUPABASE_BUCKET).upload(md_file_location, md_content.encode('utf-8'))
        logger.info(f"Markdown file uploaded to Supabase Storage at {md_file_location}")

        artifact_dir = OUTPUT_DIR / f"{md_filename.stem}_artifacts"
        if artifact_dir.exists():
            for image_path in artifact_dir.glob("*"):
                if image_path.exists():
                    with open(image_path, "rb") as img_file:
                        img_content = img_file.read()
                    image_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_artifacts/{image_path.name}"
                    supabase.storage.from_(SUPABASE_BUCKET).upload(image_location, img_content)
                    logger.info(f"Artifact image uploaded to Supabase Storage at {image_location}")

        json_file_path = OUTPUT_DIR / f"{md_filename.stem}_nodes.json"
        if json_file_path.exists():
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                json_content = json_file.read()
            json_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_nodes.json"
            supabase.storage.from_(SUPABASE_BUCKET).upload(json_location, json_content.encode('utf-8'))
            logger.info(f"JSON nodes file uploaded to Supabase Storage at {json_location}")
        else:
            logger.error(f"JSON file not found: {json_file_path}")

        try:
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Temporary directory {TEMP_DIR} removed.")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {TEMP_DIR}: {e}")

        return {
            "message": f"File '{file.filename}' berhasil diunggah dan diproses.",
            "markdown_file": md_filename.name,
            "json_nodes_file": f"{md_filename.stem}_nodes.json"
        }
    else:
        return {"message": "Gagal memproses file."}


@router.get("/session")
def list_sessions(current_user: str = Depends(get_current_user)):
    email = current_user
    user_data = supabase.table("users").select("id").eq("email", email).execute()
    if user_data.data:
        user_id = user_data.data[0]["id"]
    else:
        raise HTTPException(status_code=404, detail="User not found")

    response = supabase.table("sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return response.data


@router.get("/")
def home():
    return {"message": "API is running!"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

