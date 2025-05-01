from fastapi import APIRouter, Depends, HTTPException
import sys
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _2_image.main import process_markdown_files, download_all_files_from_folder
from auth.dependencies import get_current_user
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


# BASE_DIR = Path(__file__).resolve().parent
# INPUT_DIR = BASE_DIR / "input_md"
# OUTPUT_DIR = BASE_DIR.parent / "_3_chunking" / "input_md"

# INPUT_FOLDER = r"C:\\Grader\\RAG-Develop-main\\_2_image\\input_md"
# OUTPUT_FOLDER = r"C:\\Grader\\RAG-Develop-main\\_2_image\\output_md"


env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_DIR = TEMP_DIR / "parsed"
OUTPUT_DIR = TEMP_DIR / "img"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/session/{session_id}/img_description")
async def process_markdown(
    session_id: str,
    current_user: str = Depends(get_current_user)  
):
    """
    Endpoint to process markdown files with images.
    """
    user_id = current_user 
    input_folder = TEMP_DIR / "parsed"
    output_folder = TEMP_DIR / "img"
    
    result = process_markdown_files(input_folder=input_folder, output_folder=output_folder, current_user=current_user, session_id=session_id)

    if "Error" in result.get("message", ""):
        raise HTTPException(status_code=500, detail=result["message"])

    return {
        "message": result["message"],
        "output_folder": str(output_folder)
    }

@router.get("/")
def home():
    return {"message": "API is running!"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8100)