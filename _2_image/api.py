from fastapi import APIRouter
import sys
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _2_image.main import process_markdown_files

router = APIRouter()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_md"
OUTPUT_DIR = BASE_DIR.parent / "_3_chunking" / "input_md"

# INPUT_FOLDER = r"C:\\Grader\\RAG-Develop-main\\_2_image\\input_md"
# OUTPUT_FOLDER = r"C:\\Grader\\RAG-Develop-main\\_2_image\\output_md"

@router.post("/image_description")
def process_markdown():
    """
    Endpoint to process markdown files with images.
    """
    if not os.path.exists(INPUT_DIR):
        return {"error": "Input folder does not exist"}

    process_markdown_files(INPUT_DIR, OUTPUT_DIR)
    return {"message": "Processing completed", "output_folder": OUTPUT_DIR}

@router.get("/")
def home():
    return {"message": "API is running!"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8100)