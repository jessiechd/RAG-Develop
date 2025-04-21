from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import logging
import sys
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _1_parsing.main import convert_and_save, extract_nodes

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_pdfs"
OUTPUT_DIR = BASE_DIR.parent / "_2_image" / "input_md"


@app.post("/parsing")
async def upload_and_process_file(file: UploadFile = File(...)):
    file_location = INPUT_DIR / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    convert_and_save()

    md_filename = Path(file.filename).with_suffix(".md")
    md_file_path = OUTPUT_DIR / md_filename

    if md_file_path.exists():
        extract_nodes(md_file_path)
        return {
            "message": f"File '{file.filename}' berhasil diunggah dan diproses.",
            "markdown_file": md_filename.name,
            "json_nodes_file": f"{md_filename.stem}_nodes.json"
        }
    else:
        return {"message": "Gagal memproses file."}

@app.get("/")
def home():
    return {"message": "API is running!"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

