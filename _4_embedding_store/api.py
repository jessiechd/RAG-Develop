from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import os
import uvicorn
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _4_embedding_store.main import store_chunks_in_supabase

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "input_json"

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.get("/embedding")
def process_json_files():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".json"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as json_file:
                json_chunks = json.load(json_file)
            store_chunks_in_supabase(json_chunks)
            print(f"Processed and stored: {filename}")
    return {"message": "All text and table embeddings stored successfully in Supabase!"}


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8300)

