from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import shutil
import json
import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from _1_parsing.main import convert_and_save, extract_nodes, download_file_from_supabase_parsing
from _2_image.main import process_markdown_files
from _3_chunking.main import process_markdown, list_markdown_files_from_supabase
from _4_embedding_store.main import store_chunks_in_supabase, list_chunked_json_files_from_supabase, download_file_from_supabase
from auth.dependencies import get_current_user
import logging
from dotenv import load_dotenv
from supabase import Client, create_client
from pydantic import BaseModel
from datetime import datetime
from collections import defaultdict


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
TEMP_DIR_parsing = BASE_DIR.parent / "_1_parsing" / "temp"
os.makedirs(TEMP_DIR_parsing, exist_ok=True)

INPUT_DIR_parsing = TEMP_DIR_parsing / "input_pdfs"
OUTPUT_DIR_parsing = TEMP_DIR_parsing / "parsed"
os.makedirs(INPUT_DIR_parsing, exist_ok=True)
os.makedirs(OUTPUT_DIR_parsing, exist_ok=True)

TEMP_DIR_img = BASE_DIR.parent / "_2_image" / "temp"
os.makedirs(TEMP_DIR_img, exist_ok=True)
INPUT_DIR_img = TEMP_DIR_img / "parsed"
OUTPUT_DIR_img = TEMP_DIR_img / "img"
os.makedirs(INPUT_DIR_img, exist_ok=True)
os.makedirs(OUTPUT_DIR_img, exist_ok=True)

TEMP_DIR_chunking = BASE_DIR.parent / "_3_chunking" / "temp"
os.makedirs(TEMP_DIR_chunking, exist_ok=True)
INPUT_DIR_chunking = TEMP_DIR_chunking / "img"
OUTPUT_DIR_chunking = TEMP_DIR_chunking / "chunking"
os.makedirs(INPUT_DIR_chunking, exist_ok=True)
os.makedirs(OUTPUT_DIR_chunking, exist_ok=True)

TEMP_DIR_embedding = BASE_DIR.parent / "_4_embedding_store" / "temp"
os.makedirs(TEMP_DIR_embedding, exist_ok=True)
INPUT_DIR_embedding = TEMP_DIR_embedding / "embedding"
os.makedirs(INPUT_DIR_embedding, exist_ok=True)


def get_user_uuid_by_email(email: str) -> str:
    """Function to get the UUID for a user by email from the users table in Supabase."""
    response = supabase.table("users").select("id").eq("email", email).execute()
    if response.data:
        return response.data[0]["id"]
    else:
        raise ValueError(f"User with email {email} not found.")
        

# def list_files_recursively(storage, path):
#     # Ambil file dalam folder pertama
#     files = storage.from_("docs").list(path=path)

#     # Simpan file yang ditemukan
#     all_files = files

#     # Periksa apakah ada folder dalam hasil
#     subfolders = [f["name"] for f in files if f["name"].endswith('/')]
    
#     for subfolder in subfolders:
#         # Rekursif untuk setiap subfolder
#         all_files.extend(list_files_recursively(storage, f"{path}{subfolder}"))
    
#     return all_files

# def list_files_with_url(storage, path):
#     # Ambil file dalam folder
#     files = storage.from_("docs").list(path=path)

#     # Ambil URL untuk setiap file
#     for file in files:
#         # Get the public URL directly (without the need to access a dictionary)
#         file["download_url"] = storage.from_("docs").get_public_url(file["name"])
    
#     return files


# def list_files_recursively_with_urls(storage, path):
#     items = storage.from_("docs").list(path=path)
#     all_files = []

#     print(f"Listing items in {path}: {items}")  # Cek setiap folder dan file yang ditemukan

#     for item in items:
#         item_path = f"{path}/{item['name']}" if not path.endswith("/") else f"{path}{item['name']}"
        
#         print(f"Processing item: {item_path}")  # Debug output untuk setiap item
        
#         if item['name'].endswith("/"):  # Folder ditemukan
#             print(f"Found folder: {item['name']}")
#             folder_contents = {
#                 "folder_name": item['name'],
#                 "contents": list_files_recursively_with_urls(storage, item_path)  # Rekursi
#             }
#             all_files.append(folder_contents)
#         else:  # File ditemukan
#             print(f"Found file: {item['name']}")
#             item["full_path"] = item_path
#             item["download_url"] = storage.from_("docs").get_public_url(item_path)
#             all_files.append(item)
    
#     return all_files

def build_folder_tree(files, storage):
    tree = {}

    for file in files:
        parts = file['name'].split('/')
        current = tree

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Ini file
                current.setdefault('files', []).append({
                    "name": part,
                    "full_path": file['name'],
                    "download_url": storage.from_("docs").get_public_url(file['name'])
                })
            else:
                # Ini folder
                current = current.setdefault('folders', {}).setdefault(part, {})

    def format_tree(tree_dict):
        result = []
        for folder_name, content in tree_dict.get('folders', {}).items():
            result.append({
                "folder_name": folder_name,
                "contents": format_tree(content)
            })
        for file in tree_dict.get('files', []):
            result.append(file)
        return result

    return format_tree(tree)


def list_files_with_folder_structure(storage, prefix: str):
    all_items = storage.from_("docs").list(path="")  # Hati-hati: ini non-rekursif
    matching_items = [item for item in all_items if item["name"].startswith(prefix)]
    return build_folder_tree(matching_items, storage)


def list_files_recursively_by_prefix(storage, path=""):
    items = storage.from_("docs").list(path=path)
    all_files = []

    for item in items:
        item_path = f"{path}/{item['name']}" if path else item['name']

        # Deteksi folder
        if 'metadata' not in item or not item['metadata']:
            folder_contents = list_files_recursively_by_prefix(storage, item_path)
            if folder_contents:
                all_files.append({
                    "folder_name": item['name'],
                    "contents": folder_contents
                })
        else:
            file_url = storage.from_("docs").get_public_url(item_path)
            all_files.append({
                "name": item['name'],
                "full_path": item_path,
                "download_url": file_url
            })

    return all_files

def list_folder_structure(storage, path=""):
    """
    Fungsi untuk menelusuri folder dan file secara rekursif.
    """
    items = storage.from_("docs").list(path=path)
    result = []

    for item in items:
        item_name = item["name"]
        item_path = f"{path.rstrip('/')}/{item_name}"  # Menghindari '/' ekstra di akhir path

        # Cek apakah item tersebut adalah folder
        if item_name.endswith("/"):
            # Jika item adalah folder, lanjutkan rekursi untuk menelusuri isi folder tersebut
            folder_contents = list_folder_structure(storage, item_path)
            result.append({
                "folder_name": item_name.rstrip("/"),  # Menghilangkan '/' di nama folder
                "contents": folder_contents
            })
        else:
            # Jika item adalah file, tambahkan file ke dalam result
            result.append({
                "name": item_name,
                "full_path": item_path
            })

    return result


class SessionCreate(BaseModel):
    session_name: str | None = None 

@router.get("/sessionadmin")
def list_sessions(current_user: dict = Depends(get_current_user)):
    """
    Fungsi untuk mengembalikan daftar session untuk admin atau user.
    """
    user_id = current_user["email"]
    role = current_user["role"]
    storage = supabase.storage

    all_sessions = []

    if role == "admin":
        # Ambil semua folder di root (untuk admin)
        top_folders = storage.from_("docs").list(path="")

        for folder in top_folders:
            folder_name = folder["name"]
            prefix = folder_name if folder_name.endswith("/") else f"{folder_name}/"
            files = list_folder_structure(storage, prefix)
            if files:
                all_sessions.append({
                    "user_folder": folder_name,
                    "contents": files
                })
    else:
        # Untuk user biasa, ambil folder khusus sesuai user_id
        prefix = f"user_{user_id}/"
        files = list_folder_structure(storage, prefix)
        all_sessions.append({
            "user_folder": prefix,
            "contents": files
        })

    if not all_sessions:
        raise HTTPException(status_code=404, detail="No sessions found.")

    return all_sessions




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
    current_user: dict = Depends(get_current_user)
):

    
    chunking_results = []
    embedding_results = []

    user_id = current_user["email"]
    user_uuid = get_user_uuid_by_email(user_id) 
    try:
        file_location_parsing = f"user_{user_id}/{session_id}/input_pdfs/{file.filename}"
        file_content = await file.read()

        # Upload file for parsing
        supabase.storage.from_(SUPABASE_BUCKET).upload(file_location_parsing, file_content)
        logger.info(f"File uploaded to Supabase Storage at {file_location_parsing}")

        now_str = datetime.utcnow().isoformat()
        supabase.table("documents_track").insert({
            "session_id": session_id,
            "user_id": user_uuid,
            "file_name": file.filename,
            "file_path": file_location_parsing,
            "uploaded_at": now_str
        }).execute()

        file_name = file.filename
        file_path = f"input_pdfs/{file_name}"
        downloaded_file = download_file_from_supabase_parsing(file_path, current_user=user_id, session_id=session_id)

        if not downloaded_file:
            raise HTTPException(status_code=500, detail="File download failed.")

        convert_and_save(user_id=user_id, session_id=session_id)
        md_filename = Path(file.filename).with_suffix(".md")
        md_file_path = OUTPUT_DIR_parsing / f"{md_filename.stem}.md" 

        if md_file_path.exists():
            extract_nodes(md_file_path, user_id, session_id)

            # Upload markdown file
            with open(md_file_path, "r", encoding="utf-8") as md_file:
                md_content = md_file.read()
            md_file_location = f"user_{user_id}/{session_id}/parsed/{md_filename.name}"
            supabase.storage.from_(SUPABASE_BUCKET).upload(md_file_location, md_content.encode("utf-8"))

            # Upload artifacts
            artifact_dir = OUTPUT_DIR_parsing / f"{md_filename.stem}_artifacts"
            if artifact_dir.exists():
                for image_path in artifact_dir.glob("*"):
                    with open(image_path, "rb") as img_file:
                        img_content = img_file.read()
                    image_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_artifacts/{image_path.name}"
                    supabase.storage.from_(SUPABASE_BUCKET).upload(image_location, img_content)

            # Upload JSON nodes file
            json_file_path = OUTPUT_DIR_parsing / f"{md_filename.stem}_nodes.json"
            if json_file_path.exists():
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    json_content = json_file.read()
                json_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_nodes.json"
                supabase.storage.from_(SUPABASE_BUCKET).upload(json_location, json_content.encode("utf-8"))
            else:
                logger.error(f"JSON file not found: {json_file_path}")

        # Image processing
        result = process_markdown_files(INPUT_DIR_img, OUTPUT_DIR_img, user_id, session_id)
        if "Error" in result.get("message", ""):
            raise HTTPException(status_code=500, detail=result["message"])

        # Chunking
        files = list_markdown_files_from_supabase(user_id, session_id)
        if not files:
            raise HTTPException(status_code=404, detail="No markdown files found.")

        for file_name in files:
            url = process_markdown(file_name, user_id, session_id)
            chunking_results.append({"file_name": file_name, "json_url": url})

        # Embedding
        chunked_files = list_chunked_json_files_from_supabase(user_id, session_id)
        if not chunked_files:
            raise HTTPException(status_code=404, detail="No chunked JSON files found in Supabase.")


        results = []
        for file_name in chunked_files:
            logger.info(f"Downloading: {file_name}")
            file_path_e = download_file_from_supabase(file_name, user_id, session_id)
            with open(file_path_e, "r", encoding="utf-8") as json_file:
                chunks = json.load(json_file)

            store_chunks_in_supabase(chunks)

            results.append({
                "file_name": file_name,
                "status": "stored"
            })

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for temp_dir in [TEMP_DIR_parsing, TEMP_DIR_embedding]:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Temporary directory {temp_dir} removed.")
                except Exception as e:
                    logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")

    return {
        "message": "Pipeline completed successfully.",
        "chunking_results": chunking_results,
        "embedding_results": embedding_results
    }
    
