import logging
import json
import os
from pathlib import Path
import warnings
import torch
import shutil
import re
from supabase import create_client, Client
from dotenv import load_dotenv
import tempfile
import time

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())
print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.settings import settings

warnings.filterwarnings("ignore")

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0


# INPUT_DIR = BASE_DIR / "input_pdfs"
# OUTPUT_DIR = BASE_DIR.parent / "_2_image" / "input_md"
# os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_DIR = TEMP_DIR / "input_pdfs"
OUTPUT_DIR = TEMP_DIR / "parsed"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_pipeline_options(input_format):
    if input_format == InputFormat.PDF:
        return PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_table_structure=True,
                generate_page_images=True,
                generate_picture_images=True,
                images_scale=IMAGE_RESOLUTION_SCALE,
            )
        )
    elif input_format == InputFormat.DOCX:
        return WordFormatOption(pipeline_cls=SimplePipeline)
    return None

def initialize_converter():
    allowed_formats = [InputFormat.PDF, InputFormat.DOCX]
    format_options = {fmt: create_pipeline_options(fmt) for fmt in allowed_formats if create_pipeline_options(fmt)}
    return DocumentConverter(allowed_formats=allowed_formats, format_options=format_options)

def download_file_from_supabase(file_name: str, current_user: str, session_id: str) -> Path:
    """Download file from Supabase Storage and save locally."""
    try:
        supabase_path = f"user_{current_user}/{session_id}/input_pdfs/{file_name}"
        file_bytes = supabase.storage.from_(SUPABASE_BUCKET).download(supabase_path)
        
        if not file_bytes:
            logger.error(f"Failed to download {file_name} from Supabase: Empty response")
            return None

        local_download_path = INPUT_DIR / file_name
        os.makedirs(local_download_path.parent, exist_ok=True)

        with open(local_download_path, "wb") as temp_file:
            temp_file.write(file_bytes)
        
        logger.info(f"File downloaded to {local_download_path}")
        return local_download_path

    except Exception as e:
        logger.error(f"Error downloading {file_name} from Supabase: {e}")
        return None
def upload_file_to_supabase(file_path: Path, dest_file_name: str, user_id: str, session_id: str) -> str:
    try:
        with open(file_path, "rb") as f:
            data = f.read()

        file_path_with_user = f"user_{user_id}/{session_id}/{dest_file_name}".replace("\\", "/")

        content_type = (
            "text/markdown" if file_path.suffix == ".md"
            else "application/json" if file_path.suffix == ".json"
            else "image/jpeg" if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".PNG"] else "application/octet-stream"
        )

        response = supabase.storage.from_(SUPABASE_BUCKET).upsert(
            file_path_with_user,
            data,
            {"content-type": content_type}
        )


        if response and hasattr(response, 'data') and response.data.get('Key'):
            public_url = f"https://{supabase.storage.url}/object/{SUPABASE_BUCKET}/{file_path_with_user}"
            logger.info(f"File uploaded to Supabase Storage at {public_url}")
            return public_url
        else:
            logger.error(f"Failed to upload file to Supabase Storage: {response}")
            return None
    except Exception as e:
        logger.error(f"Error uploading {file_path.name} to Supabase: {e}")
        return None




def convert_and_save(user_id: str, session_id: str):
    input_paths = list(INPUT_DIR.glob("*.pdf")) + list(INPUT_DIR.glob("*.docx"))

    if not input_paths:
        logger.warning("No input files found.")
        return

    converter = initialize_converter()
    results = converter.convert_all(input_paths)

    uploaded_images = set()

    for res in results:
        file_stem = res.input.file.stem
        original_filename = res.input.file.name

        # original_file_url = upload_file_to_supabase(
        #     file_path=res.input.file,
        #     dest_subfolder="input_md",  # Sesuai struktur yang kamu mau
        #     user_id=user_id
        # )

        # if not original_file_url:
        #     logger.error(f"Failed to upload original file for {file_stem}")
        #     continue

        artifact_dir_name = f"{file_stem}_artifacts"
        artifact_dir_path = OUTPUT_DIR / artifact_dir_name
        artifact_dir_path.mkdir(parents=True, exist_ok=True)

        md_path = OUTPUT_DIR / f"{file_stem}.md"
        res.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)

        uploaded_images = set()

        for page in res.document.pages:
            if hasattr(page, 'image') and page.image:
                image_path = Path(page.image)
                if image_path.exists() and image_path.name not in uploaded_images:
                    image_dest_path = artifact_dir_path / image_path.name
                    shutil.move(image_path, image_dest_path)

                    supabase_image_path = f"parsed/{file_stem}_artifacts/{image_dest_path.name}"
                    upload_file_to_supabase(image_dest_path, supabase_image_path, user_id, session_id)

                    uploaded_images.add(image_path.name)

                    upload_result = upload_file_to_supabase(image_dest_path, supabase_image_path, user_id, session_id)

                    if upload_result:
                        logger.info(f"Uploaded image: {image_dest_path.name} → {upload_result}")
                        uploaded_images.add(image_path.name)
                    else:
                        logger.error(f"Failed to upload image: {image_dest_path.name} to Supabase path: {supabase_image_path}")

        def replace_image_path(match):
            alt_text = match.group(1)
            image_file = Path(match.group(2)).name
            new_path = f"{artifact_dir_name}/{image_file}"
            return f"![{alt_text}]({new_path})"

        with open(md_path, 'r', encoding='utf-8') as file:
            md_content = file.read()

        md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_path, md_content)


        with open(md_path, 'w', encoding='utf-8') as file:
            file.write(md_content)


        supabase_md_path = f"parsed/{file_stem}.md"
        upload_file_to_supabase(md_path, supabase_md_path, user_id, session_id)

        extract_nodes(md_path, user_id, session_id)


        try:
            os.remove(res.input.file)
            logger.info(f"Original input file {res.input.file.name} removed.")
        except Exception as e:
            logger.error(f"Failed to remove input file {res.input.file.name}: {e}")

        # === 5. Ensure Upload is Complete before Cleanup ===
        # logger.info(f"File {file_stem} and its assets have been uploaded successfully.")

        # # === 6. Make sure folder has finished processing before cleanup ===
        # time.sleep(1)  # Optional: Give time to filesystem to process

    # After processing all files, clean up the temporary folder
    # parsed_folder = OUTPUT_DIR
    # try:
    #     if parsed_folder.exists() and parsed_folder.is_dir():
    #         shutil.rmtree(parsed_folder)
    #         logger.info(f"Temporary folder {parsed_folder} has been removed.")
    # except Exception as e:
    #     logger.error(f"Failed to remove folder {parsed_folder}: {e}")


def upload_node_json_to_supabase(json_path: Path, user_id: str, dest_file_name: str, session_id: str) -> str:
    """Upload extracted node JSON to Supabase Storage under parsed folder with user_id."""
    try:
        with open(json_path, "rb") as f:
            data = f.read()

        relative_json_path = str(f"user_{user_id}/{session_id}/{dest_file_name}".replace("\\", "/"))
        
        response = supabase.storage.from_(SUPABASE_BUCKET).upsert(
            relative_json_path,
            data,
            {"content-type": "application/json"}
        )

        if response and response.get("Key"):
            public_url = f"https://{supabase.storage.url}/object/{SUPABASE_BUCKET}/{relative_json_path}"
            logger.info(f"Node JSON uploaded to Supabase Storage at {public_url}")
            return public_url
        else:
            logger.error(f"Failed to upload node JSON to Supabase Storage: {response}")
            return None

    except Exception as e:
        logger.error(f"Error uploading node JSON {dest_file_name} to Supabase: {e}")
        return None


def extract_nodes(md_file_path, user_id: str, session_id: str):
    """Extracts nodes from a markdown file, including image references, and uploads JSON with user_id."""
    output_json_path = OUTPUT_DIR / f"{md_file_path.stem}_nodes.json"

    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except Exception as e:
        logger.error(f"Error reading {md_file_path}: {e}")
        return
    
    nodes = []
    text_block = ""

    for line in markdown_content.split('\n'):
        if '![' in line and '(' in line and ')' in line:
            parts = line.split('(')
            image_path = parts[1].split(')')[0] if len(parts) > 1 else None
            node_text = parts[0].split('[')[1].split(']')[0] if '[' in parts[0] else ""
            if text_block.strip():
                nodes.append({"index": len(nodes) + 1, "text": text_block.strip(), "image_path": None})
            if image_path:
                relative_image_path = image_path.replace("\\", "/")
                nodes.append({"index": len(nodes) + 1, "text": node_text, "image_path": relative_image_path})
            else:
                nodes.append({"index": len(nodes) + 1, "text": node_text, "image_path": None})
            text_block = ""
        else:
            text_block += line + "\n"
    
    if text_block.strip():
        nodes.append({"index": len(nodes) + 1, "text": text_block.strip(), "image_path": None})

    try:
        with open(output_json_path, "w", encoding="utf-8") as fp:
            json.dump({"file_name": md_file_path.name, "nodes": nodes}, fp, indent=4)
        logger.info(f"Extracted {len(nodes)} nodes from {md_file_path.name} to {output_json_path.name}")

        supabase_json_path = f"parsed/{output_json_path.name}".replace("\\", "/")
        upload_node_json_to_supabase(output_json_path, user_id, supabase_json_path, session_id)
    except Exception as e:
        logger.error(f"Error saving JSON for {md_file_path.name}: {e}")

def main(user_id: str):
    settings.debug.profile_pipeline_timings = True
    
    convert_and_save(user_id)

    for md_file in OUTPUT_DIR.glob("*.md"):
        extract_nodes(md_file, user_id)

if __name__ == "__main__":
    main(user_id)