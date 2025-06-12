import json
import re
import os
import argparse
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

def download_file_from_supabase(file_name: str, current_user: str, session_id: str) -> Path:
    """Download file from Supabase Storage and save locally."""
    try:
        supabase_path = f"user_{current_user}/{session_id}/img/{file_name}"
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

def list_markdown_files_from_supabase(user_id: str, session_id: str):
    folder_path = f"user_{user_id}/{session_id}/img"
    try:
        res = supabase.storage.from_(SUPABASE_BUCKET).list(path=folder_path)

        if not res:
            raise Exception("No response from Supabase")

        return [item["name"] for item in res if item["name"].endswith(".md")]

    except Exception as e:
        raise Exception(f"Failed to list files from Supabase: {str(e)}")


def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def is_table(chunk):
    return bool(re.search(r'^\|.*\|\n\|[-| ]+\|\n(\|.*\|\n)*', chunk, re.MULTILINE))

def extract_and_split_table(chunk, max_rows=10):
    lines = chunk.strip().split("\n")
    header, table_rows = None, []
    for i, line in enumerate(lines):
        if re.match(r'^\|[-| ]+\|$', line):
            header = lines[i - 1].strip("|").split("|")
            header = [h.strip() for h in header]
            continue
        if header:
            row_data = line.strip("|").split("|")
            row_data = [cell.strip() for cell in row_data]
            table_rows.append(row_data)
    
    table_chunks = [
        {"headers": header, "rows": table_rows[i:i + max_rows]}
        for i in range(0, len(table_rows), max_rows)
    ]
    return table_chunks if header and table_rows else None

def extract_section_title(header):
    match = re.match(r'^(#+)\s+(.*)', header.strip())
    return match.group(2) if match else None

def detect_table_title(pre_table_text):
    lines = pre_table_text.strip().split("\n")
    return lines[-1] if lines and len(lines[-1].split()) < 10 else None

def split_text(text, section_title, max_words=400, overlap=40):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        if start == 0:
            chunk = f"## {section_title}\n{chunk}"
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

def clean_dot_leaders(text):
    return re.sub(r'\.{2,}', '', text).strip()

def is_table_of_contents(table_rows):
    score = 0
    for row in table_rows:
        row_text = " ".join(row)
        if re.search(r'\.{5,}', row_text) and re.search(r'[A-Za-z]', row_text):
            score += 1
    return score >= len(table_rows) * 0.6

def structure_toc(table_rows, current_section):
    toc = []
    for row in table_rows:
        if len(row) == 1:
            title = clean_dot_leaders(row[0])
            toc.append({
                "number": None,
                "title": title,
                "section": current_section
            })
        elif len(row) >= 2:
            number = clean_dot_leaders(row[0])
            title = clean_dot_leaders(row[1])
            toc.append({
                "number": number,
                "title": title,
                "section": current_section
            })
    return toc

def autofill_toc_numbers(toc_items):
    counter = 1
    for item in toc_items:
        if not item.get("number") or item["number"] is None:
            item["number"] = str(counter)
            counter += 1
    return toc_items

def upload_file_to_supabase(file_name: str, local_path: Path, current_user: str, session_id: str):
    """Upload updated file back to Supabase Storage."""
    try:
        if isinstance(local_path, str):
            local_path = Path(local_path)

        supabase_path = f"user_{current_user}/{session_id}/chunking/{file_name}" 
        logger.debug(f"Uploading {file_name} to Supabase at {supabase_path}")

        with open(local_path, "rb") as f:
            data = f.read()

        content_type = (
            "text/markdown" if local_path.suffix == ".md"
            else "application/json" if local_path.suffix == ".json"
            else "image/jpeg"
        )

        response = supabase.storage.from_(SUPABASE_BUCKET).upload(supabase_path, data, {"content-type": content_type})

        logger.debug(f"Supabase upload response: {response}")

        return response

    except Exception as e:
        logger.error(f"Error uploading {file_name} to Supabase: {e}")
        return None

def run_tiny_llama(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_chunk_prompt(mini_chunks):
    annotated = "\n".join([f"<CHUNK{i}> {text}" for i, text in enumerate(mini_chunks)])
    instructions = (
        "You are a smart text segmenter. Group the annotated mini-chunks into larger, semantically coherent chunks.\n"
        "Each chunk should combine 2-4 mini-chunks that belong together in meaning.\n"
        "Respond with groups using this format:\n\n"
        "Chunk 1: <CHUNK0>, <CHUNK1>\n"
        "Chunk 2: <CHUNK2>, <CHUNK3>, <CHUNK4>\n"
    )
    return f"{instructions}\n\n{annotated}"

def agentic_chunk_text(text, section_title, max_chars=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    mini_chunks, temp = [], ""
    for sent in sentences:
        if len(temp) + len(sent) < max_chars:
            temp += " " + sent
        else:
            mini_chunks.append(temp.strip())
            temp = sent
    if temp:
        mini_chunks.append(temp.strip())

    if len(mini_chunks) <= 1:
        return [text.strip()], ["Kept as one chunk"]

    if len(mini_chunks) > 12:
        mini_chunks = mini_chunks[:12]

    prompt = build_chunk_prompt(mini_chunks)
    raw_output = run_tiny_llama(prompt)

    grouped_chunks = []
    explanation = []
    for line in raw_output.splitlines():
        if line.startswith("Chunk"):
            refs = re.findall(r"<CHUNK(\d+)>", line)
            if refs:
                valid_refs = [int(i) for i in refs if int(i) < len(mini_chunks)]
                if not valid_refs:
                    continue
                group_text = " ".join([mini_chunks[i] for i in valid_refs])
                cleaned_text = re.sub(r"\s+", " ", group_text.strip())
                grouped_chunks.append(cleaned_text)
                if len(valid_refs) > 1:
                    explanation.append(f"Chunk {len(grouped_chunks)}: grouped {len(valid_refs)} mini-chunks")
                else:
                    explanation.append(f"Chunk {len(grouped_chunks)}: kept mini-chunk")
    return grouped_chunks, explanation
    
def process_markdown(file_name: str, user_id: str, session_id: str):
    # Download file dari Supabase
    file_path = download_file_from_supabase(file_name, user_id, session_id)
    if not file_path:
        raise Exception("File not found or download failed")

    markdown_text = load_markdown(file_path)
    file_name_only = os.path.basename(file_path)
    sections = re.split(r'^(#+\s+.*)', markdown_text, flags=re.MULTILINE)
    final_chunks, current_section, chunk_id = [], "Unknown", 1

    for i in range(1, len(sections), 2):
        section_title = extract_section_title(sections[i]) or current_section
        content = sections[i + 1].strip()
        current_section = section_title
        table_matches = list(re.finditer(r'(\|.*\|\n\|[-| ]+\|\n(?:\|.*\|\n)+)', content, re.MULTILINE))
        last_index = 0

        for match in table_matches:
            start, end = match.span()
            pre_table_text = content[last_index:start].strip()
            table_text = match.group(0)
            last_index = end

            table_title = detect_table_title(pre_table_text)
            if pre_table_text:
                text_chunks, _ = agentic_chunk_text(pre_table_text, section_title)
                for chunk in text_chunks:
                    final_chunks.append({
                        "chunk_id": chunk_id,
                        "content": chunk,
                        "metadata": {
                            "source": file_name_only,
                            "section": section_title,
                            "position": chunk_id,
                            "user_id": user_id,
                            "session_id": session_id
                        }
                    })
                    chunk_id += 1
                    
            rows = [line.strip() for line in table_text.split("\n") if "|" in line]
            headers = [cell.strip() for cell in rows[0].strip("|").split("|")]
            data_rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[2:]]
            
            if is_table_of_contents(data_rows):
                toc_items.extend(structure_toc(data_rows, current_section))
            else:
                    final_chunks.append({
                        "chunk_id": chunk_id,
                        "table": {
                            "headers": headers,
                            "rows": data_rows
                        },
                        "metadata": {
                            "source": file_name_only,
                            "section": section_title,
                            "table_title": table_title,
                            "position": chunk_id,
                            "user_id": user_id,
                            "session_id": session_id
                        }
                    })
                    chunk_id += 1
        
        remaining_text = content[last_index:].strip()
        if remaining_text:
            text_chunks, _ = agentic_chunk_text(remaining_text, section_title)
            for chunk in text_chunks:
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "metadata": {
                        "source": file_name_only,
                        "section": section_title,
                        "position": chunk_id,
                        "user_id": user_id,
                        "session_id": session_id
                    }
                })
                chunk_id += 1

    toc_items = autofill_toc_numbers(toc_items)
    if toc_items:
        final_chunks.insert(0, {
            "chunk_id": 0,
            "toc_items": toc_items,
            "metadata": {
                "source": file_name,
                "section": "ToC",
                "position": 0,
                "user_id": user_id,
                "session_id": session_id
            }
        })

    output_file = OUTPUT_DIR / file_name_only.replace(".md", ".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(final_chunks, json_file, indent=4, ensure_ascii=False)
    print(f"Chunking completed. JSON saved to: {output_file}")


    upload_response = upload_file_to_supabase(output_file.name, output_file, user_id, session_id )

    if upload_response and isinstance(upload_response, dict) and "Key" in upload_response:
        file_path_in_storage = upload_response["Key"]
        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_path_in_storage}"
        logger.info(f"File successfully uploaded to Supabase: {file_url}")
        return file_url

    try:
        shutil.rmtree(TEMP_DIR) 
        print(f"Folder {TEMP_DIR} has been removed successfully.")
    except Exception as e:
        print(f"Error removing folder {TEMP_DIR}: {e}")
    
    return {"message": "Success"}


if __name__ == "__main__":

    # BASE_DIR = Path(__file__).resolve().parent

    # input_folder = BASE_DIR / "input_md"
    # output_folder_c = BASE_DIR.parent / "_4_embedding_store" / "input_json"
    # # os.makedirs(output_folder, exist_ok=True)
    
    # for file_name in os.listdir(input_folder):
    #     if file_name.endswith(".md"):
    #         input_path = os.path.join(input_folder, file_name)
    #         output_path = os.path.join(output_folder_c, file_name.replace(".md", ".json"))
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    process_markdown(file_name, current_user, session_id)




