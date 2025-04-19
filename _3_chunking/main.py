import json
import re
import os
import argparse

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

def process_markdown(file_path, output_file):
    markdown_text = load_markdown(file_path)
    file_name = os.path.basename(file_path)
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
                text_chunks = split_text(pre_table_text, section_title)
                for chunk in text_chunks:
                    final_chunks.append({
                        "chunk_id": chunk_id,
                        "content": chunk,
                        "metadata": {"source": file_name, "section": section_title, "position": chunk_id}
                    })
                    chunk_id += 1
            
            table_chunks = extract_and_split_table(table_text)
            if table_chunks:
                for table_chunk in table_chunks:
                    final_chunks.append({
                        "chunk_id": chunk_id,
                        "table": table_chunk,
                        "metadata": {"source": file_name, "section": section_title, "table_title": table_title, "position": chunk_id}
                    })
                    chunk_id += 1
        
        remaining_text = content[last_index:].strip()
        if remaining_text:
            text_chunks = split_text(remaining_text, section_title)
            for chunk in text_chunks:
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "metadata": {"source": file_name, "section": section_title, "position": chunk_id}
                })
                chunk_id += 1
    
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(final_chunks, json_file, indent=4, ensure_ascii=False)
    print(f"Chunking completed. JSON saved to: {output_file}")

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    input_folder = BASE_DIR / "input_md"
    output_folder = BASE_DIR.parent / "_4_embedding_store" / "input_json"
    # os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".md"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".md", ".json"))
            process_markdown(input_path, output_path)
