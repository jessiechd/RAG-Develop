import os
import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def load_internvl_model():
    path = 'OpenGVLab/InternVL2_5-1B'
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  
        low_cpu_mem_usage=True,
        use_flash_attn=True if torch.cuda.is_available() else False,
        trust_remote_code=True
    ).eval().to(device)  

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer, device

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def load_image(image_file, input_size=448):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)
    return pixel_values

def extract_images_and_context(markdown_path):
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    image_data = []
    for i, line in enumerate(lines):
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            context_before = " ".join(lines[max(0, i-2):i]).strip()
            context_after = " ".join(lines[i+1:min(len(lines), i+3)]).strip()
            image_data.append((img_path, context_before, context_after))
    return image_data, lines

def generate_caption(model, tokenizer, image_path, context_before, context_after):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found - {image_path}")
        return "[Image description unavailable]"

    pixel_values = load_image(image_path)
    prompt = f"<image>\nContext: {context_before} ... {context_after}. Please describe the image shortly."
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response

def update_markdown(markdown_path, enriched_data, lines, output_folder, markdown_folder):

    new_lines = []
    for line in lines:
        new_lines.append(line)
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            caption = next((desc for img, _, _, desc in enriched_data if img == img_path), "[Image description unavailable]")

            new_lines.append(f"\n*Image Description:* {caption}\n")

    # os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, os.path.relpath(markdown_path, start=markdown_folder))

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def process_markdown_files(markdown_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model, tokenizer, device = load_internvl_model()

    for file in os.listdir(markdown_folder):
        if file.endswith(".md"):
            markdown_path = os.path.join(markdown_folder, file)
            filename_without_ext = os.path.splitext(file)[0]
            image_folder = os.path.join(markdown_folder, f"{filename_without_ext}_artifacts")

            if not os.path.exists(image_folder):
                print(f"Warning: Image folder '{image_folder}' not found for '{file}'")
                continue

            image_data, lines = extract_images_and_context(markdown_path)
            enriched_data = []
            for img_path, context_before, context_after in image_data:
                full_image_path = os.path.join(image_folder, img_path)
                caption = generate_caption(model, tokenizer, full_image_path, context_before, context_after)
                enriched_data.append((img_path, context_before, context_after, caption))

            output_path = os.path.join(output_folder, file)
            update_markdown(markdown_path, enriched_data, lines, output_folder, markdown_folder)
            print(f"Processed: {file}")

if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent
    markdown_folder = BASE_DIR / "input_md"
    output_folder = BASE_DIR.parent / "_3_chunking" / "input_md"

    process_markdown_files(markdown_folder, output_folder)
