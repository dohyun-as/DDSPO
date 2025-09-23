import os
import json
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 24
OUTPUT_DIR = "grids_geneval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = {
    "sana": [
        "../results/SANA/ours_beta2k_lora512/checkpoint-100/geneval_img",
        "../results/teacher/SANA_600M_1024/geneval_img"
    ]
}

def read_prompt(metadata_path: str) -> str:
    try:
        with open(metadata_path, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            return data.get("prompt", "(no prompt)")
    except Exception as e:
        print(f"⚠️ Failed to read prompt from {metadata_path}: {e}")
        return "(no prompt)"

def create_image_grid(images: List[Image.Image], prompt: str, group_name: str, index: int, out_path: str):
    group_dir = os.path.join(out_path, group_name)
    os.makedirs(group_dir, exist_ok=True)

    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    bbox = font.getbbox(prompt)
    prompt_height = bbox[3] - bbox[1] + 10

    grid_img = Image.new('RGB', (total_width, max_height + prompt_height), color=(255, 255, 255))

    x_offset = 0
    for img in images:
        grid_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    draw = ImageDraw.Draw(grid_img)
    draw.text((10, max_height + 5), prompt, font=font, fill=(0, 0, 0))

    grid_img.save(os.path.join(group_dir, f"{index:04d}.jpg"))

def process_group(group_name: str, paths: List[str]):
    print(f"📦 Processing group: {group_name}")

    subfolder_sets: List[set] = []
    for path in paths:
        subfolders = {
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f)) and os.path.exists(os.path.join(path, f, "samples", "00000.png"))
        }
        subfolder_sets.append(subfolders)

    shared_subfolders = sorted(set.intersection(*subfolder_sets))
    print(f"🖼️ Found {len(shared_subfolders)} shared folders")

    for idx, folder_name in enumerate(shared_subfolders):
        imgs = []
        prompt = ""
        for path in paths:
            base_folder = os.path.join(path, folder_name)
            img_path = os.path.join(base_folder, "samples", "00000.png")
            metadata_path = os.path.join(base_folder, "metadata.jsonl")

            with Image.open(img_path).convert("RGB") as img:
                imgs.append(img.copy())

            if not prompt:  # 첫 번째 path에서만 prompt 읽음
                prompt = read_prompt(metadata_path)

        create_image_grid(imgs, prompt, group_name, idx, OUTPUT_DIR)

if __name__ == "__main__":
    for group_name, paths in GROUPS.items():
        process_group(group_name, paths)
