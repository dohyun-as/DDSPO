import os
import json
from PIL import Image, ImageDraw, ImageFont
from typing import List
import math

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # 필요시 경로 수정
FONT_SIZE = 24
OUTPUT_DIR = "grids"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = {
    # "sdxl": [
    #     "../results/SDXL/our_beta16k_sdxl/checkpoint-100/qualitative_pickapic_img",
    #     "../results/teacher/SDXL-1/qualitative_pickapic_img"
    # ],
    "sd14": [
        "../results/teacher/std-v1-4/qualitative_pickapic_img",
        "../results/final_ours_sd14_align/align_dpo_beta8k/checkpoint-100/qualitative_pickapic_img",
        "../results/final_ours_sd14_align/align_our_beta16k/checkpoint-100/qualitative_pickapic_img"
    ],
    # "sana": [
    #     "../results/SANA/ours_beta2k_lora512/checkpoint-100/qualitative_pickapic_img",
    #     "../results/teacher/SANA_600M_1024/qualitative_pickapic_img"
    # ]
}

def load_metadata(path: str):
    with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def create_image_grid(images: List[Image.Image], prompt: str, group_name: str, index: int, out_path: str):
    # 그룹별 디렉토리 생성
    group_dir = os.path.join(out_path, group_name)
    os.makedirs(group_dir, exist_ok=True)

    widths, heights = zip(*(i.size for i in images))
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


def process_group(group_name: str, ckpt_paths: List[str]):
    print(f"📦 Processing group: {group_name}")
    # Load all metadata
    metadatas = [load_metadata(path) for path in ckpt_paths]

    # 공통 프롬프트 key 추출
    shared_keys = set(metadatas[0].keys())
    for md in metadatas[1:]:
        shared_keys.intersection_update(md.keys())
    shared_keys = sorted(shared_keys)

    print(f"🖼️ Found {len(shared_keys)} shared prompts")

    for idx, key in enumerate(shared_keys):
        prompt = metadatas[0][key]
        imgs = []
        for path in ckpt_paths:
            img_path = os.path.join(path, "images", key)
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)

        create_image_grid(imgs, prompt, group_name, idx, OUTPUT_DIR)

if __name__ == "__main__":
    for group_name, paths in GROUPS.items():
        process_group(group_name, paths)
