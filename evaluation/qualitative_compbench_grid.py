import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 24
OUTPUT_DIR = "grids_compbench"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = {
    "sdxl": [
        "../results/SDXL/our_beta16k_sdxl/checkpoint-100/compbench_img/complex/samples",
        "../results/teacher/SDXL-1/compbench_img/complex/samples"
    ],
    "sd14": [
        "../results/final_ours_sd14_align/align_dpo_beta8k/checkpoint-100/compbench_img/complex/samples",
        "../results/final_ours_sd14_align/align_our_beta16k/checkpoint-100/compbench_img/complex/samples",
        "../results/teacher/std-v1-4/compbench_img/complex/samples"
    ],
    "sana": [
        "../results/SANA/ours_beta2k_lora512/checkpoint-100/compbench_img/complex/samples",
        "../results/teacher/SANA_600M_1024/compbench_img/complex/samples"
    ]
}

def extract_prompt(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    return parts[0].replace("_", " ") if len(parts) > 1 else base.replace("_", " ")

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

    # 각 경로별 {prompt: filename} 딕셔너리 생성
    prompt_maps: List[Dict[str, str]] = []
    for path in paths:
        mapping = {}
        for f in os.listdir(path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                prompt = extract_prompt(f)
                mapping[prompt] = f  # 같은 프롬프트일 경우, 마지막 것만 유지됨
        prompt_maps.append(mapping)

    # 공통 prompt 찾기
    shared_prompts = sorted(set.intersection(*[set(d.keys()) for d in prompt_maps]))
    print(f"🖼️ Found {len(shared_prompts)} shared prompts")

    for idx, prompt in enumerate(shared_prompts):
        imgs = []
        for path, mapping in zip(paths, prompt_maps):
            filename = mapping[prompt]
            img_path = os.path.join(path, filename)
            with Image.open(img_path).convert("RGB") as img:
                imgs.append(img.copy())

        create_image_grid(imgs, prompt, group_name, idx, OUTPUT_DIR)

if __name__ == "__main__":
    for group_name, paths in GROUPS.items():
        process_group(group_name, paths)
