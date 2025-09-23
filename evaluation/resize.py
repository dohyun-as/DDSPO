import os
from PIL import Image

def change_img_size(input_folder, output_folder, resz=256):
    os.makedirs(output_folder, exist_ok=True)
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    for i, filename in enumerate(img_list):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((resz, resz))
        img.save(os.path.join(output_folder, filename))
        img.close()
        if i % 1000 == 0:
            print(f"{i}/{len(img_list)} | {filename} resized to {resz}")

if __name__ == "__main__":
    input_folder = "/workspace/Diffusion_align/evaluation/mscoco_val2014/val2014"        # 예: "./val2014"
    output_folder = "/workspace/Diffusion_align/evaluation/mscoco_val2014/val2014_256"   # 예: "./val2014_256"
    target_size = 256

    change_img_size(input_folder, output_folder, resz=target_size)