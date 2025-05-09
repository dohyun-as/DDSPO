import argparse
import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default="generated_images",
        help="Directory containing generated images and metadata.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output file (pickscore.txt). If not set, defaults to <image_dir>/pickscore.txt"
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="CLIP processor path or model ID"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yuvalkirstain/PickScore_v1",
        help="PickScore model path or model ID"
    )
    return parser.parse_args()


@torch.no_grad()
def calc_probs(prompt, images, processor, model, device):
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=[prompt],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    image_embs = model.get_image_features(**image_inputs)
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

    text_embs = model.get_text_features(**text_inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    return scores.cpu().tolist()


def main():
    args = parse_args()
    img_dir = os.path.join(args.image_dir,"images")
    metadata_path = os.path.join(args.image_dir, "metadata.json")
    output_path = args.output_file or os.path.join(args.image_dir, "pickscore.txt")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {img_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.processor)
    model = AutoModel.from_pretrained(args.model).eval().to(device)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scores = []

    for fname, prompt in metadata.items():
        img_path = os.path.join(img_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image {img_path}: {e}")
            continue

        try:
            score = calc_probs(prompt, [img], processor, model, device)[0]
            scores.append(score)
            print(f"{fname} → score: {score:.4f}")
        except Exception as e:
            print(f"Failed to score image {img_path}: {e}")
            continue

    if scores:
        mean_score = sum(scores) / len(scores)
        with open(output_path, "w") as f:
            f.write(f"{mean_score:.6f}\n")
        print(f"\n✅ Average PickScore: {mean_score:.6f} saved to {output_path}")
    else:
        print("❌ No valid scores computed.")


if __name__ == "__main__":
    main()
