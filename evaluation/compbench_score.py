import argparse
import os
import csv

def extract_score_from_file(filepath, keyword):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        for line in f:
            if keyword in line:
                try:
                    return float(line.strip().split(keyword)[-1].strip())
                except ValueError:
                    return None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="List of result directories to parse")
    parser.add_argument("--output", type=str, default="compiled_scores.csv", help="Output CSV file path")
    args = parser.parse_args()

    rows = []
    header = ["path", "color", "shape", "texture", "spatial", "non-spatial", "complex"]

    for path in args.paths:
        scores = [path]

        # BLIP VQA scores
        for key in ["color", "shape", "texture"]:
            file_path = os.path.join(path, key, "annotation_blip", "blip_vqa_score.txt")
            score = extract_score_from_file(file_path, "BLIP-VQA score:")
            scores.append(score)

        # Object detection (spatial)
        file_path = os.path.join(path, "spatial", "labels", "annotation_obj_detection_2d", "avg_score.txt")
        scores.append(extract_score_from_file(file_path, "score avg:"))

        # CLIP (non-spatial)
        file_path = os.path.join(path, "non_spatial", "annotation_clip", "score_avg.txt")
        scores.append(extract_score_from_file(file_path, "score avg:"))

        # 3-in-1 (complex)
        file_path = os.path.join(path, "complex", "annotation_3_in_1", "vqa_score.txt")
        scores.append(extract_score_from_file(file_path, "score avg:"))

        rows.append(scores)

    # Save to CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved scores to {args.output}")

if __name__ == "__main__":
    main()
