import json
from pathlib import Path

# 파일 경로 리스트
input_files = [
    "output_rank0.jsonl",
    "output_rank1.jsonl",
    "output_rank2.jsonl"
]

# 출력 경로
output_path = Path("../data/captions/diffusiondb_llama/diffusiondb_llama.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

# 모든 라인을 담을 리스트
merged_data = []

# 각 파일 읽어서 데이터 추가
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            merged_data.append(json.loads(line.strip()))

# 병합된 데이터를 jsonl 형식으로 저장
with open(output_path, "w", encoding="utf-8") as f:
    for item in merged_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Merged {len(merged_data)} items into {output_path}")
