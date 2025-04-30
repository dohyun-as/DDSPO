import pandas as pd
import random
import json
from pathlib import Path
from urllib.request import urlretrieve

# Load the parquet file
parquet_url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
local_parquet = "metadata.parquet"

Path(local_parquet).parent.mkdir(parents=True, exist_ok=True)
if not Path(local_parquet).exists():
    print("Downloading metadata.parquet...")
    urlretrieve(parquet_url, local_parquet)

df = pd.read_parquet(local_parquet)
prompts = df["prompt"].dropna().unique().tolist()

entries = []

def generate_negative_prompts(prompt, max_trials=5):
    words = prompt.split()
    if len(words) < 2:
        return []

    trials = 0
    candidates = set()
    while trials < max_trials:
        trials += 1
        ratio = random.uniform(0.4, 0.7) 
        num_to_remove = max(1, int(ratio  * len(words)))
        indices_to_remove = set(random.sample(range(len(words)), num_to_remove))
        neg_prompt = ' '.join([w for i, w in enumerate(words) if i not in indices_to_remove])
        if neg_prompt != prompt:
            candidates.add(neg_prompt)
        if len(candidates) >= 5:
            break
    return list(candidates)

for idx, prompt in enumerate(prompts):
    negs = generate_negative_prompts(prompt)
    entry = {
        "id": f"{idx+1:08}",
        "tag": "DiffusionDB",
        "prompt": prompt,
        "neg_prompts": negs
    }
    entries.append(entry)
    # if len(entries) >= 100:  # Adjust this number as needed
    #     break

# Save to JSONL
output_path = "diffusiondb_removal.jsonl"
with open(output_path, "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Saved {len(entries)} entries to {output_path}")
