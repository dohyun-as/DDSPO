import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from urllib.request import urlretrieve
from tqdm import tqdm
import re, json, ast
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

# Settings
template_path = "template_prompt.txt"     # Full system + user prompt structure with {input_prompt}
output_path = f"output_rank{accelerator.process_index}.jsonl"
error_log_path = f"error_rank{accelerator.process_index}.jsonl"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# device = "cuda" if torch.cuda.is_available() else "cpu"
auth_token = "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" #your huggingface token
cache_dir = "../cache"  # Cache directory for model and tokenizer



# --------------------------------------
# 1. Load tokenizer and model
# --------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_auth_token=auth_token,
    trust_remote_code=True,
    cache_dir=cache_dir
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=auth_token,
    trust_remote_code=True,
    cache_dir=cache_dir
).to(device)
model.eval()


# --------------------------------------
# 2. Load the base prompt template
#    (system + user 컨텍스트가 들어있는 template)
# --------------------------------------
with open(template_path, "r", encoding="utf-8") as f:
    base_context = f.read()

# --------------------------------------
# 3. 실제 프롬프트를 템플릿 형식에 맞게 감싸줄 함수
# --------------------------------------
prompt_template = (
    "Below is the new prompt to process.\n"
    "prompt: {}\n"
    "<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

def build_prompt(input_prompt: str) -> str:
    """ base_context + (prompt_template.format(input_prompt)) 형태로 완성된 prompt 반환 """
    return base_context + prompt_template.format(input_prompt)

# --------------------------------------
# 4. DiffusionDB metadata.parquet 불러오기
# --------------------------------------
parquet_url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
local_parquet = "metadata.parquet"

Path(local_parquet).parent.mkdir(parents=True, exist_ok=True)
if not Path(local_parquet).exists():
    print("Downloading metadata.parquet...")
    urlretrieve(parquet_url, local_parquet)

df = pd.read_parquet(local_parquet)
prompts = df["prompt"].dropna().unique().tolist()

# 프로세스 수와 현재 프로세스 인덱스
world_size = accelerator.num_processes
rank = accelerator.process_index

# prompts 나누기 (shard)

processed_prompts = set()
for file in Path(".").glob("output_rank*.jsonl"):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                processed_prompts.add(item["prompt"])
            except Exception as e:
                print(f"Skipping line in {file} due to error: {e}")

# 아직 처리되지 않은 것만 남기기
prompts = [p for p in prompts if p not in processed_prompts]

prompts = prompts[rank::world_size]
# --------------------------------------
# 5. 모델 응답 파싱을 위한 헬퍼 함수
# --------------------------------------
def extract_last_response_info(full_text):
    # Find the second occurrence of "assistant"
    assistant_indices = [m.start() for m in re.finditer(r"\bassistant\b", full_text)]
    if len(assistant_indices) < 2:
        raise ValueError("Second 'assistant' block not found in the text.")
    
    # Slice from second "assistant" to the end
    response_text = full_text[assistant_indices[1]:]

    # Extract Important words
    important_match = re.search(r"Important words: \[(.*?)\]", response_text)
    if important_match:
        important_words_str = important_match.group(1)
        important_words = [w.strip().strip('"').strip("'") for w in important_words_str.split(",")]
    else:
        important_words = []

    # Extract Final output JSON
    final_output_match = re.search(r"Final output:\s*({.*})", response_text, re.DOTALL)
    if final_output_match:
        final_output_str = final_output_match.group(1)
        final_output = json.loads(final_output_str)
    else:
        final_output = {}

    return important_words, final_output

# --------------------------------------
# 6. 배치 단위로 LLM에 질의하는 함수
# --------------------------------------
def query_llm_batched(batch_prompts, max_new_tokens=256, temperature=0.7):
    """
    batch_prompts: 이미 prompt template이 적용된 문자열들의 리스트
    """

    inputs = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    ).to(device)
    # print(f"inputs: {inputs}")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    # print(f"outputs: {outputs}")
    # 각 배치에 대해 디코딩
    decoded_responses = [
        tokenizer.decode(output, skip_special_tokens=True) 
        for output in outputs
    ]
    return decoded_responses

# --------------------------------------
# 7. 모든 prompt에 대해 처리 & 결과 저장 (배치 사용)
#    batch_size를 조절하여 속도를 높일 수 있음
# --------------------------------------
batch_size = 64  # 원하는 배치 사이즈로 조절하세요

idx_counter = 1
if Path(output_path).exists():
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                id_num = int(item["id"])
                idx_counter = max(idx_counter, id_num + 1)
            except Exception:
                continue
print(f"Starting from ID: {idx_counter}")
with open(output_path, "a", encoding="utf-8") as fw, open(error_log_path, "a", encoding="utf-8") as fe:
    # prompts를 batch_size만큼씩 나누어서 처리
    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Processing prompts"):
        end_idx = start_idx + batch_size
        batch_prompts_raw = prompts[start_idx:end_idx]

        # (1) batch 내부의 모든 prompt에 template 적용
        batch_prompts_for_model = [build_prompt(p) for p in batch_prompts_raw]

        # (2) 모델 응답 생성 (배치 추론)
        try:
            responses = query_llm_batched(batch_prompts_for_model)
        except Exception as e:
            print(f"Error during model inference: {e}")
            # 배치 전체가 에러일 경우, 각각 에러로 기록
            for bp in batch_prompts_raw:
                error_line = {
                    "prompt": bp,
                    "error": str(e)
                }
                fe.write(json.dumps(error_line, ensure_ascii=False) + "\n")
            continue
        accelerator.wait_for_everyone()
        # (3) 배치 응답 각각 파싱 및 결과 저장
        for prompt_text, response_text in zip(batch_prompts_raw, responses):
            try:
                # print(f"Response: {response_text}")
                important_words, final_output_dict = extract_last_response_info(response_text)
                # print(f"Important words: {important_words}")
                # print(f"Final output: {final_output_dict}")

                output_line = {
                    "id": f"{idx_counter:08d}",
                    "prompt": prompt_text,
                    "important_words": important_words,
                    "neg_prompts": final_output_dict.get("neg_prompts", [])
                }
                fw.write(json.dumps(output_line, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error during model inference: {e}")
                error_line = {
                    "prompt": prompt_text,
                    "response_text": response_text,
                    "error": str(e)
                }
                fe.write(json.dumps(error_line, ensure_ascii=False) + "\n")

            idx_counter += 1