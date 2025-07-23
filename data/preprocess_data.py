
import os
import json
from datasets import Dataset
from tqdm import tqdm


EN_PATH = 'data/train.en.txt'
VI_PATH = 'data/train.vi.txt'


with open(EN_PATH, encoding='utf-8') as f:
    en_lines = f.readlines()
with open(VI_PATH, encoding='utf-8') as f:
    vi_lines = f.readlines()


def build_examples(en_lines, vi_lines):
    examples = []
    for src, tgt in tqdm(zip(en_lines, vi_lines), total=len(en_lines), desc="Đang xử lý dữ liệu"):
        src = src.strip()
        tgt = tgt.strip()
        instruction = f"Translate this sentence to Vietnamese without any punctuation marks."
        examples.append({
            "instruction": instruction,
            "input": src,
            "output": tgt,
        })
    return examples

data = build_examples(en_lines, vi_lines)

with open("data/data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)