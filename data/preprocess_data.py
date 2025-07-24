import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

EN_PATH = 'DATA/train.en.txt'
VI_PATH = 'DATA/train.vi.txt'


with open(EN_PATH, encoding='utf-8') as f:
    en_lines = f.readlines()
with open(VI_PATH, encoding='utf-8') as f:
    vi_lines = f.readlines()


def build_examples(en_lines, vi_lines):
    examples = []
    for src, tgt in tqdm(zip(en_lines, vi_lines), total=len(en_lines), desc="Đang xử lý dữ liệu"):
        src = src.strip()
        tgt = tgt.strip()
        instruction = "Translate this sentence to Vietnamese without any punctuation marks."
        examples.append({
            "instruction": instruction,
            "input": src,
            "output": tgt,
        })
    return examples

data = build_examples(en_lines, vi_lines)


with open("data/data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)


with open("data/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"✅ Đã lưu {len(train_data)} mẫu train và {len(test_data)} mẫu test.")
