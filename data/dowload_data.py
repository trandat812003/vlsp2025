import gdown
import os


output_folder = "./DATA"
os.makedirs("DATA", exist_ok=True)

files = [
    {"id": "1nybC7S9MZGxmwbyRAkYMLH9Wa42BiCNU", "name": "train.en.txt"},
    {"id": "1Gj9zhEDPLTsmmcXlKpbd-lbS0aTMajAJ", "name": "train.vi.txt"}
]


for f in files:
    url = f"https://drive.google.com/uc?export=download&id={f['id']}"
    output_path = os.path.join(output_folder, f["name"])
    gdown.download(url, output_path, quiet=False)
