import re
import json
from pathlib import Path
import argparse

# convert the model output from txt to json
def load_output_from_txt(file_path:str):
    with open(file_path, "r") as file:
        text = ''.join(file.readlines())
    
    file_name = str(Path(file_path).name)
    parent_dir = str(Path(file_path).resolve().parent)

    pattern = re.compile(
    r"Generated Summary (\d+):\s*"       # 捕获编号 N
    r"(.*?)"                             # 非贪婪地捕获 Generated Summary 内容
    r"\s*Gold Metareview \1:\s*"         # 匹配相同编号的 Gold Metareview
    r"(.*?)(?=(?:Generated Summary \d+:|$))",  # 捕获 Gold Metareview 内容，直到下一个块或文件末尾
    re.S                                 # 让 . 能匹配换行
    )

    matches = pattern.findall(text)
    output_dir = f"{parent_dir}/{file_name}.json"
    with open(output_dir, "w") as f2:
        json.dump(matches, f2, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_folder", type=str, default="output")

    args = parser.parse_args()
    root = Path(args.path_to_folder)

    for file_path in root.rglob('*.txt'):
        load_output_from_txt(file_path)
