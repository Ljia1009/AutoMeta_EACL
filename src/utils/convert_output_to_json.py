import re
import json

from pathlib import Path
import argparse

# convert the model output from txt to json
def load_output_from_txt(file_path:str, type:str):
    with open(file_path, "r") as file:
        text = ''.join(file.readlines())
    
    file_name = str(Path(file_path).name)
    parent_dir = str(Path(file_path).resolve().parent)
    if type == "finetune":
        pattern = re.compile(
            r"Generated:\s*(.*?)\s*"           
            r"Gold:\s*(.*?)"                   
            r"(?=(?:\s*Generated:|$))",        
            re.S                               
        )
    else:
        pattern = re.compile(
        r"Generated Summary (\d+):\s*"       
        r"(.*?)"                             
        r"\s*Gold Metareview \1:\s*"         
        r"(.*?)(?=(?:Generated Summary \d+:|$))", 
        re.S                                 
        )

    matches = pattern.findall(text)
    output_dir = f"{parent_dir}/{file_name}.json"
    with open(output_dir, "w") as f2:
        json.dump(matches, f2, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_folder", type=str, default="output")
    parser.add_argument("--type", type=str)


    args = parser.parse_args()
    root = Path(args.path_to_folder)

    for file_path in root.rglob('*.txt'):
        load_output_from_txt(file_path, args.type)
