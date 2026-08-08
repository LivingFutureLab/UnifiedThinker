import os 
import json 
import glob 
import random


if __name__ == '__main__':
    data_files = list(glob.glob("benchmark/image-generation/WISE/data/*.json"))
    data_files = [df for df in data_files if "_rewrite" not in df]
    print("data_files: {}".format(data_files))
    
    datas = []
    for df in data_files:
        tmps = json.load(open(df))
        for d in tmps:
            datas.append({
                "prompt": d['Prompt'],
                "prompt_cot": d['Explanation'],
                "category": d["Category"],
                "subcategory": d['Subcategory'],
                "task_type": "reason_t2i",
                "domain": "reason_t2i"
            })
    
    random.shuffle(datas)
    
    outfile = "benchmark/image-generation/WISE/wise_for_rft.jsonl"
    with open(outfile, 'w') as f:
        for d in datas:
            f.write("{}\n".format(json.dumps(d, ensure_ascii=False)))
    print("write to {}".format(outfile))
    