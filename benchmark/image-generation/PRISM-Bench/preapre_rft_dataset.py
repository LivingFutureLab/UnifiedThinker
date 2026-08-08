#coding=utf-8
import os, json, glob 
import random

jfiles = list(glob.glob("benchmark/image-generation/PRISM-Bench/captions/*/*.jsonl"))
datas = []
for jf in jfiles:
    category = os.path.splitext(os.path.basename(jf))[0]
    with open(jf, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == "": continue
            d = json.loads(line)
            prompt = d['prompt']
            datas.append({
                "edit_prompt": prompt,  # 和 image-edit 使用相同的 key
                "t2i_eval_category": category,
                "task_type": "t2i",
                "domain": "t2i",
            })

outfile = "benchmark/image-generation/PRISM-Bench/prism_bench_for_rft.jsonl"
with open(outfile, 'w') as f:
    for d in datas:
        f.write("{}\n".format(json.dumps(d, ensure_ascii=False)))
print("write to {}".format(outfile))

random.shuffle(datas)
outfile = "benchmark/image-generation/PRISM-Bench/prism_bench_for_rft_tiny_16.jsonl"
with open(outfile, 'w') as f:
    for d in datas[:16]:
        f.write("{}\n".format(json.dumps(d, ensure_ascii=False)))
print("write to {}".format(outfile))
