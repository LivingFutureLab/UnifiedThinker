## GEdit-bench


Step-1: 推理图片
```
bash benchmarks/edit/infer_qwen_image_edit.sh
```


Step-2: 调用 gpt-4o 评分
```
export HF_ENDPOINT=https://hf-mirror.com

bash benchmarks/image_generation/GEdit-Bench/run_gedit_score.sh
```


## ImgEdit

Step-1: 推理图片
```
bash benchmarks/edit/infer_qwen_image_edit.sh
```

Step-2: 评分
```
benchmarks/image_generation/ImgEdit-basic/basic_bench.sh
```