#!/usr/bin/env python3
"""
Generate RISEBench outputs with Unified Thinker (Thinker -> CoT -> Editor).

This is the generation half that the released repo was missing: benchmark/ only
ships the scoring scripts. Output layout matches what RISEBench's qwen_eval.py
expects:

    <output>/images/<category>/<index>.png

Reasoning is produced by the UnifiedThinker checkpoint loaded as its own model,
NOT by pipe.text_encoder (which is Qwen-Image-Edit's own frozen Qwen2.5-VL).

Usage (from repo root):
    PYTHONPATH=$(pwd) python3 benchmark/image-generation/RISEBench/gen_risebench.py \
        --data   <path>/datav2_total_w_subtask.json \
        --input  <path>/data \
        --output outputs/UnifiedThinker-7B \
        --model_path model/Qwen-Image-Edit-2509 \
        --thinker_path model/UnifiedThinker-7B
"""
import argparse
import json
import os
import os.path as osp
import re
import time

import torch
from PIL import Image
from termcolor import colored
from transformers import AutoProcessor

from src.pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline
from src.data.odps_t2i_edit_data import preprocess_image
from inference.qwen_edit_think.demo_qwen_edit_think import (
    vlm_prompt_thinking,
    load_thinker,
)


def extract_answer_from_cot(prompt_cot, fallback):
    """Same rule as inference/infer_single.py."""
    if prompt_cot is None:
        return fallback, False
    m = re.search(r"<answer>(.*?)</answer>", prompt_cot, re.DOTALL)
    if m:
        return m.group(1).strip(), True
    # No <answer> tag (e.g. the Thinker ran to the token limit without closing the
    # tag). Fall back to the raw instruction rather than feeding the whole CoT to
    # the diffusion model — the full reasoning text is not a usable edit prompt.
    return fallback, False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="datav2_total_w_subtask.json")
    p.add_argument("--input", required=True, help="RISEBench data/ dir with input images")
    p.add_argument("--output", required=True, help="outputs/<MODEL_NAME>")
    p.add_argument("--model_path", default="model/Qwen-Image-Edit-2509")
    p.add_argument("--processor_path", default="model/UnifiedThinker-7B")
    p.add_argument("--thinker_path", default=None)
    p.add_argument("--categories", default=None,
                   help="comma separated subset, e.g. temporal_reasoning,causal_reasoning")
    p.add_argument("--limit", type=int, default=None, help="only first N items (debug)")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_offload", action="store_true")
    p.add_argument("--no_thinking", action="store_true",
                   help="skip the Thinker and feed the raw instruction (baseline)")
    p.add_argument("--phase", choices=["cot", "image", "both"], default="both",
                   help="Run the Thinker and the editor in separate passes. Holding "
                        "both models at once costs ~73GB, which thrashes the page "
                        "cache on smaller hosts; two passes keep only one model "
                        "resident and are markedly faster there.")
    return p.parse_args()


def load_cots(path):
    """index -> record, from a previous --phase cot run."""
    cots = {}
    if osp.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cots[rec["index"]] = rec
    return cots


def main():
    args = parse_args()
    thinker_path = args.thinker_path or args.processor_path
    device = "cuda"

    items = json.load(open(args.data))
    if args.categories:
        keep = set(args.categories.split(","))
        items = [it for it in items if it["category"] in keep]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items to generate")

    os.makedirs(args.output, exist_ok=True)
    cot_log_path = osp.join(args.output, "thinker_cot.jsonl")

    need_thinker = args.phase in ("cot", "both") and not args.no_thinking
    need_pipe = args.phase in ("image", "both")

    # ---- models -------------------------------------------------------------
    thinker = vlm_processor = pipe = None
    if need_thinker:
        print(f"Loading UnifiedThinker from {thinker_path}...")
        thinker = load_thinker(thinker_path, torch_dtype=torch.bfloat16).to(device)
        vlm_processor = AutoProcessor.from_pretrained(args.processor_path)

    if need_pipe:
        print(f"Loading editor from {args.model_path}...")
        pipe = QwenImageEditThinkPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16)
        # With only the editor resident, the whole pipeline fits on an 80GB card.
        if args.no_offload or args.phase == "image":
            pipe = pipe.to(device)
        else:
            pipe.enable_model_cpu_offload()
    print("models ready")

    # phase=image reuses the CoTs produced by phase=cot; phase=cot uses the same
    # file to resume after an interruption
    cached_cots = load_cots(cot_log_path) if args.phase in ("image", "cot") else {}
    if cached_cots:
        print(f"loaded {len(cached_cots)} cached CoTs from {cot_log_path}")

    n_done = n_skip = n_fail = n_no_answer = 0
    t_start = time.time()

    for i, item in enumerate(items):
        index, category = item["index"], item["category"]
        out_dir = osp.join(args.output, "images", category)
        os.makedirs(out_dir, exist_ok=True)
        out_path = osp.join(out_dir, f"{index}.png")

        if osp.exists(out_path) and args.phase != "cot":
            n_skip += 1
            continue
        if args.phase == "cot" and index in cached_cots:
            n_skip += 1
            continue

        try:
            img_path = osp.join(args.input, item["image"])
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image, max_area=1024 * 1024, adjust_ar=False)
            width, height = image.size

            instruction = item["instruction"]
            prompt_cot = None
            has_answer = True

            if args.no_thinking:
                prompt = instruction
            elif args.phase == "image":
                rec = cached_cots.get(index)
                if rec is None:
                    raise RuntimeError(
                        f"no cached CoT for {index}; run --phase cot first")
                prompt_cot = rec["cot"]
                prompt = rec["final_prompt"]
                has_answer = rec["answer_tag_found"]
            else:
                prompt_cot = vlm_prompt_thinking(
                    [image], instruction, vlm_processor, thinker,
                    max_new_tokens=args.max_new_tokens,
                )
                prompt, has_answer = extract_answer_from_cot(prompt_cot, instruction)
                if not has_answer:
                    n_no_answer += 1

            # phase=cot only records the reasoning, no diffusion
            if args.phase != "cot":
                inputs = {
                    "image": [image],
                    "prompt": prompt,
                    "prompt_cot": None,
                    "generator": torch.manual_seed(args.seed),
                    "true_cfg_scale": args.guidance_scale,
                    "negative_prompt": " ",
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": 1.0,
                    "num_images_per_prompt": 1,
                    "fix_ref_img_pixel_area": False,
                }
                with torch.inference_mode():
                    out = pipe(**inputs, height=height, width=width)
                out.images[0].save(out_path)

            if args.phase != "image":
                with open(cot_log_path, "a") as f:
                    f.write(json.dumps({
                        "index": index,
                        "category": category,
                        "instruction": instruction,
                        "cot": prompt_cot,
                        "final_prompt": prompt,
                        "answer_tag_found": has_answer,
                    }, ensure_ascii=False) + "\n")

            n_done += 1
            elapsed = time.time() - t_start
            rate = elapsed / max(n_done, 1)
            left = (len(items) - n_skip - n_done) * rate
            print(colored(
                f"[{i+1}/{len(items)}] {index}  done={n_done} skip={n_skip} "
                f"fail={n_fail} no_answer={n_no_answer}  "
                f"{rate:.1f}s/img  eta={left/3600:.1f}h", "green"))

        except Exception as e:
            n_fail += 1
            print(colored(f"[{i+1}/{len(items)}] {index} FAILED: {type(e).__name__}: {e}",
                          "red"))
            continue

    print("=" * 70)
    print(f"generated {n_done}, skipped {n_skip}, failed {n_fail}, "
          f"missing <answer> tag {n_no_answer}")
    print(f"total {(time.time()-t_start)/3600:.2f}h -> {args.output}")


if __name__ == "__main__":
    main()
