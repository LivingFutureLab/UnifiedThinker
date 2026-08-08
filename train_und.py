#coding=utf-8
# jianchong.zq for und task only
import os
import json
import argparse
import torch
import diffusers
import transformers
import deepspeed
import shutil
from transformers import AutoProcessor
from safetensors import safe_open

from termcolor import colored
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from peft import LoraConfig
from diffusers.optimization import get_scheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration

from src.data.load import load_data_und
from src.optim.load_optim import load_optim

from src.utils.env_utils import (
    init_accelerator,
    init_logger,
    in_notebook,
    import_class,
    init_seed,
)
from src.utils.io_utils import count_parameters
from src.model.utils import download_model_weight_oss, download_model_weight

if not in_notebook():
    import ml_tracker

  
def main(args):
    # (1) init environ

    # 1.1 init log dir, ckpt dir
    logging_dir = os.path.join(args.oss_path, args.exp_name, "logs")
    save_dir = os.path.join(args.oss_path, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # 1.2 init accelerator
    accelerator, device = init_accelerator(args, save_dir, logging_dir)
    #ds_zero_stage = accelerator.deepspeed_plugin.deepspeed_config['zero_optimization']['stage']

    # 1.3 init ml tracker
    if not in_notebook():
        ml_tracker.init(id=args.exp_name)
        # ml_tracker = None

    # 1.4 init logger
    logger = init_logger(__name__, logging_dir)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    cfg_path = os.path.join(save_dir, "config.yaml")
    if cfg_path.startswith("oss://tstar-image-dataset/"):
        cfg_path = cfg_path.replace("oss://tstar-image-dataset/", "/data/oss_bucket_0/")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    OmegaConf.save(args, cfg_path)
    print(f"Successfully saved config to {cfg_path}")

    # 1.5 set precision
    weight_dtype = (
        torch.float16
        if args.mixed_precision == "fp16"
        else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    )
    logger.info(f"Default model weight dtype: {weight_dtype}")

    # 1.6 set global seed for reproducibility
    init_seed(args)

    logger.info("(1) ---------- init env done! ----------")

    # (2) load data
    if "local" in args.train_data_und.data_type:
        args.dataloader.shuffle = True  # 在每个 epoch 开始时，都对整个数据集的索引进行一次随机打乱
    dataset_und, dataloader_und = load_data_und(args)
    logger.info("(2) ---------- load data done! ----------")

    # (3) load model    
    if args.qwenvl_path.startswith("model."): # mos
        args.qwenvl_path = args.qwenvl_path.rstrip("/")
        qwenvl_path = download_model_weight(args.qwenvl_path)
        args.qwenvl_path = qwenvl_path
        args.train_data_und.data_params.qwenvl_pretrained = args.qwenvl_path
    else:
        assert os.path.exists(args.qwenvl_path), "{} not exist.".format(args.qwenvl_path)
    accelerator.wait_for_everyone()
    
    if "qwen2.5-vl" in args.qwenvl_path.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "qwen3-vl" in args.qwenvl_path.lower():
        if "qwen3-vl-8b" in args.qwenvl_path.lower():
            model_class = Qwen3VLForConditionalGeneration
        elif "qwen3-vl-30b-a3b" in args.qwenvl_path.lower():
            model_class = Qwen3VLMoeForConditionalGeneration
        else:
            raise ValueError(f"not supported model: {args.qwenvl_path}")
    else:
        raise ValueError(f"not supported model: {args.qwenvl_path}")
    
    text_encoder = model_class.from_pretrained(
        args.qwenvl_path, torch_dtype=torch.bfloat16
    )
        
    text_encoder.requires_grad_(False)
    if args.model.text_encoder_lora:
        text_encoder_lora_config = LoraConfig(
            r=args.model.lora_r,
            lora_alpha=args.model.lora_alpha,
            lora_dropout=0.01,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        text_encoder.language_model.add_adapter(text_encoder_lora_config)
    else:
        text_encoder.requires_grad_(True)
        text_encoder.visual.requires_grad_(False)    # freeze vit    
    text_encoder.language_model.gradient_checkpointing_enable()
                       
    logger.info("(3) ---------- load model done! ----------")

    # (4) set optimizer & scheduler
    params_to_optimize = filter(
        lambda p: p.requires_grad, text_encoder.parameters()
    )

    if args.zero_stage == "zero2":
        cnt_str = count_parameters(text_encoder)
        print(colored(f"trainable params count: {cnt_str}", "green", attrs=["bold"]))

    optim_func = load_optim(args.optim.optim_class)

    # 计算学习率
    if args.optim.scale_lr:
        args.optim.optim_params.lr *= (
            args.gradient_accumulation_steps * args.train_bs * accelerator.num_processes
        )

    optimizer = optim_func(params_to_optimize, **args.optim.optim_params)
    logger.info("(4) ---------- load optimizer done! ----------")

    lr_scheduler = get_scheduler(
        args.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.optim.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # (6) prepare everything with our `accelerator`.
    text_encoder, optimizer, lr_scheduler = accelerator.prepare(
        text_encoder,
        optimizer,
        lr_scheduler,
    )
    
    if "local" in args.train_data_und.data_type:
        dataloader_und = accelerator.prepare(dataloader_und)
        

    # start train
    global_step = 0

    # set progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,  # TODO: resume train
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    def buffered_length_sorted_generator(dataloader, buffer_size=256, dataset_type="und"):
        """
        一个生成器, 它从dataloader中缓冲一批数据, 按长度排序后yield。
        """
        def _get_sort_key(item) -> int:
            if dataset_type == "und":
                # 文本长度
                return item['input_ids'].size(1)
            elif dataset_type == "gen":
                # 图像像素总数 (主图 + 参考图)
                tgt_image_pixels = item['image'][0].size[0] * item['image'][0].size[1]
                ref_images_pixels = sum(
                    ref_img.size[0] * ref_img.size[1] 
                    for ref_img in item["raw_condition_images"][0]
                )
                return tgt_image_pixels + ref_images_pixels
            else:
                raise ValueError(f"Internal error: Invalid dataset_type '{dataset_type}' in _get_sort_key")
        
        buffer = []
        
        for batch in dataloader:
            # 立即检查每个batch
            if dataset_type == "und":
                assert batch["input_ids"].size(0) == 1, "batch size must be 1"
            elif dataset_type == "gen":
                # t2i or edit, only single image generation
                assert len(batch["image"]) == 1, "batch size must be 1"
            else:
                raise ValueError(f"Unsupported dataset_type: '{dataset_type}'. Choose 'und' or 'gen'.")
        
            buffer.append(batch)
            
            if len(buffer) >= buffer_size:
                buffer.sort(key=_get_sort_key)
                
                for item in buffer:
                    yield item
                
                buffer = []

        # Don't forget to yield the remaining items
        if buffer:
            buffer.sort(key=_get_sort_key)
            for item in buffer:
                yield item
                 
    for epoch in range(args.max_train_epochs):
        if getattr(dataset_und, "new_epoch", None):
            dataset_und.new_epoch()
        
        sorted_dataloader_und = buffered_length_sorted_generator(dataloader_und, buffer_size=256, dataset_type="und")
            
        for i, batch in enumerate(sorted_dataloader_und):
            if global_step >= args.max_train_steps:
                break
            
            with accelerator.accumulate(text_encoder):
                text_encoder.train()
                
                ########################### train step ############################
                loss_dict_for_sync = {}
                
                record_ids = batch.pop("record_ids", None)
                for k in batch:
                    batch[k] = batch[k].to(device=text_encoder.device)
                    if k not in ["input_ids", "labels", "image_grid_thw"]:
                        batch[k] = batch[k].to(dtype=text_encoder.dtype)    
                outputs = text_encoder(**batch)
                loss = outputs.loss
                if torch.isnan(loss): # TODO: check
                    print(f"\n[Warning] Understanding loss is NaN, resetting to 0.0.")
                    print(f"  - Record IDs: {record_ids}")
                    #loss = torch.tensor(0.0, device=text_encoder.device, requires_grad=True)
                    
                    if hasattr(outputs, 'logits') and outputs.logits is not None and (not torch.any(torch.isfinite(outputs.logits))):
                        loss = torch.mean(outputs.logits) * 0.0
                    else:
                        params = text_encoder.parameters()
                        loss = sum(p.sum() for p in params) * 0.0
                            
                loss_dict_for_sync["loss"] = loss
                accelerator.backward(loss)
                        
                grad_norm_tensor = None
                if accelerator.sync_gradients:
                    # 仅在梯度同步时进行裁剪和记录, clip_grad_norm_ 返回的已经是同步后的张量
                    grad_norm_tensor = accelerator.clip_grad_norm_(
                        params_to_optimize, args.max_grad_norm
                    )
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                synced_losses = accelerator.gather_for_metrics(loss_dict_for_sync)
                loss_metrics = {k: v.mean().detach().item() for k, v in synced_losses.items()}
                if grad_norm_tensor is not None:
                    if torch.is_tensor(grad_norm_tensor):
                        loss_metrics["grad_norm"] = grad_norm_tensor.detach().item()
                    else:
                        loss_metrics["grad_norm"] = grad_norm_tensor     
                loss = loss_metrics 
                ############################### train step ###############################
                
                accelerator.wait_for_everyone()
                
                if accelerator.sync_gradients:
                    global_step += 1
        
                    accelerator.wait_for_everyone()
                       
                    # save model 
                    if args.zero_stage == "zero2":
                        # save model
                        if global_step % args.checkpointing_steps == 0 or global_step == 10:
                            accelerator.wait_for_everyone()
                            #save_path = os.path.join(save_dir, "ckpt", f"epoch-{epoch}-step-{global_step}")
                            save_path = os.path.join(save_dir, "ckpt", f"step-{global_step}")
                            
                            if in_notebook():
                                tmp_save_path = save_path
                            else:
                                assert save_path.startswith("oss://tstar-image-dataset/"), "wrong of model_path: {}".format(save_path)
                                tmp_save_path = os.path.join("./tmp_ckpt", save_path.replace("oss://tstar-image-dataset/", ""))
                            
                            os.makedirs(tmp_save_path, exist_ok=True)
                            unwrap_text_encoder = accelerator.unwrap_model(text_encoder)
                            
                            # 使用 accelerator.save_model 进行分片保存。这个函数需要被所有进程调用，它会自动处理分片逻辑。
                            print(f"Process {accelerator.process_index}: Starting to save sharded model to {tmp_save_path}...")
                            accelerator.save_model(
                                model=unwrap_text_encoder,
                                save_directory=tmp_save_path,
                                safe_serialization=True  # 推荐使用 safetensors 格式，更安全、更快
                            )
                            # 上传 oss
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:      
                                if not in_notebook():
                                    from src.model.utils import upload_model_weight_oss
                                    upload_model_weight_oss(tmp_save_path, save_path)
                                    print(f"Main process: Upload checkpoint directory to {save_path} successfully!")
                            
                            # 清理本地临时文件
                            accelerator.wait_for_everyone()
                            if accelerator.is_local_main_process: 
                                try:
                                    shutil.rmtree(tmp_save_path)
                                    print(f"Main process: Delete the temporary directory: {tmp_save_path}")
                                except OSError as e:
                                    print(f"Main process: Error when deleting the temporary directory: {tmp_save_path}. {e}")
                            accelerator.wait_for_everyone()
                    
                    # # save dataset state
                    # if global_step % args.checkpointing_steps == 0 or global_step == 10:
                    #     accelerator.wait_for_everyone()
                    #     datasets_to_save = {
                    #         "dataset_und": dataset_und,
                    #     }
                    #     for dataset_name, dataset in datasets_to_save.items():
                    #         save_path = os.path.join(save_dir, "ckpt", f"step-{global_step}", dataset_name)
                    #         if in_notebook():
                    #             tmp_save_path = save_path
                    #         else:
                    #             assert save_path.startswith("oss://tstar-image-dataset/"), "wrong of model_path: {}".format(save_path)
                    #             tmp_save_path = save_path.replace("oss://tstar-image-dataset/", "/data/oss_bucket_0/")
                    #         os.makedirs(tmp_save_path, exist_ok=True)
                    #         save_dataset_state(dataset, accelerator, tmp_save_path)
                                
                    accelerator.wait_for_everyone()
                    now = datetime.now()
                    formatted_time = now.strftime("%m/%d-%H:%M:%S.%f")[:-3]
                    logs = {
                        "t": formatted_time,
                        **loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "grad_acc": args.gradient_accumulation_steps,
                        "global_batch": global_batch
                    }
                    progress_bar.set_postfix(**logs)
                    progress_bar.update(1)
                    
                    if (
                        not in_notebook()
                        and accelerator.sync_gradients
                        and ml_tracker is not None
                    ):
                        ml_tracker.log(logs)
                    logger.debug(json.dumps(logs))
                    
                    ## check 训练参数正常更新
                    if args.zero_stage == "zero2" and global_step % 10 == 0:
                        # debug, 看 lora 参数更新是否符合预期
                        if args.model.text_encoder_lora:
                            lora_delta_w = (text_encoder.language_model.layers[26].mlp.up_proj.lora_B.default.weight.detach() @ text_encoder.language_model.layers[26].mlp.up_proj.lora_A.default.weight.detach()).abs().max().item()
                            print("  text_encoder.language_model.layers[26].mlp.up_proj, lora deltaW, max value: {}".format(lora_delta_w))
                        else:
                            tmp_w = text_encoder.language_model.layers[26].mlp.up_proj.weight.detach().abs().max()
                            print("  text_encoder.language_model.layers[26].mlp.up_proj, max value: {:.6f}".format(tmp_w))
                
                accelerator.wait_for_everyone()   
                

if __name__ == "__main__":
    # ******** for debug in notebook, rank and world_size need to set ********
    if in_notebook():
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    else:
        os.environ["NCCL_MIN_NCHANNELS"] = "16"
    # ************************************************************************

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--tables", type=str, default="")
    parser.add_argument(
        "--zero_stage", "-z", type=str, default="zero3", choices=["zero2", "zero3"]
    )
    parser.add_argument("--run_timestamp", type=str, help="Unique timestamp for the run")
    parser.add_argument("--pdb_debug", action="store_true")

    args = parser.parse_args()
    
    if args.pdb_debug:
        import pdb; pdb.set_trace()

    config = OmegaConf.load(args.config)
    base_name = args.config.split("/")[-1].split(".")[0]
    
    if hasattr(config, "global_batch") and (config.global_batch is not None) and (config.global_batch % (config.train_bs * int(os.environ["WORLD_SIZE"])) == 0):
        global_batch = config.global_batch
        config.gradient_accumulation_steps = global_batch // (config.train_bs * int(os.environ["WORLD_SIZE"]))
        print(colored("reset gradient_accumulation_steps to {}".format(config.gradient_accumulation_steps), "green", attrs=["bold"]))
    else:
        global_batch = (
            config.train_bs
            * config.gradient_accumulation_steps
            * int(os.environ["WORLD_SIZE"])
        )
                
    str_lr = str(config.lr).replace(".", "")
    if args.run_timestamp is None:
        config["exp_name"] = f"{base_name}_bs{global_batch}_lr{str_lr}".replace("-", "_") + "_{}".format(datetime.now().strftime('%Y%m%d_%H'))
    else:
        config["exp_name"] = f"{base_name}_bs{global_batch}_lr{str_lr}_".replace("-", "_") + args.run_timestamp
    config["zero_stage"] = args.zero_stage
    
    print(config["oss_path"])
    print(config)
    if args.pdb_debug:
        config.dataloader.num_workers = 1
        

    main(config)
