#coding=utf-8
import os
import json
import argparse
import torch
import diffusers
import transformers
import deepspeed
import shutil

from termcolor import colored
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf

from diffusers.optimization import get_scheduler

from src.data.load import load_data_gen
from src.model.load_model import load_model, prepare_model
from src.optim.load_optim import load_optim

from src.utils.env_utils import (
    init_accelerator,
    init_logger,
    in_notebook,
    import_class,
    init_seed,
)
from src.utils.io_utils import count_parameters
from train_unified_gen_und import save_dataset_state, download_model_weight, download_model_weight_oss, count_parameters_zero3

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
    if "local" in args.train_data_gen.data_type:
        args.dataloader.shuffle = True  # 在每个 epoch 开始时，都对整个数据集的索引进行一次随机打乱
    dataset_gen, dataloader_gen = load_data_gen(args)
    logger.info("(2) ---------- load data done! ----------")

    # (3) load model
    if args.model.loader.loader_params.model_path.startswith("oss://"):     # download from oss
        model_path = download_model_weight_oss(args.model.loader.loader_params.model_path)
        args.model.loader.loader_params.model_path = model_path
    elif args.model.loader.loader_params.model_path.startswith("model."):   # download from mos
        model_path = download_model_weight(args.model.loader.loader_params.model_path)
        args.model.loader.loader_params.model_path = model_path
    else:
        assert os.path.exists(args.model.loader.loader_params.model_path), "{} not exist.".format(args.model.loader.loader_params.model_path)
    accelerator.wait_for_everyone()
    model_dict = load_model(**args.model.loader)
    model_dict = prepare_model(
        model_dict,
        device=device,
        weight_dtype=weight_dtype,
        **args.model.prepare,
    )
    logger.info("(3) ---------- load model done! ----------")

    # (4) set optimizer & scheduler
    params_to_optimize = filter(
        lambda p: p.requires_grad, model_dict["train_model"].parameters()
    )
    optim_func = load_optim(args.optim.optim_class)
    # 计算学习率
    if args.optim.scale_lr:
        args.optim.optim_params.lr *= (
            args.gradient_accumulation_steps * args.train_bs * accelerator.num_processes
        )
    optimizer = optim_func(params_to_optimize, **args.optim.optim_params)
    
    if args.zero_stage == "zero2":
        cnt_str = count_parameters(model_dict["train_model"])
        print(colored(f"trainable params count: {cnt_str}", "green", attrs=["bold"]))
        cnt_str_textencoder = count_parameters(model_dict["text_encoder"])
        cnt_str_transformer = count_parameters(model_dict["transformer"])
        print(colored(f"    text_encoder trainable params count: {cnt_str_textencoder}", "green", attrs=["bold"]))
        print(colored(f"    transformer trainable params count: {cnt_str_transformer}", "green", attrs=["bold"]))
        
    logger.info("(4) ---------- load optimizer done! ----------")

    lr_scheduler = get_scheduler(
        args.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.optim.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # (6) prepare everything with our `accelerator`.
    model_dict["train_model"], optimizer, lr_scheduler = accelerator.prepare(
        model_dict["train_model"],
        optimizer,
        lr_scheduler,
    )
    
    if args.zero_stage == "zero3":
        trainable_cnt_str, cnt_str = count_parameters_zero3(model_dict["train_model"])
        logger.info(f"MIKA deepspeed-engine trainable params count: {trainable_cnt_str}, total params count: {cnt_str}")
        logger.info(colored(f"MIKA deepspeed-engine trainable params count: {trainable_cnt_str}, total params count: {cnt_str}", "green", attrs=["bold"]))
        
    if "local" in args.train_data_gen.data_type:
        dataloader_gen = accelerator.prepare(dataloader_gen)

    # (7) setup trainer
    TrainerClass = import_class(args.trainer.trainer_class)
    trainer = TrainerClass(
        args,
        accelerator,
        model_dict,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        weight_dtype=weight_dtype,
        device=device,
        **args.trainer.trainer_params,
    )

    # start train
    global_step = 0

    # set progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,  # TODO: resume train
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    def buffered_length_sorted_generator(dataloader, buffer_size=256, dataset_type="gen"):
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
        if getattr(dataset_gen, "new_epoch", None):
            dataset_gen.new_epoch()
            
        if hasattr(args, "enable_buffered_length_sorted") and args.enable_buffered_length_sorted:
            print(colored("\nEnable_buffered_length_sorted", "green", attrs=["bold"]))
            sorted_dataloader_gen = buffered_length_sorted_generator(dataloader_gen, buffer_size=256, dataset_type="gen")
        else:
            sorted_dataloader_gen = dataloader_gen
            
        for i, batch_gen in enumerate(sorted_dataloader_gen):
            if global_step >= args.max_train_steps:
                    break
                
            with accelerator.accumulate(model_dict["train_model"]):
                model_dict["train_model"].train()
                loss = trainer.train_step({"gen": batch_gen}, gstep=global_step)
            
            accelerator.wait_for_everyone()
            
            if accelerator.sync_gradients:
                #progress_bar.update(1)
                global_step += 1
                
                # save model under zero2
                if args.zero_stage == "zero2":
                    # save model
                    if global_step % args.checkpointing_steps == 0 or global_step == 10:
                        accelerator.wait_for_everyone()
                        save_path = os.path.join(save_dir, "ckpt", f"step-{global_step}")
                        
                        if in_notebook():
                            tmp_save_path = save_path
                        else:
                            assert save_path.startswith("oss://tstar-image-dataset/"), "wrong of model_path: {}".format(save_path)
                            tmp_save_path = os.path.join("./tmp_ckpt", save_path.replace("oss://tstar-image-dataset/", ""))
                        
                        os.makedirs(tmp_save_path, exist_ok=True)
                        train_model = accelerator.unwrap_model(model_dict["train_model"])
                        
                        # 使用 accelerator.save_model 进行分片保存。这个函数需要被所有进程调用，它会自动处理分片逻辑。
                        print(f"Process {accelerator.process_index}: Starting to save sharded model to {tmp_save_path}...")
                        accelerator.save_model(
                            model=train_model,
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
                
                # print log 
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
                    if hasattr(args.model.prepare.pre_params, "text_encoder_lora") and args.model.prepare.pre_params.text_encoder_lora:
                        lora_delta_w = (model_dict["train_model"].text_encoder.language_model.layers[26].mlp.up_proj.lora_B.default.weight.detach() @ model_dict["train_model"].text_encoder.language_model.layers[26].mlp.up_proj.lora_A.default.weight.detach()).abs().max().item()
                        print("  text_encoder.language_model.layers[26].mlp.up_proj, lora_deltaW, max value: {:.6f}".format(lora_delta_w))
                    else:
                        tmp_w = model_dict["text_encoder"].language_model.layers[26].mlp.up_proj.weight.detach().abs().max()
                        print("  text_encoder.language_model.layers[26].mlp.up_proj, max value: {:.6f}".format(tmp_w))
                    
                    if hasattr(args.model.prepare.pre_params, "transformer_lora") and args.model.prepare.pre_params.transformer_lora:
                        lora_delta_w = (model_dict["train_model"].transformer.transformer_blocks[58].attn.to_v.lora_B.default.weight.detach() @ model_dict["train_model"].transformer.transformer_blocks[58].attn.to_v.lora_A.default.weight.detach()).abs().max().item()
                        print("  transformer.transformer_blocks[58].attn.to_v, lora_deltaW, max value: {:.6f}".format(lora_delta_w))
                    else:
                        tmp_w = model_dict["transformer"].transformer_blocks[58].attn.to_v.weight.detach().abs().max()
                        print("  transformer.transformer_blocks[58].attn.to_v, max value: {:.6f}".format(tmp_w))
                
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
