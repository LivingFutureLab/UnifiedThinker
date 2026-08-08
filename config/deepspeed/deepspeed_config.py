from accelerate import DeepSpeedPlugin
from accelerate.utils.deepspeed import HfDeepSpeedConfig


def get_ds_plugin(args):
    if args.zero_stage == 'zero2':
        stage = 2
    elif args.zero_stage == 'zero3':
        stage = 3
    else:
        raise ValueError(f"Invalid deepspeed stage: {args.zero_stage}, which should be 'zero2' or 'zero3'")
    
    ds_conf_dict = {
        "train_micro_batch_size_per_gpu": args.train_bs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {"device": "none", "nvme_path": None},
            "offload_param": {"device": "none", "nvme_path": None},
            "stage3_gather_16bit_weights_on_model_save": True,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        
        # "flops_profiler": {
        #     "enabled": True,
        #     "profile_step": 1,
        #     "module_depth": -1,
        #     "top_modules": 1,
        #     "detailed": True,
        #     "output_file": None,
        # },
        
        "zero_allow_untested_optimizer": True,
    }
    if args.mixed_precision == "fp16":
        ds_conf_dict["fp16"] = {
            "enabled": True,
            "auto_cast": True,
        }
    elif args.mixed_precision == "bf16":
        ds_conf_dict["bf16"] = {"enabled": True}
                
    ds_config = HfDeepSpeedConfig(config_file_or_dict=ds_conf_dict)
    ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
    return ds_plugin
