import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
    
import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.utils.logging import get_logger
from roll.distributed.scheduler.initialize import init

try:
    import flash_attn
except Exception as e:
    print("error of {}".format(str(e)))

#from rlvr_image_think.rlvr_image_think_pipeline_v2 import RLVRImageThinkPipeline, RLVRImageThinkConfig # 还有问题，需要调试
from rlvr_image_think.rlvr_image_think_pipeline_v1 import RLVRImageThinkPipeline, RLVRImageThinkConfig

logger = get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="./")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    parser.add_argument("--run_timestamp", type=str, help="Unique timestamp for the run")
    args = parser.parse_args()
    
    #config_path = os.path.relpath(os.path.join(rootdir, "scripts/qwen_image_edit_think_rlvr"),  start=os.path.dirname(__file__))
    
    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=RLVRImageThinkConfig, data=OmegaConf.to_container(cfg, resolve=True))
    
    if ppo_config.system_envs.get("RAY_DEBUG", "") == "legacy":
        print("在对应后端的 Platform 环境配置中设置 'RAY_DEBUG' 为 'legacy', 这样就可以使用 pdb 进行单步调试")
        import pdb; pdb.set_trace()
    
    # if args.run_timestamp is not None:
    #     ppo_config.checkpoint_config['output_dir'] = os.path.join(ppo_config.checkpoint_config['output_dir'], f"exp_{args.run_timestamp}")
    if ppo_config.checkpoint_config["type"] == "file_system":
        os.makedirs(ppo_config.checkpoint_config['output_dir'], exist_ok=True)
        logger.info(f"ckpt output_dir: {ppo_config.checkpoint_config['output_dir']}")

    init()

    pipeline = RLVRImageThinkPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
