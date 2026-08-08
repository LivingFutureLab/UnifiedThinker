import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
    
import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from rlvr_image_think.rlvr_vlm_pipeline import RLVRConfig, RLVRVLMPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="./")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    if ppo_config.system_envs.get("RAY_DEBUG", "") == "legacy":
        print("在对应后端的 Platform 环境配置中设置 'RAY_DEBUG' 为 'legacy', 这样就可以使用 pdb 进行单步调试")
        import pdb; pdb.set_trace()
        
    init()

    pipeline = RLVRVLMPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
