import sys
import os

# 添加模块所在目录到Python路径
# sys.path.append("/home/hanyi.zz/ScaleAligner/roll/agentic/env/alfworld")

def get_environment(env_type):
    if env_type == 'AlfredTWEnv':
        from roll.pipeline.agentic.env.alfworld.src.agents.environment.alfred_tw_env import AlfredTWEnv
        return AlfredTWEnv
    elif env_type == 'AlfredThorEnv':
        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
        return AlfredThorEnv
    elif env_type == 'AlfredHybrid':
        from alfworld.agents.environment.alfred_hybrid import AlfredHybrid
        return AlfredHybrid
    else:
        raise NotImplementedError(f"Environment {env_type} is not implemented.")
