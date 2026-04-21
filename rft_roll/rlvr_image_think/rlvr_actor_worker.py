#jianchong.zq: 继承 'roll/pipeline/rlvr/actor_worker.py', 方便进行调试

import logging
import numpy as np
import torch 

from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto

from roll.pipeline.rlvr.actor_worker import ActorWorker

logger = logging.getLogger(__name__)

DEBUG = True

class ImageThinkActorWorker(ActorWorker):
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def generate(self, data: DataProto):
        if DEBUG:
            logger.info("---------- PDB debugger enabled for generate() ----------")
            import pdb; pdb.set_trace()
            # data check ...
        
        return super().generate(data)
        