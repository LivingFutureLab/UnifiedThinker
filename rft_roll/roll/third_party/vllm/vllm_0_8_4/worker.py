import gc
from typing import Optional

import torch
from roll.platforms import current_platform

from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger

logger = get_logger()

Worker = current_platform.get_vllm_worker_class()

class Worker084(WorkerHelper, Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
