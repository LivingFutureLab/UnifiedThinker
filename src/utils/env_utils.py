import logging
import os
import time
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import Accelerator
from accelerate.logging import get_logger
from config.deepspeed.deepspeed_config import get_ds_plugin
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored

logger_initialized = {}


def init_accelerator(args, save_dir, logging_dir):
    accelerator_project_config = ProjectConfiguration(
        project_dir=save_dir, logging_dir=logging_dir
    )
    
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
        deepspeed_plugin=get_ds_plugin(args),
    )
    return accelerator, accelerator.device


def init_logger(name=None, logging_dir=None, log_level=logging.INFO, file_mode="w"):
    if not name:
        name = "global"

    formatter = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(
        format=formatter,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    file_handler = logging.FileHandler(os.path.join(logging_dir, "run.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(formatter))

    logger = get_logger(__name__)
    logger.logger.addHandler(file_handler)

    return logger


def import_class(name):
    try:
        nameList = name.split(".")
        mod = __import__(nameList[0])
        for nam in nameList[1:]:
            mod = getattr(mod, nam)
        return mod
    except ImportError as err:
        print("Cant find the module", err.args)


def in_notebook():
    return os.environ.get("SUMMARY_DIR", None) is None


def init_seed(args):
    if args.seed is not None:
        set_seed(args.seed)
