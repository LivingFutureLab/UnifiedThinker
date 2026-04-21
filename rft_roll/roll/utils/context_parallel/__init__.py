# from roll.utils.context_parallel.globals import get_ulysses_group, set_upg_manager
# from roll.utils.context_parallel.monkey_patch import apply_ulysses_patch, unapply_ulysses_patch


# __all__ = ["set_upg_manager", "get_ulysses_group", "apply_ulysses_patch", "unapply_ulysses_patch"]


#jiachong.zq
from roll.utils.context_parallel.globals import get_ulysses_group, set_upg_manager
__all__ = ["set_upg_manager", "get_ulysses_group"]

import transformers
from packaging.version import parse as V
IS_TRANSFORMERS_GTE_4_53 = V(transformers.__version__) >= V("4.53.0")


if IS_TRANSFORMERS_GTE_4_53:
    pass
else:
    from roll.utils.context_parallel.monkey_patch import apply_ulysses_patch, unapply_ulysses_patch
    __all__.extend(["apply_ulysses_patch", "unapply_ulysses_patch"])
