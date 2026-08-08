def load_optim(optim_class):
    if optim_class == "AdamW":
        from torch.optim import AdamW

        optim_func = AdamW
    elif optim_class == "8bit_AdamW":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optim_func = bnb.optim.AdamW8bit
    elif optim_class == "Adam":
        from torch.optim import Adam

        optim_func = Adam
    elif optim_class == "Prodigy":  # ominicontrol use it.
        from prodigyopt import Prodigy

        optim_func = Prodigy
    elif optim_class == "CAME":
        from came_pytorch import CAME

        optim_func = CAME
    else:
        raise NotImplementedError("Now only support [Adam, AdamW, Prodigy, 8bit-AdamW]")
    return optim_func