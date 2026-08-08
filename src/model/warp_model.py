import torch
import torch.nn as nn


class WarpModel(nn.Module):
    def __init__(self, model_dict, warp_key=[]):
        super().__init__()

        for key in warp_key:
            setattr(self, key, model_dict[key])
    
    def forward(self, sub_model_name, **kwargs):
        if getattr(self, sub_model_name, None) is None:
            raise ValueError(f'{sub_model_name} is not in model_dict')
        return getattr(self, sub_model_name)(**kwargs)


class WarpModel2(nn.Module):
    def __init__(self, model_dict, warp_key=[]):
        """
        使用 nn.ModuleDict 来封装子模型。
        Args:
            model_dict (dict): 一个键为字符串，值为 nn.Module 子类的字典。
                               例如: {'encoder': Encoder(), 'decoder': Decoder()}
        """
        super().__init__()
        self.models = nn.ModuleDict()
        for key in warp_key:
            self.models[key] = model_dict[key]
    def forward(self, sub_model_name, **kwargs):
        if sub_model_name not in self.models:
            raise ValueError(f"'{sub_model_name}' is not a valid sub_model. "
                             f"Available models: {list(self.models.keys())}")
        return self.models[sub_model_name](**kwargs)