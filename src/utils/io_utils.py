import hashlib

def generate_text_hash(text, algorithm='sha256'):
  """
  生成给定文本的哈希值。

  Args:
    text: 需要哈希的文本字符串。
    algorithm: 使用的哈希算法名称（字符串），例如 'sha256', 'sha512', 'md5' 等。
               默认为 'sha256'。
               注意：md5 和 sha1 已不再被认为是安全的加密哈希算法。

  Returns:
    文本的哈希值字符串（十六进制表示），如果算法不支持则返回 None。
  """
  try:
    # 创建一个哈希对象
    # 注意：hashlib 要求输入是 bytes 类型，所以需要对字符串进行编码
    # 通常使用 UTF-8 编码
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))

    # 获取十六进制格式的哈希值
    hash_value = hasher.hexdigest()

    return hash_value

  except ValueError:
    print(f"错误：不支持的哈希算法 '{algorithm}'")
    return None
  except Exception as e:
    print(f"发生错误：{e}")
    return None

import subprocess


def count_parameters(model):
    """
    统计 PyTorch 模型中可训练参数的总数，并以人可读的单位返回。

    参数:
        model (torch.nn.Module): 需要统计参数的 PyTorch 模型。

    返回:
        str: 表示参数总数的人可读字符串 (例如："1.2M", "50K", "1.5G")。
    """
    # 统计所有需要梯度的（即通常是可训练的）参数总数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 将参数数量转换为人可读的单位
    if total_params < 1000:
        return f"{total_params}"
    elif total_params < 1_000_000:
        # .1f 表示保留一位小数
        return f"{total_params / 1000:.1f}K"
    elif total_params < 1_000_000_000:
        return f"{total_params / 1_000_000:.1f}M"
    else:
        return f"{total_params / 1_000_000_000:.1f}B"

def count_parameters_zero3(model):
    """
    统计 PyTorch 模型中可训练参数的总数，并以人可读的单位返回。

    参数:
        model (torch.nn.Module): 需要统计参数的 PyTorch 模型。

    返回:
        str: 表示参数总数的人可读字符串 (例如："1.2M", "50K", "1.5G")。
    """
    # 统计所有需要梯度的（即通常是可训练的）参数总数
    from deepspeed import DeepSpeedEngine
    if isinstance(model, DeepSpeedEngine):
        def _get_model_parameters(model):
            num_params = 0
            trainable_num_params = 0

            for p in model.module.parameters():
                # since user code might call deepspeed.zero.Init() before deepspeed.initialize(), need to check the attribute to check if the parameter is partitioned in zero 3 already or not
                n = 0
                if hasattr(p, "ds_tensor"):  # if the parameter is partitioned in zero 3
                    n += p.ds_numel
                else:  # if the parameter is not partitioned in zero 3 yet
                    n += p.numel()
                num_params += n
                if p.requires_grad:
                    trainable_num_params += n
            return trainable_num_params, num_params
        total_trainable_params, total_params = _get_model_parameters(model)
    else:
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

    # 将参数数量转换为人可读的单位
    def convert_formae(total_params):
        if total_params < 1000:
            return f"{total_params}"
        elif total_params < 1_000_000:
            # .1f 表示保留一位小数
            return f"{total_params / 1000:.1f}K"
        elif total_params < 1_000_000_000:
            return f"{total_params / 1_000_000:.1f}M"
        else:
            return f"{total_params / 1_000_000_000:.1f}B"
    return convert_formae(total_trainable_params), convert_formae(total_params)
    
if __name__ == '__main__':
    print(generate_text_hash("hello world", "sha256"))