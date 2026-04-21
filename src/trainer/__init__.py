import os
import importlib

# 获取当前目录下的所有 Python 文件
_current_dir = os.path.dirname(__file__)
_files = [f[:-3] for f in os.listdir(_current_dir) 
          if f.endswith('.py') and f != '__init__.py']

# 动态导入所有模块
for _module in _files:
    importlib.import_module(f'.{_module}', package=__package__)

# 从 base_trainer 导入
from . import base_trainer