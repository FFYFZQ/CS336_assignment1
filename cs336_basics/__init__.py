import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")


# 导入 Tokenizer 类
from .Tokenization import Tokenizer
from .linear_layer import Linear
from .embedding_layer import Embedding
from .RMSNorm import RMSNorm
from .feedforward import SwiGLU_FFN
from .rope import RoPE


