"""vLLM plugin entry point for attn_trace.

vLLM 启动时通过 setup.py 声明的 `vllm.general_plugins` entry point 发现本模块，
在主进程和每个 EngineCore 子进程各调用一次 ``register()``。
"""
from attn_trace import enable


def register() -> None:
    """安装只读钩子。幂等——多次调用只挂一次。"""
    enable()
