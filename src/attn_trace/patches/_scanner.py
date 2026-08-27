"""AttentionLayerBase subclass discovery + auto-hook install.

vLLM 0.23 只在 `Attention` / `MLAAttention` / `CrossAttention` /
`ChunkedLocalAttention` / `EncoderOnlyAttention` / `StaticSinkAttention`
六个"vanilla"类里定义 `__init__` 和 `get_kv_cache_spec`。但很多模型
（DeepSeek V4、DeepSeek V4 Flash、DSA 系列等）会**直接继承 AttentionLayerBase**
再自己实现，跳过上面 6 个类。硬编码列表在这种模型上会全部落空。

这个模块提供两个能力：

1. `walk_attention_subclasses()`：递归返回当前进程里 AttentionLayerBase 的所有
   已加载子类（包括中间层）。
2. `install_subclass_hook(cb)`：在 AttentionLayerBase 上安装一个
   `__init_subclass__` 钩子，模型文件被 import 后新定义的子类也会被回调。
   通过 "walk 初次 + 钩子后续" 两级覆盖，保证任何时刻加载的子类都能被 wrap。
"""

import importlib
from typing import Callable, Iterable

from ..logutil import get_logger

logger = get_logger()


def get_attention_layer_base():
    """按顺序尝试几个可能的路径，返回 AttentionLayerBase 类；找不到抛 ImportError。"""
    for mod_path in (
        "vllm.model_executor.layers.attention_layer_base",
        "vllm.model_executor.layers.attention.attention",
        "vllm.v1.attention.backend",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, "AttentionLayerBase", None)
        if cls is not None:
            return cls
    raise ImportError("AttentionLayerBase not found in any known vLLM location")


def walk_attention_subclasses(base_cls=None) -> list[type]:
    """递归收集 base_cls 的所有子孙类（去重）。默认 base_cls = AttentionLayerBase。"""
    if base_cls is None:
        base_cls = get_attention_layer_base()
    seen: set[type] = set()
    result: list[type] = []
    stack = list(base_cls.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        result.append(cls)
        stack.extend(cls.__subclasses__())
    return result


def install_subclass_hook(callback: Callable[[type], None], base_cls=None) -> bool:
    """在 AttentionLayerBase 上装 `__init_subclass__` 钩子。

    - 现存子类：调用者自己先跑一次 walk_attention_subclasses 处理。
    - 未来子类：本函数负责让 callback 在每个新子类被定义时被调用一次。

    返回 True 表示钩子装成功；False 表示 AttentionLayerBase 不可用。
    """
    if base_cls is None:
        try:
            base_cls = get_attention_layer_base()
        except ImportError as e:
            logger.warning("attn_trace._scanner: %s", e)
            return False

    # 防重复挂钩
    if getattr(base_cls, "_attn_trace_hook_installed", False):
        return True

    # 保留链上原有的 __init_subclass__（ABC 有自己的、可能还有别的）
    orig = base_cls.__dict__.get("__init_subclass__")
    orig_fn = orig.__func__ if isinstance(orig, classmethod) else orig

    def _new_hook(cls, **kwargs):
        # 先跑原逻辑（如果有）
        if orig_fn is not None:
            try:
                orig_fn(cls, **kwargs)
            except TypeError:
                # 有些实现签名可能不同，兜底跑无 kwargs 版本
                try:
                    orig_fn(cls)
                except Exception:
                    pass
            except Exception:
                pass
        # 再跑 callback
        try:
            callback(cls)
        except Exception as e:
            logger.warning("attn_trace: subclass hook callback failed for %s: %s", cls, e)

    base_cls.__init_subclass__ = classmethod(_new_hook)
    base_cls._attn_trace_hook_installed = True
    logger.info("attn_trace: installed __init_subclass__ hook on %s", base_cls.__name__)
    return True


def wrap_method_if_defined(
    cls: type,
    method_name: str,
    wrapper_factory: Callable[[Callable, type], Callable],
    marker_attr: str,
) -> bool:
    """如果 cls 自己（不是继承来的）定义了 method_name，就把它替换成 wrapper_factory 产出的包装。

    - marker_attr 用于幂等：包装后的函数会带上 setattr(wrapper, marker_attr, True)。
    - 返回 True 表示这次真的 patch 了；False 表示 cls 没有自己的 method 或已被 patch。
    """
    if method_name not in cls.__dict__:
        return False
    orig = cls.__dict__[method_name]
    if getattr(orig, marker_attr, False):
        return False
    wrapper = wrapper_factory(orig, cls)
    setattr(wrapper, marker_attr, True)
    setattr(cls, method_name, wrapper)
    return True


def unique_class_paths(classes: Iterable[type]) -> list[str]:
    """把类列表拉平成 'module.Name' 字符串，便于日志显示。"""
    return sorted({f"{c.__module__}.{c.__name__}" for c in classes})
