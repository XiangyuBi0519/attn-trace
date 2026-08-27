"""Patch: `AttentionLayerBase.get_kv_cache_spec` on every subclass.

Emits `[LAYER_KV_SPEC]` for every attention/cache layer as it's asked what
KV cache spec to use. Covers ANY class that inherits from
`AttentionLayerBase` — including custom model-side classes like
`DSAAttention`, `AscendCompressorStateCache`, `AscendDeepseekV4IndexerCache`,
`AscendDeepseekV4SWACache`, plus the vanilla `Attention` / `MLAAttention`
/ etc.

Uses subclass auto-discovery (walk + `__init_subclass__` hook) rather than
a hardcoded class list, so models loaded after `register()` are also
covered automatically.
"""

from ..logutil import get_logger
from . import _scanner

logger = get_logger()

_MARKER = "_attn_trace_spec_wrapped"


def apply() -> None:
    # 1) 处理注册时已经加载好的所有子类
    try:
        existing = _scanner.walk_attention_subclasses()
    except ImportError as e:
        logger.warning("attn_trace.layer_spec: %s", e)
        return

    patched_now = [c for c in existing if _wrap(c)]
    if patched_now:
        logger.info(
            "attn_trace.layer_spec: wrapped get_kv_cache_spec on %d existing subclasses: %s",
            len(patched_now), _scanner.unique_class_paths(patched_now),
        )
    else:
        logger.info(
            "attn_trace.layer_spec: no existing subclass owned get_kv_cache_spec at "
            "register time (this is normal — model files load later)"
        )

    # 2) 装 __init_subclass__ 钩子，让后续 import 的模型类也被抓到
    _scanner.install_subclass_hook(_wrap_and_log)


def _wrap_and_log(cls) -> None:
    if _wrap(cls):
        logger.info(
            "attn_trace.layer_spec: late-wrapped get_kv_cache_spec on %s.%s",
            cls.__module__, cls.__name__,
        )


def _wrap(cls) -> bool:
    return _scanner.wrap_method_if_defined(
        cls, "get_kv_cache_spec", _make_wrapper, _MARKER,
    )


def _make_wrapper(orig_fn, owner_cls):
    def wrapper(self, vllm_config):
        spec = orig_fn(self, vllm_config)
        _emit(self, owner_cls, spec)
        return spec
    return wrapper


def _emit(layer, owner_cls, spec) -> None:
    if spec is None:
        return
    try:
        logger.info(
            "[LAYER_KV_SPEC] layer=%s cls=%s.%s spec=%s block_size=%s "
            "num_kv_heads=%s head_size=%s dtype=%s "
            "sliding_window=%s attention_chunk_size=%s "
            "kv_dtype=%s attn_type=%s use_mla=%s",
            getattr(layer, "layer_name", "?"),
            owner_cls.__module__, owner_cls.__name__,
            type(spec).__name__,
            getattr(spec, "block_size", None),
            getattr(spec, "num_kv_heads", None),
            getattr(spec, "head_size", None),
            getattr(spec, "dtype", None),
            getattr(spec, "sliding_window", None),
            getattr(spec, "attention_chunk_size", None),
            getattr(layer, "kv_cache_dtype", None),
            getattr(layer, "attn_type", None),
            getattr(layer, "use_mla", None),
        )
    except Exception as e:
        logger.warning("[LAYER_KV_SPEC] log failed for %s: %s", owner_cls.__name__, e)
