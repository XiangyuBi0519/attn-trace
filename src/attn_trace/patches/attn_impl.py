"""Patch: `__init__` on every AttentionLayerBase subclass.

Emits `[ATTN_IMPL_INIT]` after each attention/cache module is fully
constructed. Uses subclass auto-discovery so custom attention classes
(DeepSeek V4 `DSAAttention`, `AscendCompressorStateCache`,
`AscendDeepseekV4IndexerCache`, `AscendDeepseekV4SWACache`, etc.) that
inherit directly from `AttentionLayerBase` and never touch the vanilla
`Attention` class are still covered.
"""

from ..logutil import get_logger
from . import _scanner

logger = get_logger()

_MARKER = "_attn_trace_init_wrapped"


def apply() -> None:
    try:
        existing = _scanner.walk_attention_subclasses()
    except ImportError as e:
        logger.warning("attn_trace.attn_impl: %s", e)
        return

    patched_now = [c for c in existing if _wrap(c)]
    if patched_now:
        logger.info(
            "attn_trace.attn_impl: wrapped __init__ on %d existing subclasses: %s",
            len(patched_now), _scanner.unique_class_paths(patched_now),
        )
    else:
        logger.info(
            "attn_trace.attn_impl: no existing subclass owned __init__ at "
            "register time (model classes load later)"
        )

    _scanner.install_subclass_hook(_wrap_and_log)


def _wrap_and_log(cls) -> None:
    if _wrap(cls):
        logger.info(
            "attn_trace.attn_impl: late-wrapped __init__ on %s.%s",
            cls.__module__, cls.__name__,
        )


def _wrap(cls) -> bool:
    return _scanner.wrap_method_if_defined(
        cls, "__init__", _make_wrapper, _MARKER,
    )


def _make_wrapper(orig_fn, owner_cls):
    def wrapper(self, *args, **kwargs):
        orig_fn(self, *args, **kwargs)
        _emit(self, owner_cls)
        _emit_csa2(self, owner_cls)
    return wrapper


def _emit(layer, owner_cls) -> None:
    try:
        impl = getattr(layer, "impl", None)
        backend = getattr(layer, "attn_backend", None)
        backend_name = None
        if backend is not None:
            try:
                backend_name = backend.get_name()
            except Exception:
                backend_name = getattr(backend, "__name__", type(backend).__name__)

        logger.info(
            "[ATTN_IMPL_INIT] layer=%s cls=%s.%s "
            "backend=%s backend_cls=%s.%s "
            "impl_cls=%s.%s "
            "num_heads=%s num_kv_heads=%s head_size=%s "
            "sliding_window=%s kv_cache_dtype=%s "
            "attn_type=%s kv_sharing_target=%s",
            getattr(layer, "layer_name", None) or getattr(layer, "prefix", "?"),
            owner_cls.__module__, owner_cls.__name__,
            backend_name,
            getattr(backend, "__module__", "?"),
            getattr(backend, "__name__", "?"),
            getattr(impl, "__module__", "?") if impl is not None else "?",
            type(impl).__name__ if impl is not None else "?",
            getattr(impl, "num_heads", None) if impl is not None else None,
            getattr(impl, "num_kv_heads", None) if impl is not None else None,
            getattr(impl, "head_size", None) if impl is not None else None,
            getattr(layer, "sliding_window", None),
            getattr(layer, "kv_cache_dtype", None),
            getattr(layer, "attn_type", None),
            getattr(layer, "kv_sharing_target_layer_name", None),
        )
    except Exception as e:
        logger.warning("[ATTN_IMPL_INIT] log failed for %s: %s", owner_cls.__name__, e)


def _emit_csa2(layer, owner_cls) -> None:
    """CSA2（DeepSeek V4 / V4.1）专属视图：从 impl.role 推 Full / Reindex / Reuse
    模式，并打出跨层复用的来源层（long_kv / index_k）与 compress_ratio。

    模式判据（vllm_ascend.attention.dsa_v41 的 role）：
      Full    = is_kv_source and is_index_source （自算 KV + 自算 index）
      Reindex = not is_kv_source and is_index_source （复用 KV、自算 index）
      Reuse   = not is_kv_source and not is_index_source （KV + Top-K 都复用）

    非 CSA2 层（role 不存在）静默跳过——本函数只对带 role 的 DSA 注意力生效。
    """
    impl = getattr(layer, "impl", None)
    role = getattr(impl, "role", None) if impl is not None else None
    if role is None:
        return
    try:
        is_kv = getattr(role, "is_kv_source", None)
        is_idx = getattr(role, "is_index_source", None)
        if is_kv and is_idx:
            mode = "Full"
        elif is_idx:
            mode = "Reindex"
        elif is_kv is False and is_idx is False:
            mode = "Reuse"
        else:
            mode = "?"
        logger.info(
            "[CSA2_LAYER] layer=%s cls=%s.%s mode=%s compress_ratio=%s "
            "is_kv_source=%s is_index_source=%s has_long_context=%s "
            "long_kv_source=%s index_k_source=%s compressor_state=%s",
            getattr(layer, "layer_name", None) or getattr(layer, "prefix", "?"),
            owner_cls.__module__, owner_cls.__name__,
            mode,
            getattr(role, "compress_ratio", getattr(layer, "compress_ratio", None)),
            is_kv, is_idx,
            getattr(role, "has_long_context", None),
            getattr(impl, "long_kv_source_prefix", None),
            getattr(impl, "index_k_source_prefix", None),
            getattr(impl, "compressor_state_prefix", None),
        )
    except Exception as e:
        logger.warning("[CSA2_LAYER] log failed for %s: %s", owner_cls.__name__, e)
