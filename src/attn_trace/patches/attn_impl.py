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
            getattr(layer, "layer_name", "?"),
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
