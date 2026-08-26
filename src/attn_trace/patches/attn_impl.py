"""Patch: per-layer `Attention.__init__` (and subclasses).

Emits `[ATTN_IMPL_INIT]` after each attention module is fully constructed.
This shows, for every real layer instance, which backend class was chosen
and which concrete `impl` subclass ended up handling that layer — i.e. the
piece that decides forward-time kernels and KV read/write semantics.

`[ATTN_BACKEND_PICK]` only fires once per distinct (head_size, dtype, ...)
tuple (functools.cache) and `[LAYER_KV_SPEC]` only tells you the spec, not
the impl. This patch fills the gap between the two.
"""

import importlib

from ..logutil import get_logger

logger = get_logger()


# (module path, class name). Skipped silently if missing on this vLLM install.
# Order matters: base `Attention` first so a subclass that inherits __init__
# from it is still covered indirectly (though we only wrap classes that
# *own* __init__).
_ATTN_CLASSES = [
    ("vllm.model_executor.layers.attention.attention", "Attention"),
    ("vllm.model_executor.layers.attention.mla_attention", "MLAAttention"),
    ("vllm.model_executor.layers.attention.cross_attention", "CrossAttention"),
    ("vllm.model_executor.layers.attention.chunked_local_attention", "ChunkedLocalAttention"),
    ("vllm.model_executor.layers.attention.encoder_only_attention", "EncoderOnlyAttention"),
    ("vllm.model_executor.layers.attention.static_sink_attention", "StaticSinkAttention"),
]


def apply() -> None:
    patched = 0
    for mod_path, cls_name in _ATTN_CLASSES:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        # 只 patch 直接定义了 __init__ 的类；纯继承的子类靠父类 patch 生效即可。
        if "__init__" not in cls.__dict__:
            continue
        orig = cls.__dict__["__init__"]
        if getattr(orig, "_attn_trace_wrapped", False):
            continue

        def _make_wrapper(orig_fn, tag_cls_name):
            def wrapper(self, *args, **kwargs):
                orig_fn(self, *args, **kwargs)
                _emit_impl_log(self, tag_cls_name)
            wrapper._attn_trace_wrapped = True  # type: ignore[attr-defined]
            return wrapper

        setattr(cls, "__init__", _make_wrapper(orig, cls_name))
        patched += 1
        logger.info("attn_trace: patched %s.%s.__init__", mod_path, cls_name)

    if patched == 0:
        logger.warning(
            "attn_trace.attn_impl: no attention classes patched — "
            "vLLM layout may have changed"
        )


def _emit_impl_log(layer, cls_name: str) -> None:
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
            "[ATTN_IMPL_INIT] layer=%s cls=%s "
            "backend=%s backend_cls=%s.%s "
            "impl_cls=%s.%s "
            "num_heads=%s num_kv_heads=%s head_size=%s "
            "sliding_window=%s kv_cache_dtype=%s "
            "attn_type=%s kv_sharing_target=%s",
            getattr(layer, "layer_name", "?"),
            cls_name,
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
        logger.warning("[ATTN_IMPL_INIT] log failed for %s: %s", cls_name, e)
