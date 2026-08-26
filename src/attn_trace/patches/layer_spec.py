"""Patch: per-layer `get_kv_cache_spec` methods.

Emits `[LAYER_KV_SPEC]` for every attention layer as it's asked what KV cache
spec to use. Covers the base `Attention` plus any subclass that overrides
`get_kv_cache_spec` (MLA / Cross / ChunkedLocal / EncoderOnly / StaticSink).
"""

import importlib

from ..logutil import get_logger

logger = get_logger()


# (module path, class name) — silently skipped if the module/class is absent
# on the current vLLM install.
_LAYER_CLASSES = [
    ("vllm.model_executor.layers.attention.attention", "Attention"),
    ("vllm.model_executor.layers.attention.mla_attention", "MLAAttention"),
    ("vllm.model_executor.layers.attention.cross_attention", "CrossAttention"),
    ("vllm.model_executor.layers.attention.chunked_local_attention", "ChunkedLocalAttention"),
    ("vllm.model_executor.layers.attention.encoder_only_attention", "EncoderOnlyAttention"),
    ("vllm.model_executor.layers.attention.static_sink_attention", "StaticSinkAttention"),
]


def apply() -> None:
    patched = 0
    for mod_path, cls_name in _LAYER_CLASSES:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        # 只 patch 直接在该类字典里定义的 get_kv_cache_spec；如果只是继承来的，
        # 说明父类已被 patch，不用再套一层（否则日志会重复触发）。
        if "get_kv_cache_spec" not in cls.__dict__:
            continue
        orig = cls.__dict__["get_kv_cache_spec"]
        if getattr(orig, "_attn_trace_wrapped", False):
            continue

        def _make_wrapper(orig_fn, tag_cls_name):
            def wrapper(self, vllm_config):
                spec = orig_fn(self, vllm_config)
                _emit_layer_spec_log(self, tag_cls_name, spec)
                return spec
            wrapper._attn_trace_wrapped = True  # type: ignore[attr-defined]
            return wrapper

        setattr(cls, "get_kv_cache_spec", _make_wrapper(orig, cls_name))
        patched += 1
        logger.info("attn_trace: patched %s.%s.get_kv_cache_spec", mod_path, cls_name)

    if patched == 0:
        logger.warning(
            "attn_trace.layer_spec: no attention classes patched — "
            "vLLM layout may have changed"
        )


def _emit_layer_spec_log(layer, cls_name: str, spec) -> None:
    if spec is None:
        return
    try:
        logger.info(
            "[LAYER_KV_SPEC] layer=%s cls=%s spec=%s block_size=%s "
            "num_kv_heads=%s head_size=%s dtype=%s "
            "sliding_window=%s attention_chunk_size=%s kv_dtype=%s "
            "attn_type=%s use_mla=%s",
            getattr(layer, "layer_name", "?"),
            cls_name,
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
        logger.warning("[LAYER_KV_SPEC] log failed for %s: %s", cls_name, e)
