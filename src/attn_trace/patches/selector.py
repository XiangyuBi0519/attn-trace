"""Patch: `vllm.v1.attention.selector._cached_get_attn_backend`.

Emits `[ATTN_BACKEND_PICK]` for every distinct (head_size, dtype, use_mla,
use_sparse, ...) combination that vLLM resolves during startup.
"""

from ..logutil import get_logger

logger = get_logger()


def apply() -> None:
    import vllm.v1.attention.selector as sel

    _orig = sel._cached_get_attn_backend  # 已被 functools.cache 包过

    def wrapped(backend, attn_selector_config, num_heads=None):
        # 直接调原（cached）实现，保留原来的缓存语义
        result = _orig(
            backend=backend,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        try:
            required_layout = None
            try:
                required_layout = result.get_required_kv_cache_layout()
            except Exception:
                pass
            logger.info(
                "[ATTN_BACKEND_PICK] backend_cls=%s.%s selected=%s "
                "num_heads=%s required_kv_layout=%s attn_cfg=%s",
                getattr(result, "__module__", "?"),
                getattr(result, "__name__", "?"),
                backend,
                num_heads,
                required_layout,
                attn_selector_config,
            )
        except Exception as e:
            logger.warning("[ATTN_BACKEND_PICK] log failed: %s", e)
        return result

    sel._cached_get_attn_backend = wrapped
    logger.info("attn_trace: patched vllm.v1.attention.selector._cached_get_attn_backend")
