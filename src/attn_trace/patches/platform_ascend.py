"""Patch: `vllm_ascend.platform.NPUPlatform.get_attn_backend_cls`.

Emits `[NPU_ATTN_MAP]` showing which Ascend backend qualname is selected
for every (use_mla, use_sparse, use_compress, is_310p, selected_backend)
combination.

Ascend-only — silently skipped when vllm_ascend is not installed.
"""

from ..logutil import get_logger

logger = get_logger()


def apply() -> None:
    from vllm_ascend.platform import NPUPlatform  # ImportError → skipped by caller

    # 拿到 classmethod 对象背后的原始函数
    orig_cm = NPUPlatform.__dict__.get("get_attn_backend_cls")
    if orig_cm is None:
        logger.info("attn_trace: NPUPlatform has no get_attn_backend_cls, skip")
        return

    orig_fn = orig_cm.__func__

    def wrapped_fn(cls, selected_backend, attn_selector_config, num_heads=None):
        result = orig_fn(cls, selected_backend, attn_selector_config, num_heads)
        try:
            key = (
                getattr(attn_selector_config, "use_mla", None),
                getattr(attn_selector_config, "use_sparse", None),
            )
            use_compress = getattr(attn_selector_config, "use_compress", False)
            is_310p = False
            try:
                from vllm_ascend.utils import is_310p as _is_310p
                is_310p = bool(_is_310p())
            except Exception:
                pass
            logger.info(
                "[NPU_ATTN_MAP] selected_backend=%s key=(mla=%s,sparse=%s) "
                "use_compress=%s is_310p=%s -> %s",
                selected_backend, key[0], key[1], use_compress, is_310p, result,
            )
        except Exception as e:
            logger.warning("[NPU_ATTN_MAP] log failed: %s", e)
        return result

    NPUPlatform.get_attn_backend_cls = classmethod(wrapped_fn)
    logger.info("attn_trace: patched NPUPlatform.get_attn_backend_cls")
