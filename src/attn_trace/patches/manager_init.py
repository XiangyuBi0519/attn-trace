"""Patch: `vllm.v1.core.single_type_kv_cache_manager.get_manager_for_kv_cache_spec`.

Emits `[KV_MGR_INIT]` every time a per-group single-type manager is created,
so you can see which Manager class (FullAttention / SlidingWindow /
ChunkedLocal / Mamba / SinkFullAttention / Cross) is bound to each group.
"""

import importlib

from ..logutil import get_logger

logger = get_logger()


def apply() -> None:
    import vllm.v1.core.single_type_kv_cache_manager as stm

    _orig = stm.get_manager_for_kv_cache_spec

    def wrapped(kv_cache_spec, max_num_batched_tokens, max_model_len, **kwargs):
        manager = _orig(
            kv_cache_spec, max_num_batched_tokens, max_model_len, **kwargs
        )
        try:
            logger.info(
                "[KV_MGR_INIT] spec=%s manager=%s block_size=%s "
                "kv_group_id=%s max_admission_blocks=%s "
                "max_num_batched_tokens=%s max_model_len=%s",
                type(kv_cache_spec).__name__,
                type(manager).__name__,
                getattr(kv_cache_spec, "block_size", None),
                kwargs.get("kv_cache_group_id"),
                kwargs.get("max_admission_blocks_per_request"),
                max_num_batched_tokens,
                max_model_len,
            )
        except Exception as e:
            logger.warning("[KV_MGR_INIT] log failed: %s", e)
        return manager

    stm.get_manager_for_kv_cache_spec = wrapped

    # 同名符号也可能被 kv_cache_coordinator 提前 import；一并覆盖。
    for consumer in (
        "vllm.v1.core.kv_cache_coordinator",
        "vllm.v1.core.kv_cache_manager",
    ):
        try:
            mod = importlib.import_module(consumer)
            if hasattr(mod, "get_manager_for_kv_cache_spec"):
                setattr(mod, "get_manager_for_kv_cache_spec", wrapped)
        except Exception:
            pass

    logger.info("attn_trace: patched get_manager_for_kv_cache_spec")
