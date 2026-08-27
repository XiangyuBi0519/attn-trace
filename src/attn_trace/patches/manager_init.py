"""Patch: `vllm.v1.core.single_type_kv_cache_manager.get_manager_for_kv_cache_spec`.

Emits `[KV_MGR_INIT]` every time a per-group single-type manager is created.
Scans every already-imported module under `vllm.v1.core.*` for a
`get_manager_for_kv_cache_spec` symbol (some coordinators do
`from ... import get_manager_for_kv_cache_spec` before we get a chance to
patch the source module), so hybrid-cache coordinators that inline the
name still see our wrapped version.
"""

import sys

from ..logutil import get_logger

logger = get_logger()


def apply() -> None:
    import vllm.v1.core.single_type_kv_cache_manager as stm

    orig = stm.get_manager_for_kv_cache_spec

    def wrapped(kv_cache_spec, max_num_batched_tokens, max_model_len, **kwargs):
        manager = orig(
            kv_cache_spec, max_num_batched_tokens, max_model_len, **kwargs
        )
        try:
            logger.info(
                "[KV_MGR_INIT] spec=%s.%s manager=%s.%s block_size=%s "
                "kv_group_id=%s max_admission_blocks=%s "
                "max_num_batched_tokens=%s max_model_len=%s",
                type(kv_cache_spec).__module__, type(kv_cache_spec).__name__,
                type(manager).__module__, type(manager).__name__,
                getattr(kv_cache_spec, "block_size", None),
                kwargs.get("kv_cache_group_id"),
                kwargs.get("max_admission_blocks_per_request"),
                max_num_batched_tokens,
                max_model_len,
            )
        except Exception as e:
            logger.warning("[KV_MGR_INIT] log failed: %s", e)
        return manager

    # 覆盖源模块
    stm.get_manager_for_kv_cache_spec = wrapped

    # 扫描所有 vllm.* / vllm_ascend.* / 任何自定义 plugin 模块，把
    # `from ... import get_manager_for_kv_cache_spec` 拿到的本地符号一并替换。
    # 之前只扫 `vllm.v1.core.*` 会漏掉 vllm_ascend.core 系列，导致 hybrid
    # coordinator 拿的是未 patched 的引用，KV_MGR_INIT 因此从不 fire。
    replaced_in = []
    for name, mod in list(sys.modules.items()):
        if mod is None or mod is stm:
            continue
        # 跳过 stdlib 之外的 3rd party 无关模块（性能优化，非必需）
        if not (
            name.startswith("vllm")
            or name.startswith("kv_cache_affinity")
            or "coordinator" in name
            or "kv_cache" in name
        ):
            continue
        if getattr(mod, "get_manager_for_kv_cache_spec", None) is orig:
            mod.get_manager_for_kv_cache_spec = wrapped
            replaced_in.append(name)

    logger.info(
        "attn_trace: patched get_manager_for_kv_cache_spec "
        "(source + %d consumer namespaces: %s)",
        len(replaced_in), replaced_in,
    )
