"""Patches: `EngineCore._initialize_kv_caches`, `EngineCore.__init__`,
`resolve_kv_cache_block_sizes`.

Emits:
- `[BLOCK_SIZES]` right after vLLM resolves scheduler_block_size /
  hash_block_size. THIS IS THE ONE that decides whether your plugin's
  `Request.from_engine_core_request` re-hashed block_hashes align with what
  the pool actually stores.
- `[KV_GROUP]` for every KV cache group finalized inside `_initialize_kv_caches`.
- `[KV_STARTUP_SUMMARY]` at the end of `EngineCore.__init__`: one-line summary
  of specs, managers, groups, layers, num_blocks, prefix caching, connector.
"""

from collections import Counter

from ..logutil import get_logger

logger = get_logger()


def apply() -> None:
    _patch_resolve_block_sizes()
    _patch_initialize_kv_caches()
    _patch_engine_core_init()


# ---------------------------------------------------------------------------
# BLOCK_SIZES
# ---------------------------------------------------------------------------

def _patch_resolve_block_sizes() -> None:
    import vllm.v1.core.kv_cache_utils as ku

    _orig = ku.resolve_kv_cache_block_sizes

    def wrapped(kv_cache_config, vllm_config):
        scheduler_bs, hash_bs = _orig(kv_cache_config, vllm_config)
        try:
            groups = kv_cache_config.kv_cache_groups
            logger.info(
                "[BLOCK_SIZES] scheduler_block_size=%d hash_block_size=%d "
                "num_groups=%d group_block_sizes=%s "
                "cache_config.block_size=%d prefix_caching=%s connector=%s "
                "dcp=%s pcp=%s",
                scheduler_bs, hash_bs,
                len(groups),
                [g.kv_cache_spec.block_size for g in groups],
                vllm_config.cache_config.block_size,
                vllm_config.cache_config.enable_prefix_caching,
                vllm_config.kv_transfer_config is not None,
                vllm_config.parallel_config.decode_context_parallel_size,
                vllm_config.parallel_config.prefill_context_parallel_size,
            )
        except Exception as e:
            logger.warning("[BLOCK_SIZES] log failed: %s", e)
        return scheduler_bs, hash_bs

    ku.resolve_kv_cache_block_sizes = wrapped
    # 有些消费者以 `from ... import resolve_kv_cache_block_sizes` 提前 import，
    # 把它们的本地符号也一并替换，避免旧引用逃过 patch。
    for consumer in ("vllm.v1.engine.core",):
        try:
            import importlib
            mod = importlib.import_module(consumer)
            if hasattr(mod, "resolve_kv_cache_block_sizes"):
                setattr(mod, "resolve_kv_cache_block_sizes", wrapped)
        except Exception:
            pass
    logger.info("attn_trace: patched resolve_kv_cache_block_sizes")


# ---------------------------------------------------------------------------
# KV_GROUP
# ---------------------------------------------------------------------------

def _patch_initialize_kv_caches() -> None:
    from vllm.v1.engine.core import EngineCore

    _orig = EngineCore._initialize_kv_caches

    def wrapped(self, vllm_config):
        result = _orig(self, vllm_config)
        try:
            _log_kv_groups(result)
        except Exception as e:
            logger.warning("[KV_GROUP] log failed: %s", e)
        return result

    EngineCore._initialize_kv_caches = wrapped
    logger.info("attn_trace: patched EngineCore._initialize_kv_caches")


def _log_kv_groups(kv_cache_config) -> None:
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    logger.info(
        "[KV_GROUP] total_groups=%d total_blocks=%s",
        len(groups),
        getattr(kv_cache_config, "num_blocks", None),
    )
    for i, g in enumerate(groups):
        spec = g.kv_cache_spec
        layer_names = list(g.layer_names)
        logger.info(
            "[KV_GROUP] idx=%d spec=%s block_size=%s num_layers=%d "
            "num_kv_heads=%s head_size=%s dtype=%s "
            "sliding_window=%s attention_chunk_size=%s "
            "sample_layers=%s",
            i,
            type(spec).__name__,
            getattr(spec, "block_size", None),
            len(layer_names),
            getattr(spec, "num_kv_heads", None),
            getattr(spec, "head_size", None),
            getattr(spec, "dtype", None),
            getattr(spec, "sliding_window", None),
            getattr(spec, "attention_chunk_size", None),
            layer_names[:3] + (["..."] if len(layer_names) > 3 else []),
        )


# ---------------------------------------------------------------------------
# KV_STARTUP_SUMMARY
# ---------------------------------------------------------------------------

def _patch_engine_core_init() -> None:
    from vllm.v1.engine.core import EngineCore

    _orig = EngineCore.__init__

    def wrapped(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        try:
            _log_startup_summary(self)
        except Exception as e:
            logger.warning("[KV_STARTUP_SUMMARY] log failed: %s", e)

    EngineCore.__init__ = wrapped
    logger.info("attn_trace: patched EngineCore.__init__ (summary)")


def _log_startup_summary(engine_core) -> None:
    scheduler = getattr(engine_core, "scheduler", None)
    kv_cache_config = getattr(scheduler, "kv_cache_config", None) if scheduler else None
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []

    spec_hist = Counter(type(g.kv_cache_spec).__name__ for g in groups)
    total_layers = sum(len(g.layer_names) for g in groups)

    # Managers 位于 scheduler.kv_cache_manager.coordinator.single_type_managers
    mgr_hist = Counter()
    try:
        coord = scheduler.kv_cache_manager.coordinator  # type: ignore[union-attr]
        for m in getattr(coord, "single_type_managers", []):
            mgr_hist[type(m).__name__] += 1
    except Exception:
        pass

    vllm_config = getattr(engine_core, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)

    logger.info(
        "[KV_STARTUP_SUMMARY] num_groups=%d specs=%s managers=%s "
        "total_layers=%d num_blocks=%s hash_block_size=%s "
        "prefix_caching=%s has_kv_connector=%s "
        "data_parallel_size=%s tensor_parallel_size=%s "
        "pipeline_parallel_size=%s",
        len(groups),
        dict(spec_hist),
        dict(mgr_hist),
        total_layers,
        getattr(kv_cache_config, "num_blocks", None),
        getattr(scheduler, "hash_block_size", None),
        getattr(cache_config, "enable_prefix_caching", None),
        getattr(vllm_config, "kv_transfer_config", None) is not None,
        getattr(parallel_config, "data_parallel_size", None),
        getattr(parallel_config, "tensor_parallel_size", None),
        getattr(parallel_config, "pipeline_parallel_size", None),
    )
