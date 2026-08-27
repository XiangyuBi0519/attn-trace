"""Patches: `EngineCore.__init__` (+ EngineCoreProc / DPEngineCoreProc),
`EngineCore._initialize_kv_caches`, and `resolve_kv_cache_block_sizes`.

Emits:
- `[BLOCK_SIZES]` right after vLLM resolves scheduler_block_size /
  hash_block_size. THIS IS THE ONE that decides whether your plugin's
  `Request.from_engine_core_request` re-hashed block_hashes align with
  what the pool actually stores.
- `[KV_GROUP]` for every KV cache group finalized inside
  `_initialize_kv_caches`.
- `[KV_STARTUP_SUMMARY]` at the end of `EngineCore.__init__` (also
  covers EngineCoreProc / DPEngineCoreProc which own their own __init__
  but call `super().__init__()`; the wrapper uses try/finally so the
  summary is still emitted even if downstream init raises).
"""

import importlib
from collections import Counter

from ..logutil import get_logger

logger = get_logger()


def apply() -> None:
    _patch_resolve_block_sizes()
    _patch_initialize_kv_caches()
    _patch_engine_core_init_chain()


# ---------------------------------------------------------------------------
# BLOCK_SIZES
# ---------------------------------------------------------------------------

def _patch_resolve_block_sizes() -> None:
    import sys
    import vllm.v1.core.kv_cache_utils as ku

    orig = ku.resolve_kv_cache_block_sizes

    def wrapped(kv_cache_config, vllm_config):
        scheduler_bs, hash_bs = orig(kv_cache_config, vllm_config)
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

    # 把 `from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes`
    # 拿到的本地符号也一并替换
    for name, mod in list(sys.modules.items()):
        if not name.startswith("vllm."):
            continue
        if mod is None or mod is ku:
            continue
        if getattr(mod, "resolve_kv_cache_block_sizes", None) is orig:
            mod.resolve_kv_cache_block_sizes = wrapped

    logger.info("attn_trace: patched resolve_kv_cache_block_sizes")


# ---------------------------------------------------------------------------
# KV_GROUP
# ---------------------------------------------------------------------------

def _patch_initialize_kv_caches() -> None:
    from vllm.v1.engine.core import EngineCore

    orig = EngineCore._initialize_kv_caches

    def wrapped(self, vllm_config):
        result = orig(self, vllm_config)
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
            "[KV_GROUP] idx=%d spec=%s.%s block_size=%s num_layers=%d "
            "num_kv_heads=%s head_size=%s dtype=%s "
            "sliding_window=%s attention_chunk_size=%s "
            "sample_layers=%s",
            i,
            type(spec).__module__, type(spec).__name__,
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

def _patch_engine_core_init_chain() -> None:
    """
    Wrap __init__ on EngineCore + EngineCoreProc + DPEngineCoreProc.

    Even though the subclasses call super().__init__() (so wrapping only the
    base class would in principle work), wrap all three so:
      1) 如果某个子类 __init__ 在 super().__init__() 之后又做了更多初始化，
         我们打的 summary 反映的是 SUB 类 __init__ 结束后的完整状态；
      2) 用 try/finally 保证即使后段初始化抛异常，summary 也会被 emit 一次
         （便于定位启动失败）。
    """
    from vllm.v1.engine.core import EngineCore

    patched = []
    for mod_path, cls_name in [
        ("vllm.v1.engine.core", "EngineCore"),
        ("vllm.v1.engine.core", "EngineCoreProc"),
        ("vllm.v1.engine.core", "DPEngineCoreProc"),
    ]:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        if "__init__" not in cls.__dict__:
            continue
        orig = cls.__dict__["__init__"]
        if getattr(orig, "_attn_trace_init_wrapped", False):
            continue

        def _make_wrapper(orig_fn, tag):
            def wrapper(self, *args, **kwargs):
                exc = None
                try:
                    orig_fn(self, *args, **kwargs)
                except BaseException as e:
                    exc = e
                    raise
                finally:
                    # 每个 init 链只 emit 一次 summary，由最外层类（真正被
                    # 实例化的那个）负责。用一个实例属性做标记；如果整条链上
                    # 没人（比如没 patch 到最外层类）emit，异常场景下由最深
                    # 层兜底 emit 一次，避免完全没输出。
                    should_emit = False
                    if type(self).__name__ == tag:
                        should_emit = True
                    elif exc is not None and not getattr(
                        self, "_attn_trace_summary_emitted", False
                    ):
                        should_emit = True
                    if should_emit:
                        try:
                            _log_startup_summary(self, failed=exc is not None)
                            self._attn_trace_summary_emitted = True
                        except Exception as e2:
                            logger.warning("[KV_STARTUP_SUMMARY] log failed: %s", e2)
            wrapper._attn_trace_init_wrapped = True  # type: ignore[attr-defined]
            return wrapper

        setattr(cls, "__init__", _make_wrapper(orig, cls_name))
        patched.append(cls_name)

    logger.info(
        "attn_trace: patched EngineCore.__init__ chain (%s)",
        ", ".join(patched) if patched else "nothing patched",
    )


def _log_startup_summary(engine_core, failed: bool = False) -> None:
    scheduler = getattr(engine_core, "scheduler", None)
    kv_cache_config = getattr(scheduler, "kv_cache_config", None) if scheduler else None
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []

    spec_hist = Counter(type(g.kv_cache_spec).__name__ for g in groups)
    total_layers = sum(len(g.layer_names) for g in groups)

    mgr_hist: Counter = Counter()
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
        "[KV_STARTUP_SUMMARY] engine_class=%s failed_before_return=%s "
        "num_groups=%d specs=%s managers=%s "
        "total_layers=%d num_blocks=%s hash_block_size=%s "
        "prefix_caching=%s has_kv_connector=%s "
        "data_parallel_size=%s tensor_parallel_size=%s "
        "pipeline_parallel_size=%s",
        type(engine_core).__name__,
        failed,
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
