"""Patch: `SingleTypeKVCacheManager` subclass `__init__`s + optional
`get_manager_for_kv_cache_spec` factory.

Emits `[KV_MGR_INIT]` every time a per-group single-type manager is created.
Uses two layers so all Manager instances are captured regardless of which
factory / code path builds them:

- Base class + subclass scan (main path): walks
  `SingleTypeKVCacheManager` subclass tree and wraps every subclass's
  `__init__`. Uses `__init_subclass__` hook to catch subclasses defined
  later (Ascend hybrid managers are declared inside vllm_ascend model
  files that load well after the plugin registers). This is the same
  pattern layer_spec / attn_impl use for AttentionLayerBase.

- `get_manager_for_kv_cache_spec` factory patch (legacy path): kept as a
  belt-and-suspenders. If someone actually calls the factory the extra
  info (max_admission_blocks, max_num_batched_tokens, max_model_len)
  gets logged too.
"""

import sys

from ..logutil import get_logger
from . import _scanner

logger = get_logger()

_MARKER = "_attn_trace_mgr_init_wrapped"


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

    # ---- 主路径：包 SingleTypeKVCacheManager 及其所有子类的 __init__ ----
    _patch_manager_subclass_inits()


def _patch_manager_subclass_inits() -> None:
    """在 SingleTypeKVCacheManager 上装子类 __init__ 钩子。

    Ascend 侧 hybrid coordinator 有可能不走 get_manager_for_kv_cache_spec
    工厂，直接实例化 Manager 子类。只要 __init__ 被调，就 emit 一条
    [KV_MGR_INIT] （instance 级别，不依赖任何工厂调用）。
    """
    try:
        from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    except ImportError as e:
        logger.warning("attn_trace.manager_init: cannot import base class: %s", e)
        return

    # 1) 现存所有子孙类：拿一遍
    existing = _scanner.walk_attention_subclasses(base_cls=SingleTypeKVCacheManager)
    patched_now = [c for c in existing if _wrap_manager_init(c)]
    if patched_now:
        logger.info(
            "attn_trace.manager_init: wrapped __init__ on %d existing Manager subclasses: %s",
            len(patched_now), _scanner.unique_class_paths(patched_now),
        )
    else:
        logger.info(
            "attn_trace.manager_init: no existing Manager subclass owned __init__ at "
            "register time (subclasses may load later, hook will catch them)"
        )

    # 2) 装 __init_subclass__ 钩子，接管后续动态加载的子类（如 Ascend 变体）
    _scanner.install_subclass_hook(_wrap_and_log_manager, base_cls=SingleTypeKVCacheManager)


def _wrap_and_log_manager(cls) -> None:
    if _wrap_manager_init(cls):
        logger.info(
            "attn_trace.manager_init: late-wrapped __init__ on %s.%s",
            cls.__module__, cls.__name__,
        )


def _wrap_manager_init(cls) -> bool:
    return _scanner.wrap_method_if_defined(
        cls, "__init__", _make_manager_wrapper, _MARKER,
    )


def _make_manager_wrapper(orig_fn, owner_cls):
    def wrapper(self, *args, **kwargs):
        orig_fn(self, *args, **kwargs)
        try:
            spec = getattr(self, "kv_cache_spec", None)
            logger.info(
                "[KV_MGR_INIT] manager=%s.%s spec=%s.%s "
                "block_size=%s kv_group_id=%s enable_caching=%s "
                "dcp_world_size=%s pcp_world_size=%s "
                "max_admission_blocks=%s",
                owner_cls.__module__, owner_cls.__name__,
                type(spec).__module__ if spec else "?",
                type(spec).__name__ if spec else "?",
                getattr(spec, "block_size", None) if spec else None,
                getattr(self, "kv_cache_group_id", None),
                getattr(self, "enable_caching", None),
                getattr(self, "dcp_world_size", None),
                getattr(self, "pcp_world_size", None),
                getattr(self, "max_admission_blocks_per_request", None),
            )
        except Exception as e:
            logger.warning("[KV_MGR_INIT] log failed for %s: %s", owner_cls.__name__, e)
    return wrapper
