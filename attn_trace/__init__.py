"""
attn_trace — vLLM 0.23 attention/KV-cache manager path tracer

只读探针。不修改任何执行逻辑，只加日志。用于回答：
1. 当前模型创建了哪些 SingleTypeKVCacheManager 子类实例？
2. 每个 KVCacheGroup 覆盖哪些层？block_size 多少？spec 类型是什么？
3. 运行时 remove_skipped_blocks 会不会被真正触发（get_num_skipped_tokens > 0）？
4. free_blocks(prepend=True) 具体在什么时机、被哪个 manager 触发？

使用方式：作为 vLLM plugin 自动加载（推荐），或手动 import 后调用 enable()。
详见 README.md。
"""
import inspect
import logging
import os
import sys

_LOG_NAME = "attn_trace"
logger = logging.getLogger(_LOG_NAME)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s pid=%(process)d %(message)s"
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_enabled = False
_first_free_prepend_seen: set[str] = set()
_first_skipped_seen: set[str] = set()


def _short_layer_names(names, limit=4):
    if not names:
        return "[]"
    if len(names) <= limit:
        return str(list(names))
    return f"[{names[0]}, ..., {names[-1]}] (total={len(names)})"


def enable():
    """挂钩 vLLM 的四个关键点，打印一次性构造信息 + 首次热路径命中。

    幂等；多次调用只挂一次。子进程（multiproc executor）需要各自调用一次——
    走 vLLM plugin 入口时自动满足；手动使用时看 README。
    """
    global _enabled
    if _enabled:
        return
    _enabled = True

    try:
        from vllm.v1.core.single_type_kv_cache_manager import (
            SingleTypeKVCacheManager,
        )
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    except ImportError as e:
        logger.error("attn_trace: cannot import vLLM internals (%s); is vLLM installed?", e)
        return

    orig_st_init = SingleTypeKVCacheManager.__init__
    orig_free_blocks = BlockPool.free_blocks
    orig_remove_skipped = SingleTypeKVCacheManager.remove_skipped_blocks
    orig_coord_init = KVCacheCoordinator.__init__

    def logged_st_init(self, *args, **kwargs):
        orig_st_init(self, *args, **kwargs)
        spec = getattr(self, "kv_cache_spec", None)
        gid = getattr(self, "kv_cache_group_id", None)
        block_size = getattr(self, "block_size", None)
        # spec 可能带 sliding_window / attention_chunk_size 等关键字段
        spec_attrs = {}
        if spec is not None:
            for k in ("sliding_window", "attention_chunk_size", "num_kv_heads",
                      "head_size", "dtype", "use_mla"):
                if hasattr(spec, k):
                    spec_attrs[k] = getattr(spec, k)
        logger.info(
            "[MGR_INIT] cls=%s group_id=%s block_size=%s spec_type=%s spec_attrs=%s",
            type(self).__name__, gid, block_size,
            type(spec).__name__ if spec else None, spec_attrs,
        )

    def logged_coord_init(self, *args, **kwargs):
        orig_coord_init(self, *args, **kwargs)
        managers = getattr(self, "single_type_managers", None)
        if managers is None:
            logger.info("[COORD_INIT] cls=%s (no single_type_managers attr)", type(self).__name__)
            return
        logger.info(
            "[COORD_INIT] cls=%s num_groups=%d", type(self).__name__, len(managers)
        )
        for i, m in enumerate(managers):
            spec = getattr(m, "kv_cache_spec", None)
            layer_names = getattr(spec, "layer_names", None) if spec else None
            logger.info(
                "[COORD_INIT]   group[%d] manager=%s spec=%s layers=%s",
                i, type(m).__name__, type(spec).__name__ if spec else None,
                _short_layer_names(layer_names),
            )

    def logged_free_blocks(self, ordered_blocks, prepend=False):
        if prepend:
            # 只打一次 stack，避免刷屏；每个「调用来源」记一次
            caller = inspect.stack()[1]
            key = f"{caller.filename}:{caller.function}:{caller.lineno}"
            if key not in _first_free_prepend_seen:
                _first_free_prepend_seen.add(key)
                blocks_list = list(ordered_blocks)
                logger.info(
                    "[PREPEND_FIRST] free_blocks(prepend=True) called from %s (n=%d) "
                    "— subsequent calls from same site will be silent",
                    key, len(blocks_list),
                )
                return orig_free_blocks(self, blocks_list, prepend=prepend)
        return orig_free_blocks(self, ordered_blocks, prepend=prepend)

    def logged_remove_skipped(self, request_id, total_computed_tokens):
        n = 0
        try:
            n = self.get_num_skipped_tokens(total_computed_tokens)
        except Exception:
            pass
        key = f"{type(self).__name__}"
        if n > 0 and key not in _first_skipped_seen:
            _first_skipped_seen.add(key)
            logger.info(
                "[SKIP_FIRST] %s.get_num_skipped_tokens()=%d (total_computed=%d) — "
                "this manager WILL trigger prepend_n path; subsequent hits silent",
                type(self).__name__, n, total_computed_tokens,
            )
        return orig_remove_skipped(self, request_id, total_computed_tokens)

    SingleTypeKVCacheManager.__init__ = logged_st_init
    SingleTypeKVCacheManager.remove_skipped_blocks = logged_remove_skipped
    BlockPool.free_blocks = logged_free_blocks
    KVCacheCoordinator.__init__ = logged_coord_init

    logger.info(
        "[PROBE_ENABLED] attn_trace hooks installed on vLLM in pid=%d",
        os.getpid(),
    )


def register():
    """vLLM plugin entry point. vLLM 会在主进程 + 每个 EngineCore 子进程各调一次。"""
    enable()
