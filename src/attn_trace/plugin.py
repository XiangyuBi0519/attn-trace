import os

from .logutil import get_logger

_APPLIED = False


def register() -> None:
    """vLLM 会通过 entry-point 组 `vllm.general_plugins` 找到并调用这个函数。

    也可以从任意启动脚本里手动 import 调用；重复调用是幂等的。
    """
    global _APPLIED
    logger = get_logger()

    if _APPLIED:
        logger.debug("attn_trace already registered; skip")
        return

    if os.environ.get("ATTN_TRACE_DISABLE", "").lower() in ("1", "true", "yes"):
        logger.info("attn_trace disabled via ATTN_TRACE_DISABLE")
        return

    logger.info(
        "attn_trace: register() start (pid=%s, ATTN_TRACE_LOG_LEVEL=%s)",
        os.getpid(),
        os.environ.get("ATTN_TRACE_LOG_LEVEL", "INFO"),
    )

    from .patches import apply_all

    apply_all()
    _APPLIED = True
    logger.info("attn_trace: register() done")
