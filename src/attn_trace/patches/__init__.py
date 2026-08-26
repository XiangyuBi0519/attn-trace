from ..logutil import get_logger

logger = get_logger()


def apply_all() -> None:
    """依次应用所有 patch。任何一个失败都只 warning，不中断其它 patch，也不影响启动。"""
    from . import selector, engine_init, manager_init, layer_spec

    for mod in (selector, engine_init, manager_init, layer_spec):
        _safe_apply(mod)

    # Ascend 侧可选。没装 vllm_ascend 就静默跳过；装了但函数签名对不上则 warning。
    try:
        from . import platform_ascend  # noqa: F401
    except ImportError:
        logger.debug("attn_trace: vllm_ascend not present, skip platform_ascend patch")
    else:
        _safe_apply(platform_ascend)


def _safe_apply(mod) -> None:
    name = getattr(mod, "__name__", str(mod))
    try:
        mod.apply()
    except ImportError as e:
        logger.info("attn_trace: %s skipped (ImportError: %s)", name, e)
    except Exception as e:
        logger.warning("attn_trace: %s failed to apply: %s", name, e)
