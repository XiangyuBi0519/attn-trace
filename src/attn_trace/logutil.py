import logging
import os

_LOGGER_NAME = "attn_trace"
_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if getattr(logger, "_attn_trace_configured", False):
        return logger

    level_name = os.environ.get("ATTN_TRACE_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    # 让日志也能出现在 vllm 的 root logger 里
    logger.propagate = True

    # 保证至少有一个 stderr handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(sh)

    log_file = os.environ.get("ATTN_TRACE_LOG_FILE")
    if log_file:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(fh)
        except OSError as e:
            logger.warning("attn_trace: cannot open ATTN_TRACE_LOG_FILE=%r: %s", log_file, e)

    logger._attn_trace_configured = True  # type: ignore[attr-defined]
    return logger
