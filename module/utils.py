import sys
import warnings
import socket
from typing import Optional, Any, Dict
import os
import logging


_log_extra_fields: Dict[str, Any] = {}


def log_extra_field(field_name: str, field_value: Any) -> None:
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value


def is_float_printable(x):
    try:
        tmp = f"{x:0.2f}"
        return True
    except (ValueError, TypeError):
        return False


def pop_metadata(data):
    meta = {k: data.pop(k) for k in list(data) if k.startswith("metadata")}
    return data, meta


def setup_logging() -> None:
    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        for field_name, field_value in _log_extra_fields.items():
            setattr(record, field_name, field_value)
        return record

    logging.setLogRecordFactory(log_record_factory)

    handler: logging.Handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(levelname)-.1s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logging.basicConfig(handlers=[handler], level=logging.INFO)

    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


