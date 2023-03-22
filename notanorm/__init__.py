"""Simple db accessor system: not an orm."""

import sys
from .base import DbRow

try:
    # if you want this, install mysqlclient or pymysql
    from .mysql import MySqlDb
except ImportError:
    pass
except Exception:  # pragma: no cover
    # apparently this can fail for weird reasons, not just import errors
    import logging

    logging.exception("failed mysql import for unknown reason")
    pass

from .sqlite import SqliteDb
from .base import DbBase, Op, And, Or, ReconnectionArgs
from .model import (
    DbType,
    DbCol,
    DbTable,
    DbModel,
    DbIndex,
    DbIndexField,
    DbColCustomInfo,
)
from . import errors
from .connparse import open_db

try:
    # if you want this stuff, install sqlglot
    if tuple(sys.version_info[:2]) > (3, 6):
        from .ddl_helper import model_from_ddl
        from .jsondb import JsonDb
except ImportError:  # pragma: no cover
    pass


__all__ = [
    "SqliteDb",
    "JsonDb",
    "MySqlDb",
    "DbRow",
    "DbBase",
    "DbType",
    "DbCol",
    "DbColCustomInfo",
    "DbTable",
    "DbModel",
    "DbIndex",
    "DbIndexField",
    "errors",
    "Op",
    "And",
    "Or",
    "open_db",
    "model_from_ddl",
    "ReconnectionArgs",
]
