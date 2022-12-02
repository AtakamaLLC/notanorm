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
from .base import DbBase, Op
from .model import DbType, DbCol, DbTable, DbModel, DbIndex, DbIndexField
from . import errors
from .connparse import open_db

try:
    # if you want this, install sqlglot
    if tuple(sys.version_info[:2]) > (3, 6):
        from .ddl_helper import model_from_ddl
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "SqliteDb",
    "MySqlDb",
    "DbRow",
    "DbBase",
    "DbType",
    "DbCol",
    "DbTable",
    "DbModel",
    "DbIndex",
    "DbIndexField",
    "errors",
    "Op",
    "open_db",
    "model_from_ddl",
]
