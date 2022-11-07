"""Simple db accessor system: not an orm."""

from .base import DbRow

try:
    # if you want this, install mysqlclient or pymysql
    from .mysql import MySqlDb
except ImportError:
    pass

from .sqlite import SqliteDb
from .base import DbBase, Op
from .model import DbType, DbCol, DbTable, DbModel, DbIndex
from . import errors
from .connparse import open_db

try:
    # if you want this, install sqlglot
    from .ddl_helper import model_from_ddl
except (ImportError, ModuleNotFoundError):
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
    "errors",
    "Op",
    "open_db",
    "model_from_ddl",
]
