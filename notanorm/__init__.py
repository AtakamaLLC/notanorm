"""Simple db accessor system: not an orm."""

from .base import DbRow

try:
    from .mysql import MySqlDb
except ImportError:
    pass
from .sqlite import SqliteDb
from .base import DbBase, Op
from .model import DbModel
from . import errors
from .connparse import open_db

__all__ = [
    "SqliteDb",
    "MySqlDb",
    "DbRow",
    "DbBase",
    "DbModel",
    "errors",
    "Op",
    "open_db",
]
