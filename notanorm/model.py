"""Model definition support."""

from enum import Enum
from typing import NamedTuple, Tuple, Any, Dict

__all__ = ["DbType", "DbCol", "DbIndex", "DbTable", "DbModel"]


class DbType(Enum):
    """Database types that should work on all providers."""
    TEXT = "text"
    BLOB = "blob"
    INTEGER = "int"
    FLOAT = "float"
    DOUBLE = "double"
    ANY = "any"


class DbCol(NamedTuple):
    """Database column definition that should work on all providers."""
    name: str                       # column name
    typ: DbType                     # column type
    autoinc: bool = False           # autoincrement (only integer)
    size: int = 0                   # for certain types, size is available
    notnull: bool = False           # not null
    fixed: bool = False             # not varchar
    default: Any = None             # has a default value


class DbIndex(NamedTuple):
    """Index definition."""
    fields: Tuple[str, ...]         # list of fields in the index
    unique: bool = False            # has a unique index?
    primary: bool = False           # is the primary key?


class DbTable(NamedTuple):
    """Table definition."""
    columns: Tuple[DbCol, ...]
    indexes: Tuple[DbIndex, ...] = ()


class DbModel(dict):
    """Container of table definitions."""
