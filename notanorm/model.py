"""Model definition support."""

from enum import Enum
from typing import NamedTuple, Tuple, Any, Set

__all__ = ["DbType", "DbCol", "DbIndex", "DbTable", "DbModel"]


class DbType(Enum):
    """Database types that should work on all providers."""
    TEXT = "text"
    BLOB = "blob"
    INTEGER = "int"
    FLOAT = "float"
    DOUBLE = "double"
    ANY = "any"
    BOOLEAN = "bool"


class DbCol(NamedTuple):
    """Database column definition that should work on all providers."""
    name: str                       # column name
    typ: DbType                     # column type
    autoinc: bool = False           # autoincrement (only integer)
    size: int = 0                   # for certain types, size is available
    notnull: bool = False           # not null
    fixed: bool = False             # not varchar
    default: Any = None             # has a default value

    def _as_tup(self):
        return (self.name.lower(), self.typ, self.autoinc, self.size, self.notnull, self.fixed, self.default)

    def __eq__(self, other):
        return self._as_tup() == other._as_tup()


class DbIndex(NamedTuple):
    """Index definition."""
    fields: Tuple[str, ...]         # list of fields in the index
    unique: bool = False            # has a unique index?
    primary: bool = False           # is the primary key?

    def _as_tup(self):
        return (tuple(f.lower() for f in self.fields), self.unique, self.primary)

    def __eq__(self, other):
        return self._as_tup() == other._as_tup()

    def __hash__(self):
        return hash(self._as_tup())


class DbTable(NamedTuple):
    """Table definition."""
    columns: Tuple[DbCol, ...]
    indexes: Set[DbIndex] = ()


class DbModel(dict):
    """Container of table definitions."""

    def _as_cmp(self):
        return {k.lower(): v for k, v in self.items()}

    def __eq__(self, other):
        return self._as_cmp() == other._as_cmp()
