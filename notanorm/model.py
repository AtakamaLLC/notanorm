"""Model definition support."""

from enum import Enum
from typing import NamedTuple, Tuple, Any, Set, Dict, Optional

__all__ = [
    "DbType",
    "DbCol",
    "DbIndex",
    "DbIndexField",
    "DbTable",
    "DbModel",
    "DbColCustomInfo",
]


class DbType(Enum):
    """Database types that should work on all providers."""

    TEXT = "text"
    BLOB = "blob"
    INTEGER = "int"
    FLOAT = "float"
    DOUBLE = "double"
    ANY = "any"
    BOOLEAN = "bool"


class ExplicitNone:
    def __eq__(self, other):
        return isinstance(other, ExplicitNone)

    def __str__(self):
        return str(None)

    def __repr__(self):
        return repr(None)


class DbColCustomInfo(NamedTuple):
    dialect: str  # dialect (usually matches uri_type)
    info: Any


class DbCol(NamedTuple):
    """Database column definition that should work on all providers."""

    name: str  # column name
    typ: DbType  # column type
    autoinc: bool = False  # autoincrement (only integer)
    size: int = 0  # for certain types, size is available
    notnull: bool = False  # not null
    fixed: bool = False  # not varchar
    default: Any = None  # has a default value
    custom: DbColCustomInfo = None  # dialect-specific column information

    def _as_tup(self):
        return (
            self.name.lower(),
            self.typ,
            self.autoinc,
            self.size,
            self.notnull,
            self.fixed,
            self.default,
            self.custom,
        )

    def __eq__(self, other):
        return self._as_tup() == other._as_tup()


class DbIndexField(NamedTuple):
    name: str
    prefix_len: Optional[int] = None

    def _as_tup(self) -> Tuple[str]:
        return self.name.lower(), self.prefix_len

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DbIndexField):  # pragma: no cover
            return False

        return self._as_tup() == other._as_tup()

    def __hash__(self) -> int:
        return hash(self._as_tup())


class DbIndex(NamedTuple):
    """Index definition."""

    fields: Tuple[DbIndexField, ...]
    unique: bool = False  # has a unique index?
    primary: bool = False  # is the primary key?
    name: Optional[str] = None  # only set when reading from the db, ignored on creation

    def _as_tup(self) -> Tuple[Tuple[DbIndexField, ...], bool, bool]:
        return (self.fields, self.unique, self.primary)

    def __eq__(self, other):
        return self._as_tup() == other._as_tup()

    def __hash__(self):
        return hash(self._as_tup())

    @classmethod
    def from_fields(cls, fields, unique=False, primary=False):
        assert type(fields) in (tuple, list)
        return DbIndex(
            tuple(
                DbIndexField(*tuple((ent,) if type(ent) is str else ent))
                for ent in fields
            ),
            unique,
            primary,
        )


class DbTable(NamedTuple):
    """Table definition."""

    columns: Tuple[DbCol, ...]
    indexes: Set[DbIndex] = set()


class DbModel(Dict[str, DbTable]):
    """Container of table definitions."""

    def _as_cmp(self):
        return {k.lower(): v for k, v in self.items()}

    def __eq__(self, other):
        return self._as_cmp() == other._as_cmp()
