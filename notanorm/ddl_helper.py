from collections import defaultdict
from typing import Tuple, Dict, List, Any, Type

from sqlglot import Expression, parse, exp

from .model import DbType, DbCol, DbIndex, DbTable, DbModel, ExplicitNone, DbIndexField
from .sqlite import SqliteDb
from . import errors as err

import logging

log = logging.getLogger(__name__)


# some support for different sqlglot versions
has_varb = getattr(exp.DataType.Type, "VARBINARY", None)


class DDLHelper:
    # map of sqlglot expression types to internal model types
    TYPE_MAP = {
        exp.DataType.Type.INT: DbType.INTEGER,
        exp.DataType.Type.TINYINT: DbType.INTEGER,
        exp.DataType.Type.BIGINT: DbType.INTEGER,
        exp.DataType.Type.BOOLEAN: DbType.BOOLEAN,
        exp.DataType.Type.BINARY: DbType.BLOB,
        exp.DataType.Type.VARCHAR: DbType.TEXT,
        exp.DataType.Type.CHAR: DbType.TEXT,
        exp.DataType.Type.TEXT: DbType.TEXT,
        exp.DataType.Type.VARIANT: DbType.ANY,
        exp.DataType.Type.DECIMAL: DbType.DOUBLE,
        exp.DataType.Type.DOUBLE: DbType.DOUBLE,
        exp.DataType.Type.FLOAT: DbType.FLOAT,
    }

    if has_varb:
        TYPE_MAP.update({
            exp.DataType.Type.VARBINARY: DbType.BLOB,
        })

    SIZE_MAP = {
        exp.DataType.Type.TINYINT: 1,
        exp.DataType.Type.SMALLINT: 2,
        exp.DataType.Type.INT: 4,
        exp.DataType.Type.BIGINT: 8,
    }

    FIXED_MAP = {
        exp.DataType.Type.CHAR,
    }

    def __init__(self, ddl, *dialects):
        if not dialects:
            # guess dialect
            dialects = ("mysql", "sqlite")

        first_x = None

        self.__sqlglot = None
        self.__model = None

        for dialect in dialects:
            try:
                if dialect == "sqlite":
                    self.__model_from_sqlite(ddl)
                else:
                    self.__model_from_sqlglot(ddl, dialect)
                return
            except Exception as ex:
                first_x = first_x or ex

        # earlier (more picky) dialects give better errors
        if first_x:
            raise first_x

    def __model_from_sqlglot(self, ddl, dialect):
        # sqlglot generic parser
        tmp_ddl = ddl
        if dialect == "mysql" and not has_varb:  # pragma: no cover
            # sqlglot 9 doesn't support varbinary
            tmp_ddl = ddl.replace("varbinary", "binary")
            tmp_ddl = ddl.replace("VARBINARY", "BINARY")
        res = parse(tmp_ddl, read=dialect)
        self.__sqlglot = res
        self.dialect = dialect

    def __model_from_sqlite(self, ddl):
        # sqlite memory parser
        ddl = ddl.replace("auto_increment", "autoincrement")
        tmp_db = SqliteDb(":memory:")
        tmp_db.executescript(ddl)
        self.__model = tmp_db.model()
        self.dialect = "sqlite"

    def __columns(self, ent) -> Tuple[Tuple[DbCol, ...], List[DbIndex]]:
        """Get a tuple of DbCols from a parsed statement

        Argument is a sqlglot parsed grammar of a CREATE TABLE statement.

        If a primary key is specified, return it too.
        """
        cols: List[DbCol] = []
        idxs = []
        for col in ent.find_all(exp.Anonymous):
            if col.name.lower() == "primary key":
                primary_list = [ent.name for ent in col.find_all(exp.Column)]
                idxs.append(DbIndex(fields=tuple(DbIndexField(n, prefix_len=None) for n in primary_list), primary=True, unique=False))

        for col in ent.find_all(exp.ColumnDef):
            dbcol, is_prim, is_uniq = self.__info_to_model(col)
            if is_prim:
                idxs.append(DbIndex(fields=(DbIndexField(col.name, prefix_len=None),), primary=True, unique=False))
            elif is_uniq:
                idxs.append(DbIndex(fields=(DbIndexField(col.name, prefix_len=None),), primary=False, unique=True))
            cols.append(dbcol)
        return tuple(cols), idxs

    @staticmethod
    def __info_to_index(index: Expression, dialect: str) -> Tuple[DbIndex, str]:
        """Get a DbIndex and a table name, given a sqlglot parsed index"""
        primary: exp.PrimaryKeyColumnConstraint = index.find(exp.PrimaryKeyColumnConstraint)
        unique = index.args.get("unique")
        tab = index.args["this"].args["table"]
        cols = index.args["this"].args["columns"]
        field_info: List[Dict[str, Any]] = []

        args: List[Expression] = cols.args["expressions"] if isinstance(cols, exp.Tuple) else [cols]
        args = [a.this if isinstance(a, exp.Paren) else a for a in args]

        for ent in args:
            allowed_types: Tuple[Type[Expression], ...]
            if dialect != "mysql":
                # For MySQL, a parenthesized arg here indicates an expression
                # index. For other dialects, it's just a normal way to specify
                # a column name.
                while isinstance(ent, exp.Paren):
                    ent = ent.this

                allowed_types = (exp.Column,)
            else:
                # MySQL prefix indices (e.g. CREATE INDEX ... ON tbl(col(10)))
                # show up as anonymous functions.
                allowed_types = (exp.Column, exp.Anonymous)

            if not isinstance(ent, allowed_types):
                raise err.SchemaError(f"Unsupported type in index definition: {type(ent)}({ent})")

            if dialect == "mysql" and isinstance(ent, exp.Anonymous):
                exps = ent.args["expressions"]

                if len(exps) != 1:
                    raise err.SchemaError(f"Invalid prefix index definition: {ent}")

                try:
                    prefix_len = int(exps[0].name)
                except ValueError as e:
                    raise err.SchemaError(f"Invalid prefix index length: {exps[0].name}") from e

                field_info.append({"name": ent.name, "prefix_len": prefix_len})
            else:
                field_info.append({"name": ent.name, "prefix_len": None})

        return (
            DbIndex(
                fields=tuple(DbIndexField(**f) for f in field_info), primary=bool(primary), unique=bool(unique)
            ),
            tab.name,
        )

    @classmethod
    def __info_to_model(cls, info) -> Tuple[DbCol, bool, bool]:
        """Turn a sqlglot parsed ColumnDef into a model entry."""
        typ = info.find(exp.DataType)
        fixed = typ.this in cls.FIXED_MAP
        size = cls.SIZE_MAP.get(typ.this, 0)
        typ = cls.TYPE_MAP[typ.this]
        notnull = info.find(exp.NotNullColumnConstraint)
        autoinc = info.find(exp.AutoIncrementColumnConstraint)
        is_primary = info.find(exp.PrimaryKeyColumnConstraint)
        default = info.find(exp.DefaultColumnConstraint)
        is_unique = info.find(exp.UniqueColumnConstraint)

        # sqlglot has no dedicated or well-known type for the 32 in VARCHAR(32)
        # so this is from the grammar of types:  VARCHAR(32) results in a "type.kind.args.expressions" tuple
        expr = info.args["kind"].args.get("expressions")
        if expr:
            size = int(expr[0].this)

        if default:
            lit = default.find(exp.Literal)
            bool_val = default.find(exp.Boolean)
            if default.find(exp.Null):
                # None means no default, so we have this silly thing
                default = ExplicitNone()
            elif bool_val:
                # None means no default, so we have this silly thing
                default = bool_val.this
            elif not lit:
                default = str(default.this)
            elif lit.is_string:
                default = lit.this
            else:
                default = str(lit)
        return (
            DbCol(
                name=info.name,
                typ=typ,
                notnull=bool(notnull),
                default=default,
                autoinc=bool(autoinc),
                size=size,
                fixed=fixed,
            ),
            is_primary,
            is_unique,
        )

    def model(self):
        """Get generic db model: dict of tables, each a dict of rows, each with type, unique, autoinc, primary."""
        if self.__model:
            return self.__model

        model = DbModel()
        tabs: Dict[str, Tuple[DbCol, ...]] = {}
        indxs = defaultdict(lambda: [])
        for ent in self.__sqlglot:
            tab = ent.find(exp.Table)
            assert tab, f"unknonwn ddl entry {ent}"
            idx = ent.find(exp.Index)
            if not idx:
                tabs[tab.name], idxs = self.__columns(ent)
                indxs[tab.name] += idxs
            else:
                idx, tab_name = self.__info_to_index(ent, self.dialect)
                indxs[tab_name].append(idx)

        for tab in tabs:
            dbcols: Tuple[DbCol, ...] = tabs[tab]
            model[tab] = DbTable(dbcols, set(indxs[tab]))

        self.__model = model

        return model


def model_from_ddl(ddl: str, *dialects: str) -> DbModel:
    """Convert indexes and create statements to internal model, without needing a database connection."""
    return DDLHelper(ddl, *dialects).model()
