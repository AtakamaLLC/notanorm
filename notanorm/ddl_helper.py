from collections import defaultdict
from typing import Tuple, Dict, List

from sqlglot import parse, exp

from .model import DbType, DbCol, DbIndex, DbTable, DbModel, ExplicitNone
from .sqlite import SqliteDb

import logging

log = logging.getLogger(__name__)


class DDLHelper:
    # map of sqlglot expression types to internal model types
    TYPE_MAP = {
        exp.DataType.Type.INT: DbType.INTEGER,
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

    FIXED_MAP = {
        exp.DataType.Type.CHAR,
        #  todo: add support for varbinary vs binary in sqlglot
        #        exp.DataType.Type.BINARY
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
        if dialect == "mysql":
            # bug sqlglot doesn't support varbinary
            tmp_ddl = ddl.replace("varbinary", "binary")
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

    def __columns(self, ent) -> Tuple[Tuple[DbCol, ...], DbIndex]:
        """Get a tuple of DbCols from a parsed statement

        Argument is a sqlglot parsed grammar of a CREATE TABLE statement.

        If a primary key is specified, return it too.
        """
        cols: List[DbCol] = []
        primary = None
        for col in ent.find_all(exp.Anonymous):
            if col.name.lower() == "primary key":
                primary_list = [ent.name for ent in col.find_all(exp.Column)]
                primary = DbIndex(fields=tuple(primary_list), primary=True, unique=False)
        for col in ent.find_all(exp.ColumnDef):
            dbcol, is_prim = self.__info_to_model(col, primary)
            if is_prim:
                primary = DbIndex(fields=(col.name,), primary=True, unique=False)
            cols.append(dbcol)
        return tuple(cols), primary

    @staticmethod
    def __info_to_index(index) -> Tuple[DbIndex, str]:
        """Get a DbIndex and a table name, given a sqlglot parsed index"""
        primary = index.find(exp.PrimaryKeyColumnConstraint)
        unique = index.args.get("unique")
        tab = index.args["this"].args["table"]
        cols = index.args["this"].args["columns"]
        field_names = [ent.name for ent in cols.find_all(exp.Column)]
        return (
            DbIndex(
                fields=tuple(field_names), primary=bool(primary), unique=bool(unique)
            ),
            tab.name,
        )

    @classmethod
    def __info_to_model(cls, info, primary) -> Tuple[DbCol, bool]:
        """Turn a sqlglot parsed ColumnDef into a model entry."""
        typ = info.find(exp.DataType)
        fixed = typ.this in cls.FIXED_MAP
        typ = cls.TYPE_MAP[typ.this]
        notnull = info.find(exp.NotNullColumnConstraint)
        autoinc = info.find(exp.AutoIncrementColumnConstraint)
        is_primary = info.find(exp.PrimaryKeyColumnConstraint)
        default = info.find(exp.DefaultColumnConstraint)

        # sqlglot has no dedicated or well-known type for the 32 in VARCHAR(32)
        # so this is from the grammar of types:  VARCHAR(32) results in a "type.kind.args.expressions" tuple
        expr = info.args["kind"].args.get("expressions")
        if expr:
            size = int(expr[0].this)
        else:
            size = 0

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
                tabs[tab.name], primary = self.__columns(ent)
                if primary:
                    indxs[tab.name].append(primary)
            else:
                idx, tab_name = self.__info_to_index(ent)
                indxs[tab_name].append(idx)

        for tab in tabs:
            dbcols: Tuple[DbCol, ...] = tabs[tab]
            model[tab] = DbTable(dbcols, set(indxs[tab]))

        self.__model = model

        return model


def model_from_ddl(ddl, *dialects, dialect=None):
    """Convert indexes and create statements to internal model, without needing a database connection."""
    return DDLHelper(ddl, *dialects).model()
