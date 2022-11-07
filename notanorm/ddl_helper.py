from collections import defaultdict
from typing import Tuple, Dict, List

import sqlglot
from sqlglot import parse, exp

from .model import DbType, DbCol, DbIndex, DbTable, DbModel

import logging

log = logging.getLogger(__name__)


class DDLHelper:
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
            dialects = ("sqlite", "mysql")

        last_x = None

        for dialect in dialects:
            if dialect == "mysql":
                # bug sqlglot doesn't support varbinary
                ddl = ddl.replace("varbinary", "binary")
            try:
                res = parse(ddl, read=dialect)
                self.ddl = res
                break
            except sqlglot.ParseError as ex:
                last_x = ex

        if last_x:
            raise last_x

    def __columns(self, ent) -> Tuple[Tuple[DbCol, ...], DbIndex]:
        cols: List[DbCol] = []
        primary = None
        for col in ent.find_all(exp.Anonymous):
            if col.name == "primary key":
                primary_list = [ent.name for ent in col.find_all(exp.Column)]
                primary = DbIndex(fields=tuple(primary_list), primary=True, unique=False)
        for col in ent.find_all(exp.ColumnDef):
            dbcol, is_prim = self.__info_to_model(col)
            if is_prim:
                primary = DbIndex(fields=(col.name,), primary=True, unique=False)
            cols.append(dbcol)
        return tuple(cols), primary

    @staticmethod
    def __info_to_index(index):
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
    def __info_to_model(cls, info) -> Tuple[DbCol, bool]:
        typ = info.find(exp.DataType)
        fixed = typ.this in cls.FIXED_MAP
        typ = cls.TYPE_MAP[typ.this]
        notnull = info.find(exp.NotNullColumnConstraint)
        autoinc = info.find(exp.AutoIncrementColumnConstraint)
        is_primary = info.find(exp.PrimaryKeyColumnConstraint)
        default = info.find(exp.DefaultColumnConstraint)
        expr = info.args["kind"].args.get("expressions")
        if expr:
            size = int(expr[0].this)
        else:
            size = 0
        if default:
            lit = default.find(exp.Literal)
            if default.find(exp.Null):
                default = None
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
        model = DbModel()
        tabs: Dict[str, Tuple[DbCol, ...]] = {}
        indxs = defaultdict(lambda: [])
        for ent in self.ddl:
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
        return model


def model_from_ddl(ddl, dialect="mysql"):
    """Convert indexes and create statements to internal model, without needing a database connection."""
    return DDLHelper(ddl, dialect).model()
