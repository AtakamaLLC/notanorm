from collections import defaultdict

import sqlglot
from sqlglot import parse, exp

from .model import DbType, DbCol, DbTable, DbIndex, DbModel

import logging

log = logging.getLogger(__name__)


class DDLHelper:
    TYPE_MAP = {
        exp.DataType.Type.INT: DbType.INTEGER,
        exp.DataType.Type.BIGINT: DbType.INTEGER,
        exp.DataType.Type.BOOLEAN: DbType.BOOLEAN,
        exp.DataType.Type.BINARY: DbType.BLOB,
        exp.DataType.Type.VARCHAR: DbType.TEXT,
        exp.DataType.Type.TEXT: DbType.TEXT,
        exp.DataType.Type.VARIANT: DbType.ANY,
        exp.DataType.Type.DECIMAL: DbType.DOUBLE,
        exp.DataType.Type.DOUBLE: DbType.DOUBLE,
        exp.DataType.Type.FLOAT: DbType.FLOAT
    }

    def __init__(self, ddl, *dialects):
        if not dialects:
            dialects = ("sqlite", "mysql")

        last_x = None

        for dialect in dialects:
            try:
                res = parse(ddl, read=dialect)
                self.ddl = res
                break
            except sqlglot.ParseError as ex:
                last_x = ex

        if last_x:
            raise last_x

    def __columns(self, ent):
        cols = []
        for col in ent.find_all(exp.ColumnDef):
            dbcol, is_prim = self.__info_to_model(col)
            if is_prim:
                self.primary = dbcol
            cols.append(dbcol)
        return tuple(cols)

    @staticmethod
    def __info_to_index(index):
        primary = index.find(exp.PrimaryKeyColumnConstraint)
        unique = index.args.get("unique")
        tab = index.args["this"].args["table"]
        cols = index.args["this"].args["columns"]
        field_names = [ent.name for ent in cols.find_all(exp.Column)]
        return DbIndex(fields=tuple(field_names), primary=bool(primary), unique=bool(unique)), tab.name

    @classmethod
    def __info_to_model(cls, info):
        typ = info.find(exp.DataType)
        typ = cls.TYPE_MAP[typ.this]
        notnull = info.find(exp.NotNullColumnConstraint)
        autoinc = info.find(exp.AutoIncrementColumnConstraint)
        is_primary = info.find(exp.PrimaryKeyColumnConstraint)
        default = info.find(exp.DefaultColumnConstraint)
        if default:
            default = default.find(exp.Literal)
            if default.is_string:
                default = "'" + str(default) + "'"
            else:
                default = str(default)
        size = 0
        fixed = False
        return DbCol(name=info.name, typ=typ, notnull=bool(notnull),
                     default=default, autoinc=bool(autoinc),
                     size=size, fixed=fixed), is_primary

    def model(self):
        """Get generic db model: dict of tables, each a dict of rows, each with type, unique, autoinc, primary."""
        model = DbModel()
        tabs = {}
        indxs = defaultdict(lambda: [])
        for ent in self.ddl:
            tab = ent.find(exp.Table)
            assert tab, f"unknonwn ddl entry {ent}"
            idx = ent.find(exp.Index)
            if not idx:
                tabs[tab.name] = self.__columns(ent)
            else:
                idx, tab_name = self.__info_to_index(ent)
                indxs[tab_name].append(idx)
        if self.primary.name:
            DbIndex(fields=(self.primary.name,), primary=True, unique=False)
            indxs[tab_name].append(idx)

        for tab in tabs:
            model[tab] = DbTable(tabs[tab], set(indxs[tab]))
        return model