import re
import threading
from collections import defaultdict
from functools import partial
from typing import Any, Callable

from .base import DbBase, DbRow, parse_bool, Op
from .errors import UnknownColumnError, TableNotFoundError
from .model import DbType, DbCol, DbTable, DbIndex, DbModel, DbIndexField
from .ddl_helper import model_from_ddl, DDLHelper
from sqlglot import parse, exp

import logging

log = logging.getLogger(__name__)


class QueryRes:
    rowcount = 0
    lastrowid = 0

    def __init__(self, *, generator=None):
        self.__gen = generator

    def fetchall(self):
        ret = []
        if self.__gen:
            for ent in self.__gen:
                ret.append(ent)
        return ret

    def fetchone(self):
        if self.__gen:
            try:
                return next(self.__gen)
            except StopIteration:
                return None

    def close(self):
        self.__gen and self.__gen.close()


class JsonDb(DbBase):
    uri_name = "jsondb"
    placeholder = "?"
    use_pooled_locks = False
    generator_guard = False
    max_reconnect_attempts = 1
    _parse_memo = {}

    def _begin(self, conn):
        conn._tx.append([])

    def execute(self, sql, parameters=(), _script=False, write=True, **kwargs):
        todo = self._parse_memo.get(sql)
        if todo is None:
            todo = parse(sql)
            self._parse_memo[sql] = todo

        if todo and todo[0].find(exp.Create or todo[0].find(exp.AlterTable)):
            model = DDLHelper(todo).model()
            if model:
                self.__model.update(model)
            return QueryRes()

        return self.__execute(todo, parameters)

    def __execute(self, stmts: list, parameters):
        for ent in stmts:
            # this is all we support
            op = ent.find(exp.Select)
            if op:
                return QueryRes(generator=self.__op_select(op, parameters))
            op = ent.find(exp.Insert)
            if op:
                return self.__op_insert(op, parameters)
            op = ent.find(exp.Delete)
            if op:
                return self.__op_delete(op, parameters)
            op = ent.find(exp.Update)
            if op:
                return self.__op_update(op, parameters)

    def __op_insert(self, op, parameters):
        cols = []
        vals = []
        tab = op.find(exp.Table)
        scm = op.find(exp.Schema)
        for col in scm.expressions:
            col_name = col.name
            mod = self.__model.get(tab.name)
            if mod:
                found = False
                for cmod in mod.columns:
                    if cmod.name.lower() == col.name.lower():
                        col_name = cmod.name
                        found = True
                if not found:
                    raise UnknownColumnError(col.name)
            cols.append(col_name)
        valexp = op.find(exp.Values).expressions[0]
        i = 0
        for val in valexp.expressions:
            if isinstance(val, exp.Placeholder):
                vals.append(parameters[i])
                i += 1
            else:
                vals.append(val.value)

        self.__dat.setdefault(tab.name, []).append(
            {c: vals[i] for i, c in enumerate(cols)}
        )
        ret = QueryRes()
        ret.rowcount = 1
        return ret

    def __op_delete(self, op, parameters):
        tab = op.find(exp.Table)
        where = op.find(exp.Where)
        where_dict = self.__op_where(where, list(parameters))
        new = []
        ret = QueryRes()
        for row in self.__get_table(tab.name):
            if not all(v == row[k] for k, v in where_dict.items()):
                new.append(row)
                ret.rowcount += 1
        self.__dat[tab.name] = new
        return ret
        pass

    def __op_where(self, op, parameters: list, *, ret=None) -> dict:
        ret = {} if ret is None else ret
        if op:
            op = op.this
            if isinstance(op, exp.And):
                self.__op_where(op.left, parameters, ret=ret)
            elif isinstance(op, exp.Is):
                assert isinstance(op.right, exp.Null)
                ret[op.left.name] = None
            elif isinstance(op, exp.EQ):
                val = self.__val_from(op, parameters)
                if isinstance(op.left, exp.Literal):
                    raise RuntimeError("literal comparisons not supported yet")
                ret[op.left.name] = val
            elif isinstance(op, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                val = self.__val_from(op, parameters)
                cod = {exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}.get(
                    type(op)
                )
                val = Op(cod, val)
                if isinstance(op.left, exp.Literal):
                    raise RuntimeError("literal comparisons not supported yet")
                ret[op.left.name] = val
        return ret

    @staticmethod
    def __val_from(op, parameters):
        if isinstance(op.right, exp.Placeholder):
            val = parameters.pop(0)
        else:
            val = op.right.output_name
            if op.right.is_int:
                val = int(val)
            elif op.right.is_number:
                val = float(val)
        return val

    def __op_select(self, op, parameters):
        cols = []
        params = list(parameters)
        aliases = {}
        for col in op.expressions:
            if isinstance(col, exp.Alias):
                aliases[col.this.name] = col.output_name
                col = col.this
            cols.append(col.name)
        tab = op.args["from"].args["expressions"][0].name
        where = op.find(exp.Where)
        where_dict = self.__op_where(where, params)
        for row in self.__get_table(tab):
            if all(self.__op(row[k], v) for k, v in where_dict.items()):
                yield {aliases.get(k, k): v for k, v in row.items()}

    def __get_table(self, tab):
        try:
            return self.__dat[tab]
        except KeyError:
            if tab in self.__model:
                return []
            raise TableNotFoundError(tab)

    def __op(self, k, v):
        if isinstance(v, Op):
            if v.op == ">":
                return k > v.val
            if v.op == ">=":
                return k >= v.val
            if v.op == "<":
                return k < v.val
            if v.op == "<=":
                return k <= v.val
            assert False, "not supported"
        return k == v

    def aggregate(self, *_a, **_k):
        raise NotImplementedError

    def count(self, table, where=None, *, _group_by=None, **kws):
        assert not (where and kws)
        assert not _group_by, "not supported"
        where = where or kws
        return sum(
            all(v == row[k] for k, v in where.items())
            for row in self.__dat.get(table, {})
        )

    def sum(self, table, col, where=None, *, _group_by=None, **kws):
        assert not (where and kws)
        assert not _group_by, "not supported"
        where = where or kws
        return sum(
            row[col] if all(v == row[k] for k, v in where.items()) else 0
            for row in self.__dat.get(table, {})
        )

    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self.__file = args[0]
        self.__dat = {}
        self.__model = DbModel()
        self.__is_mem = self.__file == ":memory:"

    @property
    def timeout(self):
        return self.__timeout

    @timeout.setter
    def timeout(self, val):
        self.__timeout = val

    def __columns(self, table, *_):
        cols = []
        for k, v in self.__get_table(table)[0].items():
            cols.append(
                DbCol(
                    name=k,
                    typ=DbType.ANY,
                )
            )
        return tuple(cols)

    def __indexes(self, table, no_capture):
        return set()

    def model(self, no_capture=False):
        """Get sqlite db model: dict of tables, each a dict of rows, each with type, unique, autoinc, primary"""
        model = DbModel()
        for k in self.__dat:
            cols = self.__columns(k)
            indxs = set()
            model[k] = DbTable(cols, indxs)
        return model

    @classmethod
    def _column_def(cls, col: DbCol, single_primary: str):
        coldef = cls.quote_key(col.name)
        typ = cls._type_map[col.typ]
        if typ:
            coldef += " " + typ
        return coldef

    @staticmethod
    def simplify_model(model: DbModel):
        new_mod = DbModel()
        for tab, tdef in model.items():
            tdef: DbTable
            new_cols = []
            for coldef in tdef.columns:
                # sizes & fixed-width specifiers are ignored in sqlite
                newcol = DbCol(
                    name=coldef.name,
                    typ=DbType.ANY,
                )
                new_cols.append(newcol)
            new_idxes = set()
            new_tab = DbTable(columns=tuple(new_cols), indexes=new_idxes)
            new_mod[tab] = new_tab
        return new_mod

    def create_table(
        self, name, schema, ignore_existing=False, create_indexes: bool = True
    ):
        pass

    def _connect(self, *args, **kws):
        return self

    def _get_primary(self, table):
        tab = self.__model[table]
        ret = []
        for idx in tab.indexes:
            if idx.primary:
                return tuple(fd.name for fd in idx.fields)
        return None

    def version(self):
        return "1.0"

    def _create_index(self, *_, **__):
        pass

    def close(self):
        pass
