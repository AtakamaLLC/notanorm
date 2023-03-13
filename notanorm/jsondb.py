import contextlib
import copy
import json
import os
import random
import threading
import base64
import time
from typing import Callable, Any

import sqlglot.errors

import notanorm.errors
from .base import DbBase, Op
from .errors import NoColumnError, TableNotFoundError, TableExistsError
from .model import DbType, DbCol, DbTable, DbIndex, DbModel
from .ddl_helper import DDLHelper
from sqlglot import parse, exp

import logging

log = logging.getLogger(__name__)


class QueryRes:
    rowcount = 0
    lastrowid = 0

    def __init__(self, parent: "JsonDb", *, generator=None):
        self.parent = parent
        self.generator = generator

    def execute(self, sql, parameters=()):
        self.parent._executeone(self, sql, parameters)

    def fetchall(self):
        ret = []
        if self.generator:
            for ent in self.generator:
                ret.append(ent)
        return ret

    def fetchone(self):
        if self.generator:
            try:
                return next(self.generator)
            except StopIteration:
                return None

    def close(self):
        self.generator and self.generator.close()


class JsonDb(DbBase):
    uri_name = "jsondb"
    use_pooled_locks = False
    generator_guard = False
    max_reconnect_attempts = 1
    retry_file_access = 5
    _parse_memo = {}
    default_values = " () values ()"
    closed = False

    def _begin(self, conn: "JsonDb"):
        self._tx.append(copy.deepcopy(self.__dat))

    def commit(self):
        self._tx = self._tx[0:-1]
        if not self._tx:
            self.__write()

    def serialize(self, val):
        if isinstance(val, (int, str, float, bool, type(None))):
            return val
        tname = type(val).__name__
        return {"type": tname, "value": getattr(self, "serialize_" + tname)(val)}

    def deserialize(self, obj):
        return getattr(self, "deserialize_" + obj["type"])(obj["value"])

    def serialize_bytes(self, val):
        return base64.b64encode(val).decode("utf8")

    def deserialize_bytes(self, val):
        return base64.b64decode(val)

    def __write(self):
        if not self.__is_mem:
            # tmp file is pid + thread, so they don't accumulate forever on crashes
            tmp = (
                self.__file
                + ".jtmp."
                + str(self.__pid)
                + "."
                + str(threading.get_ident())
            )
            dat = copy.deepcopy(self.__dat)
            for tab, rows in dat.items():
                for i, row in enumerate(rows):
                    for col, val in row.items():
                        dat[tab][i][col] = self.serialize(val)
            try:
                with self.__retry_fileop(lambda: open(tmp, "w")) as fp:
                    # not safe to retry this!
                    json.dump(dat, fp)
                self.__retry_fileop(lambda: os.replace(tmp, self.__file))
            except Exception as ex:
                with contextlib.suppress(Exception):
                    os.unlink(tmp)
                raise ex
        self.__dirty = False

    def __retry_fileop(self, func: Callable) -> Any:
        last_ex = None
        for _ in range(self.retry_file_access):
            try:
                return func()
            except PermissionError as ex:
                last_ex = ex
                sleep_time = self.reconnect_backoff_start
                sleep_time = random.uniform(sleep_time * 0.5, sleep_time * 1.5)
                time.sleep(sleep_time)
        raise last_ex

    def refresh(self):
        if not self.__is_mem:
            try:
                with self.__retry_fileop(lambda: open(self.__file, "r")) as fp:
                    dat = json.load(fp)
                    for tab, rows in dat.items():
                        for i, row in enumerate(rows):
                            for col, val in row.items():
                                if isinstance(val, dict):
                                    dat[tab][i][col] = self.deserialize(val)
                    self.__dat = dat
            except FileNotFoundError:
                self.__dirty = True

    def rollback(self):
        self.__dat = self._tx[-1]
        self._tx = self._tx[0:-1]

    def cursor(self):
        return QueryRes(self)

    def _executemany(self, cursor, sql: str):
        return self._executeone(cursor, sql, ())

    def _executeone(self, cursor, sql: str, parameters):
        if self.closed:
            raise notanorm.errors.DbClosedError

        todo = self._parse_memo.get(sql)
        if todo is None:
            todo = parse(sql, "sqlite")
            self._parse_memo[sql] = todo

        assert not todo[0].find(exp.AlterTable), "alter not needed for json"

        if todo:
            if todo[0].find(exp.Create):
                model = DDLHelper(todo, py_defaults=True).model()
                if model:
                    for k, v in model.items():
                        self.create_table(k, v)

        return self.execute_cursor_stmts(cursor, todo, parameters)

    def execute_cursor_stmts(self, cursor: QueryRes, stmts: list, parameters):
        for ent in stmts:
            # this is all we support
            op = ent.find(exp.Select)
            if op:
                cursor.generator = self.__op_select(op, parameters)
            self.__dirty = True
            op = ent.find(exp.Insert)
            if op:
                return self.__op_insert(cursor, op, parameters)
            op = ent.find(exp.Delete)
            if op:
                return self.__op_delete(cursor, op, parameters)
            op = ent.find(exp.Update)
            if op:
                return self.__op_update(cursor, op, parameters)
            op = ent.find(exp.Drop)
            if op:
                return self.__op_drop(cursor, op, parameters)

    def __op_drop(self, ret, op, parameters):
        tab = op.find(exp.Table)
        if tab:
            if op.args["kind"] == "index":
                found = False
                for tab_name, info in self.__model.items():
                    for idx in tuple(info.indexes):
                        if idx.name == tab.name:
                            self.drop_index_by_name(tab_name, idx.name)
                            found = True
                if not found:
                    raise notanorm.errors.OperationalError(
                        "index '%s' not found" % tab.name
                    )
            else:
                self.drop(tab.name)
        return ret

    def drop_index_by_name(self, table: str, index_name: str):
        mod = self.__get_tab_mod(table)
        idx = [i for i in mod.indexes if i.name == index_name]
        if not idx:
            raise notanorm.errors.OperationalError
        mod.indexes.discard(idx[0])

    def __op_insert(self, res, op, parameters):
        cols = []
        vals = []
        tab = op.find(exp.Table)
        scm = op.find(exp.Schema)
        mod = self.__get_tab_mod(tab.name)
        for col in scm.expressions:
            col_name = self.__col_name(col, mod)
            cols.append(col_name)
        parameters = list(parameters)
        valexp = op.find(exp.Values).expressions[0]
        for val in valexp.expressions:
            v = self.__val_from(val, parameters)
            vals.append(v)

        tdat = self.__dat.setdefault(tab.name, [])

        if mod:
            for i, col in enumerate(mod.columns):
                if col.name not in cols:
                    if col.default:
                        cols.append(col.name)
                        vals.append(col.default)
                    if col.autoinc:
                        nxt = max(r[col.name] for r in tdat) + 1 if tdat else 1
                        cols.append(col.name)
                        vals.append(nxt)
                if col.autoinc:
                    res.lastrowid = vals[i]

        row = {c: vals[i] for i, c in enumerate(cols)}
        self.__check_integ(tab.name, row, mod)
        tdat.append(row)
        res.rowcount = 1

    def __col_name(self, col, mod):
        col_name = col.name
        if mod:
            found = False
            for cmod in mod.columns:
                if cmod.name.lower() == col.name.lower():
                    col_name = cmod.name
                    found = True
            if not found:
                raise NoColumnError(col.name)
        return col_name

    def __check_integ_idx(self, tab, row, idx: DbIndex):
        check = tuple((f, row[f.name]) for f in idx.fields)
        rows = self.__get_tab_dat(tab)
        for ent in rows:
            if tuple((f, ent[f.name]) for f in idx.fields) == check:
                raise notanorm.errors.IntegrityError
        return False

    def __check_integ(self, tab, row, mod):
        if mod:
            for cmod in mod.indexes:
                if cmod.unique or cmod.primary:
                    self.__check_integ_idx(tab, row, cmod)
        return False

    def __op_update(self, ret, op, parameters):
        sets = {}
        parameters = list(parameters)
        tab = op.find(exp.Table)
        mod = self.__get_tab_mod(tab.name)
        for set_exp in op.expressions:
            col = set_exp.left
            col_name = self.__col_name(col, mod)
            val = self.__val_from(set_exp.right, parameters)
            sets[col_name] = val
        where = op.find(exp.Where)
        where_dict = self.__op_where(where, list(parameters))
        for row in self.__dat.setdefault(tab.name, []):
            if all(self.__op(row[k], v) for k, v in where_dict.items()):
                row.update(sets)
                ret.rowcount += 1
        return ret

    def __op_delete(self, ret, op, parameters):
        tab = op.find(exp.Table)
        where = op.find(exp.Where)
        where_dict = self.__op_where(where, list(parameters))
        new = []
        for row in self.__get_tab_dat(tab.name):
            if not all(v == row[k] for k, v in where_dict.items()):
                new.append(row)
                ret.rowcount += 1
        self.__dat[tab.name] = new

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
                val = self.__val_from(op.right, parameters)
                if isinstance(op.left, exp.Literal):  # pragma: no cover
                    raise RuntimeError("literal comparisons not supported yet")
                ret[op.left.name] = val
            elif isinstance(op, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                val = self.__val_from(op.right, parameters)
                cod = {exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}.get(
                    type(op)
                )
                val = Op(cod, val)
                if isinstance(op.left, exp.Literal):  # pragma: no cover
                    raise RuntimeError("literal comparisons not supported yet")
                ret[op.left.name] = val
        return ret

    @staticmethod
    def __val_from(op, parameters):
        if isinstance(op, exp.Placeholder):
            val = parameters.pop(0)
        elif isinstance(op, exp.Boolean):
            val = op.args["this"]
        else:
            val = op.output_name
            if op.is_int:
                val = int(val)
            elif op.is_number:
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
        rows = self.__get_tab_dat(tab)
        ord = op.find(exp.Order)
        if ord:
            rows = self.__sort(ord, rows)
        lim = op.find(exp.Limit)
        if lim:
            lim = self.__val_from(lim.expression, parameters)
        off = op.find(exp.Offset)
        if off:
            off = self.__val_from(off.expression, parameters)
        # bug in sqlglot flips this for sqlite parse
        if lim is not None and off is not None:
            (lim, off) = (off, lim)
        cntl = 0
        cnto = 0
        for row in rows:
            if all(self.__op(row[k], v) for k, v in where_dict.items()):
                cnto += 1
                if off is not None and cnto <= off:
                    continue
                cntl += 1
                if lim is not None and cntl > lim:
                    break
                yield {aliases.get(k, k): v for k, v in row.items()}

    def __sort(self, ord, rows):
        keys = []
        for col in ord.expressions:
            keys.append((col.this.output_name, col.args.get("desc")))

        def func(row):
            return tuple(-row[k[0]] if k[1] else row[k[0]] for k in keys)

        return sorted(rows, key=func)

    def __get_tab_dat(self, tab):
        try:
            return self.__dat[tab]
        except KeyError:
            if self.__model is None or tab in self.__model:
                return []
            raise TableNotFoundError(tab)

    def __get_tab_mod(self, tab):
        if self.__model is None:
            return None
        try:
            return self.__model[tab]
        except KeyError:
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

    def __init__(self, *args, model=None, ddl=None, **kws):
        super().__init__(*args, **kws)
        self.__pid = os.getpid()
        self.__dirty = False
        self._tx = []
        self.__dat = {}
        if ddl:
            model = DDLHelper(ddl, py_defaults=True).model()
        self.__model = model
        self.__file = args[0]
        self.__is_mem = self.__file == ":memory:"
        self.refresh()

    def __implicit_columns(self, table, *_):
        cols = []
        for k, v in self.__get_tab_dat(table)[0].items():
            cols.append(
                DbCol(
                    name=k,
                    typ=DbType.ANY,
                )
            )
        return tuple(cols)

    def model(self, no_capture=False):
        """Get sqlite db model: dict of tables, each a dict of rows, each with type, unique, autoinc, primary"""
        if self.__model is not None:
            return self.__model

        # implicit model from json
        model = DbModel()
        for k in self.__dat:
            cols = self.__implicit_columns(k)
            indxs = set()
            model[k] = DbTable(cols, indxs)
        self.__model = model
        return self.__model

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
            for idx in tdef.indexes:
                new_idxes.add(DbIndex(idx.fields, idx.unique, idx.primary, idx.name))
            new_tab = DbTable(columns=tuple(new_cols), indexes=new_idxes)
            new_mod[tab] = new_tab
        return new_mod

    def create_table(
        self, name, schema, ignore_existing=False, create_indexes: bool = True
    ):
        if self.__model is None:
            self.__model = DbModel()

        if name in self.__model:
            if not ignore_existing:
                raise TableExistsError
        else:
            if not create_indexes:
                schema = DbTable(schema.columns, set())
            self.__model[name] = schema

        if create_indexes:
            idxs = [
                DbIndex(idx.fields, idx.unique, idx.primary, os.urandom(16).hex())
                for idx in schema.indexes
            ]
            schema = DbTable(schema.columns, set(idxs))
            self.__model[name] = schema

    def _create_index(self, tab, index_name, idx: DbIndex):
        idx = DbIndex(idx.fields, idx.unique, idx.primary, index_name)
        self.__model[tab].indexes.add(idx)

    def _connect(self, *args, **kws):
        self.closed = False
        return self

    def _get_primary(self, table):
        if self.__model is None:
            return ()
        tab = self.__model[table]
        for idx in tab.indexes:
            if idx.primary:
                return tuple(fd.name for fd in idx.fields)
        return ()

    def version(self):
        return "1.0"

    def close(self):
        if self.__dirty:
            self.__write()
        self.closed = True

    @staticmethod
    def translate_error(exp):
        msg = str(exp)
        if isinstance(exp, sqlglot.errors.ParseError):
            return notanorm.errors.OperationalError(msg)
        return exp

    def rename(self, table_from, table_to):
        try:
            if self.__model is not None:
                self.__model[table_to] = self.__model.pop(table_from)
            self.__dat[table_to] = self.__dat.pop(table_from, [])
            self.clear_model_cache()
            self.__dirty = True
        except KeyError:
            raise notanorm.errors.TableNotFoundError

    def drop(self, table):
        try:
            self.__dat.pop(table, None)
            self.__dirty = True
            if self.__model is not None:
                del self.__model[table]
            self.clear_model_cache()
        except KeyError:
            raise notanorm.errors.TableNotFoundError
