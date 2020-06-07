import sqlite3

from .base import DbBase, DbRow


class SqliteDb(DbBase):
    placeholder = "?"

    # if you get one of these, you can retry
    retry_errors = (sqlite3.OperationalError, )

    # if you get one of these, it might be a duplicate key
    integrity_errors = (sqlite3.IntegrityError, )

    def __init__(self, *args, **kws):
        if args[0] == ":memory:":
            # never try to reconnect to memory dbs!
            self.retry_errors = ()
        super().__init__(*args, **kws)

    def __columns(self, table):
        tinfo = self.query("PRAGMA table_info(" + table + ")")
        if len(tinfo) == 0:
            raise KeyError(f"Table {table} not found in db {self}")

        res = self.query("PRAGMA index_list(" + table + ")")

        unique = set()
        autoinc = set()
        for row in res:
            res = self.query("PRAGMA index_info(" + row.name + ")")
            cols = [r.name for r in res]
            if row.seq:
                autoinc.add(cols[0])
            if row.unique:
                unique.add(tuple(sorted(cols)))

        clist = list()
        for col in tinfo:
            delattr(col, "cid")
            col.type = col.type.upper()
            clist.append(col.__dict__)
            col.unique = tuple([col.name]) in unique
            col.autoinc = col.name in autoinc  # or autoinc_pk and col.pk
        return clist

    def model(self):
        """Get sqlite db model: dict of tables, each a dict of rows, each with type, unique, autoinc, pk"""
        res = self.query("SELECT name, type from sqlite_master")
        model = {}
        for row in res:
            if row.type == "table":
                clist = self.__columns(row.name)
                for col in clist:
                    col["name"] = col["name"].lower()
                row.name = row.name.lower()
                model[row.name] = tuple(clist)
        # a model is a dict of tables, cols and types
        return model

    @staticmethod
    def _obj_factory(cursor, row):
        d = DbRow()
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def _connect(self, *args, **kws):
        kws["check_same_thread"] = False
        if "isolation_level" not in kws:
            kws["isolation_level"] = None
        conn = sqlite3.connect(*args, **kws)
        conn.row_factory = self._obj_factory
        return conn

    def _get_primary(self, table):
        info = self.query("pragma table_info(" + table + ");")
        prim = set()
        for x in info:
            if x.pk:
                prim.add(x.name)
        return prim
