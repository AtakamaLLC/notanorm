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

    @staticmethod
    def _obj_factory(cursor, row):
        d = DbRow()
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def _connect(self, *args, **kws):
        kws["check_same_thread"] = False
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
