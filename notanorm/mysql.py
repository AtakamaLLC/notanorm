import MySQLdb
import MySQLdb.cursors

from .base import DbBase

# driver for mysql


class MySqlDb(DbBase):
    placeholder = "%s"
    retry_errors = (MySQLdb.OperationalError, )
    integrity_errors = (MySQLdb.IntegrityError, )

    def _connect(self, *args, **kws):
        conn = MySQLdb.connect(*args, **kws)
        conn.autocommit(True)
        conn.cursor().execute("SET SESSION sql_mode = 'ANSI';")
        return conn

    def _cursor(self, conn):
        return conn.cursor(MySQLdb.cursors.DictCursor)

    def quote_key(self, key):
        return '`' + key + '`'

    def _get_primary(self, table):
        info = self.query("SHOW KEYS FROM " + table + " WHERE Key_name = 'PRIMARY'")
        prim = set()
        for x in info:
            prim.add(x.Column_name)
        return prim
