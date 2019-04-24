import logging
import sqlite3
import unittest

from notanorm import SqliteDb, DbRow

log = logging.getLogger(__name__)


class TestDb(unittest.TestCase):
    def __fname(self):
        return ":memory:"

    def setUpClass():
        log.setLevel(logging.DEBUG)

    def tearDownClass():
        log.setLevel(logging.INFO)

    def test_db_basic(self):
        fname = self.__fname()

        db = SqliteDb(fname)
        db.query("create table foo (bar)")
        db.query("insert into foo (bar) values (?)", "hi")

        self.assertEqual(db.query("select bar from foo")[0].bar, "hi")

    def test_db_count(self):
        fname = self.__fname()

        db = SqliteDb(fname)
        db.query("create table foo (bar)")
        db.query("insert into foo (bar) values (?)", "hi")

        self.assertEqual(db.count("foo"), 1)
        self.assertEqual(db.count("foo", bar="hi"), 1)
        self.assertEqual(db.count("foo", {"bar": "hi"}), 1)
        self.assertEqual(db.count("foo", {"bar": "ho"}), 0)

    def test_db_select(self):
        fname = self.__fname()

        db = SqliteDb(fname)
        db.query("create table foo (bar)")
        db.query("insert into foo (bar) values (?)", "hi")

        self.assertEqual(db.select("foo", bar="hi")[0].bar, "hi")
        self.assertEqual(db.select("foo", bar=None), [])
        self.assertEqual(db.select("foo", {"bar": "hi"})[0].bar, "hi")
        self.assertEqual(db.select("foo", {"bar": "ho"}), [])

    def test_db_class(self):
        fname = self.__fname()

        db = SqliteDb(fname)
        db.query("create table foo (bar)")
        db.query("insert into foo (bar) values (?)", "hi")

        class Foo:
            def __init__(self, bar=None):
                self.bar = bar

        obj = db.select_one("foo", bar="hi", __class=Foo)

        self.assertEqual(type(obj), Foo)

        db.register_class("foo", Foo)

        obj = db.select_one("foo", bar="hi")

        self.assertEqual(obj.bar, "hi")
        self.assertEqual(type(obj), Foo)

    def test_db_select_in(self):
        fname = self.__fname()

        db = SqliteDb(fname)
        db.query("create table foo (bar)")
        db.query("insert into foo (bar) values (?)", "hi")
        db.query("insert into foo (bar) values (?)", "ho")

        res = [DbRow({'bar': 'hi'}), DbRow({'bar': 'ho'})]

        self.assertEqual(db.select("foo", ["bar"], {"bar": ["hi", "ho"]}), res)

    def test_db_select_join(self):
        fname = self.__fname()

        db = SqliteDb(fname)
        db.query("create table foo (col, d)")
        db.query("create table baz (col, d)")
        db.query("insert into foo (col, d) values (?, ?)", "hi", "foo")
        db.query("insert into baz (col, d) values (?, ?)", "hi", "baz")
        res = db.select("foo inner join baz on foo.col=baz.col", ["baz.d"], {"foo.col": "hi"})
        self.assertEqual(res[0].d, "baz")

    def test_db_update_and_select(self):
        fname = self.__fname()

        db = SqliteDb(fname)

        db.query("create table foo (bar primary key, baz)")
        db.insert("foo", bar="hi", baz="ho")

        self.assertEqual(db.select("foo", bar="hi")[0].bar, "hi")

        # infers where clause from primary key
        db.update("foo", bar="hi", baz="up")

        self.assertEqual(db.select("foo", bar="hi")[0].baz, "up")

        # alternate interface where the first argument is a where clause dict
        db.update("foo", {"bar": "hi"}, baz="up2")

        self.assertEqual(db.select("foo")[0].baz, "up2")

        # alternate interface where the select is explicit
        self.assertEqual(db.select("foo", ['baz'])[0].baz, "up2")

    def test_db_upsert(self):
        fname = self.__fname()

        db = SqliteDb(fname)

        db.query("create table foo (bar primary key, baz)")
        db.insert("foo", bar="hi", baz="ho")

        self.assertEqual(db.select("foo", bar="hi")[0].bar, "hi")

        # updates
        db.upsert("foo", bar="hi", baz="up")

        self.assertEqual(db.select("foo", bar="hi")[0].baz, "up")

        # inserts
        db.upsert("foo", bar="lo", baz="down")

        self.assertEqual(db.select("foo", bar="hi")[0].baz, "up")
        self.assertEqual(db.select("foo", bar="lo")[0].baz, "down")

    def test_db_upsert_non_null(self):
        db = SqliteDb(self.__fname())

        db.query("create table foo (bar primary key, baz, bop)")
        db.insert("foo", bar="hi", baz="ho", bop="keep")

        # updates baz... but not bop to none
        db.upsert_non_null("foo", bar="hi", baz="up", bop=None)

        self.assertEqual(db.select_one("foo").baz, "up")
        self.assertEqual(db.select_one("foo").bop, "keep")

    def test_conn_retry(self):
        db = SqliteDb(":memory:")
        db.query("create table foo (x)")
        db._DbBase__conn_p.close()

        with self.assertRaises(sqlite3.ProgrammingError):
            db.query("create table bar (x)")

        db.retry_errors = (sqlite3.ProgrammingError, )
        db.query("create table bar (x)")

    def test_safedb_inmemorydb(self):
        # test that in memory db's are relatively safe

        db = SqliteDb(":memory:")

        db.query("create table foo (bar primary key)")
        db.query("insert into foo (bar) values (?)", 0)

        from multiprocessing.pool import ThreadPool

        def updater(i):
            db.query("update foo set bar = bar + ?", 1)

        pool = ThreadPool(processes=100)

        pool.map(updater, range(100))

        self.assertEqual(db.query("select bar from foo")[0].bar, 100)
