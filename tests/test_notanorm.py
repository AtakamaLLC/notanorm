# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import logging
import multiprocessing
import sqlite3
import threading
from multiprocessing.pool import ThreadPool

import pytest

from notanorm import SqliteDb, DbRow, DbModel, DbCol, DbType, DbTable, DbIndex, DbBase

log = logging.getLogger(__name__)


PYTEST_REG = False


@pytest.fixture
def db_sqlite():
    db = SqliteDb(":memory:")
    yield db
    db.close()

@pytest.fixture
def db_sqlite_notmem(tmp_path):
    db = SqliteDb(str(tmp_path / "db"), timeout=1)
    yield db
    db.close()

def get_mysql_db():
    from notanorm import MySqlDb
    db = MySqlDb(read_default_file="~/.my.cnf")
    db.query("DROP DATABASE IF EXISTS test_db")
    db.query("CREATE DATABASE test_db")
    db.query("USE test_db")

    return MySqlDb(read_default_file="~/.my.cnf", db="test_db")

def cleanup_mysql_db(db):
    db.query("DROP DATABASE test_db")
    db.close()

@pytest.fixture
def db_mysql():
    db = get_mysql_db()
    yield db
    cleanup_mysql_db(db)

@pytest.fixture
def db_mysql_notmem(db_mysql):
    yield db_mysql

@pytest.fixture(name="db")
def db_fixture(request, db_name):
    yield request.getfixturevalue("db_" + db_name)

@pytest.fixture(name="db_notmem")
def db_notmem_fixture(request, db_name):
    yield request.getfixturevalue("db_" + db_name + "_notmem")

def pytest_generate_tests(metafunc):
    """Converts user-argument --db to fixture parameters."""

    global PYTEST_REG               # pylint: disable=global-statement
    if not PYTEST_REG:
        if any(db in metafunc.fixturenames for db in ("db", "db_notmem")):
            db_names = metafunc.config.getoption("db", [])
            db_names = db_names or ["sqlite"]
            for mark in metafunc.definition.own_markers:
                if mark.name == "db":
                    db_names = set(mark.args).intersection(set(db_names))
                    break
            db_names = sorted(db_names)         # xdist compat
            metafunc.parametrize("db_name", db_names, scope="function")


def test_db_basic(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")
    assert db.query("select bar from foo")[0].bar == "hi"


def test_db_count(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    assert db.count("foo") == 1
    assert db.count("foo", bar="hi") == 1
    assert db.count("foo", {"bar": "hi"}) == 1
    assert db.count("foo", {"bar": "ho"}) == 0


def test_db_select(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    assert db.select("foo", bar="hi")[0].bar == "hi"
    assert db.select("foo", bar=None) == []
    assert db.select("foo", {"bar": "hi"})[0].bar == "hi"
    assert db.select("foo", {"bar": "ho"}) == []


def test_db_row_obj__dict__(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    assert db.select_one("foo").__dict__ == {"bar": "hi"}
    assert db.select_one("foo")._asdict() == {"bar": "hi"}


def test_db_class(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    class Foo:
        def __init__(self, bar=None):
            self.bar = bar

    obj = db.select_one("foo", bar="hi", __class=Foo)

    assert type(obj) == Foo

    db.register_class("foo", Foo)

    obj = db.select_one("foo", bar="hi")

    assert obj.bar == "hi"
    assert type(obj) == Foo


def test_db_select_in(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="hi")
    db.insert("foo", bar="ho")

    res = [DbRow({'bar': 'hi'}), DbRow({'bar': 'ho'})]

    assert db.select("foo", ["bar"], {"bar": ["hi", "ho"]}) == res


def test_db_select_join(db):
    db.query("create table foo (col text, d text)")
    db.query("create table baz (col text, d text)")
    db.insert("foo", col="hi", d="foo")
    db.insert("baz", col="hi", d="baz")
    res = db.select("foo inner join baz on foo.col=baz.col", ["baz.d"], {"foo.col": "hi"})
    assert res[0].d == "baz"


def test_db_update_and_select(db):
    db.query("create table foo (bar varchar(32) not null primary key, baz text)")
    db.insert("foo", bar="hi", baz="ho")

    assert db.select("foo", bar="hi")[0].bar == "hi"

    # infers where clause from primary key
    db.update("foo", bar="hi", baz="up")

    assert db.select("foo", bar="hi")[0].baz == "up"

    # alternate interface where the first argument is a where clause dict
    db.update("foo", {"bar": "hi"}, baz="up2")

    assert db.select("foo")[0].baz == "up2"

    # alternate interface where the select is explicit
    assert db.select("foo", ['baz'])[0].baz, "up2"


def test_db_upsert(db):
    db.query("create table foo (bar varchar(32) not null primary key, baz text)")
    db.insert("foo", bar="hi", baz="ho")

    assert db.select("foo", bar="hi")[0].bar == "hi"

    # updates
    db.upsert("foo", bar="hi", baz="up")

    assert db.select("foo", bar="hi")[0].baz == "up"

    # inserts
    db.upsert("foo", bar="lo", baz="down")

    assert db.select("foo", bar="hi")[0].baz == "up"
    assert db.select("foo", bar="lo")[0].baz == "down"


def test_db_upsert_non_null(db):
    db.query("create table foo (bar varchar(32) not null primary key, baz text, bop text)")
    db.insert("foo", bar="hi", baz="ho", bop="keep")

    # updates baz... but not bop to none
    db.upsert_non_null("foo", bar="hi", baz="up", bop=None)

    assert db.select_one("foo").baz == "up"
    assert db.select_one("foo").bop == "keep"


def test_model(db):
    model = DbModel({
        "foo": DbTable(columns=(
            DbCol("auto", typ=DbType.INTEGER, autoinc=True, notnull=True),
            DbCol("blob", typ=DbType.BLOB),
            DbCol("tex", typ=DbType.TEXT, notnull=True),
            DbCol("siz3v", typ=DbType.TEXT, size=3, fixed=False),
            DbCol("siz3", typ=DbType.TEXT, size=3, fixed=True),
            DbCol("flt", typ=DbType.FLOAT),
            DbCol("dbl", typ=DbType.DOUBLE),
        ), indexes=tuple([
            DbIndex(fields=("auto", ), primary=True)
        ]))
    })
    db.create_model(model)
    check = db.model()
    assert check == model


def test_conn_retry(db):
    db.query("create table foo (x integer)")
    db._DbBase__conn_p.close()                  # pylint: disable=no-member
    db.max_reconnect_attempts = 1
    with pytest.raises(Exception):
        db.query("create table foo (x integer)")
    db.max_reconnect_attempts = 2
    db.query("create table bar (x integer)")


@pytest.mark.db("sqlite")
def test_safedb_inmemorydb(db):
    # test that in memory db's are relatively safe
    db.query("create table foo (bar primary key)")
    db.query("insert into foo (bar) values (?)", 0)

    def updater(_i):
        db.query("update foo set bar = bar + ?", 1)

    pool = ThreadPool(processes=100)

    pool.map(updater, range(100))

    assert db.query("select bar from foo")[0].bar == 100

def get_db(db_name, db_conn):
    if db_name == "sqlite":
        return SqliteDb(*db_conn[0], **db_conn[1])
    else:
        from notanorm import MySqlDb
        return MySqlDb(*db_conn[0], **db_conn[1])

def cleanup_db(db):
    if db.__class__.__name__ == "MySqlDb":
        cleanup_mysql_db(db)

def _test_upsert_i(db_name, i, db_conn, mod):
    db = get_db(db_name, db_conn)

    with db.transaction() as db:
        db.upsert("foo", bar=i % mod, baz=i)
        row = db.select_one("foo", bar=i % mod)
        db.update("foo", bar=i % mod, cnt=row.cnt + 1)

def test_upsert_multiprocess(db_name, db_notmem, tmp_path):
    db = db_notmem
    db.query("create table foo (bar integer primary key, baz integer, cnt integer default 0)")

    num = 22
    ts = []
    mod = 5

    for i in range(0, num):
        currt = multiprocessing.Process(target=_test_upsert_i, args=(db_name, i, db.connection_args, mod), daemon=True)
        ts.append(currt)

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    log.debug(db.select("foo"))
    assert len(db.select("foo")) == mod
    for i in range(mod):
        ent = db.select_one("foo", bar=i)
        assert ent.cnt == int(num / mod) + (i < num % mod)

# for some reqson mysql seems connect in a way that causes multiple object to have the same underlying connection
# todo: maybe using the native "mysql connector" would enable fixing this
@pytest.mark.db("sqlite")
def test_upsert_threaded_multidb(db_notmem, db_name):
    db = db_notmem
    db.query("create table foo (bar integer primary key, baz integer, cnt integer default 0)")

    num = 22
    ts = []
    mod = 5

    def _upsert_i(up_i, up_db_name, db_conn, up_mod):
        with get_db(up_db_name, db_conn) as _db:
            _db.upsert("foo", bar=up_i % up_mod, baz=up_i)
            with _db.transaction():
                row = _db.select_one("foo", bar=up_i % up_mod)
                _db.update("foo", bar=up_i % up_mod, cnt=row.cnt + 1)

    for i in range(0, num):
        currt = threading.Thread(target=_upsert_i, args=(i, db_name, db.connection_args, mod), daemon=True)
        ts.append(currt)

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    log.debug(db.select("foo"))
    assert len(db.select("foo")) == mod
    for i in range(mod):
        ent = db.select_one("foo", bar=i)
        assert ent.cnt == int(num / mod) + (i < num % mod)

def test_transactions_any_exc(db):
    class TestExc(Exception):
        pass

    db.query("CREATE table foo (bar integer primary key)")
    db.insert("foo", bar=5)
    with pytest.raises(TestExc):
        with db.transaction() as db:
            db.delete_all("foo")
            raise TestExc()
    assert db.select("foo")[0].bar == 5

def test_transactions_deadlock(db):
    def trans_thread(orig_db):
        with orig_db.transaction() as ins_db:
            for ins in range(50, 100):
                ins_db.insert("foo", bar=ins)

    db.query("CREATE table foo (bar integer primary key)")

    thread = threading.Thread(target=trans_thread, args=(db,), daemon=True)
    thread.start()
    for i in range(50):
        db.insert("foo", bar=i)
    thread.join()


# for some reqson mysql seems connect in a way that causes multiple object to have the same underlying connection
# todo: maybe using the native "connector" would enable fixing this
@pytest.mark.db("sqlite")
def test_transaction_fail_on_begin(db_notmem: "DbBase", db_name):
    db1 = db_notmem
    db2 = get_db(db_name, db1.connection_args)

    db1.max_reconnect_attempts = 1

    with db2.transaction():
        with pytest.raises(sqlite3.OperationalError, match=r".*database.*is locked"):
            with db1.transaction():
                pass
