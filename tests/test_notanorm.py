# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import logging
import multiprocessing
import copy
import sqlite3
import threading
import time
from multiprocessing.pool import ThreadPool
from typing import Generator

import pytest

import notanorm.errors
from notanorm import SqliteDb, DbRow, DbModel, DbCol, DbType, DbTable, DbIndex, DbBase

import notanorm.errors as err

log = logging.getLogger(__name__)


PYTEST_REG = False


@pytest.fixture
def db_sqlite():
    db = SqliteDb(":memory:")
    yield db
    db.close()


@pytest.fixture
def db_sqlite_notmem(tmp_path):
    db = SqliteDb(str(tmp_path / "db"))
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
    db._DbBase__closed = False
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

    global PYTEST_REG  # pylint: disable=global-statement
    if not PYTEST_REG:
        if any(db in metafunc.fixturenames for db in ("db", "db_notmem")):
            db_names = metafunc.config.getoption("db", [])
            db_names = db_names or ["sqlite"]
            for mark in metafunc.definition.own_markers:
                if mark.name == "db":
                    db_names = set(mark.args).intersection(set(db_names))
                    break
            db_names = sorted(db_names)  # xdist compat
            metafunc.parametrize("db_name", db_names, scope="function")


def test_db_basic(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")
    assert db.query("select bar from foo")[0].bar == "hi"


def test_db_delete(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="hi")
    db.delete("foo", bar="hi")
    assert not db.select("foo")


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

    ret = db.select_one("foo")
    assert ret.__dict__ == {"bar": "hi"}
    assert ret._asdict() == {"bar": "hi"}
    assert ret.copy() == ret._asdict()
    assert copy.copy(ret) == ret._asdict()


def test_db_row_obj_case(db):
    db.query("create table foo (Bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    ret = db.select_one("foo")
    assert ret["bar"] == "hi"
    assert ret["BAR"] == "hi"
    assert ret.bar == "hi"
    assert ret.BaR == "hi"
    assert "Bar" in ret.keys()
    assert "bar" not in ret.keys()
    assert "bar" in ret


def test_db_order(db):
    db.query("create table foo (bar integer)")
    for i in range(10):
        db.insert("foo", bar=i)

    fwd = db.select("foo", order_by=["bar"])

    assert db.select("foo", order_by="bar desc") == list(reversed(fwd))
    assert next(iter(db.select("foo", order_by="bar desc"))).bar == 9


def test_db_select_gen_ex(db):
    db.query("create table foo (bar integer)")
    db.insert("foo", bar=1)
    db.insert("foo", bar=2)

    # works normally
    generator = db.select_gen("foo", order_by="bar")
    assert next(iter(generator)).bar == 1

    # it's a generator
    generator = db.select_gen("foox", order_by="bar")
    assert isinstance(generator, Generator)

    # raises error correctly
    with pytest.raises(notanorm.errors.TableNotFoundError):
        for _ in generator:
            raise ValueError

    # errors pass up correctly
    generator = db.select_gen("foo", order_by="bar")
    with pytest.raises(ValueError):
        for _ in generator:
            raise ValueError

    class Foo:
        def __init__(self, bar=None):
            self.bar = bar
            assert False, "not a good foo"

    # bad class
    db.register_class("foo", Foo)
    generator = db.select_gen("foo", order_by="bar")
    with pytest.raises(AssertionError):
        for _ in generator:
            pass


def test_db_row_obj_iter(db):
    db.query("create table foo (Bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    ret = db.select_one("foo")
    for k in ret:
        assert k == "Bar"

    assert "Bar" in ret


def test_db_row_obj_integer_access(db):
    db.query("create table foo (a text, b text, c text)")
    db.insert("foo", a="a", b="b", c="c")

    ret = db.select_one("foo")

    assert ret[0] == "a"
    assert ret[1] == "b"
    assert ret[2] == "c"

    assert len(list(ret.keys())) == 3
    assert len(list(ret.values())) == 3
    assert len(list(ret.items())) == 3


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

    res = [DbRow({"bar": "hi"}), DbRow({"bar": "ho"})]

    assert db.select("foo", ["bar"], {"bar": ["hi", "ho"]}) == res
    assert db.select("foo", bar=["hi", "ho"]) == res


def test_db_select_join(db):
    db.query("create table foo (col text, d text)")
    db.query("create table baz (col text, d text)")
    db.insert("foo", col="hi", d="foo")
    db.insert("baz", col="hi", d="baz")
    res = db.select(
        "foo inner join baz on foo.col=baz.col", ["baz.d"], {"foo.col": "hi"}
    )
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
    assert db.select("foo", ["baz"])[0].baz, "up2"


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

    # no-op
    db.upsert("foo", bar="hi")

    # no-op
    db.update("foo", bar="hi")


def test_db_insert_no_vals(db):
    db.query("create table foo (bar integer default 1)")
    db.insert("foo")


def test_db_upsert_non_null(db):
    db.query(
        "create table foo (bar varchar(32) not null primary key, baz text, bop text)"
    )
    db.insert("foo", bar="hi", baz="ho", bop="keep")

    # updates baz... but not bop to none
    db.upsert_non_null("foo", bar="hi", baz="up", bop=None)

    assert db.select_one("foo").baz == "up"
    assert db.select_one("foo").bop == "keep"


def test_model_create(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("auto", typ=DbType.INTEGER, autoinc=True, notnull=True),
                    DbCol("blob", typ=DbType.BLOB),
                    DbCol("tex", typ=DbType.TEXT, notnull=True),
                    DbCol("siz3v", typ=DbType.TEXT, size=3, fixed=False),
                    DbCol("siz3", typ=DbType.TEXT, size=3, fixed=True),
                    DbCol("flt", typ=DbType.FLOAT),
                    DbCol("dbl", typ=DbType.DOUBLE),
                ),
                indexes={
                    DbIndex(fields=("auto",), primary=True),
                    DbIndex(fields=("flt",), unique=True),
                },
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert check == model


def test_model_create_nopk(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("inty", typ=DbType.INTEGER),),
                indexes={DbIndex(fields=("inty",), primary=False)},
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert check == model


def test_model_cmp(db):
    model1 = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("Auto", typ=DbType.INTEGER, autoinc=True, notnull=True),
                ),
                indexes={DbIndex(fields=("Auto",), primary=True)},
            )
        }
    )
    model2 = DbModel(
        {
            "FOO": DbTable(
                columns=(
                    DbCol("autO", typ=DbType.INTEGER, autoinc=True, notnull=True),
                ),
                indexes={DbIndex(fields=("autO",), primary=True)},
            )
        }
    )

    assert model1["foo"].columns[0] == model2["FOO"].columns[0]
    assert model1 == model2


def test_conn_retry(db):
    db.query("create table foo (x integer)")
    db._conn_p.close()  # pylint: disable=no-member
    db.max_reconnect_attempts = 1
    with pytest.raises(Exception):
        db.query("create table foo (x integer)")
    db.max_reconnect_attempts = 2
    db.query("create table bar (x integer)")
    db.max_reconnect_attempts = 99


def test_conn_reopen(db):
    db.query("create table foo (x integer)")
    db.close()
    assert db.closed
    with pytest.raises(Exception):
        db.query("create table foo (x integer)")


def test_multi_close(db):
    db.close()
    db.close()

    class VeryClose(SqliteDb):
        def __init__(self):
            self.close()

    # always safe to call close, even if not initialized
    VeryClose()


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


# todo: maybe using the native "mysql connector" would enable fixing this
#       but really, why would mysql allow blocking transactions like sqlite?
@pytest.mark.db("sqlite")
def test_upsert_multiprocess(db_name, db_notmem, tmp_path):
    db = db_notmem
    db.query(
        "create table foo (bar integer primary key, baz integer, cnt integer default 0)"
    )

    num = 22
    ts = []
    mod = 5

    for i in range(0, num):
        currt = multiprocessing.Process(
            target=_test_upsert_i,
            args=(db_name, i, db.connection_args, mod),
            daemon=True,
        )
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
    db.query(
        "create table foo (bar integer primary key, baz integer, cnt integer default 0)"
    )

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
        currt = threading.Thread(
            target=_upsert_i, args=(i, db_name, db.connection_args, mod), daemon=True
        )
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


def test_upsert_thready_one(db_notmem):
    db = db_notmem

    db.query("create table foo (bar integer primary key, baz integer)")

    failed = False
    num = 100
    mod = 7

    def upsert_i(db, i):
        try:
            db.upsert("foo", bar=str(i % mod), baz=i)
        except Exception as e:
            nonlocal failed
            failed = True
            log.error("failed to upsert: %s", repr(e))

    ts = []
    for i in range(0, num):
        currt = threading.Thread(target=upsert_i, args=(db, i), daemon=True)
        ts.append(currt)

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    assert not failed
    assert len(db.select("foo")) == mod


def test_select_gen_not_lock(db: DbBase):
    db.query("CREATE table foo (bar integer primary key)")

    with db.transaction():
        for i in range(500):
            db.insert("foo", bar=i)

    thread_result = []

    ev1 = threading.Event()
    ev2 = threading.Event()

    def slow_q():
        nonlocal thread_result
        for row in db.select_gen("foo", order_by="bar desc"):
            thread_result.append(row.bar)
            ev1.set()
            ev2.wait()

    thread = threading.Thread(target=slow_q, daemon=True)
    thread.start()

    ev1.wait()
    with db.transaction():
        for i in range(500, 600):
            db.insert("foo", bar=i)

    ev2.set()

    start = time.time()
    fast_result = []
    for ent in db.select_gen("foo", order_by="bar desc"):
        fast_result.append(ent.bar)
    end = time.time()

    thread.join()

    assert thread_result == list(reversed(range(500)))
    assert fast_result == list(reversed(range(600)))
    # this prevents a slow machine from falsely succeeding
    assert (end - start) < 1


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


@pytest.mark.db("sqlite")
def test_readonly_fail(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="y1")
    db.query("PRAGMA query_only=ON;")
    with pytest.raises(err.DbReadOnlyError):
        db.insert("foo", bar="y2")


def test_timeout_rational(db_notmem):
    db = db_notmem
    assert db.max_reconnect_attempts > 1
    assert db.timeout > 1
    assert db.timeout < 60


def test_db_more_than_one(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar=1)
    db.insert("foo", bar=1)
    with pytest.raises(err.MoreThanOneError):
        assert db.select_one("foo")
    with pytest.raises(err.MoreThanOneError):
        assert db.select_one("foo", bar=1)
