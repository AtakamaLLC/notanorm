# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import copy
import logging
import multiprocessing
import sqlite3
import threading
import time
from multiprocessing.pool import ThreadPool, Pool as ProcessPool
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

import notanorm.errors
import notanorm.errors as err
from notanorm import SqliteDb, DbRow, DbBase, DbType, ReconnectionArgs
from notanorm.connparse import open_db, parse_db_uri

from tests.conftest import cleanup_mysql_db

log = logging.getLogger(__name__)


def test_db_basic(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")
    assert db.query("select bar from foo")[0].bar == "hi"


def test_db_delete(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="hi")
    db.delete("foo", bar="hi")
    assert not db.select("foo")


def test_db_version(db):
    assert db.version()


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


def test_db_op_gt(db):
    db.query("create table foo (bar integer)")
    db.insert("foo", bar=3)
    db.insert("foo", {"bar": 4})
    db.insert("foo", bar=5)

    assert db.select_one("foo", bar=notanorm.Op(">", 4)).bar == 5

    assert db.select_one("foo", bar=notanorm.Op("<", 4)).bar == 3

    assert {r.bar for r in db.select("foo", bar=notanorm.Op(">=", 4))} == {4, 5}

    assert {r.bar for r in db.select("foo", bar=notanorm.Op("<=", 4))} == {3, 4}


def test_op_internals() -> None:
    assert notanorm.Op(">", 4) == notanorm.Op(">", 4)
    assert notanorm.Op(">", 4) != notanorm.Op("=", 4)
    assert notanorm.Op(">", 4) != notanorm.Op(">", 5)
    assert notanorm.Op(">", 4) != object()

    assert repr(notanorm.Op(">", 4)) == "Op('>', 4)"


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


def test_db_select_gen_close_fetch(db):
    create_and_fill_test_db(db, 5)
    with pytest.raises(ValueError):
        generator = db.select_gen("foo")
        one = generator
        next(one)
        # cursor has 5 rows, raising here should close the cursor
        raise ValueError

    # mysql and other db's will fail here if we don't close the cursor
    assert db.select_any_one("foo")


def test_db_select_any_one(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 0, "oth")
    assert db.select_any_one("foo").bar is not None

    assert db.select_any_one("oth") is None
    assert db.select_one("oth") is None


def test_db_tab_not_found(db):
    db.query("create table foo (bar integer)")
    with pytest.raises(notanorm.errors.TableNotFoundError):
        db.select("foox")
    with pytest.raises(notanorm.errors.TableNotFoundError):
        db.execute("drop table foox")


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


def test_db_select_explicit_field_map(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="hi")
    db.insert("foo", bar="ho")

    res = [DbRow({"x": "hi"}), DbRow({"x": "ho"})]

    assert db.select("foo", fields={"x": "bar"}, bar=["hi", "ho"]) == res


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
    print("sqlite3 version", sqlite3.sqlite_version)

    db.query("create table foo (bar varchar(32) not null primary key, baz text)")
    db.insert("foo", bar="hi", baz="ho")

    assert db.select("foo", bar="hi")[0].bar == "hi"

    # infers where clause from primary key
    db.update("foo", bar="hi", baz="up")

    assert db.select("foo", bar="hi")[0].baz == "up"

    # alternate interface where the first argument is a where clause dict
    db.update("foo", {"bar": "hi"}, baz="up2")

    assert db.select("foo")[0].baz == "up2"

    # alternate interface where the first argument is a where clause dict (reversed primary)
    db.update("foo", {"baz": "up2"}, bar="yo")

    assert db.select("foo")[0].bar == "yo"

    # alternate interface where the first argument is a where clause dict and second is a update dict
    db.update("foo", {"baz": "up2"}, {"baz": "hi"})

    assert db.select("foo")[0].baz == "hi"

    # alternate interface where the select is explicit
    assert db.select("foo", ["baz"])[0].baz, "hi"


def test_db_upsert(db_sqlup):
    db = db_sqlup
    db.query("create table foo (bar varchar(32) not null primary key, baz text)")
    db.insert("foo", bar="hi", baz="ho")

    assert db.select("foo", bar="hi")[0].bar == "hi"

    # updates
    db.upsert("foo", bar="hi", baz="up")

    assert db.select("foo", bar="hi")[0].baz == "up"

    # inserts
    ret = db.upsert("foo", bar="lo", baz="down")

    if db_sqlup.uri_name == "sqlite":
        assert ret.lastrowid

    assert db.select("foo", bar="hi")[0].baz == "up"
    assert db.select("foo", bar="lo")[0].baz == "down"

    # no-op
    db.upsert("foo", bar="hi")

    # update everything
    db.upsert_all("foo", baz="all")

    assert db.select("foo", bar="lo")[0].baz == "all"
    assert db.select("foo", bar="hi")[0].baz == "all"

    # where clause doen't update, only inserts
    db.upsert("foo", {"bar": "new"}, baz="baznew")

    assert db.select("foo", bar="new")[0].baz == "baznew"

    db.upsert("foo", {"bar": "new"}, _insert_only={"baz": "bazdef"})

    assert db.select("foo", bar="new")[0].baz == "baznew"

    db.upsert("foo", bar="n2", baz="uponly", _insert_only={"baz": "i2"})

    assert db.select("foo", bar="n2")[0].baz == "i2"

    db.upsert("foo", bar="n2", baz="uponly", _insert_only={"baz": "i2"})

    assert db.select("foo", bar="n2")[0].baz == "uponly"


def test_db_insert_no_vals(db):
    db.query("create table foo (bar integer default 1)")
    db.insert("foo")
    assert db.select_one("foo").bar == 1


def test_db_insert_lrid(db):
    db.query("create table foo (bar integer auto_increment primary key)")
    ret = db.insert("foo")
    assert ret.lastrowid


def test_db_upsert_lrid(db):
    db.query("create table foo (bar integer auto_increment primary key, baz integer)")
    ret = db.upsert("foo", bar=1, baz=2)
    assert ret.lastrowid


def test_tab_exists(db):
    db.query("create table foo (bar integer)")
    with pytest.raises(err.TableExistsError):
        db.query("create table foo (bar integer)")


def test_no_primary(db):
    db.query("create table foo (bar text)")
    with pytest.raises(err.UnknownPrimaryError):
        db.upsert("foo", bar=1)


def test_db_upsert_non_null(db):
    db.query(
        "create table foo (bar varchar(32) not null primary key, baz text, bop text)"
    )
    db.insert("foo", bar="hi", baz="ho", bop="keep")

    # updates baz... but not bop to none
    db.upsert_non_null("foo", bar="hi", baz="up", bop=None)

    assert db.select_one("foo").baz == "up"
    assert db.select_one("foo").bop == "keep"


def test_conn_retry(db):
    db.query("create table foo (x integer)")
    db._conn_p.close()  # pylint: disable=no-member
    db.max_reconnect_attempts = 1
    with pytest.raises(Exception):
        db.query("create table foo (x integer)")
    db.max_reconnect_attempts = 2
    db.query("create table bar (x integer)")
    db.max_reconnect_attempts = 99


def test_reconnect_cb(tmp_path):
    def cb():
        nonlocal cb_called
        cb_called = True
        raise Exception("ensure that the cb exception is caught")

    cb_called = False
    recon_args = ReconnectionArgs(max_reconnect_attempts=2, failure_callback=cb)
    db = SqliteDb(str(tmp_path / "db"), reconnection_args=recon_args, timeout=0)
    try:
        db.query("create table foo (x integer)")
        assert not cb_called
        with patch.object(db, "_executeone", side_effect=err.DbConnectionError):
            with pytest.raises(err.DbConnectionError):
                db.query("create table bar (x integer)")
    finally:
        db.close()


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
    create_and_fill_test_db(db, 0)

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
    print("sqlite3 version", sqlite3.sqlite_version)

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

    create_and_fill_test_db(db, 0)

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
    print("sqlite3 version", sqlite3.sqlite_version)

    db1 = db_notmem
    db2 = get_db(db_name, db1.connection_args)

    db1.max_reconnect_attempts = 1

    with db2.transaction():
        with pytest.raises(sqlite3.OperationalError, match=r".*database.*is locked"):
            with db1.transaction():
                pass


def test_readonly_fail(db, db_name: str):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="y1")

    if db_name == "sqlite":
        db.query("PRAGMA query_only=ON;")
    elif db_name == "mysql":
        db.query("SET SESSION TRANSACTION READ ONLY;")
    else:
        raise NotImplementedError

    with pytest.raises(err.DbReadOnlyError):
        db.insert("foo", bar="y2")


@pytest.mark.db("sqlite")
def test_collation(db):
    def collate(v1, v2):
        return 1 if v1 > v2 else -1 if v1 < v2 else 0

    db._conn().create_collation("COMP", collate)
    db.query("create table foo (bar text collate COMP)")
    # collation + multi-thread mode = possible deadlock
    db.use_collation_locks = True

    with db.transaction():
        for i in range(5000):
            db.insert("foo", bar=str(i))

    evt = threading.Event()

    def select_gen():
        evt.wait()
        log.warning("gen-start")
        for row in db.select_gen("foo", order_by="bar"):
            log.info("gen: %s", row.bar)
        log.warning("gen-end")

    def select_one():
        evt.wait()
        log.warning("one-start")
        for i in range(5000):
            row = db.select_one("foo", bar=str(i))
            log.info("one: %s", row.bar)
        log.warning("one-end")

    select_gen_thread = threading.Thread(target=select_gen, daemon=True)
    select_one_thread = threading.Thread(target=select_one, daemon=True)

    select_gen_thread.start()
    select_one_thread.start()

    evt.set()

    select_gen_thread.join()
    select_one_thread.join()


def test_missing_column(db):
    db.query("create table foo (bar text)")
    with pytest.raises(err.NoColumnError):
        db.insert("foo", nocol="y2")


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


def test_db_integ(db):
    if isinstance(db, SqliteDb):
        db.query("pragma foreign_keys=on;")
    db.query("create table foo (bar integer primary key)")
    db.query("create table zop (bar integer, foreign key (bar) references foo (bar))")
    db.insert("foo", bar=1)
    db.insert("zop", bar=1)
    with pytest.raises(err.IntegrityError):
        db.insert("zop", bar=2)


def test_db_annoying_col_names(db):
    db.query('create table "group" (bar integer primary key, "group" integer)')
    db.insert("group", bar=1, group=1)
    db.update("group", bar=1, group=1)
    db.upsert("group", bar=1, group=1)
    db.select("group", group=1)


@pytest.mark.db("mysql")
def test_mysql_op_error(db):
    # test that connection errors don't happen when you do stuf like this
    with pytest.raises(notanorm.errors.OperationalError):
        db.query("create table foo (bar text primary key);")


def test_syntax_error(db):
    with pytest.raises(notanorm.errors.OperationalError):
        db.query("create table fo()o (bar text primary key);")


@pytest.mark.db("sqlite")
def test_no_extra_close(db):
    db.query("create table foo (bar integer primary key);")
    db.insert("foo", bar=1)

    mok = MagicMock()

    def newx(*_, **__):
        # mock cursor
        ret = MagicMock()
        ret.fetchall = lambda: []
        ret.fetchone = lambda: None
        ret.close = mok
        return ret

    db.execute = newx
    db.select("foo")
    list(db.select_gen("foo"))
    mok.close.assert_not_called()


@pytest.mark.db("sqlite")
def test_any_col(db):
    db.query("create table foo (bar whatever primary key);")
    db.insert("foo", bar=1)
    assert db.model()["foo"].columns[0].typ == DbType.ANY


def test_uri_parse():
    from notanorm import MySqlDb

    typ, args, kws = parse_db_uri("sqlite:file.db")
    assert typ == SqliteDb
    assert args == ["file.db"]
    assert kws == {}

    # escaping works
    typ, args, kws = parse_db_uri("mysql:host=whatever,password=\\,\\=::yo")
    assert typ == MySqlDb
    assert kws == {"host": "whatever", "password": ",=::yo"}

    typ, args, kws = parse_db_uri("sqlite://file.db?timeout=5.1")
    assert typ == SqliteDb
    assert args == ["file.db"]
    assert kws == {"timeout": 5.1}

    typ, args, kws = parse_db_uri("mysql:host=localhost,port=45")

    assert typ == MySqlDb
    assert kws == {"host": "localhost", "port": 45}

    typ, args, kws = parse_db_uri("mysql:localhost,port=45")

    assert typ == MySqlDb
    assert kws == {"host": "localhost", "port": 45}

    typ, args, kws = parse_db_uri("mysql://localhost?port=45")

    assert typ == MySqlDb
    assert kws == {"host": "localhost", "port": 45}

    typ, args, kws = parse_db_uri(
        "mysql://localhost?use_unicode=false&autocommit=true&buffered=FaLsE&compress=TrUe"
    )
    assert typ == MySqlDb
    assert kws == {
        "host": "localhost",
        "use_unicode": False,
        "autocommit": True,
        "buffered": False,
        "compress": True,
    }

    typ, args, kws = parse_db_uri("sqlite://file.db?check_same_thread=false")
    assert typ == SqliteDb
    assert args == ["file.db"]
    assert kws == {"check_same_thread": False}

    with pytest.raises(ValueError):
        parse_db_uri("sqlite://file.db?check_same_thread=not_a_bool")


def test_open_db():
    db = open_db("sqlite://:memory:")
    db.execute("create table foo (bar)")


def test_cap_exec(db):
    with db.capture_sql(execute=True) as stmts:
        db.execute("create table foo(inty integer)")
        db.insert("foo", inty=4)

    assert stmts[0] == ("create table foo(inty integer)", ())
    if db.uri_name == "sqlite":
        assert stmts[1] == ('insert into "foo"("inty") values (?)', (4,))


def test_exec_script(db):
    db.executescript(
        """
        create table foo (x integer);
        create table bar (y integer);
    """
    )
    db.insert("foo", x=1)
    db.insert("bar", y=2)


def create_and_fill_test_db(db, num, tab="foo", **fds):
    if not fds:
        fds = {
            "bar": "integer primary key",
            "baz": "integer not null",
            "cnt": "integer default 0",
        }
    fd_sql = ",".join(db.quote_key(fd) + " " + typ for fd, typ in fds.items())
    db.query(f"CREATE table {tab} ({fd_sql})")
    for ins in range(num):
        vals = {
            nm: ins
            if typ in ("integer primary key", "integer")
            else 0
            if typ == "integer not null"
            else None
            for nm, typ in fds.items()
        }
        db.insert(tab, **vals)


@pytest.mark.db("sqlite")
def test_sqlite_unsafe_gen(db_notmem):
    db = db_notmem
    create_and_fill_test_db(db, 5)
    db.generator_guard = True
    with pytest.raises(err.UnsafeGeneratorError):
        for row in db.select_gen("foo"):
            db.upsert("foo", bar=row.bar, baz=row.baz + 1)

    # ok, select inside select
    for row in db.select_gen("foo"):
        db.select("foo")

    for row in db.select_gen("foo"):
        list(db.select_gen("foo"))


@pytest.mark.db("sqlite")
def test_sqlite_guard_thread(db_notmem):
    db = db_notmem
    create_and_fill_test_db(db, 5)
    db.generator_guard = True
    cool = False
    event = threading.Event()

    def updatey():
        nonlocal cool
        try:
            db.upsert("foo", bar=row.bar, baz=row.baz + 1)
            cool = True
        finally:
            event.set()

    for row in db.select_gen("foo"):
        threading.Thread(target=updatey, daemon=True).start()
        assert event.wait(3)
        break

    assert cool


def upserty(uri, i):
    db = open_db(uri)
    try:
        db.generator_guard = True
        for row in db.select_gen("foo"):
            db.upsert("foo", bar=row.bar, baz=row.baz + 1)
        # this is ok: we passed
        return i
    except err.UnsafeGeneratorError:
        # this is ok: we created a consistent error
        return -1


def test_subq(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    assert len(db.select("foo", bar=db.subq("oth", ["bar"], bar=[1, 3]), baz=0)) == 2


def test_nested_subq(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    assert len(
        db.select(
            db.subq(db.subq("foo", bar=db.subq("oth", ["bar"], bar=[1, 3])), baz=0)
        )
    )


def test_subq_limited_fields_join(db):
    # use weird field name on purpose
    create_and_fill_test_db(db, 5, select="integer primary key", baz="integer")
    create_and_fill_test_db(db, 5, "oth")

    # no limit
    db.select(db.join(db.subq("foo", select=[1, 3]), "oth", baz="baz"))

    # subq limited to just 'select', so we can't join on 'baz', even tho you can without the limit
    with pytest.raises(err.UnknownColumnError):
        db.select(
            db.join(db.subq("foo", fields=["select"], select=[1, 3]), "oth", baz="baz")
        )

    # 'select' as a field name works tho
    db.select(
        db.join(db.subq("foo", fields=["select"], select=[1, 3]), "oth", select="bar")
    )


def test_joinq_ambig_unknown_col_join(db):
    create_and_fill_test_db(db, 5, "a")
    create_and_fill_test_db(db, 5, "b")
    create_and_fill_test_db(db, 5, "c")

    with pytest.raises(err.UnknownColumnError):
        # nested join... selecting "bar" automagically selects "bar" from the underlying table
        # but it's ambiguous, so it should not succeed
        db.select(db.join(db.join("a", "b", bar="bar"), "c", bar="bar"))

    # being specific is fine
    db.select(db.join(db.join("a", "b", bar="bar"), "c", a__bar="bar"))

    # unknown col error 'bzz'
    with pytest.raises(err.UnknownColumnError):
        db.select(db.join(db.join("a", "b", bar="bar"), "c", bzz="bar"))


def test_joinq_non_ambig_col(db):
    create_and_fill_test_db(db, 5, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 5, "b", bid="integer primary key", cid="integer")
    create_and_fill_test_db(db, 5, "c", cid="integer primary key", did="integer")

    assert db.select(db.join(db.join("a", "b", bid="bid"), "c", cid="cid"))


def test_joinq_model_changes(db_sqlite_notmem):
    db = db_sqlite_notmem

    create_and_fill_test_db(db, 5, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 5, "b", bid="integer primary key", cid="integer")

    assert db.select(db.join("a", "b", bid="bid"))

    file = db.uri.split(":", 1)[1]

    dbx = sqlite3.connect(file)

    # other process has changed things!
    dbx.execute("drop table b")
    dbx.execute("create table b (b_id integer primary_key, c_id integer)")
    dbx.execute("insert into b values(1, 1)")

    with pytest.raises(err.OperationalError):
        db.select(db.join("a", "b", bid="b_id"))

    db.clear_model_cache()

    db.select(db.join("a", "b", bid="b_id"))


def test_joinq_left(db):
    create_and_fill_test_db(db, 5, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 3, "b", bid="integer primary key", cid="integer")

    assert "left" in db.left_join("a", "b", bid="bid").sql

    rows = db.select(db.left_join("a", "b", bid="bid"))

    assert len(rows) == 5
    assert len(list(row for row in rows if row.cid is not None)) == 3


def test_joinq_right(db_mysql):
    # sqlite does not support right joins
    db = db_mysql
    create_and_fill_test_db(db, 3, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 5, "b", bid="integer primary key", cid="integer")

    assert "right" in db.right_join("a", "b", bid="bid").sql

    rows = db.select(db.right_join("a", "b", bid="bid"))

    assert len(rows) == 5
    assert len(list(row for row in rows if row.aid is not None)) == 3


def test_select_subq(db):
    create_and_fill_test_db(db, 5)
    assert len(db.select(db.subq("foo", bar=[1, 3]), bar=1)) == 1


def test_join_simple(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    assert len(db.select(db.join("foo", "oth", bar="bar"), {"foo.bar": 1})) == 1


def test_join_subqs(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    assert (
        len(
            db.select(
                db.join("foo", db.subq("oth", bar=[1, 3]), bar="bar"), {"foo.bar": 1}
            )
        )
        == 1
    )


def test_subqify_join(db):
    create_and_fill_test_db(db, 5, "x", xid="integer primary key", yid="integer")
    create_and_fill_test_db(db, 5, "y", yid="integer primary key", zid="integer")
    create_and_fill_test_db(
        db, 5, "z", zid="integer primary key", oth="integer default 0"
    )
    j1 = db.join("x", "y", yid="yid")
    assert len(db.select(j1, xid=1)) == 1
    sub = db.subq(j1, xid=[1, 3])
    assert len(db.select(sub, xid=1)) == 1
    row = db.select_one(sub, xid=1)
    assert row.zid == 1
    j2 = db.join(sub, "z", zid="zid")
    assert len(db.select(j2, xid=1)) == 1


def test_multi_join_nested_left_right(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    create_and_fill_test_db(db, 5, "thrd")
    create_and_fill_test_db(db, 5, "mor")
    j1 = db.join("foo", "oth", bar="bar")
    log.debug("j1: %s", j1.sql)
    assert len(db.select(j1, {"foo.bar": 1})) == 1
    j2a = db.join(j1, "thrd", _on={"foo.bar": "thrd.bar"})
    log.debug("j2a: %s", j2a.sql)
    assert len(db.select(j2a, {"foo.bar": 1})) == 1
    j2b = db.join("thrd", j1, bar="foo.bar")
    log.debug("j2b: %s", j2b.sql)
    assert len(db.select(j2b, {"thrd.bar": 1})) == 1
    j3a = db.join(j2a, "mor", _on={"foo.bar": "mor.bar"})
    log.debug("j3a: %s", j3a.sql)
    assert len(db.select(j3a, {"foo.bar": 1})) == 1
    j3b = db.join("mor", j2b, bar="thrd.bar")
    log.debug("j3b: %s", j3b.sql)
    assert len(db.select(j3b, {"thrd.bar": 1})) == 1


def test_join_explicit_mappings(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    j1 = db.subq(
        db.join("foo", "oth", bar="bar", fields={"z": "oth.bar", "x": "foo.bar"})
    )
    j2 = db.join("foo", j1, bar="x")
    assert db.select_one(j2, z=1).x == 1


def test_join_explicit_fields(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    j1 = db.subq(db.join("foo", "oth", bar="bar", fields=["oth.bar"]))
    j2 = db.join("foo", j1, bar="bar")
    assert db.select_one(j2, bar=1).bar == 1


def test_join_2_subqs_same_tab(db):
    create_and_fill_test_db(db, 5)
    s1 = db.subq("foo", bar=[1, 2, 3])
    s2 = db.subq("foo", bar=[2, 3, 4])
    jn = db.join(s1, s2, bar="bar")
    assert len(db.select(jn)) == 2


def test_join_2_subqs(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    s1 = db.subq("foo", bar=[1, 2, 3])
    s2 = db.subq("foo", bar=[2, 3, 4])
    jn = db.join(s1, s2, bar="bar")
    assert len(db.select(jn)) == 2


def test_multi_join_auto_left(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    create_and_fill_test_db(db, 5, "thrd")

    j1 = db.join("foo", "oth", bar="bar")
    j2a = db.join("thrd", j1, bar="foo.bar")

    assert len(db.select(j2a, {"foo.bar": 1})) == 1
    assert len(db.select(j2a, foo__bar=1)) == 1


def test_join_fd_names(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    # this creates a mapping of fields
    j1 = db.join("foo", "oth", bar="bar")
    # this uses the mapping and just picks one
    row = db.select_one(j1, bar=1)
    log.debug("row: %s", row)
    assert row.bar == 1
    assert row.foo__baz == 0
    row.foo__baz = 4
    assert row.foo__baz == 4
    assert row["foo.baz"] == 4


def test_where_or(db):
    create_and_fill_test_db(db, 5)
    db.update("foo", bar=3, baz=2)
    assert len(db.select("foo", _where=[{"bar": 1}, {"baz": 2}])) == 2
    db.delete("foo", [{"bar": 1}, {"baz": 2}])
    assert len(db.select("foo")) == 3


def test_del_raises(db):
    create_and_fill_test_db(db, 5)
    db.delete("foo", bar=2)
    assert len(db.select("foo")) == 4
    with pytest.raises(ValueError):
        db.delete("foo")
    with pytest.raises(ValueError):
        db.delete("foo", {"bar": 3}, baz=0)


def test_generator_proc(db_notmem):
    db = db_notmem

    uri = db.uri
    log.debug("using uri" + uri)

    create_and_fill_test_db(db, 20)
    db.close()

    proc_num = 4

    pool = ProcessPool(processes=proc_num)

    import functools

    func = functools.partial(upserty, uri)

    expected = list(range(proc_num * 2))

    if db.uri_name == "sqlite":
        expected = [-1] * proc_num * 2

    assert pool.map(func, range(proc_num * 2)) == expected


def test_db_direct_clone(db_notmem):
    db = db_notmem.clone()
    db.query("CREATE table foo (bar integer primary key)")
    db_notmem.insert("foo", bar=1)


def test_db_uri_clone(db_notmem):
    db = open_db(db_notmem.uri)
    db.query("CREATE table foo (bar integer primary key)")
    db_notmem.insert("foo", bar=1)


def test_quote_key(db: DbBase) -> None:
    from_inst = db.quote_key("key")
    assert from_inst
    assert from_inst != "key"

    # Check that quote_key is available as a classmethod on all impls
    assert type(db).quote_key("key") == from_inst


def test_db_larger_types(db):
    db.query("create table foo (bar mediumblob)")
    # If mediumblob is accidentally translated to blob, the max size in mysql is
    # 2**16. If mysql is running in strict mode, the insert will fail.
    # Otherwise, the comparison will fail.
    db.query("insert into foo (bar) values (%s)" % db.placeholder, b"a" * (2**16 + 4))
    assert db.query("select bar from foo")[0].bar == b"a" * (2**16 + 4)


def test_limit_rowcnt(db: DbBase):
    create_and_fill_test_db(db, 5)
    assert len(db.select("foo", _limit=3)) == 3
    assert len(db.select("foo", _limit=3, order_by="bar desc")) == 3
    assert len(list(db.select_gen("foo", _limit=1, order_by="bar desc"))) == 1
    assert len(list(db.select_gen("foo", _limit=0, order_by="bar desc"))) == 0

    if db.uri_name != "mysql":
        # mysql doesn't support limits in where subqueries.  probably you shouldn't use them then, if you want stuff to be compat
        assert (
            len(db.select("foo", bar=db.subq("foo", ["bar"], bar=[1, 2, 3], _limit=2)))
            == 2
        )


def test_limit_offset(db: DbBase):
    create_and_fill_test_db(db, 5)
    assert len(db.select("foo", _limit=(1, 3))) == 3
    assert db.select("foo", _limit=(2, 2), order_by="bar")[0].bar == 2
    assert len(db.select("foo", _limit=(1, 3), order_by="bar desc")) == 3
    assert list(db.select_gen("foo", _limit=(0, 3), order_by="bar desc"))[0].bar == 4
    assert len(list(db.select_gen("foo", _limit=(4, 0), order_by="bar desc"))) == 0


def test_type_translation_mysql_dialect(db: DbBase):
    # mysql-compatible types that can be used with sqlite
    schema = """
        CREATE table foo (
            a text,
            b longtext,
            c mediumtext,
            d integer,
            e tinyint,
            f smallint,
            g bigint,
            h int primary key
        )
        """

    schema_model = notanorm.model_from_ddl(schema, "mysql")

    db.execute(schema)
    exec_model = db.model()

    assert db.simplify_model(exec_model) == db.simplify_model(schema_model)


def test_clob_invalid():
    schema = """
        CREATE table no_clob (
            a clob
        )
        """

    with pytest.raises(ValueError):
        _ = notanorm.model_from_ddl(schema, "sqlite")


def test_rename_drop(db):
    db.execute("create table foo (bar int)")
    db.insert("foo", bar=1)
    db.rename("foo", "foo2")
    db.insert("foo2", bar=1)
    assert sum(r.bar for r in db.select("foo2")) == 2
    with pytest.raises(err.TableNotFoundError):
        db.drop("foo")
    db.drop("foo2")
    with pytest.raises(err.TableNotFoundError):
        db.select("foo2")


def test_drop_index(db):
    db.execute("create table foo (bar integer)")
    db.execute("create unique index ix_foo_uk on foo(bar)")
    idx = db.model()["foo"].indexes.pop()
    assert idx.name
    db.drop_index("foo", idx)
    assert not db.model()["foo"].indexes
