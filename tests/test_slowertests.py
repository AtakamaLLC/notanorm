# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import logging
import multiprocessing
import sqlite3
import threading
from multiprocessing.pool import Pool as ProcessPool
from unittest.mock import patch

import pytest

import notanorm.errors as err
from notanorm import SqliteDb, DbBase, ReconnectionArgs
from notanorm.connparse import open_db

from tests.conftest import cleanup_mysql_db
from tests.test_notanorm import create_and_fill_test_db

log = logging.getLogger(__name__)


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
    multiproc_coverage()

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


def multiproc_coverage():
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()


def upserty(uri, i):
    multiproc_coverage()

    db = open_db(uri)
    for row in db.select_gen("foo"):
        if row.bar == 0:
            for row in db.select_gen("foo"):
                db.insert("oth", bar=i * 100 + row.bar, baz=0)
        else:
            db.update("oth", bar=i * 100 + row.bar, baz=1)
    return i


def test_generator_proc(db_notmem):
    db = db_notmem

    uri = db.uri
    log.debug("using uri" + uri)

    proc_num = 4
    mult = 4
    create_and_fill_test_db(db, proc_num * mult)
    create_and_fill_test_db(db, 0, "oth")

    db.close()

    with ProcessPool(processes=proc_num) as pool:
        import functools

        func = functools.partial(upserty, uri)

        expected = list(range(proc_num * mult))

        assert pool.map(func, range(proc_num * mult)) == expected

        expected = list(
            i * 100 + j for i in range(proc_num * mult) for j in range(proc_num * mult)
        )

        db = open_db(uri)
        assert [row.bar for row in db.select("oth")] == expected


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
